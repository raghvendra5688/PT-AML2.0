"""
tabpfn_predict_embed.py

Steps 1–4 of the best-model analysis pipeline (no SHAP):

  1. Load best TabPFN ablation model + scaler
  2. Load BeatAML train/test pickles; build feature subset
  3. Predictions + 95 % CI on train and test sets  [cached to CSV]
  4. Embedding-proxy importance via transformer embeddings  [cached to CSV]

Outputs written to OUT_DIR (../Results/tabpfn/best_model_analysis):
  feature_subset.csv
  test_predictions_with_CI.csv
  predictions_scatter.pdf
  CI_width_distribution.pdf
  embedding_attention_importance.csv
  embedding_importance_bar_train.pdf
  embedding_importance_bar_test.pdf
  embedding_norm_distribution.pdf
"""

import os
import sys
import gc
import time
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misc import load_model, calculate_regression_metrics

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)


# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH  = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_baseline_all_features_Drug_Embed_best.pk"
)
SCALER_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_baseline_all_features_Drug_Embed_scaler.pk"
)
TRAIN_PATH   = "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
TEST_PATH    = "../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
ABLATION_CSV = "../Data/ablation_feature_columns.csv"
OUT_DIR      = "../Results/tabpfn/best_model_analysis"

REF_CSV = "../Results/tabpfn/ablation/baseline_all_features_Drug_Embed_predictions.csv"

# ── Constants ──────────────────────────────────────────────────────────────────

CI_LOW, CI_HIGH = 0.025, 0.975
TOP_N  = 30
SEED   = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXCLUDE_COLS = [
    "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample",
    "dbgap_rnaseq_sample", "inhibitor", "type", "status", "paper_inclusion",
    "min_conc", "max_conc", "intercept", "beta", "beta_z", "beta_p",
    "aic", "pearson_chisq", "deviance", "converged",
    "ic10", "ic25", "ic50", "ic75", "ic90",
    "all_gt_50", "all_lt_50", "curve_type",
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)

def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def save_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    _log(f"  [saved CSV]  {path}")

def save_plot(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    _log(f"  [saved plot] {path}")


# ── Predict helper ─────────────────────────────────────────────────────────────

def _predict_chunked(model, X: pd.DataFrame, chunk_size: int = 35_000, **kwargs):
    """Predict in chunks to cap peak GPU memory."""
    if len(X) <= chunk_size:
        return model.predict(X, **kwargs)
    chunks = [model.predict(X.iloc[i:i + chunk_size], **kwargs)
              for i in range(0, len(X), chunk_size)]
    if isinstance(chunks[0], (list, tuple)):
        return [np.concatenate([c[k] for c in chunks]) for k in range(len(chunks[0]))]
    return np.concatenate(chunks)


# ── Step 1: Load model ────────────────────────────────────────────────────────

def load_best_model():
    _log("\n── Step 1: Loading best TabPFN ablation model ───────────────────")
    model  = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    _log(f"  Model  : {MODEL_PATH}")
    _log(f"  n_estimators={model.n_estimators}  fit_mode={model.fit_mode}"
         f"  n_features_in_={model.n_features_in_}  device={model.device}")
    _log(f"  y_train_mean_={model.y_train_mean_:.4f}  y_train_std_={model.y_train_std_:.4f}")
    _log(f"  Scaler : {SCALER_PATH}")
    _log("  [DONE] Step 1.")
    return model, scaler


# ── Step 2: Load data ──────────────────────────────────────────────────────────

def get_feature_subset(train_df: pd.DataFrame):
    """Replicate the exact feature selection from tabpfn_ablation_study.py."""
    all_columns  = train_df.columns.tolist()
    column_types = train_df.dtypes.tolist()

    feature_metadata   = pd.read_csv(ABLATION_CSV)
    feature_groups_raw = {
        t: g["Column_Label"].tolist()
        for t, g in feature_metadata.groupby("Type")
    }

    tsne_in_data  = [c for c in feature_groups_raw.get("Tsne", []) if c in all_columns]
    metadata_cols = ["CID"] + [
        all_columns[i] for i in range(len(all_columns))
        if str(column_types[i]) in ("object", "category")
    ]
    nan_cols = [c for c in all_columns if train_df[c].isnull().any()]
    metadata_cols = list(dict.fromkeys(
        metadata_cols + nan_cols + EXCLUDE_COLS + tsne_in_data
    ))

    all_feature_cols    = sorted(set(all_columns) - set(metadata_cols) - {"auc"})
    drug_embedding_cols = [c for c in feature_groups_raw.get("Drug_Embed", [])
                           if c in all_feature_cols]

    feature_groups = {k: v for k, v in feature_groups_raw.items()
                      if k not in ("Tsne", "Drug_Embed", "Drug_PC")}
    feature_groups["Clinical_CellType_Module"] = (
        feature_groups.pop("Clinical", []) +
        feature_groups.pop("CellType", []) +
        feature_groups.pop("Module", [])
    )

    all_patient_cols = []
    for g in sorted(feature_groups):
        cols = [c for c in feature_groups[g] if c in all_feature_cols]
        all_patient_cols.extend(cols)

    return list(drug_embedding_cols) + all_patient_cols


def load_data(scaler):
    _log("\n── Step 2: Loading and preprocessing data ───────────────────────")
    train_df = pd.read_pickle(TRAIN_PATH, compression="zip")
    test_df  = pd.read_pickle(TEST_PATH,  compression="zip")
    _log(f"  Train raw: {train_df.shape}  Test raw: {test_df.shape}")

    feature_subset = get_feature_subset(train_df)
    _log(f"  Feature subset: {len(feature_subset)} columns")

    meta_cols  = [c for c in train_df.columns if c not in feature_subset and c != "auc"]
    meta_train = train_df[meta_cols].copy()
    meta_test  = test_df[[c for c in meta_cols if c in test_df.columns]].copy()

    X_train = pd.DataFrame(scaler.transform(train_df[feature_subset]), columns=feature_subset)
    X_test  = pd.DataFrame(scaler.transform(test_df[feature_subset]),  columns=feature_subset)
    y_train = train_df["auc"].to_numpy().flatten()
    y_test  = test_df["auc"].to_numpy().flatten()

    _log(f"  X_train {X_train.shape}  y_train [{y_train.min():.1f}, {y_train.max():.1f}]"
         f"  mean={y_train.mean():.2f}")
    _log(f"  X_test  {X_test.shape}   y_test  [{y_test.min():.1f}, {y_test.max():.1f}]"
         f"  mean={y_test.mean():.2f}")

    pd.DataFrame({"feature": feature_subset}).to_csv(
        os.path.join(OUT_DIR, "feature_subset.csv"), index=False)
    _log("  [DONE] Step 2.")
    return X_train, y_train, X_test, y_test, meta_train, meta_test, feature_subset


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check_predictions(model, X_test, y_test, n=100):
    _log("\n── Sanity check: aggregate agreement with reference predictions ──")
    if not os.path.exists(REF_CSV):
        _log(f"  [SKIP] Reference not found: {REF_CSV}")
        return
    ref = pd.read_csv(REF_CSV, sep="\t")
    label_diffs = np.abs(y_test[:n] - ref["labels"].values[:n])
    if label_diffs.max() > 1e-3:
        _log(f"  [ERROR] True-label mismatch (max diff={label_diffs.max():.4f}) — row order wrong")
        return
    _log(f"  Row-order PASSED (max label diff={label_diffs.max():.2e})")

    y_pred_n  = model.predict(X_test.iloc[:n])
    ref_preds = ref["predictions"].values[:n]
    true_n    = y_test[:n]

    mae_new = float(np.mean(np.abs(y_pred_n  - true_n)))
    mae_ref = float(np.mean(np.abs(ref_preds - true_n)))
    r_new   = float(np.corrcoef(y_pred_n,  true_n)[0, 1])
    r_ref   = float(np.corrcoef(ref_preds, true_n)[0, 1])
    r_agree = float(np.corrcoef(y_pred_n, ref_preds)[0, 1])
    bias    = float(np.mean(y_pred_n - ref_preds))

    _log(f"  {'metric':<20s}  {'ref':>8}  {'new':>8}")
    _log(f"  {'MAE vs true':<20s}  {mae_ref:8.3f}  {mae_new:8.3f}")
    _log(f"  {'Pearson r vs true':<20s}  {r_ref:8.4f}  {r_new:8.4f}")
    _log(f"  {'Pearson r (new vs ref)':<20s}  {'—':>8}  {r_agree:8.4f}")
    _log(f"  {'Mean bias (new-ref)':<20s}  {'—':>8}  {bias:+8.3f}")

    ok = r_agree > 0.95
    _log(f"  {'PASS' if ok else 'WARNING'} — "
         f"new vs ref agreement: r={r_agree:.4f}  "
         f"({'≥0.95 expected' if ok else '<0.95 — investigate'})")


# ── Step 3: Predictions + CI ──────────────────────────────────────────────────

def predict_with_confidence(model, X_train, y_train, X_test, y_test, meta_test):
    _log("\n── Step 3: Predictions + 95 % confidence intervals ──────────────")

    pred_csv = os.path.join(OUT_DIR, "test_predictions_with_CI.csv")
    if os.path.exists(pred_csv):
        _log(f"  [cache] {pred_csv} found — loading, skipping inference.")
        df = pd.read_csv(pred_csv)
        _log("  [DONE] Step 3 (from cache).")
        return None, df["pred_mean"].values, df["ci_low"].values, df["ci_high"].values

    _log(f"  [3a] Predicting train set ({len(X_train)} samples) …")
    y_train_pred  = _predict_chunked(model, X_train)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    _log(f"  Train | MAE={train_metrics[0]}  RMSE={train_metrics[1]}"
         f"  R²={train_metrics[2]}  Pearson r={train_metrics[3]}")

    _log(f"  [3b] Predicting test set ({len(X_test)} samples) – mean …")
    y_test_pred  = _predict_chunked(model, X_test)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    _log(f"  Test  | MAE={test_metrics[0]}  RMSE={test_metrics[1]}"
         f"  R²={test_metrics[2]}  Pearson r={test_metrics[3]}")

    _log(f"  [3c] Computing 95 % CI (quantiles {CI_LOW}/{CI_HIGH}) …")
    ci_preds      = _predict_chunked(model, X_test, output_type="quantiles",
                                     quantiles=[CI_LOW, CI_HIGH])
    y_ci_low, y_ci_high = ci_preds[0], ci_preds[1]
    ci_width      = y_ci_high - y_ci_low
    _log(f"  CI width | median={np.median(ci_width):.2f}  mean={np.mean(ci_width):.2f}"
         f"  range=[{ci_width.min():.2f}, {ci_width.max():.2f}]")

    pred_df = meta_test.copy().reset_index(drop=True)
    pred_df["label"]     = y_test
    pred_df["pred_mean"] = y_test_pred
    pred_df["ci_low"]    = y_ci_low
    pred_df["ci_high"]   = y_ci_high
    pred_df["ci_width"]  = ci_width
    ci_cols = ["CID", "inhibitor", "dbgap_subject_id", "label",
               "pred_mean", "ci_low", "ci_high", "ci_width"]
    save_csv(pred_df[[c for c in ci_cols if c in pred_df.columns]],
             "test_predictions_with_CI.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use("classic")
    for ax, split, labels, preds, cil, cih, met in [
        (axes[0], "Train", y_train, y_train_pred, None, None, train_metrics),
        (axes[1], "Test",  y_test,  y_test_pred,  y_ci_low, y_ci_high, test_metrics),
    ]:
        ax.scatter(labels, preds, alpha=0.3, s=10, color="steelblue", label="samples")
        if cil is not None:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(labels), min(300, len(labels)), replace=False)
            ax.errorbar(labels[idx], preds[idx],
                        yerr=[preds[idx] - cil[idx], cih[idx] - preds[idx]],
                        fmt="none", alpha=0.12, color="orange", lw=0.8, label="95 % CI")
        mn = min(float(labels.min()), float(preds.min()))
        mx = max(float(labels.max()), float(preds.max()))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlim(0, 300); ax.set_ylim(0, 300)
        ax.set_xlabel("True AUC", fontsize=11)
        ax.set_ylabel("Predicted AUC (mean)", fontsize=11)
        ax.set_title(f"{split}  Pearson r={met[3]}  MAE={met[0]}", fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("TabPFN Ablation (baseline_all_features_Drug_Embed) – Mean Predictions",
                 fontsize=13)
    fig.tight_layout()
    save_plot(fig, "predictions_scatter.pdf")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(ci_width, bins=50, color="teal", edgecolor="white", alpha=0.85)
    ax2.axvline(np.median(ci_width), color="red", linestyle="--",
                label=f"Median = {np.median(ci_width):.1f}")
    ax2.set_xlabel("95 % CI Width", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Prediction CI Width (Test Set)", fontsize=11)
    ax2.legend()
    fig2.tight_layout()
    save_plot(fig2, "CI_width_distribution.pdf")

    _log("  [DONE] Step 3.")
    return y_train_pred, y_test_pred, y_ci_low, y_ci_high


# ── Step 4: Embedding importance ───────────────────────────────────────────────

def _plot_embed_bar(corr, feat_names, split, filename):
    top_idx = np.argsort(corr)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(np.array(feat_names)[top_idx], corr[top_idx],
            color=plt.cm.Blues(np.linspace(0.35, 0.95, TOP_N)))
    ax.set_xlabel("|Pearson r| with embedding norm", fontsize=11)
    ax.set_title(f"Top {TOP_N} Embedding-Attention Importance – {split}", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    save_plot(fig, filename)


def compute_embedding_importance(model, X_train, X_test, feat_names):
    _log("\n── Step 4: Embedding-based (attention-proxy) importance ──────────")

    embed_csv = os.path.join(OUT_DIR, "embedding_attention_importance.csv")
    if os.path.exists(embed_csv):
        _log(f"  [cache] {embed_csv} found — loading, skipping recompute.")
        df         = pd.read_csv(embed_csv).set_index("feature")
        feat_arr   = np.array(feat_names)
        train_corr = df["embed_corr_train"].reindex(feat_arr).fillna(0.0).values
        test_corr  = df["embed_corr_test"].reindex(feat_arr).fillna(0.0).values
        _log("  [DONE] Step 4 (from cache).")
        return train_corr, test_corr, df.reset_index()

    if not hasattr(model, "get_embeddings"):
        _log("  [SKIP] model.get_embeddings() not available.")
        dummy = np.zeros(len(feat_names))
        return dummy, dummy, pd.DataFrame({
            "feature":          feat_names,
            "embed_corr_train": dummy,
            "embed_corr_test":  dummy,
        })

    def _corr_with_emb_norm(X, label):
        input_mb = X.shape[0] * X.shape[1] * 4 / 1024**2
        _log(f"  [4] Embeddings for {label} ({len(X)} samples, input={input_mb:.0f} MB) …")
        try:
            emb      = model.get_embeddings(X, data_source="test")
            emb_avg  = emb.mean(axis=0) if emb.ndim == 3 else emb
            emb_norm = np.linalg.norm(emb_avg, axis=1)
            _log(f"  [4] Embedding norms: ({len(emb_norm)},)  "
                 f"range=[{emb_norm.min():.3f}, {emb_norm.max():.3f}]")
            X_arr = X.values
            X_c   = X_arr - X_arr.mean(axis=0)
            e_c   = emb_norm - emb_norm.mean()
            cov   = (X_c * e_c[:, None]).mean(axis=0)
            std_X = X_arr.std(axis=0)
            std_e = emb_norm.std()
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.where(std_X > 1e-9, np.abs(cov / (std_X * std_e + 1e-30)), 0.0)
            return corr, emb_norm
        except Exception as exc:
            _log(f"  [WARN] get_embeddings failed for {label}: {exc}")
            return np.zeros(X.shape[1]), np.zeros(len(X))

    train_corr, train_norms = _corr_with_emb_norm(X_train, "train")

    # Flush GPU memory between train and test embedding calls to prevent
    # CUDA tensor accumulation (previously caused exit 139 / SIGSEGV).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log(f"  [4] GPU cache flushed before test embeddings  "
             f"({torch.cuda.memory_allocated()/1024**2:.0f} MB allocated)")

    test_corr, test_norms = _corr_with_emb_norm(X_test, "test")

    emb_df = pd.DataFrame({
        "feature":          feat_names,
        "embed_corr_train": train_corr,
        "embed_corr_test":  test_corr,
    }).sort_values("embed_corr_test", ascending=False)
    save_csv(emb_df, "embedding_attention_importance.csv")
    _log("  [4] Top 5 by embedding correlation (test):")
    for row in emb_df.head(5).itertuples():
        _log(f"        {row.feature:<40s}  r={row.embed_corr_test:.4f}")

    _plot_embed_bar(test_corr,  feat_names, "Test",  "embedding_importance_bar_test.pdf")
    _plot_embed_bar(train_corr, feat_names, "Train", "embedding_importance_bar_train.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, norms, label in [(axes[0], train_norms, "Train"), (axes[1], test_norms, "Test")]:
        ax.hist(norms, bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Embedding L2 norm", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Transformer Embedding Norm – {label}", fontsize=10)
    fig.tight_layout()
    save_plot(fig, "embedding_norm_distribution.pdf")
    _log("  [DONE] Step 4.")
    return train_corr, test_corr, emb_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    _log("=" * 70)
    _log("  TabPFN Ablation – Predictions + Embeddings (Steps 1–4)")
    _log("  Experiment : baseline_all_features_Drug_Embed")
    gpu_info = (f"cuda:{torch.cuda.get_device_name(0)}"
                if DEVICE == "cuda" else "cpu (no CUDA)")
    _log(f"  Device     : {gpu_info}")
    _log("=" * 70)

    model, scaler = load_best_model()
    X_train, y_train, X_test, y_test, meta_train, meta_test, feat_names = load_data(scaler)

    assert X_train.shape[1] == model.n_features_in_, (
        f"Feature mismatch: data={X_train.shape[1]}  model expects {model.n_features_in_}"
    )
    _log(f"  Feature check PASSED ({X_train.shape[1]} == model.n_features_in_)")

    sanity_check_predictions(model, X_test, y_test)

    _, y_test_pred, y_ci_low, y_ci_high = predict_with_confidence(
        model, X_train, y_train, X_test, y_test, meta_test
    )

    compute_embedding_importance(model, X_train, X_test, feat_names)

    _log("\n" + "=" * 70)
    _log(f"  Steps 1–4 complete.  Outputs: {os.path.abspath(OUT_DIR)}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
