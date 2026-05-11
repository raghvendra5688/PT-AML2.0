"""
tabpfn_shap_analysis_fast.py

Fast SHAP analysis for the best TabPFN ablation model using
shapiq.TabPFNExplainer (remove-and-contextualize paradigm).

Target wall time: < 24 hours on H200.

Key differences vs tabpfn_best_model_shap_analysis.py:
  TabPFNExplainer   — removes features from training context instead of imputing
                      with background data; no MarginalImputer overhead
  SHAP_MAX_FEATURES 250 — top-250 patient features (no LS_* drug-embedding)
                      by context variance; avoids "all features constant" crashes
                      and makes each coalition refit ~6× faster than 1561 feats
  SHAP_CONTEXT_SIZE 100 — training rows given as context (80/20 split internally)
  SHAP_BUDGET        64 — coalitions sampled per sample by KernelSHAP
  SHAP_TEST_SAMPLES 100 — test samples explained
  SHAP_TRAIN_SAMPLES 20 — train samples explained

Steps 3 & 4 are cached to CSV: re-runs skip straight to Step 5 (SHAP).

Steps:
  1. Load best TabPFN ablation model + scaler
  2. Load BeatAML train/test pickles; build feature subset
  3. Predictions + 95 % CI on train and test sets  [cached to CSV]
  4. Embedding-proxy importance via transformer embeddings  [cached to CSV]
  5. SHAP variable importance (TabPFNExplainer, KernelSHAP)
  5b. Per-sample SHAP waterfall for notable test samples
  6. Combined importance summary (SHAP + embedding)

Cache files (separate from the original script):
  SHAP_TRAIN_CACHE_PATH — saved after train SHAP completes (crash-safe checkpoint)
  SHAP_CACHE_PATH       — saved after test SHAP completes (full checkpoint)
  Script resumes from the deepest checkpoint found on restart.
"""

import os
import sys
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import shapiq
from shapiq import TabPFNExplainer
from sklearn import preprocessing, metrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misc import load_model, calculate_regression_metrics

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)

from tabpfn import TabPFNRegressor


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

SHAP_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_fast_shap_cache.pk"
)
SHAP_TRAIN_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_fast_shap_train_partial.pk"
)

# ── SHAP hyper-parameters ─────────────────────────────────────────────────────

# Only patient features (no LS_* drug-embedding) with non-zero variance in the
# SHAP_CONTEXT_SIZE context rows are passed to TabPFNExplainer.  Limiting to
# SHAP_MAX_FEATURES prevents "all features constant" coalition failures and
# keeps each TabPFN refit fast.
SHAP_MAX_FEATURES  = 250
SHAP_CONTEXT_SIZE  = 100   # rows; TabPFNExplainer splits 80/20 internally
SHAP_BUDGET        = 64    # KernelSHAP coalitions per sample
SHAP_TEST_SAMPLES  = 100
SHAP_TRAIN_SAMPLES = 20

CI_LOW, CI_HIGH = 0.025, 0.975
TOP_N = 30
SEED  = 42

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


# ── Cache / SHAP plumbing ─────────────────────────────────────────────────────

def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_cache(path: str, data: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)
    _log(f"  [cache] {os.path.getsize(path)/1024**2:.1f} MB → {path}")


def _build_explainer(model, X_ctx_sel: np.ndarray, y_ctx: np.ndarray) -> TabPFNExplainer:
    """Build a TabPFNExplainer on the K-feature context subset.

    How shapiq initialises the explainer:
      1. Splits X_ctx_sel / y_ctx 80/20 (≈80 rows train, ≈20 rows baseline).
      2. Fits a fresh TabPFN on the 80% portion to establish the full-coalition
         prediction v(all features).
      3. Predicts the 20% baseline rows and averages them → baseline_value = v(∅),
         the prediction when NO features are revealed.
    A fresh TabPFNRegressor is required because the loaded model expects
    n_features_in_=1561 while X_ctx_sel has K≤250 features.
    """
    fresh_model = TabPFNRegressor(
        n_estimators=model.n_estimators,
        ignore_pretraining_limits=True,
    )
    # index="SV" → first-order Shapley values (no interactions)
    # max_order=1 → only φ_i terms, not φ_{ij} interaction terms
    # approximator="auto" → shapiq selects KernelSHAP for regression
    return TabPFNExplainer(
        model=fresh_model, data=X_ctx_sel, labels=y_ctx,
        index="SV", max_order=1, approximator="auto",
    )


def _select_shap_features(X_ctx: np.ndarray, feat_names: list, k: int):
    """Return indices and names of the top-k patient features by context variance.

    Why we filter before passing to TabPFNExplainer:
      - LS_* (drug-embedding) features are excluded so SHAP scope matches the
        embedding-importance step (patient features only).
      - Features with near-zero variance in the 100-row context would produce
        all-constant columns when a coalition masks them to their context mean,
        causing TabPFNExplainer to crash with "All features are constant".
      - Using only top-k high-variance features makes each coalition refit faster
        because TabPFN scales with the number of features.
    """
    ctx_var = X_ctx.var(axis=0)

    patient_mask = np.array([not feat_names[i].startswith("LS_")
                              for i in range(len(feat_names))])
    patient_idx  = np.where(patient_mask)[0]
    _log(f"  [5] Patient features (non-LS_*): {len(patient_idx)}/{len(feat_names)}")

    # Keep only non-constant patient features
    nonconst_idx = patient_idx[ctx_var[patient_idx] > 1e-10]
    _log(f"  [5] Non-constant patient features in context: "
         f"{len(nonconst_idx)}/{len(patient_idx)}")

    sel_idx = (nonconst_idx if len(nonconst_idx) <= k
               else nonconst_idx[np.argsort(ctx_var[nonconst_idx])[-k:]])
    sel_names = [feat_names[i] for i in sel_idx]
    _log(f"  [5] SHAP will use {len(sel_idx)} patient features (top-{k} by variance)")
    return sel_idx, sel_names


def _map_to_full(shap_k: np.ndarray, sel_idx: np.ndarray,
                 n_samples: int, n_features: int) -> np.ndarray:
    """Zero-pad K-feature SHAP matrix back to the full n_features width."""
    out = np.zeros((n_samples, n_features), dtype=np.float32)
    out[:, sel_idx] = shap_k
    return out


def _iv_to_shap_array(iv, n_features: int) -> np.ndarray:
    """Extract first-order Shapley values from a shapiq InteractionValues object.

    shapiq stores Shapley values as a dict keyed by feature-index tuples:
      iv[(i,)] = φ_i  — contribution of feature i to the prediction of sample x.
    The efficiency axiom guarantees: Σ_i φ_i = v(all) − v(∅) = pred − baseline.
    """
    return np.array([iv[(i,)] for i in range(n_features)])


def _explain_loop(explainer: TabPFNExplainer,
                  X_sub: np.ndarray,
                  label: str) -> np.ndarray:
    """Run TabPFNExplainer on each row of X_sub and return a SHAP matrix.

    How shapiq runs KernelSHAP per sample x_i (budget=SHAP_BUDGET coalitions):
      1. Sample SHAP_BUDGET random coalitions S ⊆ {0,…,K−1} with kernel weights
         w(S) ∝ (K−1) / (C(K,|S|) · |S| · (K−|S|))  — upweights small & large S.
      2. For each coalition S (the "remove-and-contextualize" step):
           a. Fit a fresh TabPFN on (X_ctx[:,S], y_ctx)   ← keeps only features in S
           b. Predict x_i[S] → v(S), the characteristic function value for S.
      3. Solve the weighted least-squares SHAP equation to find φ_i:
           min Σ_S w(S)·(Σ_{i∈S} φ_i − (v(S) − v(∅)))²
           s.t.  Σ_i φ_i = v(all) − v(∅)   [efficiency / completeness axiom]
      Returns an InteractionValues object; iv[(i,)] = φ_i.

    Rows that fail (e.g. degenerate coalition) are silently set to zero.
    """
    n_samples, n_features = X_sub.shape
    shap_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    for i, x_i in enumerate(tqdm(X_sub, desc=f"SHAP [{label}]", unit="sample")):
        try:
            iv = explainer.explain(x_i, budget=SHAP_BUDGET)
            shap_matrix[i] = _iv_to_shap_array(iv, n_features)
        except Exception as exc:
            _log(f"  [WARN] {label} sample {i}: {exc} — row set to zero.")
    return shap_matrix


# ── Step 1: Load model ────────────────────────────────────────────────────────

def load_best_model():
    _log("\n── Step 1: Loading best TabPFN ablation model ───────────────────")
    model  = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    _log(f"  Model  : {MODEL_PATH}")
    _log(f"  Scaler : {SCALER_PATH}")
    _log(f"  n_estimators={model.n_estimators}  "
         f"n_features_in_={model.n_features_in_}  "
         f"ignore_pretraining_limits={model.ignore_pretraining_limits}")
    _log(f"  y_train_mean_={model.y_train_mean_:.4f}  "
         f"y_train_std_={model.y_train_std_:.4f}")
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

REF_CSV = "../Results/tabpfn/ablation/baseline_all_features_Drug_Embed_predictions.csv"

def sanity_check_predictions(model, X_test, y_test, n=10):
    _log("\n── Sanity check: first-N predictions vs reference ───────────────")
    if not os.path.exists(REF_CSV):
        _log(f"  [SKIP] Reference not found: {REF_CSV}")
        return
    ref = pd.read_csv(REF_CSV, sep="\t")
    label_diffs = np.abs(y_test[:n] - ref["labels"].values[:n])
    if label_diffs.max() > 1e-3:
        _log(f"  [ERROR] True-label mismatch (max diff={label_diffs.max():.4f})")
        return
    _log(f"  Row-order PASSED (max label diff={label_diffs.max():.2e})")
    y_pred_n  = model.predict(X_test.iloc[:n])
    ref_preds = ref["predictions"].values[:n]
    _log(f"\n  {'idx':>4}  {'true':>9}  {'ref':>9}  {'new':>9}  {'diff':>8}")
    _log("  " + "-" * 50)
    diffs = []
    for i in range(n):
        d = y_pred_n[i] - ref_preds[i]
        diffs.append(abs(d))
        flag = "✓" if abs(d) < 1.0 else "✗ MISMATCH"
        _log(f"  {i:4d}  {y_test[i]:9.2f}  {ref_preds[i]:9.4f}"
             f"  {y_pred_n[i]:9.4f}  {d:+8.4f}  {flag}")
    _log(f"\n  {'PASS' if max(diffs)<1.0 else 'WARNING'} – "
         f"max |diff| = {max(diffs):.5f} AUC")


# ── Step 3: Predictions + CI ──────────────────────────────────────────────────

def predict_with_confidence(model, X_train, y_train, X_test, y_test, meta_test):
    _log("\n── Step 3: Predictions + 95 % confidence intervals ──────────────")

    # Cache: the CSV is written at the end of this function; if it exists from a
    # prior run we skip the expensive full-dataset inference (~2 h on H200).
    pred_csv = os.path.join(OUT_DIR, "test_predictions_with_CI.csv")
    if os.path.exists(pred_csv):
        _log(f"  [cache] {pred_csv} found — loading, skipping inference.")
        df = pd.read_csv(pred_csv)
        _log("  [DONE] Step 3 (from cache).")
        return None, df["pred_mean"].values, df["ci_low"].values, df["ci_high"].values

    _log(f"  [3a] Predicting train set ({len(X_train)} samples) …")
    y_train_pred  = model.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    _log(f"  Train | MAE={train_metrics[0]}  RMSE={train_metrics[1]}"
         f"  R²={train_metrics[2]}  Pearson r={train_metrics[3]}")

    _log(f"  [3b] Predicting test set ({len(X_test)} samples) – mean …")
    y_test_pred  = model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    _log(f"  Test  | MAE={test_metrics[0]}  RMSE={test_metrics[1]}"
         f"  R²={test_metrics[2]}  Pearson r={test_metrics[3]}")

    # TabPFN natively outputs quantile predictions from its ensemble of predictive
    # distributions, giving calibrated confidence intervals without extra computation.
    _log(f"  [3c] Computing 95 % CI (quantiles {CI_LOW}/{CI_HIGH}) …")
    ci_preds  = model.predict(X_test, output_type="quantiles", quantiles=[CI_LOW, CI_HIGH])
    y_ci_low, y_ci_high = ci_preds[0], ci_preds[1]
    ci_width  = y_ci_high - y_ci_low
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

    # Scatter: true vs predicted with CI error bars
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

def compute_embedding_importance(model, X_train, X_test, feat_names):
    _log("\n── Step 4: Embedding-based (attention-proxy) importance ──────────")

    # Cache: embedding computation takes ~1.5 h; skip if CSV already exists.
    embed_csv = os.path.join(OUT_DIR, "embedding_attention_importance.csv")
    if os.path.exists(embed_csv):
        _log(f"  [cache] {embed_csv} found — loading, skipping recompute.")
        df      = pd.read_csv(embed_csv).set_index("feature")
        feat_arr = np.array(feat_names)
        train_corr = df["embed_corr_train"].reindex(feat_arr).fillna(0.0).values
        test_corr  = df["embed_corr_test"].reindex(feat_arr).fillna(0.0).values
        _log("  [DONE] Step 4 (from cache).")
        return train_corr, test_corr, df.reset_index()

    if not hasattr(model, "get_embeddings"):
        _log("  [SKIP] model.get_embeddings() not available.")
        dummy = np.zeros(len(feat_names))
        return dummy, dummy, pd.DataFrame({
            "feature": feat_names,
            "embed_corr_train": dummy,
            "embed_corr_test":  dummy,
        })

    def _corr_with_emb_norm(X, label):
        _log(f"  [4] Embeddings for {label} ({len(X)} samples) …")
        try:
            emb      = model.get_embeddings(X, data_source="test")
            emb_avg  = emb.mean(axis=0) if emb.ndim == 3 else emb
            emb_norm = np.linalg.norm(emb_avg, axis=1)
            _log(f"  [4] Embedding shape: {emb_avg.shape}  "
                 f"|norm| [{emb_norm.min():.3f}, {emb_norm.max():.3f}]")
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
    test_corr,  test_norms  = _corr_with_emb_norm(X_test,  "test")

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


# ── Step 5: SHAP ───────────────────────────────────────────────────────────────

def compute_shap(model, X_train: pd.DataFrame, y_train: np.ndarray,
                 X_test: pd.DataFrame):
    """Run TabPFNExplainer SHAP with three-level checkpointing.

    All three paths (full cache / partial cache / from scratch) converge at the
    output-saving block at the bottom, so plots and CSVs are always up to date.
    """
    _log("\n── Step 5: SHAP variable importance (TabPFNExplainer) ───────────")
    _log(f"  context={SHAP_CONTEXT_SIZE}  budget={SHAP_BUDGET}"
         f"  train_n={SHAP_TRAIN_SAMPLES}  test_n={SHAP_TEST_SAMPLES}"
         f"  max_feats={SHAP_MAX_FEATURES}")

    feat_names = X_train.columns.tolist()
    n_features = len(feat_names)

    # ── Path A: full cache (both train + test SHAP already done) ─────────────
    if os.path.exists(SHAP_CACHE_PATH):
        _log("  [cache] Full cache found — loading SHAP values.")
        cache        = _load_pickle(SHAP_CACHE_PATH)
        feat_sel_idx = cache.get("feat_sel_idx")
        feat_sel_names = [feat_names[i] for i in feat_sel_idx] if feat_sel_idx is not None else feat_names
        X_ctx_sel    = cache["X_ctx"][:, feat_sel_idx]
        explainer    = _build_explainer(model, X_ctx_sel, cache["y_ctx"])
        shap_train, shap_test = cache["shap_train"], cache["shap_test"]
        train_idx,  test_idx  = cache["train_idx"],  cache["test_idx"]
        _log(f"  shap_train={shap_train.shape}  shap_test={shap_test.shape}"
             f"  baseline={cache['expected_value']:.4f}")

    # ── Path B: partial cache (train done, test pending) ─────────────────────
    elif os.path.exists(SHAP_TRAIN_CACHE_PATH):
        _log("  [cache] Partial cache found — resuming from test SHAP.")
        partial      = _load_pickle(SHAP_TRAIN_CACHE_PATH)
        feat_sel_idx = partial.get("feat_sel_idx")
        feat_sel_names = [feat_names[i] for i in feat_sel_idx] if feat_sel_idx is not None else feat_names
        X_ctx_sel    = partial["X_ctx"][:, feat_sel_idx]
        explainer    = _build_explainer(model, X_ctx_sel, partial["y_ctx"])
        shap_train   = partial["shap_train"]
        train_idx    = partial["train_idx"]

        # Compute test SHAP and save full cache
        np.random.seed(SEED + 1)
        test_idx       = np.random.choice(len(X_test), min(SHAP_TEST_SAMPLES, len(X_test)),
                                          replace=False)
        X_test_sub_sel = X_test.iloc[test_idx].values[:, feat_sel_idx]
        _log(f"  [5d] {len(test_idx)} test samples × {X_test_sub_sel.shape[1]} features …")
        shap_test = _map_to_full(_explain_loop(explainer, X_test_sub_sel, "test"),
                                 feat_sel_idx, len(test_idx), n_features)
        _save_cache(SHAP_CACHE_PATH, dict(
            X_ctx=partial["X_ctx"], y_ctx=partial["y_ctx"],
            expected_value=partial["expected_value"],
            shap_train=shap_train, shap_test=shap_test,
            train_idx=train_idx, test_idx=test_idx, feat_sel_idx=feat_sel_idx,
        ))

    # ── Path C: compute from scratch ──────────────────────────────────────────
    else:
        np.random.seed(SEED)
        ctx_idx = np.random.choice(len(X_train), min(SHAP_CONTEXT_SIZE, len(X_train)),
                                   replace=False)
        X_ctx   = X_train.iloc[ctx_idx].values   # (100, 1561) — full features
        y_ctx   = y_train[ctx_idx]
        _log(f"  [5a] Context: {X_ctx.shape[0]} rows × {X_ctx.shape[1]} features")

        # Restrict to top-SHAP_MAX_FEATURES patient features before building explainer.
        # This avoids the "all features constant" coalition crash and speeds up refits.
        feat_sel_idx, feat_sel_names = _select_shap_features(X_ctx, feat_names, SHAP_MAX_FEATURES)
        X_ctx_sel = X_ctx[:, feat_sel_idx]   # (100, K)

        # Build explainer: shapiq internally fits TabPFN on X_ctx_sel to get baseline_value.
        _log(f"  [5b] Building TabPFNExplainer ({X_ctx_sel.shape[1]} features) …")
        explainer      = _build_explainer(model, X_ctx_sel, y_ctx)
        expected_value = float(explainer.baseline_value)
        _log(f"  [5b] baseline_value (v(∅)) = {expected_value:.4f}")

        # Train SHAP — save partial cache immediately in case job is killed before test SHAP
        train_idx       = np.random.choice(len(X_train), min(SHAP_TRAIN_SAMPLES, len(X_train)),
                                           replace=False)
        X_train_sub_sel = X_train.iloc[train_idx].values[:, feat_sel_idx]
        _log(f"  [5c] {len(train_idx)} train samples × {X_train_sub_sel.shape[1]} features …")
        shap_train = _map_to_full(_explain_loop(explainer, X_train_sub_sel, "train"),
                                  feat_sel_idx, len(train_idx), n_features)
        _save_cache(SHAP_TRAIN_CACHE_PATH, dict(
            X_ctx=X_ctx, y_ctx=y_ctx, expected_value=expected_value,
            shap_train=shap_train, train_idx=train_idx, feat_sel_idx=feat_sel_idx,
        ))

        # Test SHAP
        np.random.seed(SEED + 1)
        test_idx       = np.random.choice(len(X_test), min(SHAP_TEST_SAMPLES, len(X_test)),
                                          replace=False)
        X_test_sub_sel = X_test.iloc[test_idx].values[:, feat_sel_idx]
        _log(f"  [5d] {len(test_idx)} test samples × {X_test_sub_sel.shape[1]} features …")
        shap_test = _map_to_full(_explain_loop(explainer, X_test_sub_sel, "test"),
                                 feat_sel_idx, len(test_idx), n_features)
        _save_cache(SHAP_CACHE_PATH, dict(
            X_ctx=X_ctx, y_ctx=y_ctx, expected_value=expected_value,
            shap_train=shap_train, shap_test=shap_test,
            train_idx=train_idx, test_idx=test_idx, feat_sel_idx=feat_sel_idx,
        ))

    # ── All paths converge: save CSVs and plots ───────────────────────────────
    _save_shap_outputs(shap_train, shap_test, feat_names, X_train, X_test,
                       train_idx, test_idx)
    _log("  [DONE] Step 5.")
    return shap_train, shap_test, explainer, feat_sel_idx, feat_sel_names


def _save_shap_outputs(shap_train, shap_test, feat_names, X_train, X_test,
                       train_idx, test_idx):
    """Save SHAP CSVs, bar charts, and beeswarm plots."""
    shap_imp_train = np.abs(shap_train).mean(axis=0)
    shap_imp_test  = np.abs(shap_test).mean(axis=0)

    save_csv(pd.DataFrame(shap_train.astype(np.float32), columns=feat_names),
             "shap_values_train_subset.csv")
    save_csv(pd.DataFrame(shap_test.astype(np.float32), columns=feat_names),
             "shap_values_test_subset.csv")

    imp_df = pd.DataFrame({
        "feature":             feat_names,
        "shap_mean_abs_train": shap_imp_train,
        "shap_mean_abs_test":  shap_imp_test,
    }).sort_values("shap_mean_abs_test", ascending=False)
    save_csv(imp_df, "shap_feature_importance.csv")
    _log("  Top 5 features by mean |SHAP| (test):")
    for row in imp_df.head(5).itertuples():
        _log(f"    {row.feature:<40s}  {row.shap_mean_abs_test:.4f}")

    X_train_sub = X_train.iloc[train_idx].values
    X_test_sub  = X_test.iloc[test_idx].values
    _plot_shap_bar(shap_imp_test,  feat_names, "Test",  "shap_importance_bar_test.pdf")
    _plot_shap_bar(shap_imp_train, feat_names, "Train", "shap_importance_bar_train.pdf")
    _plot_shap_beeswarm(shap_test,  X_test_sub,  feat_names, "shap_beeswarm_test.pdf")
    _plot_shap_beeswarm(shap_train, X_train_sub, feat_names, "shap_beeswarm_train.pdf")


def _plot_shap_bar(shap_imp, feat_names, split, filename):
    top_idx = np.argsort(shap_imp)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(np.array(feat_names)[top_idx], shap_imp[top_idx],
            color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, TOP_N)))
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Top {TOP_N} SHAP Feature Importance – {split}", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    save_plot(fig, filename)


def _plot_shap_beeswarm(shap_vals, X_sub, feat_names, filename):
    shap_imp  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(shap_imp)[-TOP_N:]
    sv_top    = shap_vals[:, top_idx]
    xv_top    = X_sub[:, top_idx]
    top_names = np.array(feat_names)[top_idx]
    fig, ax   = plt.subplots(figsize=(8, 9))
    rng = np.random.default_rng(SEED)
    for j, (sv_col, xv_col) in enumerate(zip(sv_top.T, xv_top.T)):
        y_jitter = rng.uniform(-0.35, 0.35, len(sv_col))
        norm_val = (xv_col - xv_col.min()) / (xv_col.ptp() + 1e-9)
        ax.scatter(sv_col, j + y_jitter, c=plt.cm.coolwarm(norm_val),
                   s=6, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
    ax.set_title(f"SHAP Beeswarm – Top {TOP_N} features\n"
                 "(colour: blue = low value, red = high)", fontsize=11)
    fig.tight_layout()
    save_plot(fig, filename)


# ── Step 5b: Single-sample waterfall explanations ────────────────────────────

def explain_single_test_sample(explainer, X_test_shap, shap_feat_names,
                                y_test, y_pred, y_ci_low, y_ci_high, sample_idx=0):
    """Explain one test sample and save a waterfall plot.

    X_test_shap: DataFrame with K SHAP features, all test rows.
    shap_feat_names: list of length K matching X_test_shap columns.

    shapiq.waterfall_plot reads the InteractionValues object directly and draws
    the additive decomposition:  baseline_value → +φ_1 → +φ_2 → … → prediction,
    showing how each feature pushes the model output above or below the baseline.
    """
    _log(f"\n  [5b] Explaining sample idx={sample_idx} …")
    n_k        = len(shap_feat_names)
    x_1d       = X_test_shap.iloc[sample_idx].values
    base_value = float(explainer.baseline_value)
    predicted  = float(y_pred[sample_idx])
    true_label = float(y_test[sample_idx])
    ci_lo, ci_hi = float(y_ci_low[sample_idx]), float(y_ci_high[sample_idx])

    try:
        # explainer.explain(x_1d) runs the same KernelSHAP as _explain_loop for one sample.
        iv      = explainer.explain(x_1d, budget=SHAP_BUDGET)
        shap_1d = _iv_to_shap_array(iv, n_k)
    except Exception as exc:
        _log(f"  [WARN] explain() failed for sample {sample_idx}: {exc} — skipping.")
        return None

    _log(f"  base={base_value:.2f}  pred={predicted:.2f}  true={true_label:.2f}"
         f"  CI=[{ci_lo:.2f},{ci_hi:.2f}]  err={abs(predicted-true_label):.2f}")

    single_df = pd.DataFrame({
        "feature":       shap_feat_names,
        "feature_value": x_1d,
        "shap_value":    shap_1d,
    }).sort_values("shap_value", key=np.abs, ascending=False)
    for col, val in [("sample_idx", sample_idx), ("true_label", true_label),
                     ("pred_mean", predicted), ("ci_low", ci_lo),
                     ("ci_high", ci_hi), ("base_value", base_value)]:
        single_df[col] = val
    save_csv(single_df, f"shap_single_sample_idx{sample_idx}.csv")

    try:
        ax  = shapiq.waterfall_plot(iv, feature_names=shap_feat_names,
                                    max_display=TOP_N, show=False)
        fig = ax.get_figure()
        fig.suptitle(f"SHAP Waterfall – test sample #{sample_idx}\n"
                     f"true={true_label:.1f}  pred={predicted:.1f}"
                     f"  95 % CI=[{ci_lo:.1f},{ci_hi:.1f}]", fontsize=11)
        fig.tight_layout()
        save_plot(fig, f"shap_waterfall_sample_idx{sample_idx}.pdf")
    except Exception as exc:
        _log(f"  [WARN] waterfall_plot failed: {exc}")
    return shap_1d


def _top_k_accurate(mask, abs_error, k, label):
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        _log(f"  [WARN] No candidates for '{label}'.")
        return []
    return idxs[np.argsort(abs_error[idxs])][:k].tolist()


def explain_notable_samples(explainer, X_test_shap, shap_feat_names,
                             y_test, y_pred, y_ci_low, y_ci_high):
    _log("\n── Step 5b: Single-sample SHAP waterfall (up to 4 notable samples) ─")
    abs_error   = np.abs(y_pred - y_test)
    q25, q75    = np.percentile(y_test, [25, 75])
    median_pred = np.median(y_pred)
    sens_mask   = (y_test <= q25) & (y_pred <= median_pred)
    res_mask    = (y_test >= q75) & (y_pred >= median_pred)
    _log(f"  Sensitive candidates: {sens_mask.sum()}  "
         f"Resistant candidates: {res_mask.sum()}")

    samples = {}
    for i, idx in enumerate(_top_k_accurate(sens_mask, abs_error, 2, "sensitive"), start=1):
        samples[f"sensitive_accurate_{i}"] = idx
    for i, idx in enumerate(_top_k_accurate(res_mask,  abs_error, 2, "resistant"),  start=1):
        samples[f"resistant_accurate_{i}"] = idx

    if not samples:
        _log("  [WARN] No notable samples found — skipping Step 5b.")
        return

    for label, idx in samples.items():
        _log(f"\n  -- {label} (idx={idx}  true={y_test[idx]:.1f}"
             f"  pred={y_pred[idx]:.1f}  err={abs_error[idx]:.2f}) --")
        explain_single_test_sample(explainer, X_test_shap, shap_feat_names,
                                   y_test, y_pred, y_ci_low, y_ci_high,
                                   sample_idx=idx)
    _log("  [DONE] Step 5b.")


# ── Step 6: Combined summary ───────────────────────────────────────────────────

def combined_importance_summary(shap_imp_test, embed_corr_test, feat_names):
    _log("\n── Step 6: Combined importance summary ───────────────────────────")

    def _minmax(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-9 else np.zeros_like(v)

    shap_norm  = _minmax(shap_imp_test)
    embed_norm = _minmax(embed_corr_test)
    combined   = 0.5 * shap_norm + 0.5 * embed_norm

    summary_df = pd.DataFrame({
        "feature":         feat_names,
        "shap_score":      shap_imp_test,
        "embedding_score": embed_corr_test,
        "shap_norm":       shap_norm,
        "embedding_norm":  embed_norm,
        "combined_score":  combined,
    }).sort_values("combined_score", ascending=False)
    save_csv(summary_df, "combined_importance_summary.csv")

    top = summary_df.head(TOP_N)
    fig, ax = plt.subplots(figsize=(8, 9))
    y  = np.arange(len(top))
    bh = 0.35
    ax.barh(y + bh/2, top["shap_norm"],      bh, label="SHAP (norm.)",
            color="steelblue",  alpha=0.85)
    ax.barh(y - bh/2, top["embedding_norm"], bh, label="Embedding (norm.)",
            color="darkorange", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Normalised importance score", fontsize=11)
    ax.set_title(f"Top {TOP_N} Features – SHAP vs Embedding-Attention (Test)", fontsize=12)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    save_plot(fig, "combined_importance_comparison.pdf")

    _log("  Top 10 by combined score:")
    _log(summary_df[["feature", "shap_score", "embedding_score", "combined_score"]]
         .head(10).to_string(index=False))
    _log("  [DONE] Step 6.")
    return summary_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    _log("=" * 70)
    _log("  TabPFN Ablation – Fast SHAP (TabPFNExplainer) + CI Analysis")
    _log("  Experiment : baseline_all_features_Drug_Embed")
    _log(f"  shapiq     : {shapiq.__version__}")
    _log(f"  SHAP params: context={SHAP_CONTEXT_SIZE}  budget={SHAP_BUDGET}"
         f"  train_n={SHAP_TRAIN_SAMPLES}  test_n={SHAP_TEST_SAMPLES}"
         f"  max_feats={SHAP_MAX_FEATURES}")
    _log("=" * 70)

    # Steps 1–2: load model and data
    model, scaler = load_best_model()
    X_train, y_train, X_test, y_test, meta_train, meta_test, feat_names = load_data(scaler)

    assert X_train.shape[1] == model.n_features_in_, (
        f"Feature mismatch: data={X_train.shape[1]}  model={model.n_features_in_}"
    )
    _log(f"\n  Feature check PASSED ({X_train.shape[1]} == model.n_features_in_)")

    sanity_check_predictions(model, X_test, y_test)

    # Step 3: predictions + CI (cached to CSV; ~2 h if not cached)
    _, y_test_pred, y_ci_low, y_ci_high = predict_with_confidence(
        model, X_train, y_train, X_test, y_test, meta_test
    )

    # Step 4: embedding importance (cached to CSV; ~1.5 h if not cached)
    _, test_emb_corr, _ = compute_embedding_importance(model, X_train, X_test, feat_names)

    # Step 5: SHAP — TabPFNExplainer on top-SHAP_MAX_FEATURES patient features
    shap_train, shap_test, explainer, feat_sel_idx, shap_feat_names = \
        compute_shap(model, X_train, y_train, X_test)

    # Step 5b: waterfall explanations for the most accurate sensitive/resistant samples.
    # X_test_shap is the K-feature slice of the full test set, matching the explainer.
    X_test_shap = X_test.iloc[:, feat_sel_idx].copy() # type: ignore
    explain_notable_samples(explainer, X_test_shap, shap_feat_names,
                            y_test, y_test_pred, y_ci_low, y_ci_high)

    # Step 6: combined SHAP + embedding ranking
    shap_imp_test = np.abs(shap_test).mean(axis=0)
    combined_importance_summary(shap_imp_test, test_emb_corr, feat_names)

    _log("\n" + "=" * 70)
    _log(f"  Analysis complete.  Outputs: {os.path.abspath(OUT_DIR)}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
