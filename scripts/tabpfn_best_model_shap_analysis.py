"""
tabpfn_best_model_shap_analysis.py

Ablation model: baseline_all_features_Drug_Embed (1561 features).
Uses shapiq.TabPFNExplainer (remove-and-contextualize) with n_estimators=1
dedicated SHAP model for ~20× speedup vs the prediction model.

Steps:
  1. Load best TabPFN ablation model + saved scaler
  2. Load BeatAML train/test pickles; build feature subset via ablation logic
  3. Mean predictions + 95 % CI (quantiles) on train and test sets
  4. SHAP variable importance — test set only (TabPFNExplainer, n_estimators=1)
  4b. Per-sample SHAP waterfall for notable samples (sensitive/resistant)
  5. Attention-proxy importance via TabPFN transformer embeddings
     (patient features only in plots; LS_* drug features excluded)
  6. Combined importance summary (SHAP + embedding)

Intermediate saves
  - Full cache (test SHAP only) → SHAP_CACHE_PATH
  Script resumes from checkpoint if cache file is present.
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
import scipy
import torch
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
ABLATION_CSV    = "../Data/ablation_feature_columns.csv"
OUT_DIR         = "../Results/tabpfn/best_model_analysis"

# Cache for test SHAP arrays — script resumes from here if present
SHAP_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_baseline_all_features_Drug_Embed_shap_cache.pk"
)

# ── SHAP hyper-parameters ────────────────────────────────────────────────────
# Training rows given to TabPFNExplainer as context (80 % context, 20 % baseline).
# Smaller context → faster coalition evaluations; n_train=33k makes the full set
# unusable inside SHAP (every predict call would attend over 33k rows).
SHAP_CONTEXT_SIZE = 100
# KernelSHAP coalition budget per sample.
# Log showed ~1300-2300s/sample at budget=128, n_estimators=20.
# With n_estimators=1 shap_model and budget=64: ~57s/sample est.
SHAP_BUDGET       = 64
# Test samples to explain. Train SHAP is skipped: with 33k rows most train
# coalitions produce constant features and fail with zero SHAP values.
SHAP_TEST_SAMPLES = 150

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    """Return HH:MM:SS timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    """Print with timestamp and immediately flush to log."""
    print(f"[{_ts()}] {msg}", flush=True)


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    _log(f"  [saved CSV]  {path}")
    return path


def save_plot(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    _log(f"  [saved plot] {path}")
    return path


# ── Feature selection ─────────────────────────────────────────────────────────

def get_feature_subset(train_df: pd.DataFrame):
    """Replicate the exact feature selection from tabpfn_ablation_study.py."""
    all_columns  = train_df.columns.tolist()
    column_types = train_df.dtypes.tolist()

    feature_metadata = pd.read_csv(ABLATION_CSV)
    feature_groups_raw = {}
    for type_name, group_df in feature_metadata.groupby("Type"):
        feature_groups_raw[type_name] = group_df["Column_Label"].tolist()

    tsne_cols    = feature_groups_raw.get("Tsne", [])
    tsne_in_data = [c for c in tsne_cols if c in all_columns]

    metadata_cols = ["CID"] + [
        all_columns[i] for i in range(len(all_columns))
        if str(column_types[i]) in ("object", "category")
    ]
    nan_cols = [c for c in all_columns if train_df[c].isnull().any()]
    metadata_cols = list(dict.fromkeys(metadata_cols + nan_cols + EXCLUDE_COLS + tsne_in_data))

    all_feature_cols = sorted(list(set(all_columns) - set(metadata_cols) - {"auc"}))

    drug_embedding_cols = [c for c in feature_groups_raw.get("Drug_Embed", [])
                           if c in all_feature_cols]

    feature_groups = {k: v for k, v in feature_groups_raw.items()
                      if k not in ("Tsne", "Drug_Embed", "Drug_PC")}
    feature_groups["Clinical_CellType_Module"] = (
        feature_groups.pop("Clinical", []) +
        feature_groups.pop("CellType", []) +
        feature_groups.pop("Module", [])
    )

    ablation_groups = {}
    for group_name, group_cols in feature_groups.items():
        cols_in_data = [c for c in group_cols if c in all_feature_cols]
        if cols_in_data:
            ablation_groups[group_name] = cols_in_data

    all_patient_cols = []
    for g in sorted(ablation_groups.keys()):
        all_patient_cols.extend(ablation_groups[g])

    return list(drug_embedding_cols) + all_patient_cols


# ── Step 1: Load model ────────────────────────────────────────────────────────

def load_best_model():
    _log("\n── Step 1: Loading best TabPFN ablation model ───────────────────")
    _log(f"  Model  : {MODEL_PATH}")
    _log(f"  Scaler : {SCALER_PATH}")
    model  = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    _log(f"  Model type            : {type(model).__name__}")
    _log(f"  n_estimators          : {model.n_estimators}")
    _log(f"  n_features_in_        : {model.n_features_in_}")
    _log(f"  ignore_pretraining_limits: {model.ignore_pretraining_limits}")
    _log(f"  y_train_mean_         : {model.y_train_mean_:.4f}")
    _log(f"  y_train_std_          : {model.y_train_std_:.4f}")
    _log("  [DONE] Step 1 complete.")
    return model, scaler


# ── Step 2: Load data ─────────────────────────────────────────────────────────

def load_data(scaler):
    _log("\n── Step 2: Loading and preprocessing data ───────────────────────")
    _log(f"  Training set : {TRAIN_PATH}")
    _log(f"  Test set     : {TEST_PATH}")

    train_df = pd.read_pickle(TRAIN_PATH, compression="zip")
    test_df  = pd.read_pickle(TEST_PATH,  compression="zip")
    _log(f"  Train raw shape : {train_df.shape}")
    _log(f"  Test  raw shape : {test_df.shape}")

    feature_subset = get_feature_subset(train_df)
    _log(f"  Feature subset  : {len(feature_subset)}")

    meta_cols = [c for c in train_df.columns if c not in feature_subset and c != "auc"]
    meta_train = train_df[meta_cols].copy()
    meta_test  = test_df[[c for c in meta_cols if c in test_df.columns]].copy()

    X_train = pd.DataFrame(scaler.transform(train_df[feature_subset]), columns=feature_subset)
    X_test  = pd.DataFrame(scaler.transform(test_df[feature_subset]),  columns=feature_subset)
    y_train = train_df["auc"].to_numpy().flatten()
    y_test  = test_df["auc"].to_numpy().flatten()

    _log(f"  Train feature matrix : {X_train.shape}")
    _log(f"  Test  feature matrix : {X_test.shape}")
    _log(f"  Train label range    : [{y_train.min():.1f}, {y_train.max():.1f}]"
         f"  mean={y_train.mean():.2f}")
    _log(f"  Test  label range    : [{y_test.min():.1f}, {y_test.max():.1f}]"
         f"  mean={y_test.mean():.2f}")

    feat_subset_df = pd.DataFrame({"feature": feature_subset})
    feat_subset_path = os.path.join(OUT_DIR, "feature_subset.csv")
    feat_subset_df.to_csv(feat_subset_path, index=False)
    _log(f"  Feature subset saved → {feat_subset_path}")

    _log("  [DONE] Step 2 complete.")
    return X_train, y_train, X_test, y_test, meta_train, meta_test, feature_subset


# ── Sanity check ──────────────────────────────────────────────────────────────

REF_CSV = "../Results/tabpfn/ablation/baseline_all_features_Drug_Embed_predictions.csv"

def sanity_check_predictions(model, X_test: pd.DataFrame, y_test: np.ndarray, n: int = 10) -> None:
    _log("\n── Sanity check: compare first-N predictions to reference ────────")

    if not os.path.exists(REF_CSV):
        _log(f"  [SKIP] Reference file not found: {REF_CSV}")
        return

    ref = pd.read_csv(REF_CSV, sep="\t")
    ref_preds  = ref["predictions"].values[:n]
    ref_labels = ref["labels"].values[:n]

    label_diffs = np.abs(y_test[:n] - ref_labels)
    if label_diffs.max() > 1e-3:
        _log(f"  [ERROR] True-label mismatch (max diff={label_diffs.max():.4f}) — row ordering drifted.")
        return
    _log(f"  Row-order check PASSED (max label diff = {label_diffs.max():.2e})")

    _log(f"  Predicting first {n} test samples (mean) for comparison …")
    y_pred_n = model.predict(X_test.iloc[:n])

    _log(f"\n  {'idx':>4}  {'true_AUC':>9}  {'ref_pred':>9}  {'new_pred':>9}  {'diff':>8}")
    _log("  " + "-" * 50)
    diffs = []
    for i in range(n):
        diff = y_pred_n[i] - ref_preds[i]
        diffs.append(abs(diff))
        flag = "  ✓" if abs(diff) < 1.0 else "  ✗ MISMATCH"
        _log(f"  {i:4d}  {y_test[i]:9.2f}  {ref_preds[i]:9.4f}  {y_pred_n[i]:9.4f}  {diff:+8.4f}{flag}")

    max_diff = max(diffs)
    if max_diff < 1.0:
        _log(f"\n  PASS – max |diff| = {max_diff:.5f} AUC  (all within 1.0 tolerance)")
    else:
        _log(f"\n  WARNING – max |diff| = {max_diff:.4f} AUC  (check preprocessing)")


# ── Step 3: Predictions + CI ──────────────────────────────────────────────────

def predict_with_confidence(model, X_train, y_train, X_test, y_test, meta_test):
    _log("\n── Step 3: Predictions + 95 % confidence intervals ──────────────")

    _log(f"  [3a] Predicting on training set ({len(X_train)} samples) …")
    y_train_pred  = model.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    _log(f"  Train | MAE={train_metrics[0]}  RMSE={train_metrics[1]}"
         f"  R²={train_metrics[2]}  Pearson r={train_metrics[3]}")

    _log(f"  [3b] Predicting on test set ({len(X_test)} samples) – mean …")
    y_test_pred  = model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    _log(f"  Test  | MAE={test_metrics[0]}  RMSE={test_metrics[1]}"
         f"  R²={test_metrics[2]}  Pearson r={test_metrics[3]}")

    _log(f"  [3c] Computing 95 % CI on test set (quantiles {CI_LOW}/{CI_HIGH}) …")
    ci_preds  = model.predict(X_test, output_type="quantiles", quantiles=[CI_LOW, CI_HIGH])
    y_ci_low  = ci_preds[0]
    y_ci_high = ci_preds[1]
    ci_width  = y_ci_high - y_ci_low

    _log(f"  CI width | median={np.median(ci_width):.2f}  "
         f"mean={np.mean(ci_width):.2f}  "
         f"range=[{ci_width.min():.2f}, {ci_width.max():.2f}]")

    pred_df = meta_test.copy().reset_index(drop=True)
    pred_df["label"]     = y_test
    pred_df["pred_mean"] = y_test_pred
    pred_df["ci_low"]    = y_ci_low
    pred_df["ci_high"]   = y_ci_high
    pred_df["ci_width"]  = ci_width

    ci_cols = ["CID", "inhibitor", "dbgap_subject_id", "label", "pred_mean", "ci_low", "ci_high", "ci_width"]
    ci_out  = pred_df[[c for c in ci_cols if c in pred_df.columns]]
    save_csv(ci_out, "test_predictions_with_CI.csv")

    _log("  [3e] Saving prediction scatter plots …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use("classic")

    for ax, split, labels, preds, cil, cih, met in [
        (axes[0], "Train", y_train, y_train_pred, None,      None,      train_metrics),
        (axes[1], "Test",  y_test,  y_test_pred,  y_ci_low, y_ci_high, test_metrics),
    ]:
        ax.scatter(labels, preds, alpha=0.3, s=10, color="steelblue", label="samples")
        if cil is not None:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(labels), min(300, len(labels)), replace=False)
            ax.errorbar(
                labels[idx], preds[idx],
                yerr=[preds[idx] - cil[idx], cih[idx] - preds[idx]],
                fmt="none", alpha=0.12, color="orange", lw=0.8, label="95 % CI",
            )
        mn = min(float(labels.min()), float(preds.min()))
        mx = max(float(labels.max()), float(preds.max()))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlim(0, 300); ax.set_ylim(0, 300)
        ax.set_xlabel("True AUC", fontsize=11)
        ax.set_ylabel("Predicted AUC (mean)", fontsize=11)
        ax.set_title(f"{split}  Pearson r={met[3]}  MAE={met[0]}", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("TabPFN Ablation (baseline_all_features_Drug_Embed) – Mean Predictions", fontsize=13)
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

    _log("  [DONE] Step 3 complete.")
    return y_train_pred, y_test_pred, y_ci_low, y_ci_high


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def _iv_to_shap_array(iv, n_features: int) -> np.ndarray:
    """Extract first-order Shapley values as 1-D numpy array from InteractionValues."""
    return np.array([iv[(i,)] for i in range(n_features)])


def _build_tabpfn_explainer(shap_model, X_ctx: np.ndarray, y_ctx: np.ndarray) -> TabPFNExplainer:
    """Build a TabPFNExplainer using SHAP_CONTEXT_SIZE rows as context.

    Uses a dedicated n_estimators=1 model so each coalition evaluation runs
    a single forward pass instead of 20, giving ~20× speedup vs the prediction
    model (n_estimators=20).  TabPFNExplainer splits data 80/20: ~80 rows for
    remove-and-contextualize and ~20 for the empty-coalition baseline.
    """
    return TabPFNExplainer(
        model=shap_model,
        data=X_ctx,
        labels=y_ctx,
        index="SV",
        max_order=1,
        approximator="auto",
    )


def _save_shap_cache(X_ctx, y_ctx, expected_value, shap_test, test_idx) -> None:
    cache = {
        "X_ctx":          X_ctx,
        "y_ctx":          y_ctx,
        "expected_value": expected_value,
        "shap_test":      shap_test,
        "test_idx":       test_idx,
    }
    with open(SHAP_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    _log(f"  [cache] Saved SHAP cache ({os.path.getsize(SHAP_CACHE_PATH)/1024**2:.1f} MB)"
         f" → {SHAP_CACHE_PATH}")


def _explain_loop(explainer: TabPFNExplainer, X_sub: np.ndarray, label: str) -> np.ndarray:
    """Run TabPFNExplainer on X_sub row-by-row with tqdm progress bar.

    Returns shap_matrix of shape (n_samples, n_features).
    """
    from tqdm import tqdm
    n_samples, n_features = X_sub.shape
    shap_matrix = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, x_i in enumerate(tqdm(X_sub, desc=f"SHAP [{label}]", unit="sample")):
        try:
            iv = explainer.explain(x_i, budget=SHAP_BUDGET)
            shap_matrix[i] = _iv_to_shap_array(iv, n_features)
        except Exception as exc:
            _log(f"  [WARN] {label} sample {i}: explain failed ({exc}); row set to zero.")

    return shap_matrix


# ── Step 4: SHAP ─────────────────────────────────────────────────────────────

def compute_shap(model, X_train: pd.DataFrame, y_train: np.ndarray,
                 X_test: pd.DataFrame):
    _log("\n── Step 4: SHAP variable importance (TabPFNExplainer) ───────────")
    _log(f"  Explainer  : TabPFNExplainer (remove-and-contextualize, n_estimators=1)")
    _log(f"  Context    : {SHAP_CONTEXT_SIZE} rows → ~{int(0.8*SHAP_CONTEXT_SIZE)} training"
         f" + ~{int(0.2*SHAP_CONTEXT_SIZE)} baseline")
    _log(f"  Budget     : {SHAP_BUDGET} coalitions/sample  |  Samples: {SHAP_TEST_SAMPLES} test only")

    feat_names = X_train.columns.tolist()
    n_features = len(feat_names)

    # Dedicated 1-estimator model for SHAP: ~20× faster per coalition than the
    # prediction model (n_estimators=20).  TabPFNExplainer re-fits it per coalition
    # so the original fitted weights are irrelevant here.
    shap_model = TabPFNRegressor(n_estimators=1,
                                  ignore_pretraining_limits=True)

    # ── Resume from cache ─────────────────────────────────────────────────────
    if os.path.exists(SHAP_CACHE_PATH):
        _log(f"  [cache] Cache found — skipping SHAP computation.")
        with open(SHAP_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        X_ctx          = cache["X_ctx"]
        y_ctx          = cache["y_ctx"]
        expected_value = cache["expected_value"]
        shap_test      = cache["shap_test"]
        test_idx       = cache["test_idx"]
        explainer = _build_tabpfn_explainer(shap_model, X_ctx, y_ctx)
        _log(f"  [cache] shap_test={shap_test.shape}  base={expected_value:.4f}")
        _log("  [DONE] Step 4 (from cache).")
        return shap_test, feat_names, None, explainer

    # ── Compute from scratch ──────────────────────────────────────────────────
    np.random.seed(SEED)
    ctx_idx = np.random.choice(len(X_train), min(SHAP_CONTEXT_SIZE, len(X_train)), replace=False)
    X_ctx   = X_train.iloc[ctx_idx].values
    y_ctx   = y_train[ctx_idx]
    _log(f"  [4a] Context data: {X_ctx.shape[0]} rows × {X_ctx.shape[1]} features")

    _log(f"  [4b] Building TabPFNExplainer …")
    explainer = _build_tabpfn_explainer(shap_model, X_ctx, y_ctx)
    expected_value = float(explainer.baseline_value)
    _log(f"  [4b] Baseline (empty coalition) prediction: {expected_value:.4f}")

    np.random.seed(SEED + 1)
    test_idx   = np.random.choice(len(X_test), min(SHAP_TEST_SAMPLES, len(X_test)), replace=False)
    X_test_sub = X_test.iloc[test_idx].values
    _log(f"  [4c] Computing SHAP on {len(X_test_sub)} test samples (budget={SHAP_BUDGET}) …")
    shap_test = _explain_loop(explainer, X_test_sub, "test")
    _save_shap_cache(X_ctx, y_ctx, expected_value, shap_test, test_idx)

    # ── Save CSVs and plots ───────────────────────────────────────────────────
    save_csv(pd.DataFrame(shap_test.astype(np.float32), columns=feat_names),
             "shap_values_test_subset.csv")

    shap_imp_test = np.abs(shap_test).mean(axis=0)

    imp_df = pd.DataFrame({
        "feature":            feat_names,
        "shap_mean_abs_test": shap_imp_test,
    }).sort_values("shap_mean_abs_test", ascending=False)
    save_csv(imp_df, "shap_feature_importance.csv")

    _plot_shap_bar(shap_imp_test, feat_names, "Test", "shap_importance_bar_test.pdf")
    _plot_shap_beeswarm(shap_test, X_test_sub, feat_names, "shap_beeswarm_test.pdf")

    _log(f"  [4d] Top 5 features by mean |SHAP| (test):")
    for row in imp_df.head(5).itertuples():
        _log(f"        {row.feature:<40s}  {row.shap_mean_abs_test:.4f}")

    _log("  [DONE] Step 4 complete.")
    return shap_test, feat_names, imp_df, explainer


def _plot_shap_bar(shap_imp, feat_names, split, filename):
    top_idx = np.argsort(shap_imp)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(7, 8))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, TOP_N))
    ax.barh(np.array(feat_names)[top_idx], shap_imp[top_idx], color=colors)
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

    fig, ax = plt.subplots(figsize=(8, 9))
    rng = np.random.default_rng(SEED)
    for j, (sv_col, xv_col) in enumerate(zip(sv_top.T, xv_top.T)):
        y_jitter = rng.uniform(-0.35, 0.35, len(sv_col))
        norm_val = (xv_col - xv_col.min()) / (xv_col.ptp() + 1e-9)
        ax.scatter(sv_col, j + y_jitter, c=plt.cm.coolwarm(norm_val), s=6, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
    ax.set_title(
        f"SHAP Beeswarm – Top {TOP_N} features\n"
        "(colour: blue = low feature value, red = high)", fontsize=11,
    )
    fig.tight_layout()
    save_plot(fig, filename)


# ── Step 4b: Single-sample explanations ──────────────────────────────────────

def explain_single_test_sample(
    explainer, X_test, feat_names, y_test, y_pred, y_ci_low, y_ci_high, sample_idx=0
):
    _log(f"\n  [4b] Explaining sample index={sample_idx} …")
    n_features = len(feat_names)

    x_values   = X_test.iloc[sample_idx].values
    base_value = float(explainer.baseline_value)
    predicted  = float(y_pred[sample_idx])
    true_label = float(y_test[sample_idx])
    ci_lo      = float(y_ci_low[sample_idx])
    ci_hi      = float(y_ci_high[sample_idx])

    try:
        iv = explainer.explain(x_values, budget=SHAP_BUDGET)
        shap_1d = _iv_to_shap_array(iv, n_features)
    except Exception as exc:
        _log(f"  [WARN] explain() failed for sample {sample_idx}: {exc} — skipping plots.")
        return None

    _log(f"  [4b] base={base_value:.2f}  pred_mean={predicted:.2f}"
         f"  true={true_label:.2f}  CI=[{ci_lo:.2f}, {ci_hi:.2f}]"
         f"  error={abs(predicted - true_label):.2f}")

    single_df = pd.DataFrame({
        "feature":       feat_names,
        "feature_value": x_values,
        "shap_value":    shap_1d,
    }).sort_values("shap_value", key=np.abs, ascending=False)
    single_df["sample_idx"] = sample_idx
    single_df["true_label"] = true_label
    single_df["pred_mean"]  = predicted
    single_df["ci_low"]     = ci_lo
    single_df["ci_high"]    = ci_hi
    single_df["base_value"] = base_value
    save_csv(single_df, f"shap_single_sample_idx{sample_idx}.csv")

    # Waterfall plot via shapiq (no shap.initjs(), no headless crash)
    try:
        ax = shapiq.waterfall_plot(
            iv,
            feature_names=feat_names,
            max_display=TOP_N,
            show=False,
        )
        fig = ax.get_figure()
        fig.suptitle(
            f"SHAP Waterfall – test sample #{sample_idx}\n"
            f"true={true_label:.1f}  pred_mean={predicted:.1f}  "
            f"95 % CI=[{ci_lo:.1f}, {ci_hi:.1f}]",
            fontsize=11,
        )
        fig.tight_layout()
        save_plot(fig, f"shap_waterfall_sample_idx{sample_idx}.pdf")
    except Exception as exc:
        _log(f"  [WARN] waterfall_plot failed for sample {sample_idx}: {exc}")

    return shap_1d


def _top_k_accurate(mask: np.ndarray, abs_error: np.ndarray, k: int, label: str):
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        _log(f"  [WARN] No candidates found for '{label}' — skipping.")
        return []
    return idxs[np.argsort(abs_error[idxs])][:k].tolist()


def explain_notable_samples(
    explainer, X_test, feat_names, y_test, y_pred, y_ci_low, y_ci_high
):
    _log("\n── Step 4b: Single-sample SHAP explanations ─────────────────────")
    ci_width  = y_ci_high - y_ci_low
    abs_error = np.abs(y_pred - y_test)

    q25 = np.percentile(y_test, 25)
    q75 = np.percentile(y_test, 75)
    median_pred = np.median(y_pred)

    sens_mask = (y_test <= q25) & (y_pred <= median_pred)
    res_mask  = (y_test >= q75) & (y_pred >= median_pred)

    _log(f"  Sensitive candidates (AUC ≤ {q25:.1f} & pred ≤ {median_pred:.1f}): "
         f"{sens_mask.sum()} samples")
    _log(f"  Resistant candidates (AUC ≥ {q75:.1f} & pred ≥ {median_pred:.1f}): "
         f"{res_mask.sum()} samples")

    samples = {
        "first":       0,
        "widest_CI":   int(np.argmax(ci_width)),
        "worst_error": int(np.argmax(abs_error)),
    }
    for i, idx in enumerate(_top_k_accurate(sens_mask, abs_error, 2, "sensitive"), start=1):
        samples[f"sensitive_accurate_{i}"] = idx
    for i, idx in enumerate(_top_k_accurate(res_mask, abs_error, 2, "resistant"), start=1):
        samples[f"resistant_accurate_{i}"] = idx

    for label, idx in samples.items():
        _log(f"\n  -- {label} (index {idx}  true={y_test[idx]:.1f}  pred={y_pred[idx]:.1f}) --")
        explain_single_test_sample(
            explainer, X_test, feat_names, y_test, y_pred, y_ci_low, y_ci_high,
            sample_idx=idx,
        )

    _log("  [DONE] Step 4b complete.")


# ── Step 5: Embedding importance ──────────────────────────────────────────────

def compute_embedding_importance(model, X_train, X_test, feat_names):
    _log("\n── Step 5: Embedding-based (attention-proxy) importance ──────────")

    if not hasattr(model, "get_embeddings"):
        _log("  [SKIP] model.get_embeddings() not available in this TabPFN version.")
        _log("  [SKIP] Returning zero-filled importance arrays.")
        n_features = len(feat_names)
        dummy = np.zeros(n_features)
        return dummy, dummy, pd.DataFrame({
            "feature": feat_names,
            "embed_corr_train": dummy,
            "embed_corr_test":  dummy,
        })

    def _embedding_importance(X, split_label):
        _log(f"  [5] Getting transformer embeddings for {split_label} set ({len(X)} samples) …")
        try:
            emb      = model.get_embeddings(X, data_source="test")
            emb_avg  = emb.mean(axis=0) if emb.ndim == 3 else emb
            emb_norm = np.linalg.norm(emb_avg, axis=1)
            _log(f"  [5] Embedding shape: {emb_avg.shape}"
                 f"  |norm| range=[{emb_norm.min():.3f}, {emb_norm.max():.3f}]")
            X_arr = X.values
            # Vectorised Pearson: replaces per-column scipy loop (~1500 calls)
            std_X = X_arr.std(axis=0)
            std_e = emb_norm.std()
            X_c   = X_arr - X_arr.mean(axis=0)
            e_c   = emb_norm - emb_norm.mean()
            cov   = (X_c * e_c[:, None]).mean(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.where(std_X > 1e-9, np.abs(cov / (std_X * std_e + 1e-30)), 0.0)
            return corr, emb_norm
        except Exception as exc:
            _log(f"  [WARN] get_embeddings failed for {split_label}: {exc}")
            return np.zeros(X.shape[1]), np.zeros(len(X))

    train_corr, train_emb_norm = _embedding_importance(X_train, "train")
    test_corr,  test_emb_norm  = _embedding_importance(X_test,  "test")

    feat_arr = np.array(feat_names)
    # Patient features only for plots/reports — exclude LS_* drug-embedding dimensions
    patient_mask        = np.array([not f.startswith("LS_") for f in feat_names])
    patient_feats       = feat_arr[patient_mask].tolist()
    train_corr_patient  = train_corr[patient_mask]
    test_corr_patient   = test_corr[patient_mask]

    emb_imp_df = pd.DataFrame({
        "feature":          feat_names,
        "embed_corr_train": train_corr,
        "embed_corr_test":  test_corr,
    }).sort_values("embed_corr_test", ascending=False)
    save_csv(emb_imp_df, "embedding_attention_importance.csv")

    emb_imp_patient_df = pd.DataFrame({
        "feature":          patient_feats,
        "embed_corr_train": train_corr_patient,
        "embed_corr_test":  test_corr_patient,
    }).sort_values("embed_corr_test", ascending=False)
    save_csv(emb_imp_patient_df, "embedding_attention_importance_patient.csv")

    _log(f"  [5] Top 5 patient features by embedding correlation (test):")
    for row in emb_imp_patient_df.head(5).itertuples():
        _log(f"        {row.feature:<40s}  r={row.embed_corr_test:.4f}")

    _plot_embed_bar(test_corr_patient,  patient_feats, "Test (Patient Features)",  "embedding_importance_bar_test.pdf")
    _plot_embed_bar(train_corr_patient, patient_feats, "Train (Patient Features)", "embedding_importance_bar_train.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, norms, label in [(axes[0], train_emb_norm, "Train"), (axes[1], test_emb_norm, "Test")]:
        ax.hist(norms, bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Embedding L2 norm", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Transformer Embedding Norm – {label}", fontsize=10)
    fig.tight_layout()
    save_plot(fig, "embedding_norm_distribution.pdf")

    _log("  [DONE] Step 5 complete.")
    return train_corr, test_corr, emb_imp_df


def _plot_embed_bar(corr, feat_names, split, filename):
    top_idx = np.argsort(corr)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(7, 8))
    colors  = plt.cm.Blues(np.linspace(0.35, 0.95, TOP_N))
    ax.barh(np.array(feat_names)[top_idx], corr[top_idx], color=colors)
    ax.set_xlabel("|Pearson r| with embedding norm", fontsize=11)
    ax.set_title(f"Top {TOP_N} Embedding-Attention Importance – {split}", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    save_plot(fig, filename)


# ── Step 6: Combined summary ──────────────────────────────────────────────────

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
    ax.barh(y + bh/2, top["shap_norm"],      bh, label="SHAP (norm.)",      color="steelblue",  alpha=0.85)
    ax.barh(y - bh/2, top["embedding_norm"], bh, label="Embedding (norm.)", color="darkorange", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Normalised importance score", fontsize=11)
    ax.set_title(f"Top {TOP_N} Features – SHAP vs Embedding-Attention (Test)", fontsize=12)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    save_plot(fig, "combined_importance_comparison.pdf")

    _log(f"  Top 10 features by combined score:")
    _log(
        summary_df[["feature", "shap_score", "embedding_score", "combined_score"]]
        .head(10).to_string(index=False)
    )
    _log("  [DONE] Step 6 complete.")
    return summary_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    _log("=" * 70)
    _log("  TabPFN Ablation – SHAP (shapiq) + Attention Importance + CI Analysis")
    _log("  Experiment: baseline_all_features_Drug_Embed")
    _log("=" * 70)

    model, scaler = load_best_model()

    X_train, y_train, X_test, y_test, meta_train, meta_test, feat_names = \
        load_data(scaler)

    assert X_train.shape[1] == model.n_features_in_, (
        f"Feature count mismatch: data has {X_train.shape[1]}, "
        f"model expects {model.n_features_in_}"
    )
    _log(f"\n  Feature count check PASSED ({X_train.shape[1]} == model.n_features_in_)")

    sanity_check_predictions(model, X_test, y_test, n=10)

    y_train_pred, y_test_pred, y_ci_low, y_ci_high = predict_with_confidence(
        model, X_train, y_train, X_test, y_test, meta_test
    )

    shap_test, feat_names, imp_df, explainer = \
        compute_shap(model, X_train, y_train, X_test)

    shap_imp_test = np.abs(shap_test).mean(axis=0)

    # Save CSVs + plots if compute_shap loaded from cache (imp_df is None)
    if imp_df is None:
        imp_df = pd.DataFrame({
            "feature":            feat_names,
            "shap_mean_abs_test": shap_imp_test,
        }).sort_values("shap_mean_abs_test", ascending=False)
        save_csv(imp_df, "shap_feature_importance.csv")

        with open(SHAP_CACHE_PATH, "rb") as f:
            _c = pickle.load(f)
        X_test_sub = X_test.iloc[_c["test_idx"]].values

        save_csv(pd.DataFrame(shap_test.astype(np.float32), columns=feat_names),
                 "shap_values_test_subset.csv")
        _plot_shap_bar(shap_imp_test, feat_names, "Test", "shap_importance_bar_test.pdf")
        _plot_shap_beeswarm(shap_test, X_test_sub, feat_names, "shap_beeswarm_test.pdf")

    explain_notable_samples(
        explainer, X_test, feat_names, y_test, y_test_pred, y_ci_low, y_ci_high,
    )

    train_emb_corr, test_emb_corr, emb_df = \
        compute_embedding_importance(model, X_train, X_test, feat_names)

    summary_df = combined_importance_summary(shap_imp_test, test_emb_corr, feat_names)

    _log("\n" + "=" * 70)
    _log(f"  Analysis complete.  Outputs: {os.path.abspath(OUT_DIR)}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
