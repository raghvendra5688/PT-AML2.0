"""
tabpfn_best_model_shap_analysis.py

Ablation model: baseline_all_features_Drug_Embed (1561 features).
Feature selection and scaler loading aligned with tabpfn_ablation_study.py.

Steps:
  1. Load best TabPFN ablation model + saved scaler
  2. Load BeatAML train/test pickles; build feature subset via ablation logic
  3. Mean predictions + 95 % CI (quantiles) on train and test sets
  4. Population-level SHAP variable importance (KernelExplainer)
  4b. Per-sample SHAP explanation (waterfall + force plot) for 3 notable samples
  5. Attention-proxy importance via TabPFN transformer embeddings
  6. Combined importance summary (SHAP + embedding)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import torch
import shap

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
# ~120 MB (two 5000×1561 float64 arrays + DenseData background) — stored with big models.
SHAP_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_baseline_all_features_Drug_Embed_shap_cache.pk"
)

# Pool of rows drawn before k-means summarisation; keep large for centroid quality.
SHAP_BG_POOL    = 500
# Number of k-means centroids used as KernelExplainer background.
# coalition matrix per sample = SHAP_NSAMPLES × SHAP_BG_K = 512 × 50 = 25 600 rows
# (vs. 512 × 1000 = 512 000 with raw background → OOM on H200 140 GiB).
SHAP_BG_K       = 50
SHAP_TEST_SAMPLES = 5000
# 512 coalitions gives a good SHAP approximation at a fraction of the default cost
SHAP_NSAMPLES = 512
# Max rows per single TabPFN predict call inside KernelExplainer.
# TabPFN attention is O(n_test × n_train); batching keeps each forward pass
# well below GPU memory even when SHAP assembles large coalition matrices.
SHAP_PREDICT_BATCH = 4096

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


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  [saved CSV]  {path}")
    return path


def save_plot(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [saved plot] {path}")
    return path


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

    # sorted(set(...)) matches ablation_study.py line 128 exactly
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


def load_best_model():
    print("\n── Step 1: Loading best TabPFN ablation model ───────────────────")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Scaler : {SCALER_PATH}")
    model  = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    print(f"  Model type            : {type(model).__name__}")
    print(f"  n_estimators          : {model.n_estimators}")
    print(f"  n_features_in_        : {model.n_features_in_}")
    print(f"  ignore_pretraining_limits: {model.ignore_pretraining_limits}")
    print(f"  y_train_mean_         : {model.y_train_mean_:.4f}")
    print(f"  y_train_std_          : {model.y_train_std_:.4f}")
    return model, scaler


def load_data(scaler):
    print("\n── Step 2: Loading and preprocessing data ───────────────────────")
    print(f"  Training set : {TRAIN_PATH}")
    print(f"  Test set     : {TEST_PATH}")

    train_df = pd.read_pickle(TRAIN_PATH, compression="zip")
    test_df  = pd.read_pickle(TEST_PATH,  compression="zip")
    print(f"  Train raw shape : {train_df.shape}")
    print(f"  Test  raw shape : {test_df.shape}")

    feature_subset = get_feature_subset(train_df)
    print(f"  Feature subset  : {len(feature_subset)}")

    meta_cols = [c for c in train_df.columns if c not in feature_subset and c != "auc"]
    meta_train = train_df[meta_cols].copy()
    meta_test  = test_df[[c for c in meta_cols if c in test_df.columns]].copy()

    # Use the saved scaler (fit during ablation training) — do not refit
    X_train = pd.DataFrame(scaler.transform(train_df[feature_subset]), columns=feature_subset)
    X_test  = pd.DataFrame(scaler.transform(test_df[feature_subset]),  columns=feature_subset)
    y_train = train_df["auc"].to_numpy().flatten()
    y_test  = test_df["auc"].to_numpy().flatten()

    print(f"  Train feature matrix : {X_train.shape}")
    print(f"  Test  feature matrix : {X_test.shape}")
    print(f"  Train label range    : [{y_train.min():.1f}, {y_train.max():.1f}]"
          f"  mean={y_train.mean():.2f}")
    print(f"  Test  label range    : [{y_test.min():.1f}, {y_test.max():.1f}]"
          f"  mean={y_test.mean():.2f}")

    return X_train, y_train, X_test, y_test, meta_train, meta_test, feature_subset


def predict_with_confidence(model, X_train, y_train, X_test, y_test, meta_test):
    print("\n── Step 3: Predictions + 95 % confidence intervals ──────────────")

    print(f"  [3a] Predicting on training set ({len(X_train)} samples) …")
    y_train_pred  = model.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    print(f"  Train | MAE={train_metrics[0]}  RMSE={train_metrics[1]}"
          f"  R²={train_metrics[2]}  Pearson r={train_metrics[3]}")

    print(f"  [3b] Predicting on test set ({len(X_test)} samples) – mean …")
    y_test_pred  = model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    print(f"  Test  | MAE={test_metrics[0]}  RMSE={test_metrics[1]}"
          f"  R²={test_metrics[2]}  Pearson r={test_metrics[3]}")

    # Quantiles are fetched in a separate call; the point prediction is always
    # the mean, never the median (AUC is right-skewed so median ≈ +20-40 units).
    print(f"  [3c] Computing 95 % CI on test set (quantiles {CI_LOW}/{CI_HIGH}) …")
    ci_preds  = model.predict(X_test, output_type="quantiles", quantiles=[CI_LOW, CI_HIGH])
    y_ci_low  = ci_preds[0]
    y_ci_high = ci_preds[1]
    ci_width  = y_ci_high - y_ci_low

    print(f"  CI width | median={np.median(ci_width):.2f}  "
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

    print("  [3e] Saving prediction scatter plots …")
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

    return y_train_pred, y_test_pred, y_ci_low, y_ci_high


REF_CSV = "../Results/tabpfn/ablation/baseline_all_features_Drug_Embed_predictions.csv"

def sanity_check_predictions(model, X_test: pd.DataFrame, y_test: np.ndarray, n: int = 10) -> None:
    print("\n── Sanity check: compare first-N predictions to reference ────────")

    if not os.path.exists(REF_CSV):
        print(f"  [SKIP] Reference file not found: {REF_CSV}")
        return

    ref = pd.read_csv(REF_CSV, sep="\t")
    ref_preds  = ref["predictions"].values[:n]
    ref_labels = ref["labels"].values[:n]

    label_diffs = np.abs(y_test[:n] - ref_labels)
    if label_diffs.max() > 1e-3:
        print(f"  [ERROR] True-label mismatch (max diff={label_diffs.max():.4f}) — row ordering drifted.")
        return
    print(f"  Row-order check PASSED (max label diff = {label_diffs.max():.2e})")

    print(f"  Predicting first {n} test samples (mean) for comparison …")
    y_pred_n = model.predict(X_test.iloc[:n])

    print(f"\n  {'idx':>4}  {'true_AUC':>9}  {'ref_pred':>9}  {'new_pred':>9}  {'diff':>8}")
    print("  " + "-" * 50)
    diffs = []
    for i in range(n):
        diff = y_pred_n[i] - ref_preds[i]
        diffs.append(abs(diff))
        flag = "  ✓" if abs(diff) < 1.0 else "  ✗ MISMATCH"
        print(f"  {i:4d}  {y_test[i]:9.2f}  {ref_preds[i]:9.4f}  {y_pred_n[i]:9.4f}  {diff:+8.4f}{flag}")

    max_diff = max(diffs)
    if max_diff < 1.0:
        print(f"\n  PASS – max |diff| = {max_diff:.5f} AUC  (all within 1.0 tolerance)")
    else:
        print(f"\n  WARNING – max |diff| = {max_diff:.4f} AUC  (check preprocessing)")


def _predict_fn(X_array: np.ndarray, model: TabPFNRegressor) -> np.ndarray:
    # KernelExplainer passes large coalition matrices in one shot; TabPFN's
    # cross-attention is O(n_test × n_train) so a single 512k-row call OOMs.
    # Batching keeps each forward pass within GPU memory budget.
    chunks = [
        model.predict(pd.DataFrame(
            X_array[i:i + SHAP_PREDICT_BATCH],
            columns=model._shap_feature_names_,
        ))
        for i in range(0, len(X_array), SHAP_PREDICT_BATCH)
    ]
    return np.concatenate(chunks)


def _save_shap_cache(X_bg, expected_value, shap_train, shap_test, train_idx, test_idx) -> None:
    import pickle
    cache = {
        "X_bg":          X_bg,
        "expected_value": expected_value,
        "shap_train":    shap_train,
        "shap_test":     shap_test,
        "train_idx":     train_idx,
        "test_idx":      test_idx,
    }
    with open(SHAP_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"  [cache] Saved SHAP cache ({os.path.getsize(SHAP_CACHE_PATH)/1024**2:.1f} MB) → {SHAP_CACHE_PATH}")


def _load_shap_cache(model):
    import pickle
    print(f"  [cache] Loading SHAP cache from {SHAP_CACHE_PATH} …")
    with open(SHAP_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    X_bg          = cache["X_bg"]
    expected_value = cache["expected_value"]
    shap_train    = cache["shap_train"]
    shap_test     = cache["shap_test"]
    train_idx     = cache["train_idx"]
    test_idx      = cache["test_idx"]
    # Rebuild explainer from saved background (calls model.predict on 50 rows — fast).
    explainer = shap.KernelExplainer(lambda x: _predict_fn(x, model), X_bg, link="identity")
    # Override with the cached base value so results are identical to the original run.
    explainer.expected_value = expected_value
    print(f"  [cache] Loaded  shap_train={shap_train.shape}  shap_test={shap_test.shape}"
          f"  base={expected_value:.4f}")
    return explainer, shap_train, shap_test, train_idx, test_idx


def compute_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    print("\n── Step 4: SHAP variable importance ─────────────────────────────")

    feat_names = X_train.columns.tolist()
    model._shap_feature_names_ = feat_names

    if os.path.exists(SHAP_CACHE_PATH):
        print(f"  [cache] Cache found — skipping k-means, KernelExplainer build, and shap_values().")
        explainer, shap_train, shap_test, train_idx, test_idx = _load_shap_cache(model)
        X_train_sub = X_train.iloc[train_idx].values
        X_test_sub  = X_test.iloc[test_idx].values
    else:
        np.random.seed(SEED)
        bg_idx   = np.random.choice(len(X_train), min(SHAP_BG_POOL, len(X_train)), replace=False)
        X_bg_raw = X_train.iloc[bg_idx].values
        print(f"  [4a] Summarising {len(X_bg_raw)} background samples → {SHAP_BG_K} k-means centroids …")
        X_bg = shap.kmeans(X_bg_raw, SHAP_BG_K)
        print(f"  [4a] Background: {X_bg.data.shape[0]} centroids × {X_bg.data.shape[1]} features  "
              f"(coalition matrix per sample = {SHAP_NSAMPLES} × {SHAP_BG_K} = "
              f"{SHAP_NSAMPLES * SHAP_BG_K:,} rows)")

        print(f"  [4b] Building KernelExplainer …")
        explainer = shap.KernelExplainer(lambda x: _predict_fn(x, model), X_bg, link="identity")
        print(f"  [4b] KernelExplainer expected value (base): {explainer.expected_value:.4f}")

        train_idx   = np.random.choice(len(X_train), min(SHAP_TEST_SAMPLES, len(X_train)), replace=False)
        X_train_sub = X_train.iloc[train_idx].values
        print(f"  [4c] Computing SHAP on {len(X_train_sub)} training samples"
              f"  (nsamples={SHAP_NSAMPLES}) …")
        shap_train = explainer.shap_values(X_train_sub, nsamples=SHAP_NSAMPLES, silent=True)

        test_idx   = np.random.choice(len(X_test), min(SHAP_TEST_SAMPLES, len(X_test)), replace=False)
        X_test_sub = X_test.iloc[test_idx].values
        print(f"  [4d] Computing SHAP on {len(X_test_sub)} test samples"
              f"  (nsamples={SHAP_NSAMPLES}) …")
        shap_test = explainer.shap_values(X_test_sub, nsamples=SHAP_NSAMPLES, silent=True)

        _save_shap_cache(X_bg, explainer.expected_value, shap_train, shap_test, train_idx, test_idx)

    save_csv(pd.DataFrame(shap_train, columns=feat_names), "shap_values_train_subset.csv")
    save_csv(pd.DataFrame(shap_test,  columns=feat_names), "shap_values_test_subset.csv")

    shap_imp_train = np.abs(shap_train).mean(axis=0)
    shap_imp_test  = np.abs(shap_test).mean(axis=0)

    imp_df = pd.DataFrame({
        "feature":             feat_names,
        "shap_mean_abs_train": shap_imp_train,
        "shap_mean_abs_test":  shap_imp_test,
    }).sort_values("shap_mean_abs_test", ascending=False)
    save_csv(imp_df, "shap_feature_importance.csv")

    _plot_shap_bar(shap_imp_test,  feat_names, "Test",  "shap_importance_bar_test.pdf")
    _plot_shap_bar(shap_imp_train, feat_names, "Train", "shap_importance_bar_train.pdf")
    _plot_shap_beeswarm(shap_test,  X_test_sub,  feat_names, "shap_beeswarm_test.pdf")
    _plot_shap_beeswarm(shap_train, X_train_sub, feat_names, "shap_beeswarm_train.pdf")

    print(f"  [4h] Top 5 features by mean |SHAP| (test):")
    for row in imp_df.head(5).itertuples():
        print(f"        {row.feature:<40s}  {row.shap_mean_abs_test:.4f}")

    return shap_imp_train, shap_imp_test, feat_names, imp_df, explainer


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


def explain_single_test_sample(
    explainer, X_test, feat_names, y_test, y_pred, y_ci_low, y_ci_high, sample_idx=0
):
    print(f"\n  [4b] Explaining sample index={sample_idx} …")

    # iloc[[idx]] keeps 2-D shape (1, n_features); iloc[idx] would give a 1-D Series.
    X_single   = X_test.iloc[[sample_idx]].values
    x_values   = X_single[0]
    shap_2d    = explainer.shap_values(X_single, nsamples=SHAP_NSAMPLES, silent=True)
    shap_1d    = shap_2d[0]
    base_value = float(explainer.expected_value)
    predicted  = float(y_pred[sample_idx])
    true_label = float(y_test[sample_idx])
    ci_lo      = float(y_ci_low[sample_idx])
    ci_hi      = float(y_ci_high[sample_idx])

    print(f"  [4b] base={base_value:.2f}  pred_mean={predicted:.2f}"
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

    explanation = shap.Explanation(
        values=shap_1d, base_values=base_value, data=x_values, feature_names=feat_names,
    )
    fig, ax = plt.subplots(figsize=(9, 10))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=TOP_N, show=False)
    ax.set_title(
        f"SHAP Waterfall – test sample #{sample_idx}\n"
        f"true={true_label:.1f}  pred_mean={predicted:.1f}  "
        f"95 % CI=[{ci_lo:.1f}, {ci_hi:.1f}]",
        fontsize=11,
    )
    fig.tight_layout()
    save_plot(fig, f"shap_waterfall_sample_idx{sample_idx}.pdf")

    shap.initjs()
    shap.force_plot(
        base_value, shap_1d, x_values,
        feature_names=feat_names,
        matplotlib=True, show=False,
        contribution_threshold=0.05,
    )
    plt.title(
        f"SHAP Force Plot – test sample #{sample_idx}  "
        f"(pred_mean={predicted:.1f}, true={true_label:.1f})",
        fontsize=10, pad=14,
    )
    plt.tight_layout()
    save_plot(plt.gcf(), f"shap_force_plot_sample_idx{sample_idx}.pdf")

    return shap_1d


def _top_k_accurate(mask: np.ndarray, abs_error: np.ndarray, k: int, label: str):
    """Return indices of the k most accurately predicted samples within mask."""
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        print(f"  [WARN] No candidates found for '{label}' — skipping.")
        return []
    return idxs[np.argsort(abs_error[idxs])][:k].tolist()


def explain_notable_samples(
    explainer, X_test, feat_names, y_test, y_pred, y_ci_low, y_ci_high
):
    print("\n── Step 4b: Single-sample SHAP explanations ─────────────────────")
    ci_width  = y_ci_high - y_ci_low
    abs_error = np.abs(y_pred - y_test)

    # Thresholds: bottom 25 % = sensitive (high drug response), top 25 % = resistant.
    # Prediction alignment ensures the model is correctly calling the phenotype.
    q25 = np.percentile(y_test, 25)
    q75 = np.percentile(y_test, 75)
    median_pred = np.median(y_pred)

    sens_mask = (y_test <= q25) & (y_pred <= median_pred)   # true sensitive, pred sensitive
    res_mask  = (y_test >= q75) & (y_pred >= median_pred)   # true resistant, pred resistant

    print(f"  Sensitive candidates (AUC ≤ {q25:.1f} & pred ≤ {median_pred:.1f}): "
          f"{sens_mask.sum()} samples")
    print(f"  Resistant candidates (AUC ≥ {q75:.1f} & pred ≥ {median_pred:.1f}): "
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
        print(f"\n  -- {label} (index {idx}  true={y_test[idx]:.1f}  pred={y_pred[idx]:.1f}) --")
        explain_single_test_sample(
            explainer, X_test, feat_names, y_test, y_pred, y_ci_low, y_ci_high,
            sample_idx=idx,
        )


def compute_embedding_importance(model, X_train, X_test, feat_names):
    """Importance proxy: |Pearson r| between each feature and the L2 norm of
    the sample's average transformer embedding across estimators."""
    print("\n── Step 5: Embedding-based (attention-proxy) importance ──────────")

    def _embedding_importance(X, split_label):
        print(f"  [5] Getting transformer embeddings for {split_label} set ({len(X)} samples) …")
        emb      = model.get_embeddings(X, data_source="test")
        emb_avg  = emb.mean(axis=0) if emb.ndim == 3 else emb
        emb_norm = np.linalg.norm(emb_avg, axis=1)
        print(f"  [5] Embedding shape: {emb_avg.shape}"
              f"  |norm| range=[{emb_norm.min():.3f}, {emb_norm.max():.3f}]")
        X_arr = X.values
        corr = np.array([
            abs(scipy.stats.pearsonr(X_arr[:, j], emb_norm)[0])
            if np.std(X_arr[:, j]) > 1e-9 else 0.0
            for j in range(X_arr.shape[1])
        ])
        return corr, emb_norm

    train_corr, train_emb_norm = _embedding_importance(X_train, "train")
    test_corr,  test_emb_norm  = _embedding_importance(X_test,  "test")

    emb_imp_df = pd.DataFrame({
        "feature":          feat_names,
        "embed_corr_train": train_corr,
        "embed_corr_test":  test_corr,
    }).sort_values("embed_corr_test", ascending=False)
    save_csv(emb_imp_df, "embedding_attention_importance.csv")

    print(f"  [5] Top 5 features by embedding correlation (test):")
    for row in emb_imp_df.head(5).itertuples():
        print(f"        {row.feature:<40s}  r={row.embed_corr_test:.4f}")

    _plot_embed_bar(test_corr,  feat_names, "Test",  "embedding_importance_bar_test.pdf")
    _plot_embed_bar(train_corr, feat_names, "Train", "embedding_importance_bar_train.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, norms, label in [(axes[0], train_emb_norm, "Train"), (axes[1], test_emb_norm, "Test")]:
        ax.hist(norms, bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Embedding L2 norm", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Transformer Embedding Norm – {label}", fontsize=10)
    fig.tight_layout()
    save_plot(fig, "embedding_norm_distribution.pdf")

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


def combined_importance_summary(shap_imp_test, embed_corr_test, feat_names):
    print("\n── Step 6: Combined importance summary ───────────────────────────")

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

    print(f"  Top 10 features by combined score:")
    print(
        summary_df[["feature", "shap_score", "embedding_score", "combined_score"]]
        .head(10).to_string(index=False)
    )
    return summary_df


def main():
    ensure_dirs()
    print("=" * 70)
    print("  TabPFN Ablation – SHAP + Attention Importance + CI Analysis")
    print("  Experiment: baseline_all_features_Drug_Embed")
    print("=" * 70)

    model, scaler = load_best_model()

    X_train, y_train, X_test, y_test, meta_train, meta_test, feat_names = \
        load_data(scaler)

    assert X_train.shape[1] == model.n_features_in_, (
        f"Feature count mismatch: data has {X_train.shape[1]}, "
        f"model expects {model.n_features_in_}"
    )
    print(f"\n  Feature count check PASSED ({X_train.shape[1]} == model.n_features_in_)")

    # Run before SHAP so any preprocessing mismatch surfaces early (SHAP takes hours).
    sanity_check_predictions(model, X_test, y_test, n=10)

    y_train_pred, y_test_pred, y_ci_low, y_ci_high = predict_with_confidence(
        model, X_train, y_train, X_test, y_test, meta_test
    )

    shap_imp_train, shap_imp_test, feat_names, shap_df, explainer = \
        compute_shap(model, X_train, X_test)

    explain_notable_samples(
        explainer, X_test, feat_names, y_test, y_test_pred, y_ci_low, y_ci_high,
    )

    train_emb_corr, test_emb_corr, emb_df = \
        compute_embedding_importance(model, X_train, X_test, feat_names)

    summary_df = combined_importance_summary(shap_imp_test, test_emb_corr, feat_names)

    print("\n" + "=" * 70)
    print(f"  Analysis complete.  Outputs: {os.path.abspath(OUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
