"""
tabpfn_shap_v2.py

Steps 5–6 of the best-model analysis pipeline.  Takes embedding-importance
input produced by tabpfn_predict_embed.py and computes SHAP variable
importance using the full training context and all features.

Steps executed here:
  5.  SHAP variable importance  (TabPFNExplainer, PermutationSamplingSV)
  5b. Per-sample SHAP waterfall for notable test samples
  6.  Combined importance summary (SHAP + embedding-attention)

Prerequisites — files written by tabpfn_predict_embed.py must exist in OUT_DIR:
  test_predictions_with_CI.csv
  embedding_attention_importance.csv

Design notes
────────────
• Approximator: PermutationSamplingSV, NOT KernelSHAP.
  KernelSHAP solves a weighted LS system with `budget` equations and
  `n_features` unknowns.  For n_features=1561 the system needs budget >> 3122
  for stability — infeasible with the remove-and-contextualize paradigm
  (each coalition = one model.fit(33 K rows) call).
  PermutationSamplingSV estimates marginal contributions via random feature
  orderings; it does NOT require budget >> n_features and yields stable
  per-feature estimates even with a handful of complete orderings.

• Budget: SHAP_BUDGET = 8192 coalitions per sample.
  Each complete permutation of 1561 features costs 1561 coalition evaluations,
  so 8192 budget ≈ 5 full orderings.  Five independent estimates per feature
  are sufficient for stable mean attribution at the level of the top features.
  Timing: each coalition evaluation re-contextualises TabPFN on 33 K rows
  (~3–10 s on H200 with tabpfn 8.0.2).  8192 coalitions × ~5 s = ~11 h per
  sample.  For 120 samples (20 train + 100 test) this is infeasible in one
  36 h job; recommended approach is to submit separate jobs for train/test
  subsets.  Set SHAP_TRAIN_SAMPLES and SHAP_TEST_SAMPLES accordingly.

• Context: full training set (33 K rows) passed via `data=X_train` to
  TabPFNExplainer.  `x_test=X_test` is passed to prevent the default 80/20
  split of training context and to compute the empty-prediction baseline on
  held-out test data.

• Model: the saved best model (not a fresh proxy).  TabPFNImputer calls
  model.fit() for each coalition anyway (remove-and-contextualize), so the
  coalition predictions reflect the same TabPFN foundation weights and
  n_estimators as the original model — only the feature subset changes.

Caches (v3):
  SHAP_TRAIN_CACHE_PATH — after train SHAP (crash-safe checkpoint)
  SHAP_CACHE_PATH       — after test SHAP (full checkpoint)
  Script resumes from the deepest checkpoint found on restart.
"""

import os
import sys
import gc
import time
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import shapiq
from shapiq import TabPFNExplainer

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

# Cache paths (v3 — separate namespace from old v2 caches)
SHAP_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_shap_v3_full_cache.pk"
)
SHAP_TRAIN_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_shap_v3_train_partial.pk"
)

# ── SHAP hyper-parameters ─────────────────────────────────────────────────────
#
# Stability: PermutationSamplingSV does NOT need budget >> n_features (unlike
# KernelSHAP).  Each complete permutation of 1561 features costs 1561 coalitions.
# SHAP_BUDGET = 8192 → ≈ 5 complete orderings → stable top-feature attribution.
#
# Timing per sample (tabpfn 8.0.2, H200, 33 K context):
#   ~3–10 s/coalition × 8192 = 7–23 h/sample.
#
# Recommended job split:
#   Job A (train):  SHAP_TRAIN_SAMPLES = 5  → ~35–115 h total  (reduce as needed)
#   Job B (test):   SHAP_TEST_SAMPLES  = 20 → ~7–23 h × 20 = long; split further
#
# Adjust SHAP_BUDGET downward (e.g. 2048) for shorter test runs; results will
# be noisier but qualitatively correct (unlike KernelSHAP with low budget).
#
SHAP_BUDGET          = 8192   # coalitions per sample; ≈ 5 permutations of 1561 features
SHAP_TEST_SAMPLES    = 100    # how many test samples to explain
SHAP_TRAIN_SAMPLES   = 20     # how many train samples to explain

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

def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_cache(path: str, data: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)
    _log(f"  [cache] {os.path.getsize(path)/1024**2:.1f} MB → {path}")


# ── Steps 1 + 2: Load model and data ──────────────────────────────────────────

def get_feature_subset(train_df: pd.DataFrame) -> list:
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


def load_model_and_data():
    _log("\n── Steps 1+2: Loading model, scaler, and data ───────────────────")
    model  = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    _log(f"  Model  : n_estimators={model.n_estimators}  fit_mode={model.fit_mode}"
         f"  n_features_in_={model.n_features_in_}  device={model.device}")

    train_df = pd.read_pickle(TRAIN_PATH, compression="zip")
    test_df  = pd.read_pickle(TEST_PATH,  compression="zip")

    feature_subset = get_feature_subset(train_df)
    _log(f"  Feature subset: {len(feature_subset)} features")

    X_train = pd.DataFrame(scaler.transform(train_df[feature_subset]), columns=feature_subset)
    X_test  = pd.DataFrame(scaler.transform(test_df[feature_subset]),  columns=feature_subset)
    y_train = train_df["auc"].to_numpy().flatten()
    y_test  = test_df["auc"].to_numpy().flatten()

    assert X_train.shape[1] == model.n_features_in_, (
        f"Feature mismatch: data={X_train.shape[1]}  model expects {model.n_features_in_}"
    )
    _log(f"  X_train {X_train.shape}  X_test {X_test.shape}  "
         f"feature check PASSED ({X_train.shape[1]} == model.n_features_in_)")
    _log("  [DONE] Steps 1+2.")
    return model, X_train, y_train, X_test, y_test, feature_subset


# ── Load Step 3/4 outputs produced by tabpfn_predict_embed.py ────────────────

def load_predictions_and_embeddings(feat_names: list):
    _log("\n── Loading Step 3/4 outputs from tabpfn_predict_embed.py ────────")

    pred_csv  = os.path.join(OUT_DIR, "test_predictions_with_CI.csv")
    embed_csv = os.path.join(OUT_DIR, "embedding_attention_importance.csv")

    for p in [pred_csv, embed_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — run tabpfn_predict_embed.py first."
            )

    pred_df = pd.read_csv(pred_csv)
    y_test_pred = pred_df["pred_mean"].values
    y_ci_low    = pred_df["ci_low"].values
    y_ci_high   = pred_df["ci_high"].values
    _log(f"  Predictions loaded: {len(y_test_pred)} test samples  "
         f"pred range=[{y_test_pred.min():.1f}, {y_test_pred.max():.1f}]")

    emb_df   = pd.read_csv(embed_csv).set_index("feature")
    feat_arr = np.array(feat_names)
    embed_corr_test = emb_df["embed_corr_test"].reindex(feat_arr).fillna(0.0).values
    _log(f"  Embedding importance loaded: {len(embed_corr_test)} features  "
         f"max={embed_corr_test.max():.4f}")

    _log("  [DONE] Loading prerequisites.")
    return y_test_pred, y_ci_low, y_ci_high, embed_corr_test


# ── Step 5: TabPFNExplainer ────────────────────────────────────────────────────

def _build_explainer(model, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray) -> TabPFNExplainer:
    """Build TabPFNExplainer on the FULL training context.

    x_test=X_test is passed to:
      (a) prevent TabPFNExplainer's default 80/20 split of training data, so
          ALL training rows are used as context (not just 80 %).
      (b) compute the empty-prediction baseline on held-out test rows.

    approximator="permutation" (PermutationSamplingSV):
      Estimates marginal contributions via random feature orderings.  Does NOT
      solve a linear system, so it avoids the underdetermined-system instability
      that KernelSHAP produces when budget << 2 × n_features.
    """
    n_train, n_feat = X_train.shape
    _log(f"  [5c] Building TabPFNExplainer: context={n_train} rows × {n_feat} features")
    _log(f"       approximator=permutation  budget={SHAP_BUDGET}  "
         f"≈{SHAP_BUDGET // n_feat} complete orderings of {n_feat} features")

    t0 = time.time()
    explainer = TabPFNExplainer(
        model=model,
        data=X_train,
        labels=y_train,
        x_test=X_test,           # avoids 80/20 split; baseline computed on test
        index="SV",
        max_order=1,
        approximator="permutation",
        verbose=False,
    )
    elapsed = time.time() - t0
    _log(f"  [5c] Explainer built in {elapsed:.1f}s  "
         f"baseline={explainer.baseline_value:.4f}")
    return explainer


def _iv_to_shap_array(iv, n_features: int) -> np.ndarray:
    """Extract first-order Shapley values from an InteractionValues object."""
    return np.array([iv[(i,)] for i in range(n_features)])


def _explain_loop(explainer: TabPFNExplainer,
                  X_sub: np.ndarray,
                  label: str) -> np.ndarray:
    """Explain each row; return SHAP matrix (n_samples, n_features). Failed rows → zero."""
    n_samples, n_features = X_sub.shape
    shap_matrix = np.zeros((n_samples, n_features), dtype=np.float64)
    t_start = time.time()
    for i, x_i in enumerate(tqdm(X_sub, desc=f"SHAP [{label}]", unit="sample")):
        t_sample = time.time()
        try:
            iv = explainer.explain(x_i, budget=SHAP_BUDGET)
            shap_matrix[i] = _iv_to_shap_array(iv, n_features)
            shap_sum = shap_matrix[i].sum()
            _log(f"  [{label}] sample {i:3d}: "
                 f"sum={shap_sum:.2f}  "
                 f"max|v|={np.abs(shap_matrix[i]).max():.3f}  "
                 f"elapsed={time.time()-t_sample:.0f}s")
        except Exception as exc:
            _log(f"  [WARN] {label} sample {i}: {exc} — row set to zero.")
    total = time.time() - t_start
    _log(f"  [{label}] {n_samples} samples in {total/3600:.2f}h  "
         f"({total/n_samples:.0f}s/sample avg)")
    return shap_matrix


def compute_shap(model, X_train: pd.DataFrame, y_train: np.ndarray,
                 X_test: pd.DataFrame):
    """Run TabPFNExplainer SHAP with two-level crash-safe checkpointing."""
    _log("\n── Step 5: SHAP variable importance (TabPFNExplainer) ───────────")
    _log(f"  approximator=permutation  budget={SHAP_BUDGET}"
         f"  train_n={SHAP_TRAIN_SAMPLES}  test_n={SHAP_TEST_SAMPLES}")
    _log(f"  context: FULL training set ({len(X_train)} rows × {X_train.shape[1]} features)")

    feat_names  = X_train.columns.tolist()
    n_features  = len(feat_names)
    X_train_np  = X_train.values.astype(np.float64)
    X_test_np   = X_test.values.astype(np.float64)

    # ── Path A: full cache ────────────────────────────────────────────────────
    if os.path.exists(SHAP_CACHE_PATH):
        _log("  [cache] Full cache found — loading SHAP values.")
        cache      = _load_pickle(SHAP_CACHE_PATH)
        shap_train = cache["shap_train"]
        shap_test  = cache["shap_test"]
        train_idx  = cache["train_idx"]
        test_idx   = cache["test_idx"]
        expected_value = cache["expected_value"]
        explainer  = _build_explainer(model, X_train_np, y_train, X_test_np)
        _log(f"  shap_train={shap_train.shape}  shap_test={shap_test.shape}"
             f"  baseline={expected_value:.4f}")

    # ── Path B: partial cache (train done, test pending) ─────────────────────
    elif os.path.exists(SHAP_TRAIN_CACHE_PATH):
        _log("  [cache] Partial cache found — resuming from test SHAP.")
        partial        = _load_pickle(SHAP_TRAIN_CACHE_PATH)
        shap_train     = partial["shap_train"]
        train_idx      = partial["train_idx"]
        expected_value = partial["expected_value"]
        explainer      = _build_explainer(model, X_train_np, y_train, X_test_np)

        rng      = np.random.default_rng(SEED + 1)
        test_idx = rng.choice(len(X_test), min(SHAP_TEST_SAMPLES, len(X_test)), replace=False)
        _log(f"  [5d] Computing test SHAP: {len(test_idx)} samples × {n_features} features …")
        shap_test = _explain_loop(explainer, X_test_np[test_idx], "test")

        _save_cache(SHAP_CACHE_PATH, dict(
            shap_train=shap_train, shap_test=shap_test,
            train_idx=train_idx, test_idx=test_idx,
            expected_value=expected_value,
        ))

    # ── Path C: compute from scratch ──────────────────────────────────────────
    else:
        explainer      = _build_explainer(model, X_train_np, y_train, X_test_np)
        expected_value = float(explainer.baseline_value)

        rng       = np.random.default_rng(SEED)
        train_idx = rng.choice(len(X_train), min(SHAP_TRAIN_SAMPLES, len(X_train)), replace=False)
        _log(f"  [5d] Computing train SHAP: {len(train_idx)} samples × {n_features} features …")
        shap_train = _explain_loop(explainer, X_train_np[train_idx], "train")

        _save_cache(SHAP_TRAIN_CACHE_PATH, dict(
            shap_train=shap_train, train_idx=train_idx, expected_value=expected_value,
        ))

        rng      = np.random.default_rng(SEED + 1)
        test_idx = rng.choice(len(X_test), min(SHAP_TEST_SAMPLES, len(X_test)), replace=False)
        _log(f"  [5e] Computing test SHAP: {len(test_idx)} samples × {n_features} features …")
        shap_test = _explain_loop(explainer, X_test_np[test_idx], "test")

        _save_cache(SHAP_CACHE_PATH, dict(
            shap_train=shap_train, shap_test=shap_test,
            train_idx=train_idx, test_idx=test_idx,
            expected_value=expected_value,
        ))

    _log(f"  SHAP stats (test): "
         f"mean|v|={np.abs(shap_test).mean():.3f}  "
         f"max|v|={np.abs(shap_test).max():.3f}  "
         f"baseline={expected_value:.4f}")
    _log("  [DONE] Step 5.")
    return shap_train, shap_test, explainer, train_idx, test_idx, expected_value, feat_names


# ── SHAP output plots and CSVs ─────────────────────────────────────────────────

def save_shap_outputs(shap_train: np.ndarray, shap_test: np.ndarray,
                      feat_names: list,
                      X_train: pd.DataFrame, X_test: pd.DataFrame,
                      train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Write SHAP CSVs and bar/beeswarm plots (skipped if already present)."""
    if os.path.exists(os.path.join(OUT_DIR, "shap_v3_feature_importance.csv")):
        _log("  [cache] SHAP v3 outputs already exist — skipping regeneration.")
        return

    shap_imp_train = np.abs(shap_train).mean(axis=0)
    shap_imp_test  = np.abs(shap_test).mean(axis=0)

    save_csv(pd.DataFrame(shap_train.astype(np.float32), columns=feat_names),
             "shap_v3_values_train_subset.csv")
    save_csv(pd.DataFrame(shap_test.astype(np.float32), columns=feat_names),
             "shap_v3_values_test_subset.csv")

    imp_df = pd.DataFrame({
        "feature":             feat_names,
        "shap_mean_abs_train": shap_imp_train,
        "shap_mean_abs_test":  shap_imp_test,
    }).sort_values("shap_mean_abs_test", ascending=False)
    save_csv(imp_df, "shap_v3_feature_importance.csv")

    _log("  Top 10 features by mean |SHAP| (test):")
    for row in imp_df.head(10).itertuples():
        _log(f"    {row.feature:<40s}  {row.shap_mean_abs_test:.4f} AUC")

    _plot_shap_bar(shap_imp_test,  feat_names, "Test",  "shap_v3_importance_bar_test.pdf")
    _plot_shap_bar(shap_imp_train, feat_names, "Train", "shap_v3_importance_bar_train.pdf")
    _plot_shap_beeswarm(shap_test,  X_test.iloc[test_idx].values,  feat_names,
                        "shap_v3_beeswarm_test.pdf")
    _plot_shap_beeswarm(shap_train, X_train.iloc[train_idx].values, feat_names,
                        "shap_v3_beeswarm_train.pdf")


def _plot_shap_bar(shap_imp: np.ndarray, feat_names: list,
                   split: str, filename: str) -> None:
    top_idx = np.argsort(shap_imp)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(np.array(feat_names)[top_idx], shap_imp[top_idx],
            color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, TOP_N)))
    ax.set_xlabel("Mean |SHAP value| (AUC units)", fontsize=11)
    ax.set_title(f"Top {TOP_N} SHAP Feature Importance – {split}", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    save_plot(fig, filename)


def _plot_shap_beeswarm(shap_vals: np.ndarray, X_sub: np.ndarray,
                         feat_names: list, filename: str) -> None:
    shap_imp = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(shap_imp)[-TOP_N:]
    sv_top   = shap_vals[:, top_idx]
    xv_top   = X_sub[:, top_idx]
    top_names = np.array(feat_names)[top_idx]

    fig, ax = plt.subplots(figsize=(8, 9))
    rng = np.random.default_rng(SEED)
    for j, (sv_col, xv_col) in enumerate(zip(sv_top.T, xv_top.T)):
        y_jitter = rng.uniform(-0.35, 0.35, len(sv_col))
        norm_val = (xv_col - xv_col.min()) / ((xv_col.max() - xv_col.min()) + 1e-9)
        ax.scatter(sv_col, j + y_jitter, c=plt.cm.coolwarm(norm_val),
                   s=6, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (AUC units — impact on prediction)", fontsize=11)
    ax.set_title(f"SHAP Beeswarm – Top {TOP_N} features\n"
                 "(colour: blue = low feature value, red = high)", fontsize=11)
    fig.tight_layout()
    save_plot(fig, filename)


# ── Step 5b: Single-sample waterfall ──────────────────────────────────────────

def explain_notable_samples(explainer: TabPFNExplainer,
                             X_test: pd.DataFrame,
                             feat_names: list,
                             y_test: np.ndarray,
                             y_pred: np.ndarray,
                             y_ci_low: np.ndarray,
                             y_ci_high: np.ndarray) -> None:
    _log("\n── Step 5b: Single-sample SHAP waterfall (notable test samples) ─")
    abs_error   = np.abs(y_pred - y_test)
    q25, q75    = np.percentile(y_test, [25, 75])
    median_pred = np.median(y_pred)

    sens_mask = (y_test <= q25) & (y_pred <= median_pred)
    res_mask  = (y_test >= q75) & (y_pred >= median_pred)
    _log(f"  Sensitive candidates: {sens_mask.sum()}  Resistant: {res_mask.sum()}")

    def _top_k_accurate(mask, k, label):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            _log(f"  [WARN] No candidates for '{label}'.")
            return []
        return idxs[np.argsort(abs_error[idxs])][:k].tolist()

    samples = {
        "first":        0,
        "widest_CI":    int(np.argmax(y_ci_high - y_ci_low)),
        "worst_error":  int(np.argmax(abs_error)),
    }
    for i, idx in enumerate(_top_k_accurate(sens_mask, 2, "sensitive"), start=1):
        samples[f"sensitive_{i}"] = idx
    for i, idx in enumerate(_top_k_accurate(res_mask,  2, "resistant"),  start=1):
        samples[f"resistant_{i}"] = idx

    X_test_np = X_test.values.astype(np.float64)
    for label, idx in samples.items():
        _log(f"\n  -- {label}  idx={idx}  true={y_test[idx]:.1f}"
             f"  pred={y_pred[idx]:.1f}  err={abs_error[idx]:.2f} --")
        _explain_single(explainer, X_test_np[idx], feat_names,
                        idx, y_test[idx], y_pred[idx],
                        y_ci_low[idx], y_ci_high[idx],
                        float(explainer.baseline_value))
    _log("  [DONE] Step 5b.")


def _explain_single(explainer, x_1d: np.ndarray, feat_names: list,
                    sample_idx: int, true_val: float, pred_val: float,
                    ci_lo: float, ci_hi: float, base_value: float) -> None:
    out_csv = os.path.join(OUT_DIR, f"shap_v3_single_sample_idx{sample_idx}.csv")
    if os.path.exists(out_csv):
        _log(f"  [cache] {out_csv} — skipping.")
        return

    try:
        iv      = explainer.explain(x_1d, budget=SHAP_BUDGET)
        shap_1d = _iv_to_shap_array(iv, len(feat_names))
    except Exception as exc:
        _log(f"  [WARN] explain() failed for sample {sample_idx}: {exc}")
        return

    shap_sum = shap_1d.sum()
    _log(f"  base={base_value:.2f}  pred={pred_val:.2f}  true={true_val:.2f}"
         f"  CI=[{ci_lo:.2f},{ci_hi:.2f}]  SHAP_sum={shap_sum:.2f}"
         f"  (pred−base={pred_val-base_value:.2f})")

    df = pd.DataFrame({
        "feature":       feat_names,
        "feature_value": x_1d,
        "shap_value":    shap_1d,
    }).sort_values("shap_value", key=np.abs, ascending=False)
    for col, val in [("sample_idx", sample_idx), ("true_label", true_val),
                     ("pred_mean", pred_val), ("ci_low", ci_lo),
                     ("ci_high", ci_hi), ("base_value", base_value)]:
        df[col] = val
    save_csv(df, f"shap_v3_single_sample_idx{sample_idx}.csv")

    try:
        ax  = shapiq.waterfall_plot(iv, feature_names=feat_names, max_display=10, show=False)
        fig = ax.get_figure()
        fig.suptitle(f"SHAP Waterfall – test sample #{sample_idx}\n"
                     f"true={true_val:.1f}  pred={pred_val:.1f}"
                     f"  95 % CI=[{ci_lo:.1f},{ci_hi:.1f}]", fontsize=11)
        fig.tight_layout()
        save_plot(fig, f"shap_v3_waterfall_sample_idx{sample_idx}.pdf")
    except Exception as exc:
        _log(f"  [WARN] waterfall_plot failed: {exc}")


# ── Step 6: Combined importance summary ───────────────────────────────────────

def combined_importance_summary(shap_imp_test: np.ndarray,
                                 embed_corr_test: np.ndarray,
                                 feat_names: list) -> pd.DataFrame:
    _log("\n── Step 6: Combined importance summary ───────────────────────────")

    def _minmax(v: np.ndarray) -> np.ndarray:
        vr = v.max() - v.min()
        return (v - v.min()) / vr if vr > 1e-9 else np.zeros_like(v)

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
    save_csv(summary_df, "shap_v3_combined_importance_summary.csv")

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
    save_plot(fig, "shap_v3_combined_importance_comparison.pdf")

    _log("  Top 10 by combined score:")
    _log(summary_df[["feature", "shap_score", "embedding_score", "combined_score"]]
         .head(10).to_string(index=False))
    _log("  [DONE] Step 6.")
    return summary_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    _log("=" * 70)
    _log("  TabPFN – SHAP v3 (Steps 5–6)")
    _log("  Experiment : baseline_all_features_Drug_Embed")
    _log(f"  shapiq     : {shapiq.__version__}")
    _log(f"  SHAP params: approximator=permutation  budget={SHAP_BUDGET}"
         f"  train_n={SHAP_TRAIN_SAMPLES}  test_n={SHAP_TEST_SAMPLES}")
    gpu_info = (f"cuda:{torch.cuda.get_device_name(0)}"
                if DEVICE == "cuda" else "cpu (no CUDA)")
    _log(f"  Device     : {gpu_info}")
    _log("=" * 70)

    model, X_train, y_train, X_test, y_test, feat_names = load_model_and_data()

    y_test_pred, y_ci_low, y_ci_high, embed_corr_test = \
        load_predictions_and_embeddings(feat_names)

    shap_train, shap_test, explainer, train_idx, test_idx, expected_value, feat_names = \
        compute_shap(model, X_train, y_train, X_test)

    save_shap_outputs(shap_train, shap_test, feat_names,
                      X_train, X_test, train_idx, test_idx)

    explain_notable_samples(explainer, X_test, feat_names,
                            y_test, y_test_pred, y_ci_low, y_ci_high)

    shap_imp_test = np.abs(shap_test).mean(axis=0)
    combined_importance_summary(shap_imp_test, embed_corr_test, feat_names)

    _log("\n" + "=" * 70)
    _log(f"  Analysis complete.  Outputs: {os.path.abspath(OUT_DIR)}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
