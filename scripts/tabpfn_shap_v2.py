"""
tabpfn_shap_v2.py  —  Retrain TabPFN 8.0.2 + SHAP (Steps 2b, 5, 5b, 6).

The original saved model cannot be loaded under tabpfn 8.0.2 (module-path
change), so a fresh TabPFNRegressor is trained here and cached for reuse.

Steps:
  2.  Load data (existing scaler, all 1561-feature subset for MinMax-variance ranking)
  2b. Train new TabPFNRegressor on top TOP_SHAP_FEATURES=128 by MinMax variance
      [cached to NEW_MODEL_PATH]
  5.  SHAP — TabPFNExplainer, PermutationSamplingSV
      Same 128-feature space as the model → SHAP explains the actual model.
      Full 33 K training context; budget=512 ≈ 4 orderings of 128 features.
  5b. SHAP waterfall for notable test samples (selection via in-memory preds)
  6.  Combined importance summary (SHAP + embedding from tabpfn_predict_embed.py)

Feature selection:
  Top 128 features by MinMax-scaled [0,1] training variance.  Each feature is
  first mapped to [0,1] (fit on training data only), then variance is computed.
  Raw variance is dominated by features with large absolute scale; MinMax variance
  measures spread relative to each feature's own range — a scale-invariant
  criterion.  StandardScaler variance (≡ 1 for all features) is uninformative.
  Both the model and the SHAP explainer use the same 128-feature space, so SHAP
  values describe the actual trained model — no approximation artefact.

KernelSHAP vs PermutationSamplingSV (128 features):
  At budget=512 KernelSHAP is 2× overdetermined (512 vs 2×128=256 minimum) —
  marginal, and drug/omics features are highly correlated → ill-conditioned
  weighted regression → instability risk remains.  PermutationSamplingSV avoids
  matrix inversion entirely, guarantees Σφᵢ = pred − baseline for every sample,
  and variance decreases as O(1/√n_orderings).  budget=512 → 4 orderings.
  → PermutationSamplingSV is the robust choice.

SHAP context:
  Rather than all 33 K training rows, the explainer uses a stratified subsample
  of SHAP_CONTEXT_SIZE=10 000 rows (10 quantile bins of y, proportional draw) so
  the AUC distribution is preserved while reducing per-coalition fit time ≈ 3×.
  Step 2b still trains the model on the full 33 K rows.

Timing (tabpfn 8.0.2, n_estimators=8, H200):
  Each coalition = model.fit(10 K rows × 128 features) ≈ 0.15–0.7 s.
  SHAP_BUDGET=512 × 0.5 s ≈ 4 min/sample.
  120 samples (20 train + 100 test) ≈ 8 h → fits comfortably in a single 36 h job.

Caches:
  NEW_MODEL_PATH        — 128-feature TabPFN model
  SHAP_TRAIN_CACHE_PATH — train-SHAP checkpoint
  SHAP_CACHE_PATH       — full (train + test) SHAP checkpoint
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
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import shapiq
from shapiq import TabPFNExplainer
from tabpfn import TabPFNRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misc import load_model

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)


# ── Paths ─────────────────────────────────────────────────────────────────────

SCALER_PATH  = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_baseline_all_features_Drug_Embed_scaler.pk"
)
NEW_MODEL_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_v4_8_0_2_top128.pk"
)
TRAIN_PATH   = "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
TEST_PATH    = "../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
ABLATION_CSV = "../Data/ablation_feature_columns.csv"
OUT_DIR      = "../Results/tabpfn/best_model_analysis"

SHAP_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_shap_v4_top128_full_cache.pk"
)
SHAP_TRAIN_CACHE_PATH = (
    "/export/cse/rmall/Raghvendra/tabpfn_big_models/ablation/"
    "tabpfn_shap_v4_top128_train_partial.pk"
)


# ── Hyper-parameters ──────────────────────────────────────────────────────────

N_ESTIMATORS      = 8
TOP_SHAP_FEATURES = 208   # top features by raw (pre-scaling) variance
SHAP_BUDGET        = 832    # ≈ 4 orderings of 128 features; stable for Permutation
SHAP_CONTEXT_SIZE  = 4096 # stratified subsample of 33 K for coalition evaluations
SHAP_TEST_SAMPLES  = 50
SHAP_TRAIN_SAMPLES = 20

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

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_pickle(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    _log(f"  [cache] {os.path.getsize(path)/1024**2:.1f} MB → {path}")


# ── Step 2: Load data ──────────────────────────────────────────────────────────

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


def load_data():
    """Load scaled data and compute MinMax-[0,1] variances for feature ranking."""
    _log("\n── Step 2: Loading and scaling data ─────────────────────────────")
    scaler = _load_pickle(SCALER_PATH)
    _log(f"  Scaler loaded (mean_ shape={scaler.mean_.shape})")

    train_df = pd.read_pickle(TRAIN_PATH, compression="zip")
    test_df  = pd.read_pickle(TEST_PATH,  compression="zip")
    _log(f"  Train raw: {train_df.shape}  Test raw: {test_df.shape}")

    feature_subset = get_feature_subset(train_df)
    _log(f"  Feature subset: {len(feature_subset)} features")

    # MinMax [0,1] variance on training data for feature ranking.
    # Raw variance is dominated by absolute scale; StandardScaler forces var≡1.
    # MinMax variance measures relative spread within each feature's own range.
    mm = MinMaxScaler()
    X_mm = mm.fit_transform(train_df[feature_subset].values)
    minmax_vars = pd.Series(X_mm.var(axis=0), index=feature_subset)
    _log(f"  MinMax variance: min={minmax_vars.min():.6f}  max={minmax_vars.max():.6f}"
         f"  median={minmax_vars.median():.6f}")

    X_train = pd.DataFrame(
        scaler.transform(train_df[feature_subset]), columns=feature_subset)
    X_test  = pd.DataFrame(
        scaler.transform(test_df[feature_subset]),  columns=feature_subset)
    y_train = train_df["auc"].to_numpy().flatten()
    y_test  = test_df["auc"].to_numpy().flatten()

    _log(f"  X_train {X_train.shape}  y_train [{y_train.min():.1f}, {y_train.max():.1f}]"
         f"  mean={y_train.mean():.2f}")
    _log(f"  X_test  {X_test.shape}   y_test  [{y_test.min():.1f}, {y_test.max():.1f}]"
         f"  mean={y_test.mean():.2f}")

    pd.DataFrame({"feature": feature_subset}).to_csv(
        os.path.join(OUT_DIR, "feature_subset.csv"), index=False)
    _log("  [DONE] Step 2.")
    return X_train, y_train, X_test, y_test, feature_subset, minmax_vars


# ── Feature selection ──────────────────────────────────────────────────────────

def select_top_variable_features(feat_names: list, minmax_vars: pd.Series,
                                  top_k: int = TOP_SHAP_FEATURES) -> list:
    """Return top_k feature names ranked by MinMax-[0,1] training variance."""
    var_vals  = minmax_vars.reindex(feat_names).fillna(0.0).values
    top_idx   = np.argsort(var_vals)[-top_k:][::-1]   # descending
    top_feats = [feat_names[i] for i in top_idx]

    _log(f"  Top-{top_k} variable features (MinMax var)  "
         f"range [{var_vals[top_idx[-1]]:.6f}, {var_vals[top_idx[0]]:.6f}]  "
         f"(floor vs full min: {var_vals[top_idx[-1]]:.6f} vs {var_vals.min():.6f})")

    var_df = pd.DataFrame({
        "feature":        top_feats,
        "minmax_variance": var_vals[top_idx],
        "rank":           np.arange(1, top_k + 1),
    })
    save_csv(var_df, f"shap_v4_top{top_k}_variable_features.csv")
    return top_feats


# ── Step 2b: Train / load new model ───────────────────────────────────────────

def train_or_load_model(X_train_s: pd.DataFrame, y_train: np.ndarray) -> TabPFNRegressor:
    """Train TabPFN on the top-128-feature subset (or load from cache)."""
    _log(f"\n── Step 2b: Train / load TabPFN 8.0.2 model "
         f"({X_train_s.shape[1]} features) ─────")

    if os.path.exists(NEW_MODEL_PATH):
        _log(f"  [cache] {NEW_MODEL_PATH} — loading.")
        model = _load_pickle(NEW_MODEL_PATH)
        _log(f"  n_estimators={model.n_estimators}  fit_mode={model.fit_mode}"
             f"  device={model.device}")
        _log("  [DONE] Step 2b (from cache).")
        return model

    _log(f"  Training TabPFNRegressor(n_estimators={N_ESTIMATORS}, "
         f"ignore_pretraining_limits=True, fit_mode='fit_preprocessors', "
         f"device={DEVICE!r})")
    _log(f"  Data: {X_train_s.shape[0]} rows × {X_train_s.shape[1]} features")

    model = TabPFNRegressor(
        n_estimators=N_ESTIMATORS,
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        device=DEVICE,
        random_state=SEED,
    )
    t0 = time.time()
    model.fit(X_train_s, y_train)
    _log(f"  fit() done in {(time.time()-t0)/60:.1f} min")

    _save_pickle(NEW_MODEL_PATH, model)
    _log("  [DONE] Step 2b.")
    return model


# ── Load embedding (prerequisite from tabpfn_predict_embed.py) ────────────────

def load_embedding_importance(feat_names: list) -> pd.Series:
    """Load embedding-attention importance; returns Series indexed by feature name.
    Missing features and absent file → 0.  Step 6 degrades to SHAP-only ranking.
    """
    embed_csv = os.path.join(OUT_DIR, "embedding_attention_importance.csv")
    if not os.path.exists(embed_csv):
        _log(f"  [WARN] {embed_csv} not found — embedding scores set to zero.")
        _log("         Run tabpfn_predict_embed.py first to populate this file.")
        return pd.Series(np.zeros(len(feat_names)), index=feat_names)

    df = pd.read_csv(embed_csv).set_index("feature")
    embed_series = df["embed_corr_test"].reindex(feat_names).fillna(0.0)
    _log(f"  Embedding importance loaded: {len(embed_series)} features  "
         f"max={embed_series.max():.4f}")
    return embed_series


# ── Step 5: SHAP ───────────────────────────────────────────────────────────────

def select_shap_context(X_train_s: np.ndarray, y_train: np.ndarray,
                         n_context: int = SHAP_CONTEXT_SIZE) -> tuple:
    """Stratified subsample of training rows for SHAP coalition evaluation.

    Bins y into N_STRATA quantile strata and draws proportionally from each
    to preserve the AUC distribution.  Returns (X_ctx, y_ctx, ctx_idx).
    """
    if n_context >= len(X_train_s):
        _log(f"  [context] Using full training set ({len(X_train_s)} rows).")
        return X_train_s, y_train, np.arange(len(X_train_s))

    N_STRATA = 10
    rng      = np.random.default_rng(SEED)
    edges    = np.percentile(y_train, np.linspace(0, 100, N_STRATA + 1))
    edges[0] -= 1e-9   # include the exact minimum
    strata   = np.clip(np.digitize(y_train, edges) - 1, 0, N_STRATA - 1)

    ctx_idx = []
    for s in range(N_STRATA):
        mask = strata == s
        n_s  = max(1, int(round(n_context * mask.sum() / len(y_train))))
        n_s  = min(n_s, int(mask.sum()))
        ctx_idx.extend(rng.choice(np.where(mask)[0], size=n_s, replace=False).tolist())

    ctx_idx = np.array(ctx_idx)
    # Trim to exactly n_context (rounding across strata may overshoot by a few)
    if len(ctx_idx) > n_context:
        ctx_idx = rng.choice(ctx_idx, size=n_context, replace=False)
    elif len(ctx_idx) < n_context:
        remaining = np.setdiff1d(np.arange(len(X_train_s)), ctx_idx)
        extra     = rng.choice(remaining, size=n_context - len(ctx_idx), replace=False)
        ctx_idx   = np.concatenate([ctx_idx, extra])
    ctx_idx = np.sort(ctx_idx)

    _log(f"  [context] Stratified {len(ctx_idx)}/{len(X_train_s)} rows  "
         f"y_ctx [{y_train[ctx_idx].min():.1f}, {y_train[ctx_idx].max():.1f}]"
         f"  mean={y_train[ctx_idx].mean():.2f}  (full mean={y_train.mean():.2f})")
    return X_train_s[ctx_idx], y_train[ctx_idx], ctx_idx


def _build_explainer(model: TabPFNRegressor,
                     X_train_np: np.ndarray, y_train: np.ndarray,
                     X_test_np: np.ndarray) -> TabPFNExplainer:
    """Build TabPFNExplainer on the full training context.

    x_test is passed so TabPFNExplainer uses ALL training rows as context
    (default behaviour would split 80/20).  The test set is used only to
    compute the empty-prediction baseline.
    """
    n_ctx, n_feat = X_train_np.shape
    _log(f"  [5c] TabPFNExplainer: context={n_ctx} rows × {n_feat} features  "
         f"approximator=permutation  budget={SHAP_BUDGET} "
         f"(≈{SHAP_BUDGET // n_feat} orderings)")
    t0 = time.time()
    explainer = TabPFNExplainer(
        model=model,
        data=X_train_np,
        labels=y_train,
        x_test=X_test_np,
        index="SV",
        max_order=1,
        approximator="permutation",
        verbose=False,
    )
    _log(f"  [5c] built in {time.time()-t0:.1f}s  "
         f"baseline={explainer.baseline_value:.4f}")
    return explainer


def _iv_to_shap_array(iv, n_features: int) -> np.ndarray:
    return np.array([iv[(i,)] for i in range(n_features)])


def _explain_loop(explainer: TabPFNExplainer,
                  X_sub: np.ndarray, label: str) -> np.ndarray:
    """Explain each row; failed rows → zero vector."""
    n_samples, n_features = X_sub.shape
    shap_matrix = np.zeros((n_samples, n_features), dtype=np.float64)
    t_start = time.time()
    for i, x_i in enumerate(tqdm(X_sub, desc=f"SHAP [{label}]", unit="sample")):
        t0 = time.time()
        try:
            iv             = explainer.explain(x_i, budget=SHAP_BUDGET)
            shap_matrix[i] = _iv_to_shap_array(iv, n_features)
            _log(f"  [{label}] {i:3d}/{n_samples}  "
                 f"sum={shap_matrix[i].sum():+.2f}  "
                 f"max|v|={np.abs(shap_matrix[i]).max():.3f}  "
                 f"time={time.time()-t0:.0f}s")
        except Exception as exc:
            _log(f"  [WARN] {label} sample {i}: {exc} — row zeroed.")
    total_h = (time.time() - t_start) / 3600
    _log(f"  [{label}] {n_samples} samples in {total_h:.2f}h "
         f"({total_h*3600/n_samples:.0f}s/sample avg)")
    return shap_matrix


def compute_shap(model: TabPFNRegressor,
                 X_train_s: np.ndarray, y_train: np.ndarray,
                 X_test_s: np.ndarray, top_feats: list):
    """SHAP with two-level crash-safe checkpointing.

    X_train_s / X_test_s are already restricted to the top-128-feature subset.
    The SHAP explainer uses a stratified 10 K-row context (not the full 33 K)
    to reduce per-coalition fit time; the model in Step 2b still trained on 33 K.
    """
    _log("\n── Step 5: SHAP variable importance (TabPFNExplainer) ───────────")
    n_features = X_train_s.shape[1]

    # Build the 10 K stratified context (deterministic; same result on every resume)
    X_ctx, y_ctx, _ = select_shap_context(X_train_s, y_train)

    # Path A: full cache
    if os.path.exists(SHAP_CACHE_PATH):
        _log("  [cache] Full cache — loading.")
        cache          = _load_pickle(SHAP_CACHE_PATH)
        shap_train     = cache["shap_train"]
        shap_test      = cache["shap_test"]
        train_idx      = cache["train_idx"]
        test_idx       = cache["test_idx"]
        expected_value = cache["expected_value"]
        explainer      = _build_explainer(model, X_ctx, y_ctx, X_test_s)
        _log(f"  shap_train={shap_train.shape}  shap_test={shap_test.shape}"
             f"  baseline={expected_value:.4f}")
        _log("  [DONE] Step 5 (from cache).")
        return shap_train, shap_test, explainer, train_idx, test_idx, expected_value

    # Path B: train done, test pending
    if os.path.exists(SHAP_TRAIN_CACHE_PATH):
        _log("  [cache] Partial cache — resuming test SHAP.")
        partial        = _load_pickle(SHAP_TRAIN_CACHE_PATH)
        shap_train     = partial["shap_train"]
        train_idx      = partial["train_idx"]
        expected_value = partial["expected_value"]
        explainer      = _build_explainer(model, X_ctx, y_ctx, X_test_s)

        rng      = np.random.default_rng(SEED + 1)
        test_idx = rng.choice(len(X_test_s), min(SHAP_TEST_SAMPLES, len(X_test_s)), replace=False)
        _log(f"  [5d] Test SHAP: {len(test_idx)} samples × {n_features} features …")
        shap_test = _explain_loop(explainer, X_test_s[test_idx], "test")
        _save_pickle(SHAP_CACHE_PATH, dict(
            shap_train=shap_train, shap_test=shap_test,
            train_idx=train_idx, test_idx=test_idx, expected_value=expected_value,
        ))

    # Path C: from scratch
    else:
        explainer      = _build_explainer(model, X_ctx, y_ctx, X_test_s)
        expected_value = float(explainer.baseline_value)

        rng       = np.random.default_rng(SEED)
        train_idx = rng.choice(len(X_train_s), min(SHAP_TRAIN_SAMPLES, len(X_train_s)), replace=False)
        _log(f"  [5d] Train SHAP: {len(train_idx)} samples × {n_features} features …")
        shap_train = _explain_loop(explainer, X_train_s[train_idx], "train")
        _save_pickle(SHAP_TRAIN_CACHE_PATH, dict(
            shap_train=shap_train, train_idx=train_idx, expected_value=expected_value,
        ))

        rng      = np.random.default_rng(SEED + 1)
        test_idx = rng.choice(len(X_test_s), min(SHAP_TEST_SAMPLES, len(X_test_s)), replace=False)
        _log(f"  [5e] Test SHAP: {len(test_idx)} samples × {n_features} features …")
        shap_test = _explain_loop(explainer, X_test_s[test_idx], "test")
        _save_pickle(SHAP_CACHE_PATH, dict(
            shap_train=shap_train, shap_test=shap_test,
            train_idx=train_idx, test_idx=test_idx, expected_value=expected_value,
        ))

    _log(f"  SHAP stats (test): mean|v|={np.abs(shap_test).mean():.4f}  "
         f"max|v|={np.abs(shap_test).max():.4f}  baseline={expected_value:.4f}")
    _log("  [DONE] Step 5.")
    return shap_train, shap_test, explainer, train_idx, test_idx, expected_value


# ── SHAP outputs ───────────────────────────────────────────────────────────────

def save_shap_outputs(shap_train, shap_test, top_feats,
                      X_train_s, X_test_s, train_idx, test_idx):
    tag = "shap_v4"
    if os.path.exists(os.path.join(OUT_DIR, f"{tag}_feature_importance.csv")):
        _log(f"  [cache] {tag} outputs already exist — skipping.")
        return

    shap_imp_train = np.abs(shap_train).mean(axis=0)
    shap_imp_test  = np.abs(shap_test).mean(axis=0)

    save_csv(pd.DataFrame(shap_train.astype(np.float32), columns=top_feats),
             f"{tag}_values_train_subset.csv")
    save_csv(pd.DataFrame(shap_test.astype(np.float32), columns=top_feats),
             f"{tag}_values_test_subset.csv")

    imp_df = pd.DataFrame({
        "feature":             top_feats,
        "shap_mean_abs_train": shap_imp_train,
        "shap_mean_abs_test":  shap_imp_test,
    }).sort_values("shap_mean_abs_test", ascending=False)
    save_csv(imp_df, f"{tag}_feature_importance.csv")

    _log("  Top 10 features by mean |SHAP| (test):")
    for row in imp_df.head(10).itertuples():
        _log(f"    {row.feature:<40s}  {row.shap_mean_abs_test:.4f} AUC")

    _plot_shap_bar(shap_imp_test,  top_feats, "Test",  f"{tag}_importance_bar_test.pdf")
    _plot_shap_bar(shap_imp_train, top_feats, "Train", f"{tag}_importance_bar_train.pdf")
    _plot_shap_beeswarm(shap_test,  X_test_s[test_idx],   top_feats, f"{tag}_beeswarm_test.pdf")
    _plot_shap_beeswarm(shap_train, X_train_s[train_idx], top_feats, f"{tag}_beeswarm_train.pdf")


def _plot_shap_bar(shap_imp, feat_names, split, filename):
    top_idx = np.argsort(shap_imp)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(np.array(feat_names)[top_idx], shap_imp[top_idx],
            color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, TOP_N)))
    ax.set_xlabel("Mean |SHAP value| (AUC units)", fontsize=11)
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
        norm_val = (xv_col - xv_col.min()) / ((xv_col.max() - xv_col.min()) + 1e-9)
        ax.scatter(sv_col, j + y_jitter, c=plt.cm.coolwarm(norm_val),
                   s=6, alpha=0.5, linewidths=0)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (AUC units)", fontsize=11)
    ax.set_title(f"SHAP Beeswarm – Top {TOP_N} features  "
                 "(blue = low value, red = high)", fontsize=11)
    fig.tight_layout()
    save_plot(fig, filename)


# ── Step 5b: Waterfall for notable test samples ────────────────────────────────

def _explain_single(explainer, x_1d, top_feats, sample_idx,
                    true_val, pred_val, base_value):
    tag = f"shap_v4_single_sample_idx{sample_idx}"
    if os.path.exists(os.path.join(OUT_DIR, f"{tag}.csv")):
        _log(f"  [cache] {tag}.csv — skipping.")
        return
    try:
        iv      = explainer.explain(x_1d, budget=SHAP_BUDGET)
        shap_1d = _iv_to_shap_array(iv, len(top_feats))
    except Exception as exc:
        _log(f"  [WARN] explain() failed for sample {sample_idx}: {exc}")
        return

    _log(f"  base={base_value:.2f}  pred={pred_val:.2f}  true={true_val:.2f}  "
         f"SHAP_sum={shap_1d.sum():.2f}  (pred−base={pred_val-base_value:.2f})")

    df = pd.DataFrame({
        "feature":       top_feats,
        "feature_value": x_1d,
        "shap_value":    shap_1d,
    }).sort_values("shap_value", key=np.abs, ascending=False)
    for col, val in [("sample_idx", sample_idx), ("true_label", true_val),
                     ("pred_val", pred_val), ("base_value", base_value)]:
        df[col] = val
    save_csv(df, f"{tag}.csv")

    try:
        ax  = shapiq.waterfall_plot(iv, feature_names=top_feats, max_display=10, show=False)
        fig = ax.get_figure()
        fig.suptitle(f"SHAP Waterfall – test sample #{sample_idx}  "
                     f"true={true_val:.1f}  pred={pred_val:.1f}  "
                     f"baseline={base_value:.1f}", fontsize=11)
        fig.tight_layout()
        save_plot(fig, f"{tag}.pdf")
    except Exception as exc:
        _log(f"  [WARN] waterfall_plot failed: {exc}")


def explain_notable_samples(explainer, X_test_s: np.ndarray, top_feats,
                             y_test, y_test_pred):
    """Select notable samples using in-memory predictions (not saved to disk).

    Picks: first, worst prediction error, sensitive_1/2 (true ≤ Q25, pred ≤ median),
    resistant_1/2 (true ≥ Q75, pred ≥ median).
    """
    _log("\n── Step 5b: SHAP waterfall for notable test samples ─────────────")
    base_value  = float(explainer.baseline_value)
    abs_error   = np.abs(y_test_pred - y_test)
    q25, q75    = np.percentile(y_test, [25, 75])
    median_pred = np.median(y_test_pred)
    sens_mask   = (y_test <= q25) & (y_test_pred <= median_pred)
    res_mask    = (y_test >= q75) & (y_test_pred >= median_pred)

    def _top_k(mask, k, label):
        idxs = np.where(mask)[0]
        if not len(idxs):
            _log(f"  [WARN] no candidates for '{label}'")
            return []
        return idxs[np.argsort(abs_error[idxs])][:k].tolist()

    samples = {
        "first":      0,
        "worst_pred": int(np.argmax(abs_error)),
    }
    for i, idx in enumerate(_top_k(sens_mask, 2, "sensitive"), 1):
        samples[f"sensitive_{i}"] = idx
    for i, idx in enumerate(_top_k(res_mask, 2, "resistant"), 1):
        samples[f"resistant_{i}"] = idx

    for label, idx in samples.items():
        _log(f"\n  -- {label}  idx={idx}  true={y_test[idx]:.1f}"
             f"  pred={y_test_pred[idx]:.1f}  err={abs_error[idx]:.2f} --")
        _explain_single(explainer, X_test_s[idx], top_feats, idx,
                        y_test[idx], y_test_pred[idx], base_value)
    _log("  [DONE] Step 5b.")


# ── Step 6: Combined importance summary ───────────────────────────────────────

def combined_importance_summary(shap_imp_test, embed_series: pd.Series, top_feats):
    """Combine SHAP and embedding importance restricted to the 128-feature SHAP subset.

    embed_series is indexed by feature name (1561 features); aligned to top_feats here.
    """
    _log("\n── Step 6: Combined importance summary ───────────────────────────")

    embed_top = embed_series.reindex(top_feats).fillna(0.0).values

    def _minmax(v):
        vr = v.max() - v.min()
        return (v - v.min()) / vr if vr > 1e-9 else np.zeros_like(v)

    shap_norm  = _minmax(shap_imp_test)
    embed_norm = _minmax(embed_top)
    combined   = 0.5 * shap_norm + 0.5 * embed_norm

    summary_df = pd.DataFrame({
        "feature":         top_feats,
        "shap_score":      shap_imp_test,
        "embedding_score": embed_top,
        "shap_norm":       shap_norm,
        "embedding_norm":  embed_norm,
        "combined_score":  combined,
    }).sort_values("combined_score", ascending=False)
    save_csv(summary_df, "shap_v4_combined_importance_summary.csv")

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
    save_plot(fig, "shap_v4_combined_importance_comparison.pdf")

    _log("  Top 10 by combined score:")
    _log(summary_df[["feature", "shap_score", "embedding_score", "combined_score"]]
         .head(10).to_string(index=False))
    _log("  [DONE] Step 6.")
    return summary_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    _log("=" * 70)
    _log("  TabPFN 8.0.2 – Retrain + SHAP v4 (Steps 2b, 5, 5b, 6)")
    _log(f"  shapiq={shapiq.__version__}  "
         f"approximator=permutation  budget={SHAP_BUDGET}"
         f"  top_features={TOP_SHAP_FEATURES}"
         f"  train_n={SHAP_TRAIN_SAMPLES}  test_n={SHAP_TEST_SAMPLES}")
    gpu_info = (f"cuda:{torch.cuda.get_device_name(0)}"
                if DEVICE == "cuda" else "cpu (no CUDA)")
    _log(f"  Device={gpu_info}")
    _log("=" * 70)

    # Step 2: load full 1561-feature data + MinMax variances for ranking
    X_train, y_train, X_test, y_test, feat_names, minmax_vars = load_data()

    # Feature selection — top 128 by MinMax [0,1] variance; used by both model and SHAP
    _log(f"\n── Feature selection: top {TOP_SHAP_FEATURES} by MinMax variance ──────────")
    top_feats = select_top_variable_features(feat_names, minmax_vars)
    X_train_s = X_train[top_feats].values.astype(np.float64)
    X_test_s  = X_test[top_feats].values.astype(np.float64)
    _log(f"  X_train_s {X_train_s.shape}  X_test_s {X_test_s.shape}")

    # Step 2b: train model on the same 128-feature space
    model = train_or_load_model(X_train[top_feats], y_train)

    embed_series = load_embedding_importance(feat_names)

    # Step 5: SHAP (128-feature subset, full 33K context)
    shap_train, shap_test, explainer, train_idx, test_idx, expected_value = \
        compute_shap(model, X_train_s, y_train, X_test_s, top_feats)

    save_shap_outputs(shap_train, shap_test, top_feats,
                      X_train_s, X_test_s, train_idx, test_idx)

    # Step 5b: notable samples — predictions from 128-feature model (in-memory only)
    _log("\n── Test predictions for notable sample selection (in-memory only) ─")
    y_test_pred = model.predict(X_test[top_feats])
    _log(f"  pred range=[{y_test_pred.min():.1f}, {y_test_pred.max():.1f}]"
         f"  mean={y_test_pred.mean():.2f}")

    explain_notable_samples(explainer, X_test_s, top_feats, y_test, y_test_pred)

    # Step 6: combined importance
    shap_imp_test = np.abs(shap_test).mean(axis=0)
    combined_importance_summary(shap_imp_test, embed_series, top_feats)

    _log("\n" + "=" * 70)
    _log(f"  Complete.  Outputs: {os.path.abspath(OUT_DIR)}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
