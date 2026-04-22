# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: BeatAML2.0
#     language: python
#     name: python3
# ---

# +
import os
import itertools
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, GroupKFold
from joblib import Parallel, delayed
import torch

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

try:
    from huggingface_hub import login as hf_login
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
    _hf_token = os.environ.get("HF_TOKEN", "")
    if _hf_token:
        hf_login(token=_hf_token)
    else:
        print("Warning: HF_TOKEN not found in environment or .env file.")
except ImportError:
    HAS_TABPFN = False
    print("Warning: tabpfn not installed. Script will exit.")

N_CV_SPLITS = 5
N_GPUS      = torch.cuda.device_count()
DEVICE      = f"cuda:0" if N_GPUS > 0 else "cpu"
print(f"Detected {N_GPUS} GPU(s). Primary device: {DEVICE}")

from misc import load_model, calculate_regression_metrics
# -

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ABLATION_MODEL_DIR    = "../Models/tabpfn_models/ablation"
RESULT_DIR            = "../Results/tabpfn/ablation/"
RESULT_FILE           = "tabpfn_ablation_cv_results.csv"
DATA_FILE             = "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
FEATURE_METADATA_FILE = "../Data/ablation_feature_columns.csv"
STRATIFY_OPTIONS      = ["inhibitor", "dbgap_rnaseq_sample", "random"]
KFOLD_RANDOM_STATE    = 42

# Same exclusion list as tabpfn_ablation_study.py and cv_evaluation.py
EXCLUDE_COLS = [
    "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample",
    "inhibitor", "type", "status", "paper_inclusion", "min_conc", "max_conc",
    "intercept", "beta", "beta_z", "beta_p", "aic", "pearson_chisq", "deviance",
    "converged", "ic10", "ic25", "ic50", "ic75", "ic90", "all_gt_50", "all_lt_50",
    "curve_type",
]


# ---------------------------------------------------------------------------
# Build ablation feature groups  (mirrors tabpfn_ablation_study.py)
# ---------------------------------------------------------------------------
def build_ablation_feature_groups(feature_metadata_path):
    """
    Replicates the feature group construction in tabpfn_ablation_study.py:
      1. Load ablation_feature_columns.csv and group columns by Type.
      2. Remove Tsne.
      3. Separate Drug_Embed and Drug_PC into their own option dicts.
      4. Merge Clinical + CellType + Module → Clinical_CellType_Module.

    Returns
    -------
    patient_feature_groups : dict  — patient group name → list of column labels
    drug_feature_options   : dict  — drug option name  → list of column labels
    tsne_cols              : list  — Tsne column labels (excluded from preprocessing)
    """
    feature_metadata = pd.read_csv(feature_metadata_path)
    feature_groups = {}
    for type_name, grp in feature_metadata.groupby("Type"):
        feature_groups[type_name] = grp["Column_Label"].tolist()

    print("All groups from ablation_feature_columns.csv:")
    for gname, cols in sorted(feature_groups.items()):
        print(f"  {gname}: {len(cols)} features")

    tsne_cols       = feature_groups.pop("Tsne",      [])
    drug_embed_cols = feature_groups.pop("Drug_Embed", [])
    drug_pc_cols    = feature_groups.pop("Drug_PC",    [])

    clinical_cols = feature_groups.pop("Clinical", [])
    celltype_cols = feature_groups.pop("CellType",  [])
    module_cols   = feature_groups.pop("Module",    [])
    feature_groups["Clinical_CellType_Module"] = clinical_cols + celltype_cols + module_cols

    drug_feature_options = {
        "Drug_PC":            drug_pc_cols,
        "Drug_Embed":         drug_embed_cols,
        "Drug_PC+Drug_Embed": drug_pc_cols + drug_embed_cols,
    }

    print("\nPatient feature groups (after merging Clinical/CellType/Module):")
    for gname, cols in sorted(feature_groups.items()):
        print(f"  {gname}: {len(cols)} features")
    print("\nDrug feature options: " +
          ", ".join(f"{k}: {len(v)}" for k, v in drug_feature_options.items()))
    print(f"Tsne cols excluded from preprocessing: {len(tsne_cols)}")

    return feature_groups, drug_feature_options, tsne_cols


# ---------------------------------------------------------------------------
# Preprocessing  (mirrors tabpfn_ablation_study.py's preprocess_data)
# ---------------------------------------------------------------------------
def preprocess_data_full(big_train_df, tsne_cols, label_col="auc"):
    """
    Extract the full feature matrix from training data, mirroring preprocess_data()
    in tabpfn_ablation_study.py.  No scaler is applied here — scaling is done
    per-experiment on the relevant feature subset (see main loop below).

    Returns
    -------
    X            : DataFrame of all valid features (unscaled)
    y            : numpy array of labels
    metadata     : DataFrame of metadata columns (includes patient/inhibitor IDs)
    feature_cols : sorted list of feature column names
    """
    all_columns  = big_train_df.columns.tolist()
    column_types = big_train_df.dtypes.tolist()

    metadata_cols = ["CID"] + [
        all_columns[i] for i in range(len(all_columns))
        if str(column_types[i]) in ("object", "category")
    ]
    nan_cols     = [c for c in all_columns if big_train_df[c].isnull().any()]
    tsne_in_data = [c for c in tsne_cols if c in all_columns]

    metadata_cols = list(dict.fromkeys(metadata_cols + nan_cols + EXCLUDE_COLS + tsne_in_data))
    metadata_cols = [c for c in metadata_cols if c in all_columns]

    feature_cols = sorted(list(set(all_columns) - set(metadata_cols) - {label_col}))
    print(f"  Metadata cols: {len(metadata_cols)}, Feature cols: {len(feature_cols)}")

    metadata = big_train_df.loc[:, metadata_cols].reset_index(drop=True)
    X        = big_train_df.loc[:, feature_cols].copy()
    y        = big_train_df[label_col].to_numpy().flatten()

    return X, y, metadata, feature_cols


# ---------------------------------------------------------------------------
# Build all ablation experiment configurations
# ---------------------------------------------------------------------------
def build_experiments(patient_feature_groups, drug_feature_options, all_feature_cols):
    """
    Generate the full set of ablation experiments by replicating the naming
    convention observed in the saved model files:

      baseline_all_features_{drug}
          → all patient groups combined + drug features

      combo_1_{patient_group}_{drug}
          → single patient group + drug features

      combo_2_{group_A}+{group_B}_{drug}
          → two patient groups (alphabetical order) + drug features

      combo_3_{group_A}+{group_B}+{group_C}_{drug}
          → three patient groups (alphabetical order) + drug features

    Feature columns are filtered to those present in all_feature_cols.

    Returns list of (experiment_name, feature_subset) tuples.
    """
    feat_set    = set(all_feature_cols)
    group_names = sorted(patient_feature_groups.keys())   # alphabetical → matches file names

    # Filter each group and drug option to columns present in the data
    groups_filt = {
        g: [c for c in patient_feature_groups[g] if c in feat_set]
        for g in group_names
    }
    drug_opts_filt = {
        d: [c for c in cols if c in feat_set]
        for d, cols in drug_feature_options.items()
    }

    def _dedup(lst):
        seen = set()
        return [c for c in lst if not (c in seen or seen.add(c))]

    experiments = []

    # ── Baseline: all patient groups ─────────────────────────────────────────
    all_patient_cols = _dedup(
        col for g in group_names for col in groups_filt[g]
    )
    for drug_name, drug_cols in drug_opts_filt.items():
        exp_name = f"baseline_all_features_{drug_name}"
        feat_sub = _dedup(list(drug_cols) + all_patient_cols)
        experiments.append((exp_name, feat_sub))

    # ── Combo_N: all subsets of size 1 to N-1 ────────────────────────────────
    N = len(group_names)   # 4 patient groups
    for combo_size in range(1, N):
        for group_combo in itertools.combinations(group_names, combo_size):
            # itertools.combinations preserves sort order (group_names is sorted)
            patient_cols     = _dedup(col for g in group_combo for col in groups_filt[g])
            combo_patient_str = "+".join(group_combo)
            for drug_name, drug_cols in drug_opts_filt.items():
                exp_name = f"combo_{combo_size}_{combo_patient_str}_{drug_name}"
                feat_sub = _dedup(list(drug_cols) + patient_cols)
                experiments.append((exp_name, feat_sub))

    print(f"\nGenerated {len(experiments)} ablation experiment configuration(s):")
    for name, cols in experiments:
        print(f"  {name}: {len(cols)} features")

    return experiments


# ---------------------------------------------------------------------------
# TabPFN fold function  (module-level for loky — mirrors cv_evaluation.py)
# ---------------------------------------------------------------------------
def _fit_fold_tabpfn(fold, train_idx, val_idx, X_arr, y, model_params, device):
    """Fit one TabPFN fold on a specific device.  Defined at module level so
    loky can pickle it for parallel execution."""
    import torch
    from tabpfn import TabPFNRegressor
    from misc import calculate_regression_metrics as _calc_metrics

    params = {**model_params, "device": device}
    model  = TabPFNRegressor(**params)
    model.fit(X_arr[train_idx], y[train_idx])
    y_pred = model.predict(X_arr[val_idx])
    m      = _calc_metrics(y[val_idx], y_pred)
    del model
    torch.cuda.empty_cache()
    return fold, m


# ---------------------------------------------------------------------------
# 5-fold CV runner for TabPFN  (mirrors cv_evaluation.py)
# ---------------------------------------------------------------------------
def run_5fold_cv_tabpfn(model_params, X, y, groups,
                        n_splits=N_CV_SPLITS, random_state=KFOLD_RANDOM_STATE):
    """
    5-fold CV for TabPFN with automatic GPU parallelism, directly mirroring
    run_5fold_cv_tabpfn() in cv_evaluation.py.

    - n_gpus >= n_splits : each fold gets its own GPU, all folds run in parallel.
    - otherwise          : folds run sequentially on the available device.
    """
    if groups is not None:
        cv     = GroupKFold(n_splits=n_splits, shuffle=True)
        splits = list(cv.split(X, y, groups))
    else:
        cv     = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y))

    X_arr  = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    n_gpus = torch.cuda.device_count()

    print(f"  Train sizes ~{len(X_arr) * (n_splits - 1) // n_splits}, "
          f"val sizes ~{len(X_arr) // n_splits}")

    t_all = time.time()

    if n_gpus >= n_splits:
        # ── Parallel path: one GPU per fold ──────────────────────────────────
        devices = [f"cuda:{i}" for i in range(n_splits)]
        print(f"  {n_gpus} GPUs available — running {n_splits} folds in parallel "
              f"(fold {list(range(n_splits))} → {devices})")

        raw_results = Parallel(n_jobs=n_splits, backend="loky")(
            delayed(_fit_fold_tabpfn)(
                fold, train_idx, val_idx, X_arr, y, model_params, devices[fold]
            )
            for fold, (train_idx, val_idx) in enumerate(splits)
        )
        print(f"  All {n_splits} folds finished in {time.time() - t_all:.1f}s")

        raw_results.sort(key=lambda r: r[0])
        fold_metrics = []
        for fold, m in raw_results:
            fold_metrics.append(m)
            print(f"    Fold {fold + 1}/{n_splits}: "
                  f"MAE={m[0]:.3f}, RMSE={m[1]:.3f}, "
                  f"R2={m[2]:.3f}, Pearson={m[3]:.3f}, Spearman={m[4]:.3f}")

    else:
        # ── Sequential path: one fold at a time on the available device ───────
        dev = model_params.get("device", "cpu")
        print(f"  {n_gpus} GPU(s) available (need {n_splits} for full parallelism) — "
              f"running {n_splits} folds sequentially on {dev}")

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            t_fold = time.time()
            model  = TabPFNRegressor(**model_params)
            model.fit(X_arr[train_idx], y[train_idx])
            y_pred = model.predict(X_arr[val_idx])
            m      = calculate_regression_metrics(y[val_idx], y_pred)
            fold_metrics.append(m)
            print(f"    Fold {fold + 1}/{n_splits}: "
                  f"MAE={m[0]:.3f}, RMSE={m[1]:.3f}, "
                  f"R2={m[2]:.3f}, Pearson={m[3]:.3f}, Spearman={m[4]:.3f} "
                  f"({time.time() - t_fold:.1f}s)")
            del model
            torch.cuda.empty_cache()

        print(f"  All {n_splits} folds finished in {time.time() - t_all:.1f}s")

    return fold_metrics


# ---------------------------------------------------------------------------
# Result helpers  (mirrors cv_evaluation.py)
# ---------------------------------------------------------------------------
def aggregate_fold_metrics(fold_metrics):
    """Return (means, stds) arrays rounded to 3 decimal places."""
    arr   = np.array(fold_metrics)
    means = np.round(np.mean(arr, axis=0), 3)
    stds  = np.round(np.std(arr,  axis=0), 3)
    return means, stds


def build_result_row(experiment_name, n_features, stratify_by, means, stds):
    """Build a result dict with raw mean/std columns and formatted 'mean +/- std' columns."""
    metric_names = ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
    row = {
        "experiment_name": experiment_name,
        "n_features":      n_features,
        "stratify_by":     stratify_by,
    }
    for i, name in enumerate(metric_names):
        row[f"{name}_mean"] = round(float(means[i]), 3)
        row[f"{name}_std"]  = round(float(stds[i]),  3)
        row[name]           = f"{means[i]:.3f} +/- {stds[i]:.3f}"
    return row


OUTPUT_COLS = (
    ["experiment_name", "n_features", "stratify_by"] +
    [f"{n}_{s}" for n in ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
                 for s in ["mean", "std"]] +
    ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
)


# ---------------------------------------------------------------------------
# Main: run CV for all ablation experiments
# ---------------------------------------------------------------------------

# +
if not HAS_TABPFN:
    print("TabPFN not installed — exiting.")
else:
    print("\n" + "=" * 80)
    print("TabPFN Ablation Study — 5-Fold Cross-Validation")
    print("=" * 80)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. Build feature groups from ablation_feature_columns.csv
    print("\nBuilding ablation feature groups ...")
    patient_feature_groups, drug_feature_options, tsne_cols = \
        build_ablation_feature_groups(FEATURE_METADATA_FILE)

    # 2. Load and preprocess training data
    print(f"\nLoading training data: {DATA_FILE}")
    t_load = time.time()
    big_train_df = pd.read_pickle(DATA_FILE, compression="zip")
    print(f"  Loaded in {time.time() - t_load:.1f}s — raw shape: {big_train_df.shape}")

    X_full, Y_all, metadata, all_feature_cols = preprocess_data_full(
        big_train_df, tsne_cols, label_col="auc"
    )
    print(f"  X shape: {X_full.shape}, y shape: {Y_all.shape}")

    # 3. Generate all experiment configurations
    experiments   = build_experiments(patient_feature_groups, drug_feature_options, all_feature_cols)
    n_experiments = len(experiments)
    n_stratify    = len(STRATIFY_OPTIONS)
    print(f"\n{n_experiments} experiment(s) x {n_stratify} stratification(s) = "
          f"up to {n_experiments * n_stratify} CV run(s) total.")

    results = []

    for exp_idx, (experiment_name, feature_subset) in enumerate(experiments):
        model_path = os.path.join(ABLATION_MODEL_DIR, f"tabpfn_{experiment_name}_best.pk")

        if not os.path.exists(model_path):
            print(f"\n[SKIP] Model not found: {model_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"[{exp_idx + 1}/{n_experiments}] Experiment: {experiment_name}")

        # Load saved model to extract hyperparameters; update device for current node
        print(f"  Loading model: {model_path}")
        saved_model  = load_model(model_path)
        model_params = saved_model.get_params()
        model_params["device"] = DEVICE
        print(f"  TabPFN params: {model_params}")

        # Align feature subset to what the saved model was actually trained on.
        # feature_names_in_ (sklearn >= 1.0) is the ground truth; fall back to
        # the derived feature_subset if unavailable.
        if hasattr(saved_model, "feature_names_in_"):
            model_feat_cols = list(saved_model.feature_names_in_)
            extra   = set(feature_subset) - set(model_feat_cols)
            missing = set(model_feat_cols) - set(feature_subset)
            if extra or missing:
                print(f"  WARNING: feature mismatch "
                      f"(extra={len(extra)}, missing={len(missing)}) between "
                      f"derived subset and model; using model's feature_names_in_")
            X_exp = X_full[model_feat_cols]
            print(f"  Aligned to {len(model_feat_cols)} features from model (feature_names_in_)")
        else:
            cols_present = [c for c in feature_subset if c in X_full.columns]
            X_exp = X_full[cols_present]
            print(f"  No feature_names_in_ on model; using {len(cols_present)} derived features")

        n_feat = X_exp.shape[1]
        print(f"  Feature count: {n_feat}")

        # Apply StandardScaler to the feature subset (matches tabpfn_ablation_study.py
        # which scales each X_train_sub independently before HPO)
        scaler   = preprocessing.StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_exp), columns=X_exp.columns)
        print(f"  StandardScaler applied.")

        for strat_idx, stratify_by in enumerate(STRATIFY_OPTIONS):
            print(f"\n  [{strat_idx + 1}/{n_stratify}] CV: stratify_by={stratify_by}")

            if stratify_by == "dbgap_rnaseq_sample":
                groups = metadata["dbgap_rnaseq_sample"].to_numpy()
                print(f"  Groups: {stratify_by} — {len(np.unique(groups))} unique patients")
            elif stratify_by == "inhibitor":
                groups = metadata["inhibitor"].to_numpy()
                print(f"  Groups: {stratify_by} — {len(np.unique(groups))} unique inhibitors")
            else:
                groups = None
                print(f"  Groups: None (random KFold)")

            t_cv = time.time()
            fold_metrics = run_5fold_cv_tabpfn(
                model_params, X_scaled, Y_all, groups,
                random_state=KFOLD_RANDOM_STATE
            )
            torch.cuda.empty_cache()
            print(f"  CV completed in {time.time() - t_cv:.1f}s")

            means, stds  = aggregate_fold_metrics(fold_metrics)
            metric_names = ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
            print("  Summary: " + ", ".join(
                f"{n}={means[i]:.3f}±{stds[i]:.3f}" for i, n in enumerate(metric_names)
            ))

            results.append(build_result_row(experiment_name, n_feat, stratify_by, means, stds))

    # Save results
    results_df = pd.DataFrame(results, columns=OUTPUT_COLS)
    out_path   = os.path.join(RESULT_DIR, RESULT_FILE)
    results_df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"\nTabPFN Ablation CV results saved to: {out_path}")
    print(results_df.to_string(index=False))
# -
