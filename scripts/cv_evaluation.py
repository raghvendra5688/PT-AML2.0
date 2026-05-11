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
import time
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model, preprocessing
from sklearn.svm import LinearSVR
from sklearn.model_selection import KFold, GroupKFold
from joblib import Parallel, delayed
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
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
    print("Warning: tabpfn not installed. TabPFN model will be skipped.")

# Total physical threads available and number of CV folds.
# All 5 folds run in parallel; each fold's RF gets THREADS_PER_FOLD cores.
NUM_THREADS      = 155
N_CV_SPLITS      = 5
THREADS_PER_FOLD = NUM_THREADS // N_CV_SPLITS   # 31 threads per fold

os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_FOLD)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_FOLD)

N_GPUS = torch.cuda.device_count()
DEVICE = f"cuda:0" if N_GPUS > 0 else "cpu"
print(f"Detected {N_GPUS} GPU(s). Primary device: {DEVICE}")

from misc import load_model, calculate_regression_metrics

try:
    from LSSVM_implementation import LSSVMRegressor
    HAS_LSSVM = True
except ImportError:
    HAS_LSSVM = False
    print("Warning: LSSVM_implementation not found. Will use fallback SVR params.")

# Fallback fixed hyperparameters used when a saved LSSVM model cannot be loaded
SVR_FIXED_PARAMS = {
    "C": 1.0,
    "epsilon": 0.1,
    "max_iter": 5000,
    "dual": "auto",
}
# -

# ---------------------------------------------------------------------------
# Columns to exclude from features (same across all three model scripts)
# ---------------------------------------------------------------------------
EXCLUDE_COLS = [
    "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample",
    "inhibitor", "type", "status", "paper_inclusion", "min_conc", "max_conc",
    "intercept", "beta", "beta_z", "beta_p", "aic", "pearson_chisq", "deviance",
    "converged", "ic10", "ic25", "ic50", "ic75", "ic90", "all_gt_50", "all_lt_50",
    "curve_type"
]

# ---------------------------------------------------------------------------
# Model configurations
#   Each entry defines everything needed to run CV for one model type.
#   - data_type_options / train_options: paired lists (same order as in the
#     original training scripts so data is loaded in identical fashion)
#   - stratify_options: CV grouping strategies to evaluate
#   - scaler_type: "standard" | None  (RF uses no scaling)
#   - model_dir: subfolder under ../Models/ where best models are stored
#   - model_factory: callable(params) → fresh estimator instance
#   - result_dir: where to write the output CSV
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "glr": {
        # Data order matches glr_model.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Embed_Feat_Var", "MolFormer_Feat_Var", "ChemBERTa_Feat_Var", "Only_PC_Feat_Var"
        ],
        "stratify_options": ["inhibitor", "dbgap_rnaseq_sample", "random"],
        "scaler_type": "standard",   # same as glr_model.py: scaler="standard"
        "model_dir": "glr_models",
        "model_factory": lambda params: linear_model.Ridge(**params),
        "result_dir": "../Results/glr/",
        "result_file": "glr_cv_results.csv",
        "kfold_random_state": 42,    # same as glr_model.py KFold(random_state=42)
    },
    "svr": {
        # Data order matches svr_model.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Only_PC_Feat_Var", "Embed_Feat_Var", "MolFormer_Feat_Var", "ChemBERTa_Feat_Var"
        ],
        "stratify_options": ["inhibitor", "dbgap_rnaseq_sample", "random"],
        "scaler_type": "standard",   # same as svr_model.py: scaler="standard"
        "model_dir": "../Models/svr_models",
        "model_factory": lambda params: LSSVMRegressor(**params) if HAS_LSSVM else LinearSVR(**params),
        "result_dir": "../Results/svr/",
        "result_file": "svr_cv_results.csv",
        "kfold_random_state": 42,    # same as svr_model.py KFold(random_state=42)
    },
    "rf": {
        # Data order matches rf_model.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Only_PC_Feat_Var", "Embed_Feat_Var", "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"
        ],
        # "random" is included because trained models for that stratification exist
        "stratify_options": ["inhibitor", "dbgap_rnaseq_sample", "random"],
        "scaler_type": None,         # same as rf_model.py: scaler=None
        "model_dir": "../Models/rf_models",
        "model_factory": lambda params: ensemble.RandomForestRegressor(**params),
        "result_dir": "../Results/rf/",
        "result_file": "rf_cv_results.csv",
        "kfold_random_state": 42,    # same as rf_model.py KFold(random_state=42)
    },
    "catboost": {
        # Data order matches catboost_model_refactored.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Only_PC_Feat_Var", "Embed_Feat_Var", "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"
        ],
        # Stratify order matches catboost_model_refactored.py
        "stratify_options": ["random", "dbgap_rnaseq_sample", "inhibitor"],
        "scaler_type": None,         # same as catboost_model_refactored.py: scaler=None
        "model_dir": "../Models/catboost_models",
        "result_dir": "../Results/catboost/",
        "result_file": "catboost_cv_results.csv",
        "kfold_random_state": 42,
    },
    "lgbm": {
        # Data order matches lightgbm_model_refactored.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Only_PC_Feat_Var", "Embed_Feat_Var", "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"
        ],
        # Stratify order matches lightgbm_model_refactored.py
        "stratify_options": ["random", "dbgap_rnaseq_sample", "inhibitor"],
        "scaler_type": None,         # same as lightgbm_model_refactored.py: scaler=None
        "model_dir": "../Models/lgbm_models",
        "result_dir": "../Results/lgbm/",
        "result_file": "lgbm_cv_results.csv",
        "kfold_random_state": 42,
    },
    "xgb": {
        # Data order matches xgboost_model.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Only_PC_Feat_Var", "Embed_Feat_Var", "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"
        ],
        # Stratify order matches xgboost_model.py
        "stratify_options": ["inhibitor", "dbgap_rnaseq_sample", "random"],
        "scaler_type": None,         # same as xgboost_model.py: scaler=None
        "model_dir": "../Models/xgb_models",
        "result_dir": "../Results/xgb/",
        "result_file": "xgb_cv_results.csv",
        "kfold_random_state": 42,
    },
    "tabpfn": {
        # Data order matches tabpfn_model.py
        "train_options": [
            "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
            "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
        ],
        "data_type_options": [
            "Only_PC_Feat_Var", "Embed_Feat_Var", "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"
        ],
        # Stratify order matches tabpfn_model.py
        "stratify_options": ["inhibitor", "dbgap_rnaseq_sample", "random"],
        "scaler_type": "standard",   # same as tabpfn_model.py: scaler="standard"
        "model_dir": "../Models/tabpfn_models",
        "result_dir": "../Results/tabpfn/",
        "result_file": "tabpfn_cv_results.csv",
        "kfold_random_state": 42,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_feature_columns(df, label_col="auc"):
    """
    Reproduce the exact column selection logic of preprocess_data() in the
    original model scripts (glr_model.py / svr_model.py / rf_model.py).

    Steps (identical to the originals):
      1. Collect object/category columns as metadata.
      2. Add any column that contains at least one NaN value as metadata.
      3. Add the fixed EXCLUDE_COLS list as metadata.
      4. De-duplicate while preserving order (dict.fromkeys).
      5. Filter to columns that actually exist in the dataframe so that
         df.loc[:, metadata_cols] never raises a KeyError.
      6. feature_cols = all_columns - metadata_cols - label_col
         (np.setdiff1d returns a sorted array, matching the originals).

    Returns
    -------
    metadata_cols : list of str   - columns used as metadata / groups
    feature_cols  : list of str   - columns fed to the model (sorted)
    """
    all_columns = df.columns.tolist()
    all_columns_set = set(all_columns)
    column_types = df.dtypes.tolist()

    # 1. Object / category columns (same list comprehension as originals)
    metadata_cols = ["CID"] + [
        all_columns[i] for i in range(len(all_columns))
        if str(column_types[i]) == "object" or str(column_types[i]) == "category"
    ]

    # 2. Columns that contain at least one NaN (original uses big_train_df which
    #    equals train_data, so using df here is identical)
    nan_cols = [
        all_columns[i] for i in range(len(all_columns))
        if df[all_columns[i]].isnull().any()
    ]

    # 3–4. Merge and de-duplicate
    metadata_cols = metadata_cols + nan_cols + EXCLUDE_COLS
    metadata_cols = list(dict.fromkeys(metadata_cols))

    # 5. Keep only columns that actually exist in df so loc never raises KeyError
    #    (some EXCLUDE_COLS entries may be absent in certain data variants)
    metadata_cols = [c for c in metadata_cols if c in all_columns_set]

    # 6. Feature columns: sorted set difference, then remove label (same as originals)
    feature_cols = list(np.setdiff1d(all_columns, metadata_cols))
    feature_cols = list(np.setdiff1d(feature_cols, label_col))   # bare string → 0-d array, removes "auc"
    return metadata_cols, feature_cols


def preprocess_train(df, label_col="auc", scaler_type=None):
    """
    Extract feature matrix X, label vector y, and metadata from df.

    Applies scaling on the full training set (same approach as the original
    scripts, which fit the scaler on X_train before running HPO cross-validation).
    Returns X as a DataFrame with feature column names preserved.
    """
    metadata_cols, feature_cols = get_feature_columns(df, label_col)
    print(f"  Metadata cols: {len(metadata_cols)}, Feature cols: {len(feature_cols)}")

    # Safe: metadata_cols only contains columns that exist in df (filtered above)
    metadata = df.loc[:, metadata_cols].reset_index(drop=True)
    X = df.loc[:, feature_cols].copy()
    y = df[label_col].to_numpy().flatten()

    if scaler_type == "standard":
        print(f"  Applying StandardScaler to {len(feature_cols)} features ...")
        scaler = preprocessing.StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
        print(f"  Scaling done.")
    elif scaler_type == "minmax":
        print(f"  Applying MinMaxScaler to {len(feature_cols)} features ...")
        scaler = preprocessing.MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
        print(f"  Scaling done.")
    else:
        # No scaling — identical to rf_model.py's scaler=None path:
        #   scaler = None → if scaler: block skipped →
        #   X_train = pd.DataFrame(X_train, columns=feature_cols)
        print(f"  No scaling applied.")
        X = pd.DataFrame(X, columns=feature_cols)

    return X, y, metadata, feature_cols


def align_features_to_model(X, feature_cols, best_model):
    """
    Ensure the feature matrix X matches exactly the columns the saved model
    was trained on, in the correct order.

    sklearn >= 1.0 stores feature_names_in_ on the estimator when it was fit
    on a DataFrame.  If that attribute is present we use it as the ground truth;
    any mismatch with our preprocessed feature_cols is reported as a warning.

    If feature_names_in_ is absent (older sklearn or custom estimators) we fall
    back to the feature_cols produced by get_feature_columns(), which use the
    same np.setdiff1d logic as the originals and should therefore be consistent.

    Returns X aligned to the model's expected feature set and order.
    """
    if hasattr(best_model, "feature_names_in_"):
        model_cols = list(best_model.feature_names_in_)

        # Warn on any discrepancy between preprocessing and model
        preproc_set = set(feature_cols)
        model_set   = set(model_cols)
        if preproc_set != model_set:
            missing = model_set - preproc_set
            extra   = preproc_set - model_set
            print(f"  WARNING: feature set mismatch between preprocessing and model!")
            if missing:
                print(f"    In model but not in preprocessed data: {sorted(missing)}")
            if extra:
                print(f"    In preprocessed data but not in model:  {sorted(extra)}")

        print(f"  Aligning to {len(model_cols)} features stored in model (feature_names_in_)")
        return X[model_cols]   # reorder / subset to exactly what the model expects

    # No feature_names_in_: trust get_feature_columns() which replicates the originals
    print(f"  No feature_names_in_ on model; using {len(feature_cols)} preprocessed features")
    return X


def _fit_fold_generic(fold, train_idx, val_idx, X_arr, y, model_type, model_params):
    """
    Fit one fold for rf / catboost / lgbm / xgb and return metrics.  Defined at
    module level so loky can pickle it for parallel execution.  X is passed
    as a numpy array to avoid DataFrame serialisation overhead.
    """
    if model_type == "rf":
        model = ensemble.RandomForestRegressor(**model_params)
    elif model_type == "catboost":
        model = CatBoostRegressor(**model_params)
    elif model_type == "lgbm":
        model = lgb.LGBMRegressor(**model_params)
    elif model_type == "xgb":
        model = xgb.XGBRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.fit(X_arr[train_idx], y[train_idx])
    y_pred = model.predict(X_arr[val_idx])
    return fold, calculate_regression_metrics(y[val_idx], y_pred)


def _fit_fold_tabpfn(fold, train_idx, val_idx, X_arr, y, model_params, device):
    """
    Fit one TabPFN fold on a specific GPU device.  Defined at module level so
    loky can pickle it for parallel execution — one fold per GPU.

    Imports are local so each spawned worker process resolves them independently
    without depending on the parent's global state (e.g. HF login).
    """
    import torch
    from tabpfn import TabPFNRegressor
    from misc import calculate_regression_metrics as _calc_metrics

    params = {**model_params, "device": device}
    model = TabPFNRegressor(**params)
    model.fit(X_arr[train_idx], y[train_idx])
    y_pred = model.predict(X_arr[val_idx])
    m = _calc_metrics(y[val_idx], y_pred)
    del model
    torch.cuda.empty_cache()
    return fold, m


def run_5fold_cv_generic(model_type, model_params, X, y, groups,
                         n_splits=N_CV_SPLITS, random_state=42):
    """
    Perform 5-fold CV for rf / catboost / lgbm / xgb with all folds running in
    parallel.

    Uses GroupKFold when groups are provided so that no patient or inhibitor
    appears in both train and validation within a fold (avoids data leakage).
    Uses KFold(random_state) for the 'random' stratification.

    GroupKFold with shuffle=True randomly assigns groups to folds, preventing
    any ordering bias that might arise from the order groups appear in the data.
    """
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits, shuffle=True)
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y))

    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)

    print(f"  Launching {n_splits} folds in parallel "
          f"({THREADS_PER_FOLD} threads/fold x {n_splits} folds = {NUM_THREADS} total threads)")
    print(f"  Train sizes ~{len(X_arr) * (n_splits - 1) // n_splits}, "
          f"val sizes ~{len(X_arr) // n_splits}")

    t_all = time.time()
    raw_results = Parallel(n_jobs=n_splits, backend="loky")(
        delayed(_fit_fold_generic)(fold, train_idx, val_idx, X_arr, y, model_type, model_params)
        for fold, (train_idx, val_idx) in enumerate(splits)
    )
    print(f"  All {n_splits} folds finished in {time.time() - t_all:.1f}s")

    raw_results.sort(key=lambda r: r[0])
    fold_metrics = []
    for fold, metrics in raw_results:
        m = metrics
        fold_metrics.append(m)
        print(f"    Fold {fold + 1}/{n_splits}: "
              f"MAE={m[0]:.3f}, RMSE={m[1]:.3f}, "
              f"R2={m[2]:.3f}, Pearson={m[3]:.3f}, Spearman={m[4]:.3f}")

    return fold_metrics


def run_5fold_cv_tabpfn(model_params, X, y, groups,
                        n_splits=N_CV_SPLITS, random_state=42):
    """
    5-fold CV for TabPFN with automatic GPU parallelism.

    - If torch.cuda.device_count() >= n_splits: each fold is dispatched to its
      own GPU (fold i → cuda:i) and all folds run simultaneously via loky.
      This gives ~5x wall-clock speedup when 5 GPUs are available.
    - Otherwise: folds run sequentially on the single available device, with
      GPU cache cleared after every fold to prevent memory accumulation.
    """
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits, shuffle=True)
        splits = list(cv.split(X, y, groups))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y))

    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
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
        # ── Sequential path: one fold at a time on the available device ──────
        dev = model_params.get("device", "cpu")
        print(f"  {n_gpus} GPU(s) available (need {n_splits} for full parallelism) — "
              f"running {n_splits} folds sequentially on {dev}")

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            t_fold = time.time()
            model = TabPFNRegressor(**model_params)
            model.fit(X_arr[train_idx], y[train_idx])
            y_pred = model.predict(X_arr[val_idx])
            m = calculate_regression_metrics(y[val_idx], y_pred)
            fold_metrics.append(m)
            print(f"    Fold {fold + 1}/{n_splits}: "
                  f"MAE={m[0]:.3f}, RMSE={m[1]:.3f}, "
                  f"R2={m[2]:.3f}, Pearson={m[3]:.3f}, Spearman={m[4]:.3f} "
                  f"({time.time() - t_fold:.1f}s)")
            del model
            torch.cuda.empty_cache()

        print(f"  All {n_splits} folds finished in {time.time() - t_all:.1f}s")

    return fold_metrics


def aggregate_fold_metrics(fold_metrics):
    """Return (means, stds) arrays rounded to 3 decimal places."""
    arr = np.array(fold_metrics)
    means = np.round(np.mean(arr, axis=0), 3)
    stds = np.round(np.std(arr, axis=0), 3)
    return means, stds


def build_result_row(data_type, stratify_by, means, stds):
    """Build a result dict with raw mean/std columns and formatted 'mean +/- std' columns.

    Raw numeric columns are stored as Python floats rounded to 3 decimal places
    so the CSV never contains extra floating-point digits.
    """
    metric_names = ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
    row = {"data_type": data_type, "stratify_by": stratify_by}
    for i, name in enumerate(metric_names):
        row[f"{name}_mean"] = round(float(means[i]), 3)
        row[f"{name}_std"]  = round(float(stds[i]),  3)
        row[name] = f"{means[i]:.3f} +/- {stds[i]:.3f}"
    return row


OUTPUT_COLS = (
    ["data_type", "stratify_by"] +
    [f"{n}_{s}" for n in ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
     for s in ["mean", "std"]] +
    ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
)


# ---------------------------------------------------------------------------
# Main: run CV for all configured model types
# ---------------------------------------------------------------------------

# +
# Run catboost, lgbm, xgb, tabpfn
for model_type in ["tabpfn"]:

    if model_type == "tabpfn" and not HAS_TABPFN:
        print(f"\n[SKIP] tabpfn not installed — skipping TabPFN CV.")
        continue

    cfg = MODEL_CONFIGS[model_type]

    print("\n" + "=" * 80)
    print(f"{model_type.upper()} 5-Fold Cross-Validation")
    print("=" * 80)

    os.makedirs(cfg["result_dir"], exist_ok=True)
    results = []

    n_data_types = len(cfg["data_type_options"])
    n_stratify   = len(cfg["stratify_options"])
    print(f"  {n_data_types} data type(s) x {n_stratify} stratification(s) = "
          f"up to {n_data_types * n_stratify} CV run(s) total.")

    for input_option, data_type in enumerate(cfg["data_type_options"]):
        train_file = cfg["train_options"][input_option]

        print(f"\n[{input_option + 1}/{n_data_types}] Loading training data for {data_type} ...")
        print(f"  File: {train_file}")
        t_load = time.time()
        big_train_df = pd.read_pickle(train_file, compression="zip")
        print(f"  Loaded in {time.time() - t_load:.1f}s — raw shape: {big_train_df.shape}")

        print(f"  Preprocessing ...")
        X_preprocessed, Y_all, metadata_X_train, feature_cols = preprocess_train(
            big_train_df, label_col="auc", scaler_type=cfg["scaler_type"]
        )
        print(f"  X shape: {X_preprocessed.shape}, y shape: {Y_all.shape}")

        for strat_idx, stratify_by in enumerate(cfg["stratify_options"]):
            # Model filename: {model_type}_{data_type}_{stratify_by}_optuna_best.pk
            model_path = os.path.join(
                cfg["model_dir"],
                f"{model_type}_{data_type}_{stratify_by}_optuna_best.pk"
            )

            if not os.path.exists(model_path):
                print(f"  [SKIP] Model not found: {model_path}")
                continue

            print(f"\n  [{strat_idx + 1}/{n_stratify}] Running CV: "
                  f"data_type={data_type}, stratify_by={stratify_by}")

            # Load the saved model and extract hyperparameters.
            # Adjust thread count so that THREADS_PER_FOLD cores are used per fold,
            # saturating all NUM_THREADS cores across the N_CV_SPLITS parallel folds.
            print(f"  Loading model from: {model_path}")
            saved_model = load_model(model_path)
            model_params = saved_model.get_params()

            if model_type == "rf":
                model_params["n_jobs"] = THREADS_PER_FOLD
                model_params["verbose"] = 0
            elif model_type == "catboost":
                # CatBoost uses thread_count (not n_jobs); suppress all output
                model_params["thread_count"] = THREADS_PER_FOLD
                model_params["verbose"] = False
                model_params["allow_writing_files"] = False
            elif model_type == "lgbm":
                model_params["n_jobs"] = THREADS_PER_FOLD
                model_params["verbosity"] = -1
            elif model_type == "xgb":
                model_params["n_jobs"] = THREADS_PER_FOLD
                model_params["verbosity"] = 0
            elif model_type == "tabpfn":
                # TabPFN uses GPU — update device to the currently detected one
                model_params["device"] = DEVICE

            print(f"  {model_type.upper()} params "
                  f"(threads_per_fold={THREADS_PER_FOLD}, {N_CV_SPLITS} folds in parallel): "
                  f"{model_params}")

            # Align features to what this specific model was trained on.
            # Always derived from the freshly preprocessed X_preprocessed so that
            # each stratify_by iteration starts from the same unmodified base.
            X_cv = align_features_to_model(X_preprocessed, feature_cols, saved_model)

            # Determine groups for CV — mirrors HPO stratification in each training script.
            # GroupKFold ensures no patient/inhibitor leaks across train and validation folds.
            if stratify_by == "dbgap_rnaseq_sample":
                groups = metadata_X_train["dbgap_rnaseq_sample"].to_numpy()
                print(f"  Groups: {stratify_by} — {len(np.unique(groups))} unique patients")
            elif stratify_by == "inhibitor":
                groups = metadata_X_train["inhibitor"].to_numpy()
                print(f"  Groups: {stratify_by} — {len(np.unique(groups))} unique inhibitors")
            else:
                groups = None
                print(f"  Groups: None (random KFold)")

            t_cv = time.time()
            if model_type == "tabpfn":
                fold_metrics = run_5fold_cv_tabpfn(
                    model_params, X_cv, Y_all, groups,
                    random_state=cfg["kfold_random_state"]
                )
                torch.cuda.empty_cache()
            else:
                fold_metrics = run_5fold_cv_generic(
                    model_type, model_params, X_cv, Y_all, groups,
                    random_state=cfg["kfold_random_state"]
                )
            print(f"  CV completed in {time.time() - t_cv:.1f}s")

            means, stds = aggregate_fold_metrics(fold_metrics)
            metric_names = ["MAE", "RMSE", "R2", "PearsonR", "SpearmanR"]
            print("  Summary: " + ", ".join(
                f"{n}={means[i]:.3f}±{stds[i]:.3f}" for i, n in enumerate(metric_names)
            ))

            results.append(build_result_row(data_type, stratify_by, means, stds))

    # Save results
    results_df = pd.DataFrame(results, columns=OUTPUT_COLS)
    out_path = os.path.join(cfg["result_dir"], cfg["result_file"])
    results_df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"\n{model_type.upper()} CV results saved to: {out_path}")
    print(results_df.to_string(index=False))
# -
