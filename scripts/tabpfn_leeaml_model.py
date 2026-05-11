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
import datetime
import pandas as pd
import numpy as np
import torch
import optuna

from sklearn import preprocessing
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))

from tabpfn import TabPFNRegressor

os.makedirs("logs", exist_ok=True)
os.makedirs("../Results/tabpfn_leeaml/", exist_ok=True)
os.makedirs("../Results/tabpfn_leeaml/optuna/", exist_ok=True)

from misc import save_model, load_model, calculate_regression_metrics

torch.cuda.empty_cache()

def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

device = get_device()
print("Device:", device)

# -

# Columns to exclude from features (metadata / curve fitting / identifiers)
EXCLUDE_COLS = [
    "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample",
    "inhibitor", "type", "status", "paper_inclusion",
    "min_conc", "max_conc", "intercept", "beta", "beta_z", "beta_p",
    "aic", "pearson_chisq", "deviance", "converged",
    "ic10", "ic25", "ic50", "ic75", "ic90",
    "all_gt_50", "all_lt_50", "curve_type",
    # LeeAML-specific identifiers
    "sample_id", "SMILES", "InChIKey", "InChIKey_x", "InChIKey_y",
    "Name_x", "Name_y", "CanonicalSMILES", "Targets",
]

LABEL_COL = "auc"
DATA_TYPE = "Embed_Feat_Var"
TRAIN_PATH = "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
TEST_PATH  = "../Data/leeaml/LeeAML_Common_Set_Var_with_Drug_Embedding_Patient_Info.pkl"


def preprocess_data(train_df, test_df, label_col=LABEL_COL, scaler_type="standard"):
    """
    Build feature matrix using only columns common to both train and test.
    Excludes metadata, object/category, and NaN-containing columns.
    Stratification metadata is preserved separately.
    """
    common_cols = sorted(set(train_df.columns) & set(test_df.columns))

    # Identify columns to exclude: metadata list + object/category dtypes + NaN-containing
    obj_cols_train = [c for c in common_cols if str(train_df[c].dtype) in ("object", "category")]
    obj_cols_lee   = [c for c in common_cols if str(test_df[c].dtype)  in ("object", "category")]
    nan_cols_train = [c for c in common_cols if train_df[c].isnull().any()]
    nan_cols_lee   = [c for c in common_cols if test_df[c].isnull().any()]

    drop_set = set(EXCLUDE_COLS) | set(obj_cols_train) | set(obj_cols_lee) \
             | set(nan_cols_train) | set(nan_cols_lee) | {label_col}

    feature_cols = [c for c in common_cols if c not in drop_set]
    print(f"Total common columns:  {len(common_cols)}")
    print(f"Dropped (meta/obj/nan):{len(drop_set)}")
    print(f"Feature columns used:  {len(feature_cols)}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].to_numpy(dtype=float).flatten()

    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[label_col].to_numpy(dtype=float).flatten()

    # Metadata kept for stratification and output
    meta_train = train_df[[c for c in ["dbgap_rnaseq_sample", "inhibitor", "primary_key"]
                            if c in train_df.columns]].copy()
    meta_test  = test_df[[c for c in ["sample_id", "inhibitor", "primary_key"]
                           if c in test_df.columns]].copy()

    if scaler_type == "standard":
        scaler = preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)

    return X_train, y_train, X_test, y_test, meta_train, meta_test, scaler, feature_cols


def hyperparameter_optimization(X_train, y_train, meta_train, stratify_by, scaler, device):
    """Optuna search for best TabPFNRegressor hyperparameters using patient-level CV."""

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 4, 32),
            "softmax_temperature": trial.suggest_float("softmax_temperature", 0.7, 1.8),
            "device":             device,
            "random_state":       42,
        }
        model = TabPFNRegressor(**params)

        # Stratification: patient splits (primary choice), inhibitor, or random
        if stratify_by == "dbgap_rnaseq_sample" and "dbgap_rnaseq_sample" in meta_train.columns:
            groups = meta_train["dbgap_rnaseq_sample"].to_numpy()
        elif stratify_by == "inhibitor" and "inhibitor" in meta_train.columns:
            groups = meta_train["inhibitor"].to_numpy()
        else:
            groups = None

        if groups is not None:
            cv = GroupKFold(n_splits=5, shuffle=True)
            scores = cross_val_score(model, X_train, y_train, groups=groups,
                                     scoring="neg_mean_absolute_error", cv=cv)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train,
                                     scoring="neg_mean_absolute_error", cv=cv)

        del model
        torch.cuda.empty_cache()
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best -MAE:", study.best_value)
    print("Best params:", study.best_params)
    for t in study.trials:
        print(f"  Trial {t.number}: value={t.value:.4f}, params={t.params}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study.trials_dataframe().to_csv(
        f"logs/optuna_trials_tabpfn_leeaml_{DATA_TYPE}_{stratify_by}_{ts}.csv", index=False
    )

    # Fit best model on full training set
    best_params = {**study.best_params, "device": device, "random_state": 42}
    best_model = TabPFNRegressor(**best_params)
    best_model.fit(X_train, y_train)

    MODEL_SAVE_DIR = "../Models/tabpfn_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_model(best_model, f"{MODEL_SAVE_DIR}/tabpfn_leeaml_{DATA_TYPE}_{stratify_by}_optuna_best.pk")
    save_model(scaler,     f"{MODEL_SAVE_DIR}/tabpfn_leeaml_{DATA_TYPE}_{stratify_by}_scaling.pk")

    del best_model
    torch.cuda.empty_cache()


def make_predictions(model, X_test, y_test, meta_test, stratify_by, all_metrics):
    y_pred = model.predict(X_test)
    _, _, r2, pearson_r, spearman_r = calculate_regression_metrics(y_test, y_pred)
    print(f"[{stratify_by}] R2={r2}  Pearson={pearson_r}  Spearman={spearman_r}")

    out_df = meta_test.copy()
    out_df["predictions"] = y_pred
    out_df["labels"]      = y_test
    out_path = (f"../Results/tabpfn_leeaml/optuna/"
                f"{DATA_TYPE}_{stratify_by}_optuna_predictions.csv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Predictions written to: {out_path}")

    all_metrics.append({
        "data_type":   DATA_TYPE,
        "stratify_by": stratify_by,
        "r2":          r2,
        "pearson_r":   pearson_r,
        "spearman_r":  spearman_r,
    })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
print("Loading training data:", TRAIN_PATH)
train_df = pd.read_pickle(TRAIN_PATH, compression="zip")

print("Loading LeeAML test data:", TEST_PATH)
test_df  = pd.read_pickle(TEST_PATH, compression="zip")

X_train, y_train, X_test, y_test, meta_train, meta_test, scaler, feature_cols = \
    preprocess_data(train_df, test_df, label_col=LABEL_COL, scaler_type="standard")

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)

stratify_by = "dbgap_rnaseq_sample"
all_metrics = []

print(f"\n{'='*60}")
print(f"Running Optuna optimisation | stratify_by = {stratify_by}")
print(f"{'='*60}")

hyperparameter_optimization(X_train, y_train, meta_train, stratify_by, scaler, device)

model_path = f"../Models/tabpfn_models/tabpfn_leeaml_{DATA_TYPE}_{stratify_by}_optuna_best.pk"
best_model = load_model(model_path)

make_predictions(best_model, X_test, y_test, meta_test, stratify_by, all_metrics)

del best_model
torch.cuda.empty_cache()

# Save summary metrics
metrics_path = f"../Results/tabpfn_leeaml/optuna/{DATA_TYPE}_test_metrics_summary.csv"
pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
print(f"\nMetrics summary written to: {metrics_path}")
print(pd.DataFrame(all_metrics).to_string(index=False))
