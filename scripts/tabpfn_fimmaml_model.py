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
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

from huggingface_hub import login
_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    raise EnvironmentError(f"HF_TOKEN not found. Check that it is set in {_env_path}")
login(token=_hf_token)

from tabpfn import TabPFNRegressor

os.makedirs("logs", exist_ok=True)
os.makedirs("../Results/tabpfn_fimmaml/", exist_ok=True)
os.makedirs("../Results/tabpfn_fimmaml/optuna/", exist_ok=True)

from misc import save_model, load_model, calculate_regression_metrics

torch.cuda.empty_cache()

def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

device = get_device()
print("Device:", device)

# -

# Columns to exclude from features (metadata / curve-fit parameters / identifiers)
EXCLUDE_COLS = [
    "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample",
    "inhibitor", "type", "status", "paper_inclusion",
    "min_conc", "max_conc", "intercept", "beta", "beta_z", "beta_p",
    "aic", "pearson_chisq", "deviance", "converged",
    "ic10", "ic25", "ic50", "ic75", "ic90",
    "all_gt_50", "all_lt_50", "curve_type",
    # Drug / sample identifiers
    "sample_id", "SMILES", "InChIKey", "InChIKey_x", "InChIKey_y",
    "Name", "Name_x", "Name_y", "CanonicalSMILES", "Targets",
    # BeatAML label kept separate
    "auc",
    # FIMM-AML label kept separate
    "DSS", "sDSS",
    # FIMM-AML drug info columns (may duplicate drug features)
    "BeatAML_Name",
]

# BeatAML trains on raw AUC; FIMM-AML is evaluated by comparing predicted AUC vs raw DSS.
TRAIN_LABEL_COL = "auc"
TEST_LABEL_COL  = "DSS"

# Four drug-embedding / feature combinations to evaluate
DATA_CONFIGS = {
    #"Embed_Feat_Var": {
    #    "train_path": "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    #    "test_path":  "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    #},
    #"PC_Feat_Var": {
    #    "train_path": "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    #    "test_path":  "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    #},
    "MolFormer_Feat_Var": {
        "train_path": "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
        "test_path":  "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
    },
    "ChemBERTa_Feat_Var": {
        "train_path": "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
        "test_path":  "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    },
}


def preprocess_data(train_df, test_df, scaler_type="standard"):
    """
    Build feature matrix from columns common to both train (BeatAML) and test (FIMM-AML).
    All features are z-scored (StandardScaler fit on train, applied to test) to avoid
    scale bias across heterogeneous feature types.
    Labels are kept as raw values: BeatAML AUC for training, FIMM-AML DSS for evaluation.
    Pearson R and Spearman R compare predicted AUC against actual DSS.
    Patient-level stratification metadata is preserved separately.
    """
    common_cols = sorted(set(train_df.columns) & set(test_df.columns))

    obj_cols_train = [c for c in common_cols if str(train_df[c].dtype) in ("object", "category")]
    obj_cols_fimm  = [c for c in common_cols if str(test_df[c].dtype)  in ("object", "category")]
    nan_cols_train = [c for c in common_cols if train_df[c].isnull().any()]
    nan_cols_fimm  = [c for c in common_cols if test_df[c].isnull().any()]

    drop_set = (set(EXCLUDE_COLS)
                | set(obj_cols_train) | set(obj_cols_fimm)
                | set(nan_cols_train) | set(nan_cols_fimm))

    feature_cols = [c for c in common_cols if c not in drop_set]
    print(f"  Total common columns : {len(common_cols)}")
    print(f"  Dropped (meta/obj/nan): {len(drop_set)}")
    print(f"  Feature columns used : {len(feature_cols)}")

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    # Z-score all features: scaler fit on training set only to prevent data leakage
    if scaler_type == "standard":
        feat_scaler = preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        feat_scaler = preprocessing.MinMaxScaler()
    else:
        feat_scaler = None

    if feat_scaler is not None:
        X_train = pd.DataFrame(feat_scaler.fit_transform(X_train), columns=feature_cols)
        X_test  = pd.DataFrame(feat_scaler.transform(X_test),      columns=feature_cols)

    # Raw labels — AUC for training, DSS for evaluation (no scaling)
    y_train = train_df[TRAIN_LABEL_COL].to_numpy(dtype=float).flatten()
    y_test  = test_df[TEST_LABEL_COL].to_numpy(dtype=float).flatten()

    # Metadata for stratification and output
    meta_train = train_df[[c for c in ["dbgap_rnaseq_sample", "inhibitor", "primary_key"]
                            if c in train_df.columns]].copy()
    meta_test  = test_df[[c for c in ["sample_id", "inhibitor", "primary_key"]
                           if c in test_df.columns]].copy()

    return X_train, y_train, X_test, y_test, meta_train, meta_test, feat_scaler, feature_cols


def hyperparameter_optimization(X_train, y_train, meta_train, data_type, feat_scaler, device):
    """Optuna search for best TabPFNRegressor hyperparameters using patient-level GroupKFold CV."""

    def objective(trial):
        params = {
            "n_estimators":              trial.suggest_int("n_estimators", 4, 12),
            "softmax_temperature":       trial.suggest_float("softmax_temperature", 0.7, 1.8),
            "device":                    device,
            "random_state":              42,
            "ignore_pretraining_limits": True,
            "memory_saving_mode":        False,
        }
        model = TabPFNRegressor(**params)

        # Patient-level stratification using BeatAML sample IDs
        if "dbgap_rnaseq_sample" in meta_train.columns:
            groups = meta_train["dbgap_rnaseq_sample"].to_numpy()
            cv = GroupKFold(n_splits=5)
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

    print(f"  Best -MAE: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    for t in study.trials:
        print(f"    Trial {t.number}: value={t.value:.4f}, params={t.params}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study.trials_dataframe().to_csv(
        f"logs/optuna_trials_tabpfn_fimmaml_{data_type}_{ts}.csv", index=False
    )

    # Refit best model on full training set
    best_params = {**study.best_params, "device": device, "random_state": 42,
                   "ignore_pretraining_limits": True, "memory_saving_mode": False}
    best_model  = TabPFNRegressor(**best_params)
    best_model.fit(X_train, y_train)

    MODEL_SAVE_DIR = "../Models/tabpfn_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_model(best_model,  f"{MODEL_SAVE_DIR}/tabpfn_fimmaml_{data_type}_optuna_best.pk")
    save_model(feat_scaler, f"{MODEL_SAVE_DIR}/tabpfn_fimmaml_{data_type}_feat_scaler.pk")

    del best_model
    torch.cuda.empty_cache()


def make_predictions(model, X_test, y_test, meta_test, data_type, all_metrics):
    y_pred = model.predict(X_test)
    _, _, r2, pearson_r, spearman_r = calculate_regression_metrics(y_test, y_pred)
    print(f"  [FIMM-AML] R²={r2}  Pearson={pearson_r}  Spearman={spearman_r}")

    out_df = meta_test.copy()
    out_df["predicted_auc"] = y_pred
    out_df["DSS"]           = y_test   # raw FIMM-AML DSS — ground-truth label
    out_path = f"../Results/tabpfn_fimmaml/optuna/{data_type}_optuna_predictions.csv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Predictions written to: {out_path}")

    all_metrics.append({
        "data_type":  data_type,
        "r2":         r2,
        "pearson_r":  pearson_r,
        "spearman_r": spearman_r,
    })


# ---------------------------------------------------------------------------
# Main pipeline — iterate over all four embedding / feature types
# ---------------------------------------------------------------------------
all_metrics = []

for data_type, paths in DATA_CONFIGS.items():
    print(f"\n{'='*70}")
    print(f"DATA TYPE: {data_type}")
    print(f"  Train : {paths['train_path']}")
    print(f"  Test  : {paths['test_path']}")
    print(f"{'='*70}")

    print("Loading data...")
    train_df = pd.read_pickle(paths["train_path"], compression="zip")
    test_df  = pd.read_pickle(paths["test_path"],  compression="zip")
    print(f"  BeatAML train : {train_df.shape}")
    print(f"  FIMM-AML test : {test_df.shape}")

    X_train, y_train, X_test, y_test, meta_train, meta_test, feat_scaler, feature_cols = \
        preprocess_data(train_df, test_df, scaler_type="standard")

    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")

    print(f"\nRunning Optuna optimisation ({data_type})...")
    hyperparameter_optimization(X_train, y_train, meta_train, data_type, feat_scaler, device)

    model_path = f"../Models/tabpfn_models/tabpfn_fimmaml_{data_type}_optuna_best.pk"
    best_model = load_model(model_path)

    print(f"\nEvaluating on FIMM-AML (label: raw {TEST_LABEL_COL})...")
    make_predictions(best_model, X_test, y_test, meta_test, data_type, all_metrics)

    del best_model, train_df, test_df
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
metrics_df   = pd.DataFrame(all_metrics)
metrics_path = "../Results/tabpfn_fimmaml/optuna/all_data_types_test_metrics_summary.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"\n{'='*70}")
print("FIMM-AML evaluation summary  (train: raw AUC | test: raw DSS | all features: z-scored)")
print(f"{'='*70}")
print(metrics_df.to_string(index=False))
print(f"\nMetrics summary written to: {metrics_path}")
