# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: BeatAML2.0
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TabPFN Ablation Study - 4-Group Baseline
#
# Runs the baseline experiment with all patient groups. Each is run with Drug_PC, Drug_Embed, and Drug_PC+Drug_Embed.
# Feature groups are defined by the `Type` column in `ablation_feature_columns.csv`.

# %%
import os
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import optuna
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

import scipy
import torch
import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))

from tabpfn import TabPFNRegressor
from misc import save_model, load_model, calculate_regression_metrics

os.makedirs("logs", exist_ok=True)
os.makedirs("../Results/tabpfn/ablation/", exist_ok=True)
os.makedirs("../Models/tabpfn_models/ablation/", exist_ok=True)

def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

device = get_device()
print(device)
torch.cuda.empty_cache()

# %% [markdown]
# ## Load Feature Metadata and Define Feature Groups

# %%
feature_metadata = pd.read_csv("../Data/ablation_feature_columns.csv")
print(f"Total entries in ablation_feature_columns.csv: {len(feature_metadata)}")

# Group columns by Type
feature_groups = {}
for type_name, group_df in feature_metadata.groupby("Type"):
    feature_groups[type_name] = group_df["Column_Label"].tolist()

print("\nAll groups from CSV:")
for group_name, cols in sorted(feature_groups.items()):
    print(f"  {group_name}: {len(cols)} features")

# Remove Tsne features entirely
tsne_cols = feature_groups.pop("Tsne", [])
print(f"\nRemoved Tsne features: {tsne_cols}")

# Separate Drug_Embed and Drug_PC groups
drug_embed_cols_from_csv = feature_groups.pop("Drug_Embed", [])
print(f"Drug_Embed features from CSV: {len(drug_embed_cols_from_csv)}")

drug_pc_cols_from_csv = feature_groups.pop("Drug_PC", [])
print(f"Drug_PC features from CSV: {len(drug_pc_cols_from_csv)}")

# Merge Clinical, CellType and Module into one group
clinical_cols = feature_groups.pop("Clinical", [])
celltype_cols = feature_groups.pop("CellType", [])
module_cols = feature_groups.pop("Module", [])
feature_groups["Clinical_CellType_Module"] = clinical_cols + celltype_cols + module_cols

print("\nPatient feature groups:")
for group_name, cols in sorted(feature_groups.items()):
    print(f"  {group_name}: {len(cols)} features")

# %% [markdown]
# ## Load Training and Test Data

# %%
big_train_df = pd.read_pickle("../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl", compression="zip")
big_test_df = pd.read_pickle("../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl", compression="zip")
print(f"Training shape: {big_train_df.shape}")
print(f"Test shape: {big_test_df.shape}")


# %% [markdown]
# ## Preprocess Data
#
# Remove metadata, object/category, NaN, and Tsne columns to obtain the full feature set.

# %%
def preprocess_data(train_data, test_data, tsne_cols, label_col="auc"):
    all_columns = train_data.columns.tolist()
    column_types = train_data.dtypes.tolist()

    exclude_cols = ["primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample",
                    "inhibitor", "type", "status", "paper_inclusion",
                    "min_conc", "max_conc", "intercept", "beta", "beta_z", "beta_p",
                    "aic", "pearson_chisq", "deviance", "converged",
                    "ic10", "ic25", "ic50", "ic75", "ic90", "all_gt_50", "all_lt_50", "curve_type"]

    metadata_cols = ["CID"] + [all_columns[i] for i in range(len(all_columns))
                                if str(column_types[i]) in ("object", "category")]
    nan_cols = [c for c in all_columns if train_data[c].isnull().any()]
    tsne_in_data = [c for c in tsne_cols if c in all_columns]

    metadata_cols = list(dict.fromkeys(metadata_cols + nan_cols + exclude_cols + tsne_in_data))
    print(f"Metadata columns: {len(metadata_cols)}")

    feature_cols = sorted(list(set(all_columns) - set(metadata_cols) - {label_col}))
    print(f"Total features: {len(feature_cols)}")

    metadata_X_train = train_data.loc[:, [c for c in metadata_cols if c in all_columns]]
    X_train = train_data.loc[:, feature_cols]
    y_train = train_data[label_col].to_numpy().flatten()

    metadata_X_test = test_data.loc[:, [c for c in metadata_cols if c in test_data.columns]]
    X_test = test_data.loc[:, feature_cols]
    y_test = test_data[label_col].to_numpy().flatten()

    return X_train, y_train, X_test, y_test, metadata_X_train, metadata_X_test, feature_cols


X_train_all, Y_train, X_test_all, Y_test, metadata_X_train, metadata_X_test, all_feature_cols = \
    preprocess_data(big_train_df, big_test_df, tsne_cols)

print(f"\nTraining features shape: {X_train_all.shape}")
print(f"Test features shape: {X_test_all.shape}")

# %% [markdown]
# ## Map Features to Ablation Groups
#
# Drug embedding features are identified from the `Drug_Embed` type in `ablation_feature_columns.csv`.

# %%
# Use Drug_Embed and Drug_PC columns from CSV, filtered to those present in the feature set
drug_embedding_cols = [c for c in drug_embed_cols_from_csv if c in all_feature_cols]
print(f"Drug embedding features: {len(drug_embedding_cols)}")

drug_pc_cols = [c for c in drug_pc_cols_from_csv if c in all_feature_cols]
print(f"Drug PC features: {len(drug_pc_cols)}")

# Define drug feature options for ablation
drug_feature_options = {
    "Drug_PC": drug_pc_cols,
    "Drug_Embed": drug_embedding_cols,
    "Drug_PC+Drug_Embed": drug_pc_cols + drug_embedding_cols,
}

print("\nDrug feature options:")
for name, cols in drug_feature_options.items():
    print(f"  {name}: {len(cols)} features")

# Map patient feature groups to columns present in the data
ablation_groups = {}
for group_name, group_cols in feature_groups.items():
    cols_in_data = [c for c in group_cols if c in all_feature_cols]
    if cols_in_data:
        ablation_groups[group_name] = cols_in_data

print("\nPatient feature groups for ablation:")
for group_name, cols in sorted(ablation_groups.items()):
    print(f"  {group_name}: {len(cols)} features")
patient_feature_count = sum(len(c) for c in ablation_groups.values())
print(f"\nTotal patient features: {patient_feature_count}")
print(f"Total drug embedding features: {len(drug_embedding_cols)}")
print(f"Total drug PC features: {len(drug_pc_cols)}")
print(f"Total features in data: {len(all_feature_cols)}")


# %% [markdown]
# ## Define Hyperparameter Optimization, Prediction, and Plotting Functions

# %%
def hyperparameter_optimization(X_train, Y_train, metadata_X_train, experiment_name,
                                scaler_type="standard", device="cpu"):
    # Scale features
    if scaler_type == "standard":
        scaler = preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    else:
        X_train_scaled = X_train.copy()

    groups = metadata_X_train["dbgap_rnaseq_sample"].to_numpy()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 4, 32),
            "softmax_temperature": trial.suggest_float("softmax_temperature", 0.7, 1.8),
            "device": device,
            "random_state": 42,
            "ignore_pretraining_limits": True,
        }
        model = TabPFNRegressor(**params)
        cv = GroupKFold(n_splits=5, shuffle=True)
        scores = cross_val_score(estimator=model, X=X_train_scaled, y=Y_train,
                                 groups=groups, scoring="neg_mean_absolute_error", cv=cv)
        del model
        torch.cuda.empty_cache()
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print(f"\n[{experiment_name}] Best -MAE: {study.best_value}")
    print(f"[{experiment_name}] Best params: {study.best_params}")

    print("\nAll trial results:")
    for t in study.trials:
        print(f"Trial {t.number}: value={t.value}, params={t.params}")

    study.trials_dataframe().to_csv(
        f"logs/optuna_trials_tabpfn_ablation_{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False
    )

    best_model = TabPFNRegressor(**study.best_params, device=device, random_state=42, ignore_pretraining_limits=True)
    best_model.fit(X_train_scaled, Y_train)

    save_model(best_model, f"tabpfn_models/ablation/tabpfn_{experiment_name}_best.pk")
    save_model(scaler, f"tabpfn_models/ablation/tabpfn_{experiment_name}_scaler.pk")

    del best_model, X_train_scaled
    torch.cuda.empty_cache()

    return study, scaler


# %%
def make_predictions(experiment_name, X_test, Y_test, metadata_X_test, scaler):
    model_path = f"../Models/tabpfn_models/ablation/tabpfn_{experiment_name}_best.pk"
    best_model = load_model(model_path)

    if scaler:
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    else:
        X_test_scaled = X_test.copy()

    y_pred = best_model.predict(X_test_scaled)
    test_metrics = calculate_regression_metrics(Y_test, y_pred)
    print(f"[{experiment_name}] Test Metrics - MAE: {test_metrics[0]}, RMSE: {test_metrics[1]}, "
          f"R2: {test_metrics[2]}, Pearson: {test_metrics[3]}, Spearman: {test_metrics[4]}")

    pred_df = metadata_X_test.copy()
    pred_df["predictions"] = y_pred
    pred_df["labels"] = Y_test
    pred_df.to_csv(f"../Results/tabpfn/ablation/{experiment_name}_predictions.csv", sep="\t", index=False)
    print(f"Saved predictions for {experiment_name}.")

    del best_model
    torch.cuda.empty_cache()

    return y_pred, test_metrics


def make_prediction_plot(experiment_name, Y_test, y_pred, test_metrics):
    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(2.5, 2.5)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    plot_df = pd.DataFrame({"labels": Y_test, "predictions": y_pred})
    ax = sn.regplot(x="labels", y="predictions", data=plot_df,
                    scatter_kws={"color": "yellow", "alpha": 0.5},
                    line_kws={"color": "red"})
    title = f"TabPFN Ablation: {experiment_name}"
    ax.axes.set_title(title, fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_xlabel("Label", fontsize=10)
    ax.set_ylabel("Prediction", fontsize=10)
    ax.tick_params(labelsize=10, color="black")
    plt.text(25, 25, f'Pearson r = {test_metrics[3]}', fontsize=8)
    plt.text(25, 50, f'MAE = {test_metrics[0]}', fontsize=8)
    outfilename = f"../Results/tabpfn/ablation/{experiment_name}_prediction_plot.pdf"
    plt.savefig(outfilename, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## Run Ablation Experiment Helper

# %%
def run_ablation_experiment(experiment_name, feature_subset, X_train_all, X_test_all,
                            Y_train, Y_test, metadata_X_train, metadata_X_test, device):
    print(f"\n{'=' * 80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Number of features: {len(feature_subset)}")
    print(f"{'=' * 80}")

    X_train_sub = X_train_all[feature_subset]
    X_test_sub = X_test_all[feature_subset]

    # Hyperparameter optimization with Optuna
    study, scaler = hyperparameter_optimization(
        X_train_sub, Y_train, metadata_X_train, experiment_name,
        scaler_type="standard", device=device
    )

    # Make predictions on test set
    y_pred, test_metrics = make_predictions(
        experiment_name, X_test_sub, Y_test, metadata_X_test, scaler
    )

    # Generate prediction scatter plot
    make_prediction_plot(experiment_name, Y_test, y_pred, test_metrics)

    return test_metrics, len(feature_subset)


# Store all results
ablation_results = {}

# %% [markdown]
# ## Baseline: All Features

# %%
# Collect all patient features from ablation groups
all_patient_cols = []
for g in sorted(ablation_groups.keys()):
    all_patient_cols.extend(ablation_groups[g])

for drug_name, drug_cols in drug_feature_options.items():
    experiment_name = f"baseline_all_features_{drug_name}"
    feature_subset = list(drug_cols) + all_patient_cols

    metrics, n_feats = run_ablation_experiment(
        experiment_name, feature_subset,
        X_train_all, X_test_all, Y_train, Y_test,
        metadata_X_train, metadata_X_test, device
    )
    ablation_results[experiment_name] = {"metrics": metrics, "n_features": n_feats}

# %% [markdown]
# ## Save Results

# %%
with open("../Results/tabpfn/ablation/ablation_results_4group_baseline.pkl", "wb") as f:
    pickle.dump(ablation_results, f)
print(f"Saved {len(ablation_results)} results to ablation_results_4group_baseline.pkl")
