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
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import optuna

from catboost import CatBoostRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
import scipy
import argparse
import datetime

os.makedirs("logs", exist_ok=True)


def neg_mean_absolute_error(y_true, y_pred):
    """
    Compute negative mean absolute error (neg_mean_absolute_error).
    This is used for optimization where higher is better (maximization).
    Equivalent to sklearn's 'neg_mean_absolute_error' scoring.

    Args:
        y_true: Ground truth target values
        y_pred: Predicted target values

    Returns:
        Negative MAE score (higher is better)
    """
    return -mean_absolute_error(y_true, y_pred)


os.makedirs("../Results/catboost/", exist_ok=True)
os.makedirs("../Results/catboost/optuna/", exist_ok=True)

from misc import save_model, load_model, calculate_regression_metrics
# -


def preprocess_data(train_data, test_data, label_col="auc", scaler=None):
    """
    Preprocess training and test data by removing non-numeric and NaN columns.

    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        label_col: Name of the label column
        scaler: Type of scaler ('standard', 'minmax', or None)

    Returns:
        X_train, y_train, X_test, y_test, metadata_X_train, metadata_X_test, scaler
    """
    # Get list of all columns from training dataframe
    all_columns = train_data.columns.tolist()

    # Get types of all columns
    column_types = train_data.dtypes.tolist()

    # Columns to exclude from features
    exclude_cols = [
        "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample",
        "inhibitor", "type", "status", "paper_inclusion", "min_conc", "max_conc",
        "intercept", "beta", "beta_z", "beta_p", "aic", "pearson_chisq", "deviance",
        "converged", "ic10", "ic25", "ic50", "ic75", "ic90", "all_gt_50", "all_lt_50",
        "curve_type"
    ]

    # Put all the object or categorical columns in metadata dataframe
    metadata_cols = ["CID"] + [
        all_columns[i] for i in range(len(all_columns))
        if str(column_types[i]) == "object" or str(column_types[i]) == "category"
    ]

    # Add columns which contain NaNs to metadata
    nan_cols = [
        all_columns[i] for i in range(len(all_columns))
        if train_data[all_columns[i]].isnull().any()
    ]
    metadata_cols = metadata_cols + nan_cols + exclude_cols
    metadata_cols = list(dict.fromkeys(metadata_cols))  # Remove duplicates
    print("Metadata columns:", len(metadata_cols))

    # Get all remaining columns as feature columns and remove label column
    feature_cols = list(np.setdiff1d(all_columns, metadata_cols))
    feature_cols = list(np.setdiff1d(feature_cols, label_col))
    print("Total features:", len(feature_cols))

    metadata_X_train = train_data.loc[:, metadata_cols]
    X_train = train_data.loc[:, feature_cols]
    y_train = train_data[label_col].to_numpy().flatten()

    metadata_X_test = test_data.loc[:, metadata_cols]
    X_test = test_data.loc[:, feature_cols]
    y_test = test_data[label_col].to_numpy().flatten()

    # Apply scaler if specified
    if scaler == 'standard':
        scaler_obj = preprocessing.StandardScaler()
    elif scaler == 'minmax':
        scaler_obj = preprocessing.MinMaxScaler()
    else:
        scaler_obj = None

    if scaler_obj:
        X_train = scaler_obj.fit_transform(X_train)
        X_test = scaler_obj.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=feature_cols)
        X_test = pd.DataFrame(X_test, columns=feature_cols)

    return X_train, y_train, X_test, y_test, metadata_X_train, metadata_X_test, scaler_obj


def hyperparameter_optimization(X_train, Y_train, metadata_X_train, data_type, stratify_by, scaler, n_trials=50):
    """
    Perform hyperparameter optimization for CatBoost using Optuna.
    Optimizes all possible CatBoost parameters for best performance.

    Args:
        X_train: Training features
        Y_train: Training labels
        metadata_X_train: Metadata for training set (contains stratification columns)
        data_type: Type of data being used
        stratify_by: Stratification method ('random', 'dbgap_rnaseq_sample', 'inhibitor')
        scaler: Scaler object used for preprocessing
        n_trials: Number of Optuna trials

    Returns:
        best_model: Trained CatBoost model with best hyperparameters
    """

    def objective(trial):
        # Standard CatBoost hyperparameters for optimization
        params = {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'random_seed': 42,
            'verbose': False,
            'thread_count': 60,
            'allow_writing_files': False,

            # Number of boosting iterations
            'iterations': trial.suggest_int('iterations', 100, 1500),

            # Learning rate
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),

            # Tree depth
            'depth': trial.suggest_int('depth', 4, 10),

            # Regularization
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 100.0, log=True),

            # Sampling parameters
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),

            # Fixed standard parameters
            'bootstrap_type': 'Bernoulli',
            'grow_policy': 'SymmetricTree',
            'boosting_type': 'Plain',
        }

        model = CatBoostRegressor(**params)

        # Choose stratification type
        if stratify_by == "dbgap_rnaseq_sample":
            groups = metadata_X_train["dbgap_rnaseq_sample"].to_numpy()
            cv = GroupKFold(n_splits=5)
            scores = []
            for train_idx, valid_idx in cv.split(X_train, Y_train, groups):
                model.fit(
                    X_train.iloc[train_idx], Y_train[train_idx],
                    eval_set=(X_train.iloc[valid_idx], Y_train[valid_idx]),
                    verbose=False
                )
                y_pred = model.predict(X_train.iloc[valid_idx])
                scores.append(neg_mean_absolute_error(Y_train[valid_idx], y_pred))
            return np.mean(scores)

        elif stratify_by == "inhibitor":
            groups = metadata_X_train["inhibitor"].to_numpy()
            cv = GroupKFold(n_splits=5)
            scores = []
            for train_idx, valid_idx in cv.split(X_train, Y_train, groups):
                model.fit(
                    X_train.iloc[train_idx], Y_train[train_idx],
                    eval_set=(X_train.iloc[valid_idx], Y_train[valid_idx]),
                    verbose=False
                )
                y_pred = model.predict(X_train.iloc[valid_idx])
                scores.append(neg_mean_absolute_error(Y_train[valid_idx], y_pred))
            return np.mean(scores)

        else:  # "random" - no stratification
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, valid_idx in cv.split(X_train, Y_train):
                model.fit(
                    X_train.iloc[train_idx], Y_train[train_idx],
                    eval_set=(X_train.iloc[valid_idx], Y_train[valid_idx]),
                    verbose=False
                )
                y_pred = model.predict(X_train.iloc[valid_idx])
                scores.append(neg_mean_absolute_error(Y_train[valid_idx], y_pred))
            return np.mean(scores)

    # Create and run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest -MAE: {study.best_value}")
    print(f"Best params: {study.best_params}")

    # Save all trials info
    print("\nAll trial results:")
    for t in study.trials:
        print(f"Trial {t.number}: value={t.value}, params={t.params}")

    # Save trials to CSV
    study.trials_dataframe().to_csv(
        f"logs/optuna_trials_catboost_{data_type}_{stratify_by}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False
    )

    # Build final model with best parameters
    best_params = study.best_params.copy()
    best_params['loss_function'] = 'MAE'
    best_params['eval_metric'] = 'MAE'
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    best_params['thread_count'] = 60
    best_params['allow_writing_files'] = False

    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, Y_train)

    # Save the best model
    save_model(best_model, f"catboost_models/catboost_{data_type}_{stratify_by}_optuna_best.pk")

    return best_model


def make_predictions(best_model, X_test, Y_test, metadata_X_test, data_type, stratify_by):
    """
    Make predictions on test set and save results.

    Args:
        best_model: Trained CatBoost model
        X_test: Test features
        Y_test: Test labels
        metadata_X_test: Metadata for test set
        data_type: Type of data being used
        stratify_by: Stratification method used during training

    Returns:
        y_pred: Predictions
        test_metrics: Tuple of regression metrics (MAE, RMSE, R2, Pearson, Spearman)
    """
    y_pred = best_model.predict(X_test)
    test_metrics = calculate_regression_metrics(Y_test, y_pred)
    print(f"Test Metrics - MAE: {test_metrics[0]}, RMSE: {test_metrics[1]}, "
          f"R2: {test_metrics[2]}, Pearson: {test_metrics[3]}, Spearman: {test_metrics[4]}")

    # Save predictions with metadata
    metadata_X_test = metadata_X_test.copy()
    metadata_X_test["predictions"] = y_pred
    metadata_X_test["labels"] = Y_test
    metadata_X_test.to_csv(
        f"../Results/catboost/optuna/{data_type}_{stratify_by}_optuna_predictions.csv",
        sep="\t", index=False
    )
    print(f"Finished writing predictions for CatBoost model with data type: {data_type} "
          f"and stratified by: {stratify_by}")

    return y_pred, test_metrics


def make_prediction_plot(metadata_X_test, Y_test, y_pred, test_metrics, data_type, stratify_by):
    """
    Create and save a scatter plot of predictions vs actual values.

    Args:
        metadata_X_test: Metadata with predictions and labels
        Y_test: Test labels
        y_pred: Predictions
        test_metrics: Tuple of regression metrics
        data_type: Type of data being used
        stratify_by: Stratification method used during training
    """
    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(2.5, 2.5)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    ax = sn.regplot(
        x="labels", y="predictions", data=metadata_X_test,
        scatter_kws={"color": "green", 'alpha': 0.5},
        line_kws={"color": "red"}
    )
    title = f"CatBoost Prediction {data_type}_{stratify_by}"
    ax.axes.set_title(title, fontsize=10)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_xlabel("Label", fontsize=10)
    ax.set_ylabel("Prediction", fontsize=10)
    ax.tick_params(labelsize=10, color="black")
    plt.text(25, 25, f'Pearson r = {test_metrics[3]}', fontsize=10)
    plt.text(25, 50, f'MAE = {test_metrics[0]}', fontsize=10)

    outfilename = f"../Results/catboost/optuna/{data_type}_{stratify_by}_supervised_test_prediction.pdf"
    plt.savefig(outfilename, bbox_inches="tight")
    plt.close()


def make_importance_plot(best_model, X_train, data_type, stratify_by, top_n=30):
    """
    Create and save a feature importance plot.

    Args:
        best_model: Trained CatBoost model
        X_train: Training features (for column names)
        data_type: Type of data being used
        stratify_by: Stratification method used during training
        top_n: Number of top features to display
    """
    val = np.sort(best_model.feature_importances_)
    index = np.argsort(best_model.feature_importances_)

    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(4, 3)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    ax = fig.add_subplot(111)
    plt.bar(X_train.columns[index[-top_n:]], val[-top_n:])
    plt.xticks(rotation=90)

    title = f"Top CatBoost Feature Importance {data_type}_{stratify_by}"
    ax.axes.set_title(title, fontsize=10)
    ax.set_xlabel("Features", fontsize=10)
    ax.set_ylabel("Importance Value", fontsize=10)
    ax.tick_params(labelsize=8)

    outputfile = f"../Results/catboost/optuna/{data_type}_{stratify_by}_Coefficients.pdf"
    plt.savefig(outputfile, bbox_inches="tight")
    plt.close()


def save_metrics_summary(all_metrics, output_path="../Results/catboost/optuna/catboost_metrics_summary.csv"):
    """
    Save a summary of all metrics to a CSV file.

    Args:
        all_metrics: List of dictionaries containing metrics for each configuration
        output_path: Path to save the summary CSV
    """
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_path, index=False)
    print(f"Saved metrics summary to {output_path}")


# Dataset and stratification options
train_options = [
    "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl"
]

test_options = [
    "../Data/Test_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    "../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    "../Data/Test_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    "../Data/Test_Set_Var_with_Drug_MolFormer_Patient_Info.pkl"
]

data_type_options = ["Only_PC_Feat_Var", "Embed_Feat_Var", "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"]
stratify_options = ["random", "dbgap_rnaseq_sample", "inhibitor"]

# +
# Main execution block (similar to rf_model.py)
for input_option in range(0, len(data_type_options)):
    # Get the data for your choice: Only_PC, Embed, ChemBERTa, MolFormer
    data_type = data_type_options[input_option]
    print("Loaded training file for ", data_type)
    big_train_df = pd.read_pickle(train_options[input_option], compression="zip")
    big_test_df = pd.read_pickle(test_options[input_option], compression="zip")
    total_length = len(big_train_df.columns)

    X_train, Y_train, X_test, Y_test, metadata_X_train, metadata_X_test, scaler = preprocess_data(
        big_train_df, big_test_df, label_col="auc", scaler=None
    )

    print("Shape of training set after removing non-numeric cols and cols with NaNs")
    print(X_train.shape)
    print(X_test.shape)

    for stratify_by in stratify_options:
        print("Running for the strata: ", stratify_by)

        # Perform hyperparameter optimization
        hyperparameter_optimization(X_train, Y_train, metadata_X_train, data_type, stratify_by, scaler, n_trials=30)

        # Load best model
        model_path = "../Models/catboost_models/catboost_" + data_type + "_" + stratify_by + "_optuna_best.pk"
        best_model = load_model(model_path)

        # Make predictions
        y_pred, test_metrics = make_predictions(best_model, X_test, Y_test, metadata_X_test.copy(), data_type, stratify_by)

        # Make prediction plot
        metadata_copy = metadata_X_test.copy()
        metadata_copy["predictions"] = y_pred
        metadata_copy["labels"] = Y_test
        make_prediction_plot(metadata_copy, Y_test, y_pred, test_metrics, data_type, stratify_by)

        # Make importance plot
        make_importance_plot(best_model, X_train, data_type, stratify_by)
# -

# Visualization of data distribution (optional - run separately)
def plot_data_distribution(Y_train, Y_test, output_path="../Results/catboost/Train_Test_Y_distribution.pdf"):
    """
    Plot the distribution of training and test labels.

    Args:
        Y_train: Training labels
        Y_test: Test labels
        output_path: Path to save the plot
    """
    from matplotlib import pyplot
    pyplot.rcParams.update({'font.size': 20})
    pyplot.figure(figsize=(10, 6))

    bins = np.linspace(0, 300, 100)
    pyplot.hist(Y_train, bins, alpha=0.5, label='Y_train')
    pyplot.hist(Y_test, bins, alpha=0.5, label='Y_test')
    pyplot.legend(loc='upper right')
    pyplot.xlabel("AUC")
    pyplot.ylabel("Frequency")
    pyplot.title("AUC Distribution")
    pyplot.savefig(output_path, bbox_inches="tight")
    pyplot.close()
