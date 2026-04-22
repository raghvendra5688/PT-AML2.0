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
import numpy as np
import re 

import optuna
import xgboost as xgb
from sklearn import ensemble
from sklearn import dummy
from sklearn import linear_model
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn import kernel_approximation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, GroupKFold, cross_val_score
from scipy.stats import loguniform
import scipy
import argparse
import random

import datetime
os.makedirs("logs", exist_ok=True)
os.makedirs("../Results/xgb/", exist_ok=True)
os.makedirs("../Results/xgb/optuna/", exist_ok=True)
#os.environ["OMP_NUM_THREADS"] = "192"
#os.environ["MKL_NUM_THREADS"] = "192"

from misc import save_model, load_model, regression_results, grid_search_cv, supervised_learning_steps, calculate_regression_metrics, get_CV_results
#plt.rcParams["font.family"] = "Arial"
# -

##Create a function to preprocess train and test data
def preprocess_data(train_data, test_data, label_col = "auc", scaler = None):

    #Get list of all columns from training dataframe
    all_columns = train_data.columns.tolist()

    #Get types of all columns
    column_types = train_data.dtypes.tolist()

    #To exclude columns
    exclude_cols = ["primary_key","dbgap_subject_id","dbgap_dnaseq_sample","dbgap_rnaseq_sample","inhibitor","type","status","paper_inclusion",
                    "min_conc","max_conc","intercept","beta","beta_z","beta_p", "aic","pearson_chisq","deviance","converged","ic10","ic25","ic50","ic75",
                    "ic90","all_gt_50","all_lt_50","curve_type"]

    #Put all the object or categorical columns in metadata dataframe
    metadata_cols = ["CID"]+[all_columns[i] for i in range(len(all_columns)) if str(column_types[i])=="object" or str(column_types[i])=="category"]

    #Add to metadata columns, those columns which contain NaNs
    nan_cols = [all_columns[i] for i in range(len(all_columns)) if big_train_df[all_columns[i]].isnull().any()]
    metadata_cols = metadata_cols + nan_cols + exclude_cols
    metadata_cols = list(dict.fromkeys(metadata_cols))  #Remove duplicates
    print("Metadata columns:", metadata_cols)

    #Get the all the remaining columns as feature columns and remove "auc" column
    feature_cols = list(np.setdiff1d(all_columns,metadata_cols))
    feature_cols = list(np.setdiff1d(feature_cols, label_col))
    print("Total featurs: ",len(feature_cols))

    metadata_X_train, X_train, y_train = train_data.loc[:,metadata_cols],train_data.loc[:,feature_cols],train_data[label_col].to_numpy().flatten()
    metadata_X_test, X_test, y_test = test_data.loc[:,metadata_cols],test_data.loc[:,feature_cols],test_data[label_col].to_numpy().flatten()

    #Ask scaler type is standard or minmax and apply it
    if scaler == 'standard':
        scaler = preprocessing.StandardScaler()
    elif scaler == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns = feature_cols)
    X_test = pd.DataFrame(X_test, columns = feature_cols)

    return X_train, y_train, X_test, y_test, metadata_X_train, metadata_X_test, scaler


#Build the platform for make robust GLR models
def hyperparameter_optimization(X_train,Y_train,data_type,stratify_by,scaler):

    def objective(trial):
         
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 60
        }
    
        model = xgb.XGBRegressor(**params, verbosity=1)

        #Choose stratification type: "inhibitor" or "dbgap_rnaseq_sample"
        if stratify_by == "dbgap_rnaseq_sample":
            groups = metadata_X_train["dbgap_rnaseq_sample"].to_numpy()
        elif stratify_by == "inhibitor":
            groups = metadata_X_train["inhibitor"].to_numpy()
        else:
            groups = None
        
        if groups is not None:
            cv = GroupKFold(n_splits=5, shuffle=True)
            scores = cross_val_score(estimator=model, X=X_train, y=Y_train, groups=groups, scoring="neg_mean_absolute_error",cv=cv)
            return np.mean(scores)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, Y_train, scoring="neg_mean_absolute_error", cv=cv)
            return np.mean(scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best -MAE:", study.best_value)
    print("Best params:", study.best_params)

    # Save all trials info
    print("\nAll trial results:")
    for t in study.trials:
        print(f"Trial {t.number}: value={t.value}, params={t.params}")

    study.trials_dataframe().to_csv(f"logs/optuna_trials_xgb_{data_type}_{stratify_by}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    #Save the model
    best_model = xgb.XGBRegressor(**study.best_params, n_jobs=-1, verbosity=1)
    best_model.fit(X_train, Y_train)

    # Save the best model
    save_model(best_model, "%s_models/%s_%s_%s_optuna_best.pk" %("xgb","xgb", data_type, stratify_by))   


def make_predictions(best_model, X_test, Y_test, data_type, stratify_by):
    y_pred = best_model.predict(X_test)
    test_metrics = calculate_regression_metrics(Y_test, y_pred)
    print("Test Metrics:", test_metrics)

    metadata_X_test["predictions"] = y_pred
    metadata_X_test["labels"] = Y_test
    metadata_X_test.to_csv(f"../Results/xgb/optuna/{data_type}_{stratify_by}_optuna_predictions_drug.csv", sep="\t", index=False)
    print(f"Finished writing predictions for optuna optimized XGB model with data type: {data_type} and stratified by:{stratify_by}.")
    return y_pred, test_metrics


def make_prediction_plot(metadata_X_test,Y_test,y_pred, test_metrics, data_type, stratify_by):

    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(2.5,2.5)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    ax = sn.regplot(x="labels", y="predictions", data=metadata_X_test, scatter_kws={"color": "yellow",'alpha':0.5}, 
                line_kws={"color": "red"})
    title = "XGB Prediction "+data_type+"_"+stratify_by
    ax.axes.set_title(title,fontsize=10)
    ax.set_xlim(0,300)
    ax.set_ylim(0,300)
    ax.set_xlabel("Label",fontsize=10)
    ax.set_ylabel("Prediction",fontsize=10)
    ax.tick_params(labelsize=10, color="black")
    plt.text(25, 25, 'Pearson r =' +str(test_metrics[3]), fontsize = 10)
    plt.text(25, 50, 'MAE ='+str(test_metrics[0]),fontsize=10)
    outfilename = "../Results/xgb/optuna/"+data_type+"_"+stratify_by+"_supervised_test_prediction.pdf"
    plt.savefig(outfilename, bbox_inches="tight")


#Get the top coefficients and matching column information
def make_importance_plot(best_model, X_train, data_type, stratify_by):
    val, index = np.sort(best_model.feature_importances_), np.argsort(best_model.feature_importances_)
    
    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(4,3)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    ax = fig.add_subplot(111)
    plt.bar(X_train.columns[index[-30:]],val[-30:])
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    title = "Top XGB Coefficient for "+data_type+"_"+stratify_by
    ax.axes.set_title(title,fontsize=10)
    ax.set_xlabel("Features",fontsize=10)
    ax.set_ylabel("Coefficient Value",fontsize=10)
    ax.tick_params(labelsize=8)
    outputfile = "../Results/xgb/optuna/"+data_type+"_"+stratify_by+"_Coefficients.pdf"
    plt.savefig(outputfile, bbox_inches="tight")


#Get the setting with different X_trains and X_tests
train_options = ["../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
                 "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
                 "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
                 "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl"]
test_options = ["../Data/Test_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
                "../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
                "../Data/Test_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
                "../Data/Test_Set_Var_with_Drug_MolFormer_Patient_Info.pkl"]
data_type_options = ["Only_PC_Feat_Var","Embed_Feat_Var","ChemBERTa_Feat_Var","MolFormer_Feat_Var"]
stratify_options = ["inhibitor","dbgap_rnaseq_sample","random"]

for input_option in range(0,len(data_type_options)):
    #Get the data for your choice: Embed, MFP, ChemBERTa, All
    data_type = data_type_options[input_option]
    print("Loaded training file for ",data_type)
    big_train_df = pd.read_pickle(train_options[input_option],compression="zip")
    big_test_df = pd.read_pickle(test_options[input_option],compression="zip")
    total_length = len(big_train_df.columns)

    X_train, Y_train, X_test, Y_test, metadata_X_train, metadata_X_test, scaler = preprocess_data(big_train_df, big_test_df, label_col="auc", scaler=None)

    print("Shape of training set after removing non-numeric cols and cols with NaNs")
    print(X_train.shape)
    print(X_test.shape)

    for stratify_by in stratify_options:
        print("Running for the strata: ",stratify_by)        
        
        #Perform hyperparameter optimization
        hyperparameter_optimization(X_train,Y_train,data_type,stratify_by,scaler)

        #Load best model
        model_path = "../Models/xgb_models/xgb_"+data_type+"_"+stratify_by+"_optuna_best.pk"
        best_model = load_model(model_path)

        #Make predictions
        y_pred, test_metrics = make_predictions(best_model, X_test, Y_test, data_type,stratify_by)

        #Make prediction plot
        make_prediction_plot(metadata_X_test, Y_test, y_pred, test_metrics, data_type, stratify_by)

        #Make importance plot
        make_importance_plot(best_model, X_train, data_type, stratify_by)

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 20})
bins = np.linspace(0, 300, 100)
pyplot.hist(Y_train, bins, alpha=0.5, label='Y_train')
pyplot.hist(Y_test, bins, alpha=0.5, label='Y_test')
pyplot.legend(loc='upper right')
pyplot.xlabel("AUC")
pyplot.ylabel("Frequency")
pyplot.title("AUC distribution")
pyplot.savefig("../Results/Train_Test_Y_distribution.pdf",bbox_inches="tight")
