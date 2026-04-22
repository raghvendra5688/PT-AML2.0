# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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
import gc
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, GroupKFold, cross_val_score, GroupShuffleSplit
import random
import math
import time

import optuna
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import softmax, relu, selu, elu
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import random
import math
import datetime
from dl_model_architecture import (NN_Encoder, CNN_Encoder, LSTM_Encoder, Seq2Func, init_weights, count_parameters, training, evaluation, epoch_time, evaluation_performance)


os.makedirs("logs", exist_ok=True)
os.makedirs("../Results/lstm/", exist_ok=True)
os.makedirs("../Results/lstm/optuna/", exist_ok=True)

#Load the tokenizer and tokenize SMILES using Vocab from DeepChem
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
tokenizer = SmilesTokenizer("../Data/vocab.txt")

from misc import (save_model, load_model, regression_results, grid_search_cv, calculate_regression_metrics, supervised_learning_steps, get_CV_results)
torch.cuda.empty_cache()

#Setting up the environment
SEED = 123
random.seed(SEED)
st = random.getstate()
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# +
##Global parameters

BATCH_SIZE = 4096
N_FOLDS = 5
N_EPOCHS = 1000
PATIENCE = 100
CLIP = 1.0
OUT_DIM = 1
SMILES_INPUT_DIM = tokenizer.vocab_size

# +
#Encode the smiles
max_smiles_length=150
def encode_to_indices(x):
    return(torch.tensor(tokenizer.encode(x,max_length=max_smiles_length,padding="max_length")))

##Create a function to preprocess train and test data
def preprocess_data(train_data, test_data, label_col = "auc", scaler = None):

    #Get list of all columns from training dataframe
    all_columns = train_data.columns.tolist()

    #Get types of all columns
    column_types = train_data.dtypes.tolist()

    #Get the list of training smiles
    X_train_smiles = train_data["SMILES"].tolist()
    X_train_smiles_encoded = [encode_to_indices(x) for x in X_train_smiles]
    X_test_smiles = test_data["SMILES"].tolist()
    X_test_smiles_encoded = [encode_to_indices(x) for x in X_test_smiles]

    #Convert train and test smiles to stack of tensors
    X_train_smiles_encoded = torch.stack(X_train_smiles_encoded)
    X_test_smiles_encoded = torch.stack(X_test_smiles_encoded)

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
    print("Total features: ",len(feature_cols))

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

    #Create the training and test tensor datasets
    train = TensorDataset(X_train_smiles_encoded, torch.Tensor(np.array(X_train)),torch.Tensor(np.array(y_train)))
    test = TensorDataset(X_test_smiles_encoded, torch.Tensor(np.array(X_test)),torch.Tensor(np.array(y_test)))

    #Here patient features + drug features are representated by cell_input_dim
    cell_input_dim = X_train.shape[1]

    del X_train_smiles_encoded, X_train, X_test_smiles_encoded, X_test
    gc.collect()

    return train, y_train, test, y_test, metadata_X_train, metadata_X_test, cell_input_dim, scaler


# +
#Helper Functions
def create_dataloaders(full_train_dataset, train_idx, val_idx):
    
    train_ds = Subset(full_train_dataset, train_idx)
    val_ds = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def define_model(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout, SMILES_INPUT_DIM, smiles_emb_dim, smiles_hid_dim, smiles_out_dim, n_layers, hid_dim, OUT_DIM, init_weights):

    # -------- Model --------
    cell_enc = NN_Encoder(CELL_INPUT_DIM,cell_out_dim,cell_hid_dims,dropout)

    smiles_enc = LSTM_Encoder(SMILES_INPUT_DIM,smiles_emb_dim,smiles_hid_dim,smiles_out_dim,n_layers,dropout)

    model = Seq2Func(cell_enc,smiles_enc,hid_dim,OUT_DIM,dropout,device=DEVICE).to(DEVICE)

    model.apply(init_weights)

    return model

def train_with_val_model(model, train_loader, val_loader, optimizer, criterion, trial, fold):
    
    best_val = float("inf")
    patience_ctr = 0

    # -------- Training loop --------
    for epoch in range(N_EPOCHS):
        if (epoch + 1) % 100 == 0:
            print("Epoch: ",epoch)
            print("Val Loss: ",val_loss)
        
        training(model, train_loader, optimizer, criterion, CLIP, DEVICE)
        val_loss = evaluation(model, val_loader, criterion, DEVICE)

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1

        trial.report(val_loss, step=fold * N_EPOCHS + epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if patience_ctr >= PATIENCE:
            break

    del model, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_val

def final_data_preparation(metadata_X_train, stratify_by, train_dataset, test_dataset):

    #Choose stratification type: "inhibitor" or "dbgap_rnaseq_sample"
    if stratify_by == "dbgap_rnaseq_sample":
        groups = metadata_X_train["dbgap_rnaseq_sample"].to_numpy()
    elif stratify_by == "inhibitor":
        groups = metadata_X_train["inhibitor"].to_numpy()
    else:
        groups = None

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, valid_indices = next(gss.split(train_dataset, groups=groups))
        train_dataset_subset = Subset(train_dataset, train_indices)
        valid_dataset_subset = Subset(train_dataset, valid_indices)
    else:
        train_dataset_subset, valid_dataset_subset = data_utils.random_split(train_dataset, [0.8, 0.2], generator = torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset_subset, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset_subset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    return train_loader, valid_loader, test_loader

## Perform model training
def final_model_training(model, train_loader, valid_loader, optimizer, criterion, data_type, stratify_by):
    print(f"Performing final LSTM model training for {data_type} using {stratify_by}")
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('inf')
    patience_ctr = 0
    outputfile_model = f"../Models/lstm_models/lstm_optuna_{data_type}_{stratify_by}.pt"
    for epoch in range(N_EPOCHS):
        if (patience_ctr<PATIENCE):
            print("Counter Id: ",str(patience_ctr))
            start_time = time.time()
            train_loss = training(model, train_loader, optimizer, criterion, CLIP, DEVICE)
            valid_loss = evaluation(model, valid_loader, criterion, DEVICE)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            if valid_loss < best_valid_loss:
                patience_ctr = 0
                print("Current Val. Loss: %.3f better than prev Val. Loss: %.3f " %(valid_loss,best_valid_loss))
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), outputfile_model)
            else:
                patience_ctr += 1
        else:
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
            break;

    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(outputfile_model))
    else:
        model.load_state_dict(torch.load(outputfile_model,map_location=torch.device('cpu')))

    valid_loss = evaluation(model, valid_loader, criterion, DEVICE)
    print(f'| Best Valid Loss: {valid_loss:.3f}')

    fout_filename = "../Models/lstm_models/lstm_"+data_type+"_"+stratify_by+"_loss_plot.csv"
    fout=open(fout_filename,"w")
    for j in range(len(train_loss_list)):
        outputstring = str(train_loss_list[j])+","+str(valid_loss_list[j])+"\n"
        fout.write(outputstring)
    fout.close()

    del train_loader, valid_loader
    return model

# -

# =============================
# Optuna Objective (5-fold CV)
# =============================
def objective(trial, full_train_dataset, metadata_X_train, stratify_by, CELL_INPUT_DIM, SMILES_INPUT_DIM):

    # -------- Hyperparameters --------
    cell_out_dim = trial.suggest_int("cell_out_dim", 128, 512, step=128)
    cell_hid_dims = [
        trial.suggest_int("cell_hid_1", 256, 2048, step=256),
        trial.suggest_int("cell_hid_2", 256, 1024, step=256),
    ]

    smiles_emb_dim = trial.suggest_int("smiles_emb_dim", 64, 256, step=64)
    smiles_hid_dim = trial.suggest_int("smiles_hid_dim", 128, 512, step=128)
    smiles_out_dim = trial.suggest_int("smiles_out_dim", 128, 512, step=128)
    n_layers = trial.suggest_int("n_layers", 2, 8, step=2)

    hid_dim = trial.suggest_int("hid_dim", 128, 512, step=128)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # -------- Cross-validation --------
    #Choose stratification type: "inhibitor" or "dbgap_rnaseq_sample"
    if stratify_by == "dbgap_rnaseq_sample":
        groups = metadata_X_train["dbgap_rnaseq_sample"].to_numpy()
    elif stratify_by == "inhibitor":
        groups = metadata_X_train["inhibitor"].to_numpy()
    else:
        groups = None
    
    if groups is None:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    else:
        kf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    fold_losses = []

    print("Running a new trial")
    if groups is not None:
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_dataset, groups=groups)):

            #Build the train and validation loaders
            train_loader, val_loader = create_dataloaders(full_train_dataset, train_idx, val_idx)

            model = define_model(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout, SMILES_INPUT_DIM, smiles_emb_dim,
                                smiles_hid_dim, smiles_out_dim, n_layers, hid_dim, OUT_DIM, init_weights)
            
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            criterion = nn.L1Loss().to(DEVICE)

            best_val = train_with_val_model(model, train_loader, val_loader, optimizer, criterion, trial, fold)
                    
            fold_losses.append(best_val)

            del model
            torch.cuda.empty_cache()
    
    elif groups is None:

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_dataset)):
            
            #Build the train and validation loaders
            train_loader, val_loader = create_dataloaders(full_train_dataset, train_idx, val_idx)

            model = define_model(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout, SMILES_INPUT_DIM, smiles_emb_dim,
                                smiles_hid_dim, smiles_out_dim, n_layers, hid_dim, OUT_DIM, init_weights)
            
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            criterion = nn.L1Loss().to(DEVICE)

            best_val = train_with_val_model(model, train_loader, val_loader, optimizer, criterion, trial, fold)
                    
            fold_losses.append(best_val)

            del model
            torch.cuda.empty_cache()

    return float(np.mean(fold_losses))


# =============================
# Run Optuna + Final Training
# =============================
def run_pipeline(train_dataset, test_dataset, metadata_X_train, stratify_by, CELL_INPUT_DIM, SMILES_INPUT_DIM, data_type, scaler):

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(
        lambda t: objective(t, train_dataset, metadata_X_train, stratify_by, CELL_INPUT_DIM, SMILES_INPUT_DIM),
        n_trials=20,
    )

    print("Best CV Loss:", study.best_value)
    print("Best Params:", study.best_params)

    # Save all trials info
    print("\nAll trial results:")
    for t in study.trials:
        print(f"Trial {t.number}: value={t.value}, params={t.params}")

    study.trials_dataframe().to_csv(f"logs/optuna_trials_lstm_{data_type}_{stratify_by}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


    # =============================
    # Train final model on full train divided into train and validation
    # =============================
    p = study.best_params

    #Define model
    cell_enc = NN_Encoder(CELL_INPUT_DIM, p["cell_out_dim"], [p["cell_hid_1"], p["cell_hid_2"]], p["dropout"],)
    smiles_enc = LSTM_Encoder(SMILES_INPUT_DIM, p["smiles_emb_dim"], p["smiles_hid_dim"], p["smiles_out_dim"], p["n_layers"], p["dropout"],)
    model = Seq2Func(cell_enc, smiles_enc, p["hid_dim"], OUT_DIM, p["dropout"], device=DEVICE,).to(DEVICE)
    model.apply(init_weights)

    #Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(),lr=p["lr"],weight_decay=p["weight_decay"],)
    criterion = nn.L1Loss().to(DEVICE)

    #Define training, validation and test sets
    train_loader, valid_loader, test_loader = final_data_preparation(metadata_X_train, stratify_by, train_dataset, test_dataset)

    #Train the model
    model = final_model_training(model, train_loader, valid_loader, optimizer, criterion, data_type, stratify_by)
    
    #Evaluate the model
    test_loss = evaluation(model, test_loader, criterion, DEVICE)
    print(f"Final Test Loss: {test_loss:.4f}")

    #Get predictions
    output = evaluation_performance(model, test_loader, criterion, DEVICE)

    ##Save the scaler 
    save_model(scaler, "%s_models/%s_%s_%s_scaling_gs.pk" % ("lstm","lstm",data_type,stratify_by))

    del model, train_dataset, test_dataset
    torch.cuda.empty_cache()

    #Return output list and label list 
    return output[0], output[1]    


def make_predictions(y_pred, Y_test, data_type, stratify_by):
    test_metrics = calculate_regression_metrics(Y_test, y_pred)
    print("Test Metrics:", test_metrics)

    metadata_X_test["predictions"] = y_pred
    metadata_X_test["labels"] = Y_test
    metadata_X_test.to_csv(f"../Results/lstm/optuna/{data_type}_{stratify_by}_optuna_predictions_drug.csv", sep="\t", index=False)
    print(f"Finished writing predictions for optuna optimized LSTM model with data type: {data_type} and stratified by:{stratify_by}.")
    return test_metrics


# +

def make_prediction_plot(metadata_X_test,Y_test, y_pred, test_metrics, data_type, stratify_by):

    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(2.5,2.5)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    ax = sn.regplot(x="labels", y="predictions", data=metadata_X_test, scatter_kws={"color": "yellow",'alpha':0.5}, 
                line_kws={"color": "red"})
    title = "LSTM Prediction "+data_type+"_"+stratify_by
    ax.axes.set_title(title,fontsize=10)
    ax.set_xlim(0,300)
    ax.set_ylim(0,300)
    ax.set_xlabel("Label",fontsize=10)
    ax.set_ylabel("Prediction",fontsize=10)
    ax.tick_params(labelsize=10, color="black")
    plt.text(25, 25, 'Pearson r =' +str(test_metrics[3]), fontsize = 10)
    plt.text(25, 50, 'MAE ='+str(test_metrics[0]),fontsize=10)
    outfilename = "../Results/lstm/optuna/"+data_type+"_"+stratify_by+"_supervised_test_prediction.pdf"
    plt.savefig(outfilename, bbox_inches="tight")


# +
#Get the setting with different X_trains and X_tests
train_options = ["../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
                "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
                "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
                "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl"]
test_options = ["../Data/Test_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
                "../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl"
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

    train, Y_train, test, Y_test, metadata_X_train, metadata_X_test, CELL_INPUT_DIM, scaler = preprocess_data(big_train_df, big_test_df, label_col="auc", scaler="standard")

    for stratify_by in stratify_options:
        print("Running for the strata: ",stratify_by)        
        
        #Perform hyperparameter optimization
        y_pred, _ = run_pipeline(train,test, metadata_X_train, stratify_by, CELL_INPUT_DIM, SMILES_INPUT_DIM, data_type, scaler)

        #Make predictions
        test_metrics = make_predictions(y_pred, Y_test, data_type, stratify_by)

        #Make prediction plot
        make_prediction_plot(metadata_X_test, Y_test, y_pred, test_metrics, data_type, stratify_by)

# -
