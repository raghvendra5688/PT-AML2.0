# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
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
import gc
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, GroupKFold, cross_val_score, GroupShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import random
import time
import datetime
from torch.utils.data import Subset
from torch.nn.functional import relu,leaky_relu
from torch.nn import Linear
from torch.nn import BatchNorm1d
import networkx as nx
from rdkit import Chem
from torch_geometric import data as DATA
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from math import sqrt
from rdkit.Chem import AllChem
import optuna

from dl_model_architecture import (NN_Encoder, GCNNet, GCNNet_Enhanced, Seq2Func_Net, Seq2Func_Net_Enhanced,
                                     init_weights, count_parameters,
                                     training_net, evaluation_net, evaluation_net_performance, epoch_time)

from misc import (save_model, load_model, regression_results, grid_search_cv, calculate_regression_metrics,
                  supervised_learning_steps, get_CV_results)


os.makedirs("logs", exist_ok=True)
os.makedirs("../Results/gcn/", exist_ok=True)
os.makedirs("../Results/gcn/optuna/", exist_ok=True)
os.makedirs("../Models/gcn_models/", exist_ok=True)

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
cudaid = int(0)
DEVICE = torch.device("cuda:%d" % (cudaid) if torch.cuda.is_available() else "cpu")
print(DEVICE)

# +
##Global parameters

BATCH_SIZE = 65536
N_FOLDS = 5
N_EPOCHS = 500
PATIENCE = 100
CLIP = 1.0
OUT_DIM = 1
SMILES_INPUT_DIM = 78  # Atom feature dimension for GCN

# +
#Convert SMILES to graph representation
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if(mol is None):
        return None
    else:
        c_size = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append( feature / sum(feature) )

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return c_size, features, edge_index

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def get_smiles_func(smiles, cell_features, labels):
    #Create smiles graphs
    none_smiles = []
    data_list=[]
    for i in range(0,len(smiles)):
        smile = smiles[i]
        label = labels[i]
        cell_feature = cell_features[i]
        g = smile_to_graph(smile)
        if(g is None):
            print(smile)
            none_smiles.append(smile)
        else:
            c_size, features, edge_index = g[0],g[1],g[2]

            GCNData = DATA.Data(x=torch.FloatTensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([label]))
            GCNData.cell_src = torch.FloatTensor(cell_feature)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)
    return(data_list)


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
    nan_cols = [all_columns[i] for i in range(len(all_columns)) if train_data[all_columns[i]].isnull().any()]
    metadata_cols = metadata_cols + nan_cols + exclude_cols
    metadata_cols = list(dict.fromkeys(metadata_cols))  #Remove duplicates
    print("Metadata columns:", metadata_cols)

    #Get the all the remaining columns as feature columns and remove label column
    feature_cols = list(np.setdiff1d(all_columns,metadata_cols))
    feature_cols = list(np.setdiff1d(feature_cols, label_col))
    print("Total features: ",len(feature_cols))

    metadata_X_train, X_train, y_train = train_data.loc[:,metadata_cols],train_data.loc[:,feature_cols],train_data[label_col].to_numpy().flatten()
    metadata_X_test, X_test, y_test = test_data.loc[:,metadata_cols],test_data.loc[:,feature_cols],test_data[label_col].to_numpy().flatten()

    #Keep only numeric training and test set and those which have no Nans
    X_train_numerics_only = X_train.select_dtypes(include=np.number)
    X_test_numerics_only = X_test[X_train_numerics_only.columns]

    nan_cols = [i for i in X_train_numerics_only.columns if X_train_numerics_only[i].isnull().any()]
    X_train = X_train_numerics_only.drop(nan_cols,axis=1)
    X_test = X_test_numerics_only.drop(nan_cols,axis=1)

    print("Shape of training set after preprocessing:", X_train.shape)
    print("Shape of test set after preprocessing:", X_test.shape)

    #Ask scaler type is standard or minmax and apply it
    if scaler == 'standard':
        scaler = preprocessing.StandardScaler()
    elif scaler == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    #Get the list of training smiles
    X_train_smiles = metadata_X_train["SMILES"].tolist()
    X_train_smiles_graphs = get_smiles_func(X_train_smiles, X_train_scaled, y_train)
    X_test_smiles = metadata_X_test["SMILES"].tolist()
    X_test_smiles_graphs = get_smiles_func(X_test_smiles, X_test_scaled, y_test)

    #Here patient features + drug features are representated by cell_input_dim
    cell_input_dim = X_train_scaled.shape[1]

    del X_train_scaled, X_test_scaled
    gc.collect()

    return X_train_smiles_graphs, y_train, X_test_smiles_graphs, y_test, metadata_X_train, metadata_X_test, cell_input_dim, scaler


# +
#Helper Functions
def create_dataloaders(full_train_dataset, train_idx, val_idx):

    train_ds = [full_train_dataset[i] for i in train_idx]
    val_ds = [full_train_dataset[i] for i in val_idx]

    train_loader = PyGDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def define_model(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout, SMILES_INPUT_DIM,
                 gcn_hid_dim, smiles_out_dim, fusion_hid_dim, OUT_DIM, init_weights):

    # -------- Model --------
    cell_enc = NN_Encoder(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout)

    smiles_enc = GCNNet(SMILES_INPUT_DIM, gcn_hid_dim, smiles_out_dim, dropout)

    model = Seq2Func_Net(cell_enc, smiles_enc, fusion_hid_dim, OUT_DIM, dropout, device=DEVICE).to(DEVICE)

    model.apply(init_weights)

    return model

def train_with_val_model(model, train_loader, val_loader, optimizer, criterion, N_dim, trial, fold):

    best_val = float("inf")
    patience_ctr = 0

    # -------- Training loop --------
    for epoch in range(N_EPOCHS):
        if (epoch + 1) % 100 == 0:
            print("Epoch: ",epoch)
            print("Val Loss: ",val_loss)

        training_net(model, train_loader, optimizer, criterion, N_dim, CLIP, DEVICE)
        val_loss = evaluation_net(model, val_loader, criterion, N_dim, DEVICE)

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
        # Use range as indices for PyG dataset
        train_indices, valid_indices = next(gss.split(range(len(train_dataset)), groups=groups))
        train_dataset_subset = [train_dataset[i] for i in train_indices]
        valid_dataset_subset = [train_dataset[i] for i in valid_indices]
    else:
        # Random split for PyG dataset
        train_size = int(0.8 * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        indices = list(range(len(train_dataset)))
        random.Random(42).shuffle(indices)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        train_dataset_subset = [train_dataset[i] for i in train_indices]
        valid_dataset_subset = [train_dataset[i] for i in valid_indices]

    train_loader = PyGDataLoader(train_dataset_subset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = PyGDataLoader(valid_dataset_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader

## Perform model training
def final_model_training(model, train_loader, valid_loader, optimizer, criterion, N_dim, data_type, stratify_by):
    print(f"Performing final GCN model training for {data_type} using {stratify_by}")
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('inf')
    patience_ctr = 0
    outputfile_model = f"../Models/gcn_models/gcn_optuna_{data_type}_{stratify_by}.pt"
    for epoch in range(N_EPOCHS):
        if (patience_ctr<PATIENCE):
            print("Counter Id: ",str(patience_ctr))
            start_time = time.time()
            train_loss = training_net(model, train_loader, optimizer, criterion, N_dim, CLIP, DEVICE)
            valid_loss = evaluation_net(model, valid_loader, criterion, N_dim, DEVICE)
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

    valid_loss = evaluation_net(model, valid_loader, criterion, N_dim, DEVICE)
    print(f'| Best Valid Loss: {valid_loss:.3f}')

    fout_filename = "../Models/gcn_models/gcn_"+data_type+"_"+stratify_by+"_loss_plot.csv"
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
    # Cell encoder parameters (optimized for 80GB GPU)
    cell_out_dim = trial.suggest_int("cell_out_dim", 256, 512, step=128)
    n_cell_layers = trial.suggest_int("n_cell_layers", 2, 4)
    cell_hid_dims = [
        trial.suggest_int(f"cell_hid_{i}", 256, 512, step=128)
        for i in range(n_cell_layers)
    ]

    # GCN parameters (optimized for 80GB GPU)
    gcn_hid_dim = trial.suggest_int("gcn_hid_dim", 128, 512, step=128)
    smiles_out_dim = trial.suggest_int("smiles_out_dim", 128, 512, step=128)

    # Fusion network parameters (optimized for 80GB GPU)
    fusion_hid_dim = trial.suggest_int("fusion_hid_dim", 256, 512, step=128)

    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)

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
        kf = GroupKFold(n_splits=N_FOLDS)

    fold_losses = []

    print("Running a new trial")
    if groups is not None:
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_train_dataset)), groups=groups)):

            #Build the train and validation loaders
            train_loader, val_loader = create_dataloaders(full_train_dataset, train_idx, val_idx)

            model = define_model(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout, SMILES_INPUT_DIM,
                                gcn_hid_dim, smiles_out_dim, fusion_hid_dim, OUT_DIM, init_weights)

            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            criterion = nn.L1Loss().to(DEVICE)

            best_val = train_with_val_model(model, train_loader, val_loader, optimizer, criterion,
                                            CELL_INPUT_DIM, trial, fold)

            fold_losses.append(best_val)


            # Comprehensive GPU memory cleanup after each fold
            del model, optimizer, criterion, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
    elif groups is None:

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_train_dataset)))):

            #Build the train and validation loaders
            train_loader, val_loader = create_dataloaders(full_train_dataset, train_idx, val_idx)

            model = define_model(CELL_INPUT_DIM, cell_out_dim, cell_hid_dims, dropout, SMILES_INPUT_DIM,
                                gcn_hid_dim, smiles_out_dim, fusion_hid_dim, OUT_DIM, init_weights)

            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            criterion = nn.L1Loss().to(DEVICE)

            best_val = train_with_val_model(model, train_loader, val_loader, optimizer, criterion,
                                            CELL_INPUT_DIM, trial, fold)

            fold_losses.append(best_val)


            # Comprehensive GPU memory cleanup after each fold
            del model, optimizer, criterion, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
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

    study.trials_dataframe().to_csv(f"logs/optuna_trials_gcn_{data_type}_{stratify_by}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)


    # =============================
    # Train final model on full train divided into train and validation
    # =============================
    p = study.best_params

    #Define model
    # Reconstruct cell hidden dimensions from best params
    n_cell_layers = p["n_cell_layers"]
    cell_hid_dims = [p[f"cell_hid_{i}"] for i in range(n_cell_layers)]

    cell_enc = NN_Encoder(CELL_INPUT_DIM, p["cell_out_dim"], cell_hid_dims, p["dropout"])
    smiles_enc = GCNNet(SMILES_INPUT_DIM, p["gcn_hid_dim"], p["smiles_out_dim"], p["dropout"])
    model = Seq2Func_Net(cell_enc, smiles_enc, p["fusion_hid_dim"], OUT_DIM, p["dropout"], device=DEVICE).to(DEVICE)
    model.apply(init_weights)

    #Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(),lr=p["lr"],weight_decay=p["weight_decay"],)
    criterion = nn.L1Loss().to(DEVICE)

    #Define training, validation and test sets
    train_loader, valid_loader, test_loader = final_data_preparation(metadata_X_train, stratify_by, train_dataset, test_dataset)

    #Train the model
    model = final_model_training(model, train_loader, valid_loader, optimizer, criterion, CELL_INPUT_DIM, data_type, stratify_by)

    #Evaluate the model
    test_loss = evaluation_net(model, test_loader, criterion, CELL_INPUT_DIM, DEVICE)
    print(f"Final Test Loss: {test_loss:.4f}")

    #Get predictions
    output = evaluation_net_performance(model, test_loader, criterion, CELL_INPUT_DIM, DEVICE)

    ##Save the scaler
    save_model(scaler, "%s_models/%s_%s_%s_scaling_gs.pk" % ("gcn","gcn",data_type,stratify_by))


    #Return output list and label list

    # Comprehensive GPU memory cleanup
    del model, optimizer, criterion, train_loader, valid_loader, test_loader, train_dataset, test_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    return output[0], output[1]


def make_predictions(y_pred, Y_test, data_type, stratify_by):
    test_metrics = calculate_regression_metrics(Y_test, y_pred)
    print("Test Metrics:", test_metrics)

    metadata_X_test["predictions"] = y_pred
    metadata_X_test["labels"] = Y_test
    metadata_X_test.to_csv(f"../Results/gcn/optuna/{data_type}_{stratify_by}_optuna_predictions_drug.csv", sep="\t", index=False)
    print(f"Finished writing predictions for optuna optimized GCN model with data type: {data_type} and stratified by:{stratify_by}.")
    return test_metrics


# +

def make_prediction_plot(metadata_X_test, Y_test, y_pred, test_metrics, data_type, stratify_by):

    fig = plt.figure()
    plt.style.use('classic')
    fig.set_size_inches(2.5,2.5)
    fig.set_dpi(300)
    fig.set_facecolor("white")

    ax = sn.regplot(x="labels", y="predictions", data=metadata_X_test, scatter_kws={"color": "yellow",'alpha':0.5},
                line_kws={"color": "red"})
    title = "GCN Prediction "+data_type+"_"+stratify_by
    ax.axes.set_title(title,fontsize=10)
    ax.set_xlim(0,300)
    ax.set_ylim(0,300)
    ax.set_xlabel("Label",fontsize=10)
    ax.set_ylabel("Prediction",fontsize=10)
    ax.tick_params(labelsize=10, color="black")
    plt.text(25, 25, 'Pearson r =' +str(test_metrics[3]), fontsize = 10)
    plt.text(25, 50, 'MAE ='+str(test_metrics[0]),fontsize=10)
    outfilename = "../Results/gcn/optuna/"+data_type+"_"+stratify_by+"_supervised_test_prediction.pdf"
    plt.savefig(outfilename, bbox_inches="tight")


# +
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
    #Get the data for your choice: Embed, ChemBERTa, MolFormer
    data_type = data_type_options[input_option]
    print("Loaded training file for ",data_type)
    big_train_df = pd.read_pickle(train_options[input_option],compression="zip")
    big_test_df = pd.read_pickle(test_options[input_option],compression="zip")
    total_length = len(big_train_df.columns)

    train, Y_train, test, Y_test, metadata_X_train, metadata_X_test, CELL_INPUT_DIM, scaler = preprocess_data(big_train_df, big_test_df, label_col="auc", scaler="standard")

    for stratify_by in stratify_options:
        print("Running for the strata: ",stratify_by)

        #Perform hyperparameter optimization
        y_pred, _ = run_pipeline(train, test, metadata_X_train, stratify_by, CELL_INPUT_DIM, SMILES_INPUT_DIM, data_type, scaler)

        #Make predictions
        test_metrics = make_predictions(y_pred, Y_test, data_type, stratify_by)

        #Make prediction plot
        make_prediction_plot(metadata_X_test, Y_test, y_pred, test_metrics, data_type, stratify_by)

        # Clear GPU memory between stratification runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    # Clear GPU memory and large dataframes between data type iterations
    del big_train_df, big_test_df, train, test, Y_train, Y_test, metadata_X_train, metadata_X_test, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
# -
