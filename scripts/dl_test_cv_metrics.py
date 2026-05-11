"""
dl_test_cv_metrics.py — Test-set metrics for DL models using 5-fold CV.

For each DL model type (CNN, LSTM, GCN, GAT) the pretrained optuna model is
used to predict on the FULL external test set.  The test set is then split
into 5 patient-based folds (GroupKFold on dbgap_rnaseq_sample) and metrics
are computed for each fold's subset, giving a per-fold breakdown plus
Mean ± Std summary.

Only dbgap_rnaseq_sample stratification and Only_PC_Feat_Var data type are
evaluated.

Model / scaler paths (single pretrained model per combination):
  ../Models/{model}_models/{model}_optuna_Only_PC_Feat_Var_dbgap_rnaseq_sample.pt
  ../Models/{model}_models/{model}_Only_PC_Feat_Var_dbgap_rnaseq_sample_scaling_gs.pk

Output per model type:
  ../Results/
      {model}_Only_PC_Feat_Var_dbgap_rnaseq_sample_test_cv_metrics.csv

Usage:
    python dl_test_cv_metrics.py
"""

import gc
import math
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misc import calculate_regression_metrics

DATA_TYPE = "Only_PC_Feat_Var"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SEED           = 123
N_CV_FOLDS     = 5
OUT_DIM        = 1
STRATIFY_BY    = "dbgap_rnaseq_sample"
VOCAB_PATH     = "../Data/vocab.txt"
MAX_SMILES_LEN = 150
GRAPH_ATOM_DIM = 78
OUTPUT_DIR     = "../Results/"

TEST_DATA_PATH = "../Data/Test_Set_Var_with_Drug_Only_PC_Patient_Info.pkl"

BATCH_SIZE_MAP = {"cnn": 16384, "lstm": 4096, "gcn": 65536, "gat": 16384}

ALL_MODEL_TYPES = ["cnn", "lstm", "gcn", "gat"]

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility & device
# ──────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    _best_gpu = max(range(torch.cuda.device_count()),
                    key=lambda i: torch.cuda.mem_get_info(i)[0])
    DEVICE   = torch.device(f"cuda:{_best_gpu}")
    _free_gb = torch.cuda.mem_get_info(_best_gpu)[0] / 1024**3
    print(f"GPU {_best_gpu}  ({_free_gb:.1f} GB free)  data={DATA_TYPE}")
else:
    DEVICE = torch.device("cpu")
    print(f"CPU  data={DATA_TYPE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
EXCLUDE_COLS = [
    "primary_key", "dbgap_subject_id", "dbgap_dnaseq_sample",
    "dbgap_rnaseq_sample", "inhibitor", "type", "status", "paper_inclusion",
    "min_conc", "max_conc", "intercept", "beta", "beta_z", "beta_p",
    "aic", "pearson_chisq", "deviance", "converged",
    "ic10", "ic25", "ic50", "ic75", "ic90",
    "all_gt_50", "all_lt_50", "curve_type",
]


def extract_features(df, label_col="auc", extra_numeric_filter=False):
    """
    Identify metadata and feature columns exactly as in each training script's
    preprocess_data().

    extra_numeric_filter=True applies the additional numeric-only / NaN-drop
    step used by gcn_model_optuna.py and gat_model_optuna.py.

    Returns
    -------
    metadata_df  : pd.DataFrame
    X_raw        : pd.DataFrame  (feature columns only)
    y            : np.ndarray
    feature_cols : list[str]
    """
    all_columns  = df.columns.tolist()
    column_types = df.dtypes.tolist()

    metadata_cols = ["CID"] + [
        all_columns[i] for i in range(len(all_columns))
        if str(column_types[i]) in ("object", "category")
    ]
    nan_cols      = [c for c in all_columns if df[c].isnull().any()]
    metadata_cols = list(dict.fromkeys(metadata_cols + nan_cols + EXCLUDE_COLS))

    feature_cols = list(np.setdiff1d(all_columns, metadata_cols + [label_col]))
    X_raw        = df[feature_cols].copy()

    if extra_numeric_filter:
        X_raw        = X_raw.select_dtypes(include=np.number)
        nan_fc       = [c for c in X_raw.columns if X_raw[c].isnull().any()]
        X_raw        = X_raw.drop(nan_fc, axis=1)
        feature_cols = X_raw.columns.tolist()

    metadata_df = df[metadata_cols].reset_index(drop=True)
    y           = df[label_col].to_numpy().flatten()
    print(f"  Feature columns : {len(feature_cols)}")
    return metadata_df, X_raw, y, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# Architecture inference from state dict  (identical to dl_cv_metrics.py)
# ──────────────────────────────────────────────────────────────────────────────

def _cell_encoder_dims(sd):
    fc_keys = sorted(
        k for k in sd
        if k.startswith("cell_encoder.fcs.") and k.endswith(".0.weight")
    )
    dims          = [sd[k].shape[0] for k in fc_keys]
    cell_hid_dims = dims[:-1]
    cell_out_dim  = dims[-1]
    return cell_out_dim, cell_hid_dims


def infer_cnn_arch(sd):
    cell_out_dim, cell_hid_dims = _cell_encoder_dims(sd)
    conv_keys = sorted(
        k for k in sd
        if k.startswith("smiles_encoder.convs.") and k.endswith(".weight")
    )
    n_filters      = sd[conv_keys[0]].shape[0]
    smiles_emb_dim = sd[conv_keys[0]].shape[1]
    filter_sizes   = [sd[k].shape[2] for k in conv_keys]
    smiles_out_dim = sd["smiles_encoder.fc.weight"].shape[0]
    hid_dim        = sd["fc1.weight"].shape[0]
    return dict(
        cell_out_dim=cell_out_dim, cell_hid_dims=cell_hid_dims,
        smiles_emb_dim=smiles_emb_dim, smiles_out_dim=smiles_out_dim,
        n_filters=n_filters, filter_sizes=filter_sizes, hid_dim=hid_dim,
    )


def infer_lstm_arch(sd):
    cell_out_dim, cell_hid_dims = _cell_encoder_dims(sd)
    smiles_emb_dim = sd["smiles_encoder.embedding.weight"].shape[1]
    smiles_hid_dim = sd["smiles_encoder.rnn.weight_ih_l0"].shape[0] // 4
    n_layers       = sum(
        1 for k in sd if k.startswith("smiles_encoder.rnn.weight_hh_l")
    )
    smiles_out_dim = sd["smiles_encoder.fc.weight"].shape[0]
    hid_dim        = sd["fc1.weight"].shape[0]
    return dict(
        cell_out_dim=cell_out_dim, cell_hid_dims=cell_hid_dims,
        smiles_emb_dim=smiles_emb_dim, smiles_hid_dim=smiles_hid_dim,
        n_layers=n_layers, smiles_out_dim=smiles_out_dim, hid_dim=hid_dim,
    )


def infer_gcn_arch(sd):
    cell_out_dim, cell_hid_dims = _cell_encoder_dims(sd)
    gcn_hid_dim    = sd["smiles_encoder.fc_g1.weight"].shape[0]
    smiles_out_dim = sd["smiles_encoder.fc_g2.weight"].shape[0]
    fusion_hid_dim = sd["fc1.weight"].shape[0]
    return dict(
        cell_out_dim=cell_out_dim, cell_hid_dims=cell_hid_dims,
        gcn_hid_dim=gcn_hid_dim, smiles_out_dim=smiles_out_dim,
        fusion_hid_dim=fusion_hid_dim,
    )


def infer_gat_arch(sd, atom_dim=GRAPH_ATOM_DIM):
    cell_out_dim, cell_hid_dims = _cell_encoder_dims(sd)
    fc_g1_in       = sd["smiles_encoder.fc_g1.weight"].shape[1]
    n_heads        = int(round(math.sqrt(fc_g1_in / atom_dim)))
    smiles_hid_dim = sd["smiles_encoder.fc_g1.weight"].shape[0]
    smiles_out_dim = sd["smiles_encoder.fc_g2.weight"].shape[0]
    fusion_hid_dim = sd["fc1.weight"].shape[0]
    return dict(
        cell_out_dim=cell_out_dim, cell_hid_dims=cell_hid_dims,
        smiles_n_heads=n_heads, smiles_hid_dim=smiles_hid_dim,
        smiles_out_dim=smiles_out_dim, fusion_hid_dim=fusion_hid_dim,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_cnn(p, cell_input_dim, smiles_input_dim):
    from dl_model_architecture import NN_Encoder, CNN_Encoder, Seq2Func
    cell_enc   = NN_Encoder(cell_input_dim, p["cell_out_dim"], p["cell_hid_dims"], 0.0)
    smiles_enc = CNN_Encoder(smiles_input_dim, p["smiles_emb_dim"], p["smiles_out_dim"],
                             p["n_filters"], p["filter_sizes"], 0.0)
    return Seq2Func(cell_enc, smiles_enc, p["hid_dim"], OUT_DIM, 0.0, device=DEVICE).to(DEVICE)


def _build_lstm(p, cell_input_dim, smiles_input_dim):
    from dl_model_architecture import NN_Encoder, LSTM_Encoder, Seq2Func
    cell_enc   = NN_Encoder(cell_input_dim, p["cell_out_dim"], p["cell_hid_dims"], 0.0)
    smiles_enc = LSTM_Encoder(smiles_input_dim, p["smiles_emb_dim"], p["smiles_hid_dim"],
                               p["smiles_out_dim"], p["n_layers"], 0.0)
    return Seq2Func(cell_enc, smiles_enc, p["hid_dim"], OUT_DIM, 0.0, device=DEVICE).to(DEVICE)


def _build_gcn(p, cell_input_dim, smiles_input_dim):
    from dl_model_architecture import NN_Encoder, GCNNet, Seq2Func_Net
    cell_enc   = NN_Encoder(cell_input_dim, p["cell_out_dim"], p["cell_hid_dims"], 0.0)
    smiles_enc = GCNNet(smiles_input_dim, p["gcn_hid_dim"], p["smiles_out_dim"], 0.0)
    return Seq2Func_Net(cell_enc, smiles_enc, p["fusion_hid_dim"], OUT_DIM, 0.0,
                        device=DEVICE).to(DEVICE)


def _build_gat(p, cell_input_dim, smiles_input_dim):
    from dl_model_architecture import NN_Encoder, GATNet, Seq2Func_Net
    cell_enc   = NN_Encoder(cell_input_dim, p["cell_out_dim"], p["cell_hid_dims"], 0.0)
    smiles_enc = GATNet(smiles_input_dim, p["smiles_n_heads"],
                        p["smiles_hid_dim"], p["smiles_out_dim"], 0.0)
    return Seq2Func_Net(cell_enc, smiles_enc, p["fusion_hid_dim"], OUT_DIM, 0.0,
                        device=DEVICE).to(DEVICE)


_INFER_FN = {"cnn": infer_cnn_arch,  "lstm": infer_lstm_arch,
             "gcn": infer_gcn_arch,  "gat": infer_gat_arch}
_BUILD_FN = {"cnn": _build_cnn,      "lstm": _build_lstm,
             "gcn": _build_gcn,      "gat": _build_gat}


def load_model(model_type, cell_input_dim, smiles_input_dim):
    """Load the single pretrained optuna model in eval mode."""
    model_path = (
        f"../Models/{model_type}_models/"
        f"{model_type}_optuna_{DATA_TYPE}_{STRATIFY_BY}.pt"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    sd    = torch.load(model_path, map_location=DEVICE)
    p     = _INFER_FN[model_type](sd)
    print(f"  Inferred architecture: {p}")
    model = _BUILD_FN[model_type](p, cell_input_dim, smiles_input_dim)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_scaler(model_type):
    """Load the StandardScaler for the given model / data / stratify combination."""
    scaler_path = (
        f"../Models/{model_type}_models/"
        f"{model_type}_{DATA_TYPE}_{STRATIFY_BY}_scaling_gs.pk"
    )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as fh:
        return pickle.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# 5-fold CV split on the test set
# ──────────────────────────────────────────────────────────────────────────────

def get_test_fold_splits(n_samples, groups):
    """
    GroupKFold(N_CV_FOLDS) on the test set by dbgap_rnaseq_sample.
    Each fold's validation indices correspond to a disjoint patient subset.
    Returns list of (label, val_idx) tuples over all 5 folds.
    """
    gkf    = GroupKFold(n_splits=N_CV_FOLDS)
    splits = []
    for i, (_, val_idx) in enumerate(gkf.split(range(n_samples), groups=groups)):
        splits.append((f"Fold_{i + 1}", val_idx))
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────

def _metrics_row(model_type, fold_label, n_samples, labels, preds):
    mae, rmse, r2, pearson, spearman = calculate_regression_metrics(labels, preds)
    print(f"  [{model_type.upper()} {fold_label}]  N={n_samples}"
          f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}"
          f"  Pearson={pearson:.3f}  Spearman={spearman:.3f}")
    return {
        "Model": model_type.upper(),
        "Split": fold_label,
        "N_val": n_samples,
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Pearson_R": pearson, "Spearman_R": spearman,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SEQUENCE pipeline  (CNN / LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def run_sequence_test_cv(model_type, test_df, X_scaled, y, groups):
    """
    Tokenise SMILES once, predict on the full test set with the pretrained
    model, then compute metrics per 5-fold patient-based split.
    """
    from deepchem.feat.smiles_tokenizer import SmilesTokenizer
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from dl_model_architecture import evaluation_performance

    tokenizer        = SmilesTokenizer(VOCAB_PATH)
    smiles_input_dim = tokenizer.vocab_size

    def _encode(smi):
        return torch.tensor(
            tokenizer.encode(smi, max_length=MAX_SMILES_LEN, padding="max_length")
        )

    print(f"\n[{model_type.upper()}] Encoding SMILES ...")
    smiles_encoded = torch.stack([_encode(s) for s in test_df["SMILES"].tolist()])

    full_dataset = TensorDataset(
        smiles_encoded,
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y),
    )

    cell_input_dim = X_scaled.shape[1]
    print(f"  Loading model from ../Models/{model_type}_models/ ...")
    model     = load_model(model_type, cell_input_dim, smiles_input_dim)
    criterion = nn.L1Loss().to(DEVICE)
    batch_size = BATCH_SIZE_MAP[model_type]

    splits = get_test_fold_splits(len(y), groups)
    rows   = []
    for fold_label, val_idx in splits:
        print(f"  {fold_label}: {len(val_idx)} / {len(y)} samples"
              f"  ({100 * len(val_idx) / len(y):.1f} %)")
        loader = DataLoader(
            Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        preds, labels = evaluation_performance(model, loader, criterion, DEVICE)
        del loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows.append(_metrics_row(model_type, fold_label, len(val_idx), labels, preds))

    del model
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH pipeline  (GCN / GAT)
# ──────────────────────────────────────────────────────────────────────────────

def _smile_to_graph(smi):
    """Atom-featurisation identical to gcn_model_optuna.py / gat_model_optuna.py."""
    import networkx as nx
    from rdkit import Chem

    def _one_hot_unk(x, s):
        if x not in s:
            x = s[-1]
        return [x == e for e in s]

    def _one_hot(x, s):
        if x not in s:
            raise ValueError(f"{x} not in {s}")
        return [x == e for e in s]

    def _atom_feat(atom):
        return np.array(
            _one_hot_unk(atom.GetSymbol(),
                ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I',
                 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H',
                 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']) +
            _one_hot(atom.GetDegree(),          [0,1,2,3,4,5,6,7,8,9,10]) +
            _one_hot_unk(atom.GetTotalNumHs(),  [0,1,2,3,4,5,6,7,8,9,10]) +
            _one_hot_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) +
            [atom.GetIsAromatic()]
        )

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    c_size   = mol.GetNumAtoms()
    features = [f / (sum(f) or 1) for a in mol.GetAtoms() for f in [_atom_feat(a)]]
    edges    = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    g        = nx.Graph(edges).to_directed()
    edge_idx = [[e1, e2] for e1, e2 in g.edges]
    return c_size, features, edge_idx


def run_graph_test_cv(model_type, metadata_df, X_scaled, y, groups):
    """
    Convert SMILES to graphs, predict on the full test set with the pretrained
    model, then compute metrics per 5-fold patient-based split.
    """
    from torch_geometric import data as DATA
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from dl_model_architecture import evaluation_net_performance

    print(f"\n[{model_type.upper()}] Converting SMILES to graphs ...")
    smiles_list = metadata_df["SMILES"].tolist()
    data_list   = []
    valid_mask  = []

    for i, (smi, label, cell_feat) in enumerate(zip(smiles_list, y, X_scaled)):
        g = _smile_to_graph(smi)
        if g is None:
            print(f"  Warning: unparseable SMILES at index {i}: {smi}")
            valid_mask.append(False)
            continue
        c_size, features, edge_idx = g
        gdata             = DATA.Data(
            x          = torch.FloatTensor(features),
            edge_index = torch.LongTensor(edge_idx).transpose(1, 0),
            y          = torch.FloatTensor([label]),
        )
        gdata.cell_src = torch.FloatTensor(cell_feat)
        gdata.__setitem__("c_size", torch.LongTensor([c_size]))
        data_list.append(gdata)
        valid_mask.append(True)

    valid_mask = np.array(valid_mask)
    if not valid_mask.all():
        groups = groups[valid_mask]

    cell_input_dim   = X_scaled.shape[1]
    smiles_input_dim = GRAPH_ATOM_DIM

    print(f"  Valid graphs: {len(data_list)} / {len(y)}")
    print(f"  Loading model from ../Models/{model_type}_models/ ...")
    model     = load_model(model_type, cell_input_dim, smiles_input_dim)
    criterion = nn.L1Loss().to(DEVICE)
    batch_size = BATCH_SIZE_MAP[model_type]

    splits = get_test_fold_splits(len(data_list), groups)
    rows   = []
    for fold_label, val_idx in splits:
        print(f"  {fold_label}: {len(val_idx)} / {len(data_list)} samples"
              f"  ({100 * len(val_idx) / len(data_list):.1f} %)")
        val_ds     = [data_list[i] for i in val_idx]
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                   pin_memory=torch.cuda.is_available())
        preds, labels = evaluation_net_performance(
            model, val_loader, criterion, cell_input_dim, DEVICE
        )
        del val_loader, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        rows.append(_metrics_row(model_type, fold_label, len(val_idx), labels, preds))

    del model
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate and save results per model type
# ──────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ["MAE", "RMSE", "R2", "Pearson_R", "Spearman_R"]


def save_results(model_type, rows):
    results_df = pd.DataFrame(rows)

    mean_vals = results_df[METRIC_COLS].mean()
    std_vals  = results_df[METRIC_COLS].std(ddof=1)
    mean_n    = int(results_df["N_val"].mean())

    mean_row = {"Model": model_type.upper(), "Split": "Mean", "N_val": mean_n}
    std_row  = {"Model": model_type.upper(), "Split": "Std",  "N_val": ""}
    for col in METRIC_COLS:
        mean_row[col] = mean_vals[col]
        std_row[col]  = std_vals[col]

    summary_df = pd.DataFrame([mean_row, std_row])
    out_df     = pd.concat([results_df, summary_df], ignore_index=True)

    for col in METRIC_COLS:
        out_df[col] = out_df[col].apply(lambda x: f"{float(x):.3f}")

    out_path = f"{OUTPUT_DIR}{model_type}_{DATA_TYPE}_{STRATIFY_BY}_test_cv_metrics.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(out_df.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nLoading test data from {TEST_DATA_PATH} ...")
big_test_df = pd.read_pickle(TEST_DATA_PATH, compression="zip")
print(f"  Shape: {big_test_df.shape}")

for MODEL_TYPE in ALL_MODEL_TYPES:
    print(f"\n{'=' * 70}")
    print(f"  Model: {MODEL_TYPE.upper()}   Data: {DATA_TYPE}   Stratify: {STRATIFY_BY}")
    print(f"{'=' * 70}")

    extra_numeric = MODEL_TYPE in ("gcn", "gat")
    metadata_df, X_raw, y, _ = extract_features(
        big_test_df, extra_numeric_filter=extra_numeric
    )

    try:
        scaler   = load_scaler(MODEL_TYPE)
        X_scaled = scaler.transform(X_raw)
    except FileNotFoundError as exc:
        print(f"  SKIPPED: {exc}")
        continue

    # groups for splitting the test set by patient
    groups = metadata_df["dbgap_rnaseq_sample"].to_numpy()

    try:
        if MODEL_TYPE in ("cnn", "lstm"):
            fold_rows = run_sequence_test_cv(
                MODEL_TYPE, big_test_df, X_scaled, y, groups
            )
        else:
            fold_rows = run_graph_test_cv(
                MODEL_TYPE, metadata_df, X_scaled, y, groups
            )
    except FileNotFoundError as exc:
        print(f"  SKIPPED: {exc}")
        continue

    save_results(MODEL_TYPE, fold_rows)

print("\nAll done.")
