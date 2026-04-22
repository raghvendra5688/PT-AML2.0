"""
dl_cv_metrics.py — Held-out validation metrics for pretrained DL models.

Supports model types : cnn | lstm | gcn | gat
Data types           : Only_PC_Feat_Var | Embed_Feat_Var | ChemBERTa_Feat_Var | MolFormer_Feat_Var
Stratification       : dbgap_rnaseq_sample | inhibitor | random

Metrics saved (3 d.p.): MAE, RMSE, R2, Pearson_R, Spearman_R

EVALUATION DESIGN
-----------------
Each saved model was trained by final_data_preparation() using a single
GroupShuffleSplit (test_size=0.2, random_state=42) on the training data.
Approximately 80 % of the training samples entered the final model's training
set; the remaining 20 % were its validation set and were NEVER used for
weight updates.

Running a post-hoc GroupKFold on the same training data would overlap
~80 % of every fold's "validation" samples with the model's actual training
set, producing severely optimistic (in-sample) metrics.

This script evaluates in two steps:
  1. Reconstruct the EXACT 20 % holdout used during training (same seed=42),
     ensuring zero contamination — no holdout sample was ever seen during
     weight updates.
  2. Generate N_RANDOMIZATIONS (=5) overlapping 4/5 subsets of the holdout:
       dbgap_rnaseq_sample / inhibitor → GroupKFold (preserves group integrity)
       random                          → KFold (shuffle=True, random_state=42)
     Output includes Part_1…Part_5 rows plus Mean ± std summary.

Usage examples:
    python dl_cv_metrics.py --model_type cnn  --data_type Only_PC_Feat_Var --stratify_by dbgap_rnaseq_sample
    python dl_cv_metrics.py --model_type lstm --data_type Only_PC_Feat_Var --stratify_by inhibitor
    python dl_cv_metrics.py --model_type gcn  --data_type Embed_Feat_Var   --stratify_by random
    python dl_cv_metrics.py --model_type gat  --data_type Only_PC_Feat_Var --stratify_by dbgap_rnaseq_sample
"""

import argparse
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
import torch.utils.data as data_utils
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misc import calculate_regression_metrics

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DL model 5-fold CV quality metrics")
parser.add_argument("--model_type",  required=True,
                    choices=["cnn", "lstm", "gcn", "gat"])
parser.add_argument("--data_type",   required=True,
                    choices=["Only_PC_Feat_Var", "Embed_Feat_Var",
                             "ChemBERTa_Feat_Var", "MolFormer_Feat_Var"])
parser.add_argument("--stratify_by", required=True,
                    choices=["dbgap_rnaseq_sample", "inhibitor", "random"])
args = parser.parse_args()

MODEL_TYPE   = args.model_type
DATA_TYPE    = args.data_type
STRATIFY_BY  = args.stratify_by

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SEED             = 123
N_RANDOMIZATIONS = 5   # sub-parts used for dbgap_rnaseq_sample / random (inhibitor uses a single holdout)
OUT_DIM          = 1
VOCAB_PATH     = "../Data/vocab.txt"
MAX_SMILES_LEN = 150
GRAPH_ATOM_DIM = 78          # fixed atom-feature dimension for GCN / GAT

TRAIN_DATA_MAP = {
    "Only_PC_Feat_Var":   "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    "Embed_Feat_Var":     "../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    "ChemBERTa_Feat_Var": "../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    "MolFormer_Feat_Var": "../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
}

# Batch sizes match those used during training for each model type
BATCH_SIZE_MAP = {"cnn": 16384, "lstm": 4096, "gcn": 65536, "gat": 16384}

TRAIN_DATA_PATH = TRAIN_DATA_MAP[DATA_TYPE]
MODEL_PATH      = f"../Models/{MODEL_TYPE}_models/{MODEL_TYPE}_optuna_{DATA_TYPE}_{STRATIFY_BY}.pt"
SCALER_PATH     = f"../Models/{MODEL_TYPE}_models/{MODEL_TYPE}_{DATA_TYPE}_{STRATIFY_BY}_scaling_gs.pk"
OUTPUT_DIR      = f"../Results/{MODEL_TYPE}/"
OUTPUT_PATH     = f"{OUTPUT_DIR}{MODEL_TYPE}_{DATA_TYPE}_{STRATIFY_BY}_cv_metrics.csv"
BATCH_SIZE      = BATCH_SIZE_MAP[MODEL_TYPE]

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility & device
# ──────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    # Pick the GPU with the most free memory (mirrors gat_model_optuna.py)
    _best_gpu = max(range(torch.cuda.device_count()),
                    key=lambda i: torch.cuda.mem_get_info(i)[0])
    DEVICE   = torch.device(f"cuda:{_best_gpu}")
    _free_gb = torch.cuda.mem_get_info(_best_gpu)[0] / 1024**3
    print(f"[{MODEL_TYPE.upper()}] GPU {_best_gpu}  ({_free_gb:.1f} GB free)"
          f"  data={DATA_TYPE}  stratify={STRATIFY_BY}")
else:
    DEVICE = torch.device("cpu")
    print(f"[{MODEL_TYPE.upper()}] CPU  data={DATA_TYPE}  stratify={STRATIFY_BY}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction  (shared across all model types)
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
    Identifies metadata and feature columns exactly as in each training script's
    preprocess_data().

    extra_numeric_filter=True applies the additional numeric-only / NaN-drop
    step used by gcn_model_optuna.py and gat_model_optuna.py.

    Returns
    -------
    metadata_df : pd.DataFrame
    X_raw       : pd.DataFrame  (feature columns only)
    y           : np.ndarray
    feature_cols: list[str]
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

    # GCN / GAT: keep only numeric columns and drop any that still have NaNs
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
# Architecture inference from state dict
# ──────────────────────────────────────────────────────────────────────────────

def _cell_encoder_dims(sd):
    """
    Parse NN_Encoder architecture from the state dict.
    Returns (cell_out_dim, cell_hid_dims).
    Each fcs.i.0.weight has shape [dim_out, dim_in].
    """
    fc_keys = sorted(
        k for k in sd
        if k.startswith("cell_encoder.fcs.") and k.endswith(".0.weight")
    )
    dims          = [sd[k].shape[0] for k in fc_keys]
    cell_hid_dims = dims[:-1]
    cell_out_dim  = dims[-1]
    return cell_out_dim, cell_hid_dims


def infer_cnn_arch(sd):
    """
    CNN_Encoder keys of interest:
      convs.i.weight  shape: [n_filters, emb_dim, kernel_i]
      fc.weight       shape: [smiles_out_dim, n_filters * n_conv_layers]
    Seq2Func:
      fc1.weight      shape: [hid_dim, cell_out_dim + smiles_out_dim]
    """
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
    """
    LSTM_Encoder keys of interest:
      embedding.weight         shape: [vocab_size, smiles_emb_dim]
      rnn.weight_ih_l0         shape: [4*smiles_hid_dim, smiles_emb_dim]
      rnn.weight_hh_l{0..N-1} — count gives n_layers
      fc.weight                shape: [smiles_out_dim, n_layers * smiles_hid_dim]
    Seq2Func:
      fc1.weight               shape: [hid_dim, cell_out_dim + smiles_out_dim]
    """
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
    """
    GCNNet keys of interest:
      fc_g1.weight  shape: [gcn_hid_dim, gcn_hid_dim]   (square)
      fc_g2.weight  shape: [smiles_out_dim, gcn_hid_dim]
    Seq2Func_Net:
      fc1.weight    shape: [fusion_hid_dim, cell_out_dim + smiles_out_dim]
    """
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
    """
    GATNet keys of interest:
      fc_g1.weight  shape: [smiles_hid_dim, atom_dim * n_heads^2]
        → n_heads = sqrt(fc_g1.in_features / atom_dim)
      fc_g2.weight  shape: [smiles_out_dim, smiles_hid_dim]
    Seq2Func_Net:
      fc1.weight    shape: [fusion_hid_dim, cell_out_dim + smiles_out_dim]
    """
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
# Model builders  (dropout=0.0 — eval mode disables it regardless)
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


_INFER_FN  = {"cnn": infer_cnn_arch,  "lstm": infer_lstm_arch,
              "gcn": infer_gcn_arch,  "gat": infer_gat_arch}
_BUILD_FN  = {"cnn": _build_cnn,      "lstm": _build_lstm,
              "gcn": _build_gcn,      "gat": _build_gat}


def load_model(cell_input_dim, smiles_input_dim):
    """Load state dict, infer architecture, build and return model in eval mode."""
    # Always load directly onto the target device so no silent cross-device copies occur
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    p       = _INFER_FN[MODEL_TYPE](sd)
    print(f"  Inferred architecture: {p}")
    model   = _BUILD_FN[MODEL_TYPE](p, cell_input_dim, smiles_input_dim)
    model.load_state_dict(sd)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Holdout reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def get_all_holdout_indices(n_samples, groups):
    """
    Two-step holdout partitioning:

    Step 1 — Reconstruct the exact 20 % holdout used during training:
      • grouped → GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
      • random  → data_utils.random_split(..., generator=manual_seed(42))

    Step 2 — N_RANDOMIZATIONS overlapping (N-1)/N subsets of the holdout:
      • grouped (dbgap_rnaseq_sample / inhibitor)
                        → GroupKFold(N_RANDOMIZATIONS) — preserves group integrity
      • random          → KFold(N_RANDOMIZATIONS, shuffle=True, random_state=42)

    Returns list of (label: str, val_idx: np.ndarray) tuples.
    """
    # ── Step 1: reconstruct the exact training holdout ──────────────────────
    if STRATIFY_BY in ("dbgap_rnaseq_sample", "inhibitor"):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, holdout_idx = next(gss.split(range(n_samples), groups=groups))
    else:
        train_size = int(0.8 * n_samples)
        if MODEL_TYPE in ("gcn", "gat"):
            # gcn_model_optuna.py / gat_model_optuna.py use random.Random(42).shuffle()
            # — must match exactly to avoid evaluating on training samples.
            indices = list(range(n_samples))
            random.Random(42).shuffle(indices)
            holdout_idx = np.array(indices[train_size:])
        else:
            # cnn_model.py / lstm_model.py use data_utils.random_split with a
            # torch.Generator — reproduce the same split here.
            valid_size = n_samples - train_size
            gen = torch.Generator().manual_seed(42)
            _, val_subset = data_utils.random_split(
                range(n_samples), [train_size, valid_size], generator=gen
            )
            holdout_idx = np.array(val_subset.indices)

    # ── Step 2: N_RANDOMIZATIONS overlapping (N-1)/N subsets of the holdout ─
    splits = []
    if STRATIFY_BY in ("dbgap_rnaseq_sample", "inhibitor"):
        holdout_groups = groups[holdout_idx]
        gkf = GroupKFold(n_splits=N_RANDOMIZATIONS)
        for i, (sub_idx, _) in enumerate(
            gkf.split(holdout_idx, groups=holdout_groups)
        ):
            splits.append((f"Part_{i + 1}", holdout_idx[sub_idx]))
    else:
        kf = KFold(n_splits=N_RANDOMIZATIONS, shuffle=True, random_state=42)
        for i, (sub_idx, _) in enumerate(kf.split(holdout_idx)):
            splits.append((f"Part_{i + 1}", holdout_idx[sub_idx]))

    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────

def _metrics_row(split_label, n_val, labels, preds):
    mae, rmse, r2, pearson, spearman = calculate_regression_metrics(labels, preds)
    print(f"  [{split_label}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}"
          f"  Pearson={pearson:.3f}  Spearman={spearman:.3f}")
    return {"Split": split_label, "N_val": n_val,
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "Pearson_R": pearson, "Spearman_R": spearman}


# ──────────────────────────────────────────────────────────────────────────────
# SEQUENCE pipeline  (CNN / LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def run_sequence_cv(df, X_scaled, y, groups):
    """
    Tokenises SMILES once, builds a TensorDataset, then evaluates the
    pretrained model on the held-out 20 % split that was never used during
    final model training (reconstructed with the same seed used in training).
    Uses evaluation_performance(model, loader, criterion, DEVICE).
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

    print("  Encoding SMILES ...")
    smiles_encoded = torch.stack([_encode(s) for s in df["SMILES"].tolist()])

    full_dataset = TensorDataset(
        smiles_encoded,
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y),
    )

    cell_input_dim = X_scaled.shape[1]
    print(f"  Loading model from {MODEL_PATH} ...")
    model     = load_model(cell_input_dim, smiles_input_dim)
    criterion = nn.L1Loss().to(DEVICE)

    splits = get_all_holdout_indices(len(y), groups)
    rows   = []
    for split_label, val_idx in splits:
        print(f"  {split_label}: {len(val_idx)} / {len(y)} samples"
              f"  ({100 * len(val_idx) / len(y):.1f} %)")
        val_loader = DataLoader(
            Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        preds, labels = evaluation_performance(model, val_loader, criterion, DEVICE)
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows.append(_metrics_row(split_label, len(val_idx), labels, preds))

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH pipeline  (GCN / GAT)
# ──────────────────────────────────────────────────────────────────────────────

def run_graph_cv(df, metadata_df, X_scaled, y, groups):
    """
    Converts SMILES to PyG Data objects, then evaluates the pretrained model
    on each fold's validation split.
    Uses evaluation_net_performance(model, loader, criterion, N_dim, DEVICE).
    """
    import networkx as nx
    from rdkit import Chem
    from torch_geometric import data as DATA
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from dl_model_architecture import evaluation_net_performance

    # ── atom featurisation (identical to gcn_model_optuna.py / gat_model_optuna.py) ──
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

    def _smile_to_graph(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        c_size   = mol.GetNumAtoms()
        features = [f / (sum(f) or 1) for a in mol.GetAtoms() for f in [_atom_feat(a)]]
        edges    = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        g        = nx.Graph(edges).to_directed()
        edge_idx = [[e1, e2] for e1, e2 in g.edges]
        return c_size, features, edge_idx

    print("  Converting SMILES to graphs ...")
    smiles_list = metadata_df["SMILES"].tolist()
    data_list, valid_mask = [], []

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

    # Filter groups to match data_list in the (rare) case of unparseable SMILES
    valid_mask = np.array(valid_mask)
    if groups is not None and not valid_mask.all():
        groups = groups[valid_mask]

    cell_input_dim   = X_scaled.shape[1]
    smiles_input_dim = GRAPH_ATOM_DIM

    print(f"  Loading model from {MODEL_PATH} ...")
    model     = load_model(cell_input_dim, smiles_input_dim)
    criterion = nn.L1Loss().to(DEVICE)

    splits = get_all_holdout_indices(len(data_list), groups)
    rows   = []
    for split_label, val_idx in splits:
        print(f"  {split_label}: {len(val_idx)} / {len(data_list)} samples"
              f"  ({100 * len(val_idx) / len(data_list):.1f} %)")
        val_ds     = [data_list[i] for i in val_idx]
        val_loader = PyGDataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   pin_memory=torch.cuda.is_available())
        preds, labels = evaluation_net_performance(
            model, val_loader, criterion, cell_input_dim, DEVICE
        )
        del val_loader, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        rows.append(_metrics_row(split_label, len(val_idx), labels, preds))

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate and save results
# ──────────────────────────────────────────────────────────────────────────────

def save_results(rows):
    metric_cols = ["MAE", "RMSE", "R2", "Pearson_R", "Spearman_R"]
    results_df  = pd.DataFrame(rows)

    # Append Mean / Std only when there are multiple evaluation subsets
    if len(rows) > 1:
        mean_vals = results_df[metric_cols].mean()
        std_vals  = results_df[metric_cols].std(ddof=1)
        mean_n    = int(results_df["N_val"].mean())

        mean_row = {"Split": "Mean", "N_val": mean_n}
        std_row  = {"Split": "Std",  "N_val": ""}
        for col in metric_cols:
            mean_row[col] = mean_vals[col]
            std_row[col]  = std_vals[col]

        summary_df = pd.DataFrame([mean_row, std_row])
        results_df = pd.concat([results_df, summary_df], ignore_index=True)

    # Format all metric values to exactly 3 decimal places
    for col in metric_cols:
        results_df[col] = results_df[col].apply(lambda x: f"{float(x):.3f}")

    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved → {OUTPUT_PATH}")
    print(results_df.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nLoading training data from {TRAIN_DATA_PATH} ...")
big_train_df = pd.read_pickle(TRAIN_DATA_PATH, compression="zip")
print(f"  Shape: {big_train_df.shape}")

# GCN/GAT apply an extra numeric-only / NaN-drop step in their preprocess_data
extra_numeric = MODEL_TYPE in ("gcn", "gat")
metadata_df, X_raw, y, _ = extract_features(
    big_train_df, extra_numeric_filter=extra_numeric
)

print(f"Loading scaler from {SCALER_PATH} ...")
with open(SCALER_PATH, "rb") as fh:
    scaler = pickle.load(fh)
X_scaled = scaler.transform(X_raw)

# Groups used to reconstruct the training holdout split
if STRATIFY_BY == "dbgap_rnaseq_sample":
    groups = metadata_df["dbgap_rnaseq_sample"].to_numpy()
elif STRATIFY_BY == "inhibitor":
    groups = metadata_df["inhibitor"].to_numpy()
else:
    groups = None   # random → torch random_split is used instead

print(f"\nEvaluating on {N_RANDOMIZATIONS} sub-parts of the 20 % training holdout ...")
if MODEL_TYPE in ("cnn", "lstm"):
    fold_rows = run_sequence_cv(big_train_df, X_scaled, y, groups)
else:
    fold_rows = run_graph_cv(big_train_df, metadata_df, X_scaled, y, groups)

save_results(fold_rows)
