#!/usr/bin/env python
# coding: utf-8
# %%
import os
from collections import Counter

import numpy as np
import pandas as pd
import pubchempy as pcp
import torch
import networkx as nx
from rdkit.Chem import AllChem, Descriptors, rdmolops, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer

# %%
# Columns to exclude from RDKit descriptors
USELESS_COLS = {
    'MaxPartialCharge',
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester',
}

GRAPH_FEAT_KEYS = [
    'graph_diameter', 'avg_shortest_path', 'num_cycles', 'num_chains',
    'clustering_coefficients', 'avg_degree_centrality', 'avg_eigen_centrality',
    'avg_betweenness_centrality', 'avg_load_centrality', 'wiener_index',
    'max_degree', 'closeness_mean', 'katz_centrality_std',
    'betweenness_mean', 'betweenness_std', 'eigenvector_mean',
    'ring_4', 'heteroatom_ratio', 'ring_1', 'ring_2', 'ring_3',
    'ring_5', 'num_aromatic_rings', 'num_non_aromatic_rings',
    'average_carbon', 'average_oxygen', 'average_sulphur',
    'average_nitrogen', 'num_single_bonds', 'num_double_bonds',
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_and_validate_smiles(smiles):
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        '([R])', '([R1])', '([R2])',
    ]
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except Exception:
        return None


def compute_all_descriptors(smiles):
    desc_names = [d[0] for d in Descriptors.descList if d[0] not in USELESS_COLS]
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return [None] * len(desc_names)
    return [d[1](mol) for d in Descriptors.descList if d[0] not in USELESS_COLS]


def _append_zero_graph_feats(graph_feats):
    for key in GRAPH_FEAT_KEYS:
        graph_feats[key].append(0)


def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        _append_zero_graph_feats(graph_feats)
        return graph_feats

    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(
        nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))
    graph_feats['num_chains'].append(len(list(nx.chain_decomposition(G))))
    graph_feats['clustering_coefficients'].append(nx.average_clustering(G))
    graph_feats['avg_degree_centrality'].append(
        np.array(list(nx.degree_centrality(G).values())).mean())
    graph_feats['avg_eigen_centrality'].append(
        np.array(list(nx.katz_centrality(G).values())).mean())
    graph_feats['avg_betweenness_centrality'].append(
        np.array(list(nx.betweenness_centrality(G).values())).mean())
    graph_feats['avg_load_centrality'].append(
        np.array(list(nx.load_centrality(G).values())).mean())
    graph_feats['wiener_index'].append(nx.wiener_index(G))
    max_degree = max((d for _, d in G.degree()), default=0)
    graph_feats['max_degree'].append(max_degree)

    if nx.is_connected(G):
        closeness = list(nx.closeness_centrality(G).values())
        graph_feats['closeness_mean'].append(np.mean(closeness) if closeness else 0)
    else:
        graph_feats['closeness_mean'].append(0)

    try:
        katz = list(nx.katz_centrality(G, max_iter=1000).values())
        graph_feats['katz_centrality_std'].append(np.std(katz) if len(katz) > 1 else 0)
    except Exception:
        graph_feats['katz_centrality_std'].append(0)

    betweenness = list(nx.betweenness_centrality(G).values())
    if betweenness:
        betweenness = [b for b in betweenness if np.isfinite(b)]
        graph_feats['betweenness_mean'].append(np.mean(betweenness) if betweenness else 0)
        graph_feats['betweenness_std'].append(np.std(betweenness) if len(betweenness) > 1 else 0)
    else:
        graph_feats['betweenness_mean'].append(0)
        graph_feats['betweenness_std'].append(0)

    try:
        eigenvector = list(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6).values())
        eigenvector = [e for e in eigenvector if np.isfinite(e)]
        graph_feats['eigenvector_mean'].append(np.mean(eigenvector) if eigenvector else 0)
    except Exception:
        graph_feats['eigenvector_mean'].append(0)

    cycles = list(nx.cycle_basis(G))
    cycle_lengths = [len(c) for c in cycles]
    for ring_len in [1, 2, 3, 4, 5]:
        graph_feats[f'ring_{ring_len}'].append(
            sum(1 for l in cycle_lengths if l == ring_len))

    try:
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_counts = Counter(atom_types)
        total_atoms = len(atom_types)
        hetero_atoms = total_atoms - atom_counts.get('C', 0)
        graph_feats['heteroatom_ratio'].append(
            hetero_atoms / total_atoms if total_atoms > 0 else 0)
    except Exception:
        graph_feats['heteroatom_ratio'].append(0)

    try:
        aromatic_rings = [ring for ring in cycles
                          if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]
        graph_feats['num_aromatic_rings'].append(len(aromatic_rings))
    except Exception:
        graph_feats['num_aromatic_rings'].append(0)

    try:
        non_aromatic_rings = [ring for ring in cycles
                               if not any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]
        graph_feats['num_non_aromatic_rings'].append(len(non_aromatic_rings))
    except Exception:
        graph_feats['num_non_aromatic_rings'].append(0)

    try:
        num_carbon   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C') / n_nodes if n_nodes else 0
        num_sulfur   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'S') / n_nodes if n_nodes else 0
        num_oxygen   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O') / n_nodes if n_nodes else 0
        num_nitrogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N') / n_nodes if n_nodes else 0
        graph_feats['average_oxygen'].append(num_oxygen)
        graph_feats['average_nitrogen'].append(num_nitrogen)
        graph_feats['average_carbon'].append(num_carbon)
        graph_feats['average_sulphur'].append(num_sulfur)
    except Exception:
        graph_feats['average_oxygen'].append(0)
        graph_feats['average_nitrogen'].append(0)
        graph_feats['average_carbon'].append(0)
        graph_feats['average_sulphur'].append(0)

    try:
        single_bonds = (sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE)
                        / n_edges if n_edges else 0)
        double_bonds = (sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
                        / n_edges if n_edges else 0)
        graph_feats['num_single_bonds'].append(single_bonds)
        graph_feats['num_double_bonds'].append(double_bonds)
    except Exception:
        graph_feats['num_single_bonds'].append(0)
        graph_feats['num_double_bonds'].append(0)

    return graph_feats


def preprocessing(df, radius=2, n_bits=128):
    """Compute RDKit descriptors, graph features, and Morgan+MACCS fingerprints."""
    validated_smiles = df['SMILES'].apply(clean_and_validate_smiles).tolist()

    desc_names = [d[0] for d in Descriptors.descList if d[0] not in USELESS_COLS]
    rdkit_descriptors = [compute_all_descriptors(smi) for smi in validated_smiles]

    graph_feats = {k: [] for k in GRAPH_FEAT_KEYS}
    for smi in validated_smiles:
        compute_graph_features(smi, graph_feats)

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    for smi in validated_smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            morgan_fp = generator.GetFingerprint(mol)
            maccs_fp  = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.append(np.concatenate([np.array(morgan_fp), np.array(maccs_fp)]))
        else:
            fingerprints.append(np.zeros(n_bits + 167))

    fp_array = np.array(fingerprints)
    mfp_descriptor_names = (
        [f'Morgan_{i}' for i in range(n_bits)] +
        [f'MACCS_{i}'  for i in range(167)]
    )

    result = pd.concat([
        pd.DataFrame(rdkit_descriptors, columns=desc_names),
        pd.DataFrame(graph_feats),
        pd.DataFrame(fp_array, columns=mfp_descriptor_names),
    ], axis=1)
    result = result.replace([-np.inf, np.inf], np.nan)
    return result


def fetch_pubchem_properties(drug_names, cid_dict):
    """Fetch InChIKey, SMILES, XLogP, MolecularWeight from PubChem for each drug."""
    records = []
    for drug in drug_names:
        cid = cid_dict.get(drug)
        if pd.notna(cid):
            props = pcp.get_properties(
                ['InChIKey', 'ConnectivitySMILES', 'XLogP', 'MolecularWeight'],
                int(cid), 'cid'
            )
            row = props[0]
            row['Name'] = drug
        else:
            row = {
                'CID': None, 'InChIKey': None, 'ConnectivitySMILES': None,
                'XLogP': None, 'MolecularWeight': None, 'Name': drug,
            }
        records.append(row)
    info_df = pd.DataFrame(records)
    info_df = info_df.rename(columns={'ConnectivitySMILES': 'SMILES'})
    return info_df


def run_smiles_autoencoder(smiles_list, smiles_csv_path, output_csv_path, script_dir):
    """Write SMILES to CSV, run ls_generator2.py, and return the embedding DataFrame."""
    smiles_df = pd.DataFrame({'src': smiles_list, 'trg': smiles_list})
    smiles_df.to_csv(smiles_csv_path, index=False)
    input_name  = os.path.basename(smiles_csv_path)
    output_name = os.path.basename(output_csv_path)
    command = f"cd {script_dir} && python ls_generator2.py --input {input_name} --output {output_name}"
    os.system(command)
    return pd.read_csv(output_csv_path)


def get_molformer_embeddings(smiles_list, prefix="MF"):
    model     = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output.numpy()
    return pd.DataFrame(embeddings, columns=[f"{prefix}_{i}" for i in range(embeddings.shape[1])])


def get_chemberta_embeddings(smiles_list, prefix="CBT"):
    model     = AutoModel.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepChem/ChemBERTa-77M-MLM", trust_remote_code=True)
    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.pooler_output.numpy()
    return pd.DataFrame(embeddings, columns=[f"{prefix}_{i}" for i in range(embeddings.shape[1])])


# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR      = "../Data/FIMM-AML"
SMILES_AE_DIR = "../Data/SMILES_Autoencoder"
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

DRUG_SENSITIVITY_CSV = f"{DATA_DIR}/Drug_Sensitivity_Scores.csv"
DRUG_MAPPING_CSV     = f"{DATA_DIR}/Drug_Mapping_Data.csv"

DRUG_CIDS_CSV    = f"{DATA_DIR}/fimmaml_drug_cids.csv"
FULL_INFO_CSV    = f"{DATA_DIR}/FIMMAML_Drug_Full_Info.csv"
SMILES_EMBED_CSV = f"{DATA_DIR}/FIMMAML_Drug_Full_SMILES_Embedding.csv"
MOLFORMER_CSV    = f"{DATA_DIR}/FIMMAML_Drug_Full_MolFormer_Embedding.csv"
CHEMBERTA_CSV    = f"{DATA_DIR}/FIMMAML_Drug_Full_ChemBERTa_Embedding.csv"

SMILES_INPUT_CSV  = f"{SMILES_AE_DIR}/test_smiles.csv"
SMILES_OUTPUT_CSV = f"{SMILES_AE_DIR}/test_LS.csv"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BUILD DRUG LIST FROM MAPPING (78 FIMM↔BeatAML COMMON DRUGS)
# ══════════════════════════════════════════════════════════════════════════════

# %%
# Load Drug_Mapping_Data — the 78 FIMM-AML drugs that map to BeatAML
mapping_df = pd.read_csv(DRUG_MAPPING_CSV, sep='\t')
mapping_df.columns = ['Id', 'FIMM_Name', 'BeatAML_Name', 'Match_Type']
mapping_df = mapping_df.drop_duplicates(subset='FIMM_Name').reset_index(drop=True)
print(f"Total drugs in mapping: {len(mapping_df)}")

# %%
# Subset Drug_Sensitivity_Scores to only the 78 mapped drugs
sensitivity_df = pd.read_csv(DRUG_SENSITIVITY_CSV, sep='\t')
sensitivity_subset = sensitivity_df[
    sensitivity_df['Chemical_compound'].isin(mapping_df['FIMM_Name'])
].reset_index(drop=True)
sensitivity_subset.to_csv(f"{DATA_DIR}/Drug_Sensitivity_Scores_Common.csv", index=False)
print(f"Subset sensitivity scores: {sensitivity_subset.shape} rows")
print(f"Unique drugs in subset: {sensitivity_subset['Chemical_compound'].nunique()}")

# %%
# Get PubChem CID for each drug using the BeatAML name
# (alternate names such as AZD1152-HQPA map correctly; exact matches are identical)
drug_cid_dict = {}
for _, row in mapping_df.iterrows():
    beataml_name = row['BeatAML_Name']
    lookup_name  = beataml_name.split(" ")[0]
    cids = pcp.get_cids(lookup_name, 'name', 'substance', list_return='flat')
    if cids:
        drug_cid_dict[beataml_name] = cids[0]
        print(lookup_name, cids[0])
    else:
        drug_cid_dict[beataml_name] = None
        print(f"No CID found for: {beataml_name}")

# %%
# Save CIDs
cid_df = mapping_df.copy()
cid_df['cid'] = cid_df['BeatAML_Name'].map(drug_cid_dict)
cid_df.to_csv(DRUG_CIDS_CSV, index=False)
print(f"Saved CIDs: {DRUG_CIDS_CSV}")
print(f"  Found CID: {cid_df['cid'].notna().sum()} / {len(cid_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PUBCHEM PROPERTIES + FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

# %%
# --- RESTART FROM HERE ---
cid_df = pd.read_csv(DRUG_CIDS_CSV)
drug_cid_dict = dict(zip(cid_df['BeatAML_Name'], cid_df['cid']))
print(f"Loaded {len(drug_cid_dict)} drug CIDs")

# %%
# Fetch PubChem properties keyed on BeatAML name
beataml_names_list = cid_df['BeatAML_Name'].tolist()
drug_info_df = fetch_pubchem_properties(beataml_names_list, drug_cid_dict)

# Attach original FIMM-AML name for downstream merging
drug_info_df = drug_info_df.merge(
    cid_df[['BeatAML_Name', 'FIMM_Name']],
    left_on='Name', right_on='BeatAML_Name', how='left'
).drop(columns='BeatAML_Name')

print(f"Drugs with no SMILES: {drug_info_df['SMILES'].isnull().sum()}")
print(drug_info_df.loc[drug_info_df['SMILES'].isnull(), 'Name'].tolist())
drug_info_df = drug_info_df[drug_info_df['SMILES'].notnull()].reset_index(drop=True)
print(f"Drugs with SMILES: {len(drug_info_df)}")

# %%
# Compute physicochemical descriptors + graph features + fingerprints
results_df = preprocessing(drug_info_df, radius=2, n_bits=128)
ultimate_drug_info_df = pd.concat(
    [drug_info_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
print(f"Feature matrix shape: {ultimate_drug_info_df.shape}")
ultimate_drug_info_df.to_csv(FULL_INFO_CSV, index=False)
print(f"Saved: {FULL_INFO_CSV}")

# %%
# Verify saved file
input_drug_df = pd.read_csv(FULL_INFO_CSV)
print(input_drug_df.shape)
print(input_drug_df.columns.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EMBEDDING GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# %%
# SMILES autoencoder embeddings
src = input_drug_df["SMILES"].apply(clean_and_validate_smiles).tolist()
smiles_embedding_df = run_smiles_autoencoder(src, SMILES_INPUT_CSV, SMILES_OUTPUT_CSV, SCRIPT_DIR)
pd.concat([input_drug_df, smiles_embedding_df], axis=1).to_csv(SMILES_EMBED_CSV, index=False)
print(f"Saved: {SMILES_EMBED_CSV}")

# %%
# MolFormer embeddings
src = input_drug_df["SMILES"].apply(clean_and_validate_smiles).tolist()
molformer_df = get_molformer_embeddings(src, prefix="MF")
pd.concat([input_drug_df, molformer_df], axis=1).to_csv(MOLFORMER_CSV, index=False)
print(f"Saved: {MOLFORMER_CSV}")

# %%
# ChemBERTa embeddings
chemberta_df = get_chemberta_embeddings(src, prefix="CBT")
pd.concat([input_drug_df, chemberta_df], axis=1).to_csv(CHEMBERTA_CSV, index=False)
print(f"Saved: {CHEMBERTA_CSV}")
