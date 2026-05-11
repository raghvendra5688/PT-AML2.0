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

GRAPH_FEAT_ZERO = {k: 0 for k in GRAPH_FEAT_KEYS}


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
    """Append zero values for all graph features when SMILES is invalid."""
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
    # Validate SMILES once; reuse for both descriptor and graph computations
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
    # Rename the connectivity SMILES key to a consistent column name
    info_df = info_df.rename(columns={'ConnectivitySMILES': 'SMILES'})
    return info_df


def run_smiles_autoencoder(smiles_list, smiles_csv_path, output_csv_path, script_dir):
    """Write SMILES to CSV, run ls_generator2.py, and return the embedding DataFrame.

    ls_generator2.py always prepends '../Data/SMILES_Autoencoder/' to its --input
    and --output arguments, so we pass only the bare filenames and cd to script_dir
    first so that relative path resolves correctly.
    """
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

DATA_DIR          = "../Data/leeaml"
SMILES_AE_DIR     = "../Data/SMILES_Autoencoder"
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))

SPECIFIC_DRUGS_CSV    = f"{DATA_DIR}/leeaml_specific_drugs.csv"
SPECIFIC_CIDS_CSV     = f"{DATA_DIR}/leeaml_specific_drug_cids.csv"
SPECIFIC_FULL_CSV     = f"{DATA_DIR}/LeeAML_Specific_Drug_Full_Info.csv"
SPECIFIC_SMILES_EMBED = f"{DATA_DIR}/LeeAML_Specific_Drug_Full_SMILES_Embedding.csv"
SPECIFIC_MOLFORMER    = f"{DATA_DIR}/LeeAML_Specific_Drug_Full_MolFormer_Embedding.csv"
SPECIFIC_CHEMBERTA    = f"{DATA_DIR}/LeeAML_Specific_Drug_Full_ChemBERTa_Embedding.csv"

COMMON_DRUGS_CSV      = f"{DATA_DIR}/leeaml_common_drugs.csv"
COMMON_CIDS_CSV       = f"{DATA_DIR}/leeaml_common_drug_cids.csv"
COMMON_FULL_CSV       = f"{DATA_DIR}/LeeAML_Common_Drug_Full_Info.csv"
COMMON_SMILES_EMBED   = f"{DATA_DIR}/LeeAML_Common_Drug_Full_SMILES_Embedding.csv"
COMMON_MOLFORMER      = f"{DATA_DIR}/LeeAML_Common_Drug_Full_MolFormer_Embedding.csv"
COMMON_CHEMBERTA      = f"{DATA_DIR}/LeeAML_Common_Drug_Full_ChemBERTa_Embedding.csv"

SMILES_INPUT_CSV  = f"{SMILES_AE_DIR}/test_smiles.csv"
SMILES_OUTPUT_CSV = f"{SMILES_AE_DIR}/test_LS.csv"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LeeAML-SPECIFIC DRUGS
# ══════════════════════════════════════════════════════════════════════════════

# %%
# Read the LeeAML-specific drugs list
leeaml_df = pd.read_csv(SPECIFIC_DRUGS_CSV, sep='\t',
                         usecols=[0, 1, 2],
                         names=['leeaml_specific', 'Alternate_Name', 'Target_Genes'],
                         header=0)
leeaml_df = leeaml_df.drop_duplicates(subset='leeaml_specific').reset_index(drop=True)
drug_names_list = leeaml_df['leeaml_specific'].tolist()
print(f"Total unique specific drugs: {len(drug_names_list)}")

# %%
# Get CID for each drug from PubChem
drug_cid_dict = {}
for drug in drug_names_list:
    rev_drug_name = drug.split(" ")[0]
    cids = pcp.get_cids(rev_drug_name, 'name', 'substance', list_return='flat')
    if cids:
        drug_cid_dict[drug] = cids[0]
        print(rev_drug_name, cids[0])
    else:
        drug_cid_dict[drug] = None
        print(f"No CID found for: {drug}")

# Save CIDs (with all original columns) to CSV
cid_df = leeaml_df.copy()
cid_df['cid'] = cid_df['leeaml_specific'].map(drug_cid_dict)
cid_df.to_csv(SPECIFIC_CIDS_CSV, index=False)
print(f"Saved CIDs: {SPECIFIC_CIDS_CSV}")
print(f"  Found CID: {cid_df['cid'].notna().sum()} / {len(cid_df)}")

# %%
# --- RESTART FROM HERE ---
# Load saved CIDs
cid_df = pd.read_csv(SPECIFIC_CIDS_CSV)
drug_cid_dict   = dict(zip(cid_df['leeaml_specific'], cid_df['cid']))
drug_names_list = cid_df['leeaml_specific'].tolist()
print(f"Loaded {len(drug_cid_dict)} specific drug CIDs")

# %%
# Get PubChem properties for each specific drug
final_drug_info_df = fetch_pubchem_properties(drug_names_list, drug_cid_dict)
print(f"Specific drugs with no SMILES: {final_drug_info_df['SMILES'].isnull().sum()}")
print(final_drug_info_df.loc[final_drug_info_df['SMILES'].isnull(), 'Name'].tolist())

final_drug_info_df = final_drug_info_df[final_drug_info_df['SMILES'].notnull()].reset_index(drop=True)
print(f"Specific drugs with SMILES: {len(final_drug_info_df)}")

# %%
# Compute physicochemical descriptors + graph features + fingerprints
results_df = preprocessing(final_drug_info_df, radius=2, n_bits=128)
ultimate_drug_info_df = pd.concat(
    [final_drug_info_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
print(f"Feature matrix shape: {ultimate_drug_info_df.shape}")
ultimate_drug_info_df.to_csv(SPECIFIC_FULL_CSV, index=False)
print(f"Saved: {SPECIFIC_FULL_CSV}")

# %%
# Check the saved file
input_drug_df = pd.read_csv(SPECIFIC_FULL_CSV)
print(input_drug_df.shape)
print(input_drug_df.columns.tolist())

# %%
# SMILES autoencoder embeddings — specific drugs
src = input_drug_df["SMILES"].apply(clean_and_validate_smiles).tolist()
smiles_embedding_df = run_smiles_autoencoder(src, SMILES_INPUT_CSV, SMILES_OUTPUT_CSV, SCRIPT_DIR)
pd.concat([input_drug_df, smiles_embedding_df], axis=1).to_csv(SPECIFIC_SMILES_EMBED, index=False)

# %%
# MolFormer embeddings — specific drugs
src = input_drug_df["SMILES"].apply(clean_and_validate_smiles).tolist()
molformer_df = get_molformer_embeddings(src, prefix="MF")
pd.concat([input_drug_df, molformer_df], axis=1).to_csv(SPECIFIC_MOLFORMER, index=False)
print(f"Saved: {SPECIFIC_MOLFORMER}")

# %%
# ChemBERTa embeddings — specific drugs
chemberta_df = get_chemberta_embeddings(src, prefix="CBT")
pd.concat([input_drug_df, chemberta_df], axis=1).to_csv(SPECIFIC_CHEMBERTA, index=False)
print(f"Saved: {SPECIFIC_CHEMBERTA}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LeeAML-COMMON DRUGS (shared with BeatAML)
# ══════════════════════════════════════════════════════════════════════════════

# %%
# Read the LeeAML common drugs list
leeaml_common_df = pd.read_csv(COMMON_DRUGS_CSV, sep='\t',
                                names=['LeeAML_Drug_Name', 'Alternate_Name', 'BeatAML_Drug_Name'],
                                header=0)
leeaml_common_df = leeaml_common_df.drop_duplicates(subset='BeatAML_Drug_Name').reset_index(drop=True)
common_drug_names_list = leeaml_common_df['BeatAML_Drug_Name'].tolist()
print(f"Total unique common drugs: {len(common_drug_names_list)}")

# %%
# Get CID for each common drug from PubChem (keyed on BeatAML name for target_gene_info.csv lookup)
common_drug_cid_dict = {}
for drug in common_drug_names_list:
    rev_drug_name = drug.split(" ")[0]
    cids = pcp.get_cids(rev_drug_name, 'name', 'substance', list_return='flat')
    if cids:
        common_drug_cid_dict[drug] = cids[0]
        print(rev_drug_name, cids[0])
    else:
        common_drug_cid_dict[drug] = None
        print(f"No CID found for: {drug}")

# Save common drug CIDs
common_cid_df = leeaml_common_df.copy()
common_cid_df['cid'] = common_cid_df['BeatAML_Drug_Name'].map(common_drug_cid_dict)
missing = common_cid_df[common_cid_df['cid'].isna()]['LeeAML_Drug_Name'].tolist()
print(f"  Drugs without CID ({len(missing)}): {missing}")
common_cid_df = common_cid_df[common_cid_df['cid'].notna()].reset_index(drop=True)
common_cid_df.to_csv(COMMON_CIDS_CSV, index=False)
print(f"Saved CIDs: {COMMON_CIDS_CSV}  ({len(common_cid_df)} drugs)")

# %%
# --- RESTART FROM HERE ---
# Load saved common CIDs
common_cid_df          = pd.read_csv(COMMON_CIDS_CSV)
common_drug_cid_dict   = dict(zip(common_cid_df['BeatAML_Drug_Name'], common_cid_df['cid']))
common_drug_names_list = common_cid_df['BeatAML_Drug_Name'].tolist()
print(f"Loaded {len(common_drug_cid_dict)} common drug CIDs")

# %%
# Get PubChem properties for each common drug
common_drug_info_df = fetch_pubchem_properties(common_drug_names_list, common_drug_cid_dict)
# Attach the LeeAML drug name for downstream R merge
common_drug_info_df = common_drug_info_df.merge(
    common_cid_df[['BeatAML_Drug_Name', 'LeeAML_Drug_Name']],
    left_on='Name', right_on='BeatAML_Drug_Name', how='left'
).drop(columns='BeatAML_Drug_Name')

print(f"Common drugs with no SMILES: {common_drug_info_df['SMILES'].isnull().sum()}")
print(common_drug_info_df.loc[common_drug_info_df['SMILES'].isnull(), 'Name'].tolist())
common_drug_info_df = common_drug_info_df[common_drug_info_df['SMILES'].notnull()].reset_index(drop=True)
print(f"Common drugs with SMILES: {len(common_drug_info_df)}")

# %%
# Compute physicochemical descriptors + graph features + fingerprints — common drugs
common_results_df = preprocessing(common_drug_info_df, radius=2, n_bits=128)
ultimate_common_drug_info_df = pd.concat(
    [common_drug_info_df.reset_index(drop=True), common_results_df.reset_index(drop=True)], axis=1)
print(f"Feature matrix shape: {ultimate_common_drug_info_df.shape}")
ultimate_common_drug_info_df.to_csv(COMMON_FULL_CSV, index=False)
print(f"Saved: {COMMON_FULL_CSV}")

# %%
# Check the saved file
input_common_drug_df = pd.read_csv(COMMON_FULL_CSV)
print(input_common_drug_df.shape)
print(input_common_drug_df.columns.tolist())

# %%
# SMILES autoencoder embeddings — common drugs
src = input_common_drug_df["SMILES"].apply(clean_and_validate_smiles).tolist()
smiles_embedding_df = run_smiles_autoencoder(src, SMILES_INPUT_CSV, SMILES_OUTPUT_CSV, SCRIPT_DIR)
pd.concat([input_common_drug_df, smiles_embedding_df], axis=1).to_csv(COMMON_SMILES_EMBED, index=False)

# %%
# MolFormer embeddings — common drugs
src = input_common_drug_df["SMILES"].apply(clean_and_validate_smiles).tolist()
molformer_common_df = get_molformer_embeddings(src, prefix="MF")
pd.concat([input_common_drug_df, molformer_common_df], axis=1).to_csv(COMMON_MOLFORMER, index=False)
print(f"Saved: {COMMON_MOLFORMER}")

# %%
# ChemBERTa embeddings — common drugs
chemberta_common_df = get_chemberta_embeddings(src, prefix="CBT")
pd.concat([input_common_drug_df, chemberta_common_df], axis=1).to_csv(COMMON_CHEMBERTA, index=False)
print(f"Saved: {COMMON_CHEMBERTA}")
