#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import pandas as pd
import matplotlib
import rdkit
import pubchempy as pcp
#import modred
import huggingface_hub
import transformers
import torch
import os
from chemspipy import ChemSpider
import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

#Get API Key from environment variable and create ChemSpider object
#cspider_api_key = os.environ['CHEMSPIDER_API_KEY']
#cs = ChemSpider(cspider_api_key)


# %%
#Read the data file which contains the name of all drugs
combined_df = pd.read_csv("../BeatAML/Data/beataml_probit_curve_fits_v4_dbgap.txt", delimiter="\t")
drug_names_list = combined_df['inhibitor'].unique().tolist()


# %%
#Get the names of all the drugs and put it in a csv file
fp = open("../Data/Drug_Names.csv","w")
fp.write("Drug_Names"+"\n")
for drug in drug_names_list:
    outstr = drug+"\n"
    fp.write(outstr)
fp.close()

print(len(drug_names_list))


# %%
#Get the drug - and its corresponidng cids into a dictionary
drug_cid_dict = {}
for drug in drug_names_list:
    rev_drug_name = drug.split(" ")[0]
    cids = pcp.get_cids(rev_drug_name,'name','substance',list_return='flat')
    #cids = cs.search(rev_drug_name)
    if (len(cids)>0):
        #Put the original name of the drug with the first cid encountered in Pubchem
        drug_cid_dict[drug] = cids[0]
        print(rev_drug_name, cids[0])
    else:
        drug_cid_dict[drug] = []

# %%
#Manually add the cids for drugs which were not found automatically
drug_cid_dict['Perhexiline maleate']=4746
drug_cid_dict['Ralimetinib (LY2228820)']=11539025
drug_cid_dict['Ranolazine']=56959
drug_cid_dict['XMD 8-87']=446339
drug_cid_dict['Everolimus']=85201

print(drug_cid_dict)

# %%
#Get some properties of all the drugs using their cids
final_drug_info = []
for drug in drug_names_list:
    if (drug_cid_dict[drug]!=[]):
        tmp_compound_dict = pcp.get_properties(['InChIKey','ConnectivitySMILES','XLogP','MolecularWeight'],
                                           drug_cid_dict[drug],'cid')
        tmp_compound_dict[0]['Name'] = drug
        final_drug_info.append(tmp_compound_dict[0])
    else:
        final_drug_info.append({'CID':None, 'ConnectivitySMILES':None, 'InChIKey': None, 'XLogP':None,
                               'MolecularWeight': None, 'Name': drug})

# %%
#Create a dataframe from the final drug info
final_drug_info_df = pd.DataFrame(final_drug_info)
print(final_drug_info_df["CID"].isnull().sum())

#Which drugs do not have a CID
no_cid_drugs = final_drug_info_df[final_drug_info_df["CID"].isnull()]["Name"].to_list()
print(no_cid_drugs)

#Print the list of columns in the final drug info dataframe
print(final_drug_info_df.columns)

# %% [markdown]
# ## Columns not to consider

# %%
useless_cols = [   
    
    'MaxPartialCharge', 
    # Nan data
    'BCUT2D_MWHI',
    'BCUT2D_MWLOW',
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI',
    'BCUT2D_MRLOW',

    # Constant data
    'NumRadicalElectrons',
    'SMR_VSA8',
    'SlogP_VSA9',
    'fr_barbitur',
    'fr_benzodiazepine',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_isothiocyan',
    'fr_lactam',
    'fr_nitroso',
    'fr_prisulfonamd',
    'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'NumValenceElectrons',
    'Chi0',
    'Chi0n',
    'Chi0v',
    'Chi1',
    'Chi1n',
    'Chi1v',
    'Chi2n',
    'Kappa1',
    'LabuteASA',
    'HeavyAtomCount',
    'MolMR',
    'Chi3n',
    'BertzCT',
    'Chi2v',
    'Chi4n',
    'HallKierAlpha',
    'Chi3v',
    'Chi4v',
    'MinAbsPartialCharge',
    'MinPartialCharge',
    'MaxAbsPartialCharge',
    'FpDensityMorgan2',
    'FpDensityMorgan3',
    'Phi',
    'Kappa3',
    'fr_nitrile',
    'SlogP_VSA6',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'fr_benzene',
    'VSA_EState6',
    'NOCount',
    'fr_C_O',
    'fr_C_O_noCOO',
    'NumHDonors',
    'fr_amide',
    'fr_Nhpyrrole',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_COO2',
    'fr_halogen',
    'fr_diazo',
    'fr_nitro_arom',
    'fr_phos_ester'
]

# %% [markdown]
# ## Make SMILES canonical

# %%
RDKIT_AVAILABLE = True
from rdkit import Chem

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    # Check for any bad patterns
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    # Try to parse with RDKit if available
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles



# %%
def compute_all_descriptors(smiles):
    #Get all descriptors from RDKit
    # Exclude descriptors that are in useless_cols
    # Return a list of descriptor values in the same order as desc_names
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]

def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    n_nodes = G.number_of_nodes()

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))
    graph_feats['num_chains'].append(len(list(nx.chain_decomposition(G))))
    graph_feats['clustering_coefficients'].append(nx.average_clustering(G))
    #graph_feats['harmonic_diameter'].append(nx.harmonic_diameter(G))
    graph_feats['avg_degree_centrality'].append(np.array(list(nx.degree_centrality(G).values())).mean())
    graph_feats['avg_eigen_centrality'].append(np.array(list(nx.katz_centrality(G).values())).mean())
    graph_feats['avg_betweenness_centrality'].append(np.array(list(nx.betweenness_centrality(G).values())).mean())
    graph_feats['avg_load_centrality'].append(np.array(list(nx.load_centrality(G).values())).mean())
    graph_feats['wiener_index'].append(nx.wiener_index(G))
    max_degree = max([d for n, d in G.degree()]) if n_nodes > 0 else 0
    graph_feats['max_degree'].append(max_degree)
    # Closeness centrality
    if nx.is_connected(G):
        closeness = list(nx.closeness_centrality(G).values())
        closeness_val = np.mean(closeness) if closeness else 0
        graph_feats['closeness_mean'].append(closeness_val)
    else:
        graph_feats['closeness_mean'].append(0)

    # Katz centrality
    try:
        katz = list(nx.katz_centrality(G, max_iter=1000).values())
        graph_feats['katz_centrality_std'].append(np.std(katz) if len(katz) > 1 else 0)
    except:
        graph_feats['katz_centrality_std'].append(0)

    # Centrality measures
    betweenness = list(nx.betweenness_centrality(G).values())
    if betweenness:
        betweenness = [b for b in betweenness if np.isfinite(b)]
        graph_feats['betweenness_mean'].append(np.mean(betweenness) if betweenness else 0)
        graph_feats['betweenness_std'].append(np.std(betweenness) if len(betweenness) > 1 else 0)
    else:
        graph_feats['betweenness_mean'].append(0)
        graph_feats['betweenness_std'].append(0)

    # Eigenvector centrality
    try:
        eigenvector = list(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6).values())
        eigenvector = [e for e in eigenvector if np.isfinite(e)]
        graph_feats['eigenvector_mean'].append(np.mean(eigenvector) if eigenvector else 0)
    except:
        graph_feats['eigenvector_mean'].append(0)
                
    # Ring analysis
    cycles = list(nx.cycle_basis(G))
    cycle_lengths = [len(cycle) for cycle in cycles]
    graph_feats['ring_4'].append(sum(1 for length in cycle_lengths if length == 4))

    # Ring 1,2,3,5
    graph_feats['ring_1'].append(sum(1 for length in cycle_lengths if length == 1))
    graph_feats['ring_2'].append(sum(1 for length in cycle_lengths if length == 2))
    graph_feats['ring_3'].append(sum(1 for length in cycle_lengths if length == 3))
    graph_feats['ring_5'].append(sum(1 for length in cycle_lengths if length == 5))


    # Atom-specific features
    try:
        # Atom type distribution
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_counts = Counter(atom_types)
                    
        # Heteroatom ratio
        total_atoms = len(atom_types)
        hetero_atoms = total_atoms - atom_counts.get('C', 0)
        graph_feats['heteroatom_ratio'].append(hetero_atoms / total_atoms if total_atoms > 0 else 0)
    except Exception:
        graph_feats['heteroatom_ratio'].append(0)


    # Get aromatic rings
    try:
        aromatic_rings = [ring for ring in cycles if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]
        graph_feats['num_aromatic_rings'].append(len(aromatic_rings))
    except Exception:
        graph_feats['num_aromatic_rings'].append(0)


    # Get non-aromatic rings
    try:
        non_aromatic_rings = [ring for ring in cycles if not any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]
        graph_feats['num_non_aromatic_rings'].append(len(non_aromatic_rings))
    except Exception:
        graph_feats['num_non_aromatic_rings'].append(0)

    # Get average of number of oxygen, carbon, nitrogen, and sulfur atoms
    try:
        num_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')/n_nodes if n_nodes > 0 else 0
        num_sulfur = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')/n_nodes if n_nodes > 0 else 0
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')/n_nodes if n_nodes > 0 else 0
        num_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')/n_nodes if n_nodes > 0 else 0
        graph_feats['average_oxygen'].append(num_oxygen)
        graph_feats['average_nitrogen'].append(num_nitrogen)
        graph_feats['average_carbon'].append(num_carbon)
        graph_feats['average_sulphur'].append(num_sulfur)
    except Exception:
        graph_feats['average_oxygen'].append(0)
        graph_feats['average_nitrogen'].append(0)
        graph_feats['average_carbon'].append(0)
        graph_feats['average_sulphur'].append(0)


    # Get average of number of single and double bonds
    try:
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.SINGLE)/n_edges if n_edges > 0 else 0
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)/n_edges if n_edges > 0 else 0
        graph_feats['num_single_bonds'].append(single_bonds)
        graph_feats['num_double_bonds'].append(double_bonds)
    except Exception:
        graph_feats['num_single_bonds'].append(0)
        graph_feats['num_double_bonds'].append(0)
    
    return graph_feats


# %%
def preprocessing(df, radius = 2, n_bits = 128):
    all_smiles = df['SMILES'].apply(lambda x: clean_and_validate_smiles(x)).to_list()
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    rdkit_descriptors = [compute_all_descriptors(smi) for smi in all_smiles]

    #Get networkx graph based features
    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': [], 'num_chains': [], 'clustering_coefficients': [], 'avg_degree_centrality': [], 'avg_eigen_centrality': [], 'avg_betweenness_centrality': [],
                   'avg_load_centrality': [], 'wiener_index': [], 'max_degree': [], 'closeness_mean': [], 'katz_centrality_std': [],
                   'betweenness_mean': [], 'betweenness_std': [], 'eigenvector_mean': [], 'ring_4': [], 'heteroatom_ratio': [],
                   'ring_1': [], 'ring_2': [], 'ring_3': [], 'ring_5': [], 'num_aromatic_rings': [], 'num_non_aromatic_rings': [], 
                   'average_carbon': [], 'average_oxygen': [], 'average_sulphur': [], 'average_nitrogen': [], 'num_single_bonds': [], 
                   'num_double_bonds': []}
    
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)

    #Get smiles list and compute fingerprints
    smiles_list = df['SMILES'].to_list()
    fingerprints = []
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    #Compute fingerprints
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints
            morgan_fp = generator.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol) # type: ignore

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)
        else:
            fingerprints.append(np.zeros(n_bits  + 167))

    #Convert fingerprints to a pandas dataframe and add column names
    fp_list = np.array(fingerprints)
    mfp_descriptor_names = [f'Morgan_{i}' for i in range(n_bits)] + [f'MACCS_{i}' for i in range(167)]

    #Concatenate all descriptors    
    result = pd.concat(
        [
            pd.DataFrame(rdkit_descriptors, columns=desc_names),
            pd.DataFrame(graph_feats),
            pd.DataFrame(fp_list, columns = mfp_descriptor_names)
        ],
        axis=1
    )

    
    #Replace inf with nan
    result = result.replace([-np.inf, np.inf], np.nan)
    #result = result.replace(to_replace=[None], value=np.nan, inplace=True)
    return result

# %%
#Convert a list of dictionaries into a dataframe
final_drug_info_df = pd.DataFrame(final_drug_info)
final_drug_info_df.columns = ['CID','MolecularWeight','SMILES','InChIKey','XLogP','Name']

#Remove rows with no SMILES
final_drug_info_df = final_drug_info_df[final_drug_info_df["SMILES"].notnull()]
print(final_drug_info_df.shape)

#Calculate all descriptors and fingerprints
results_df = preprocessing(final_drug_info_df, radius=2, n_bits=128)

#Concatenate the results with the final drug info dataframe
ultimate_drug_info_df = pd.concat([final_drug_info_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
print(ultimate_drug_info_df.shape)

ultimate_drug_info_df.to_csv("../Data/Drug_Full_Info.csv",sep=",",index=False)

# %% [markdown]
# ## Use our teacher-forcing LSTM model to get embedding representation for drugs

# %%
import numpy as np
import pandas as pd
import matplotlib
import rdkit
import pubchempy as pcp
import os

# %%
#To get the embedding representation for drugs using SMILES as input
input_drug_df = pd.read_csv("../Data/Drug_Full_Info.csv")
print(input_drug_df)
print(input_drug_df.columns.tolist())

# %%
#Convert the drug smiles into format ingestible for the ls_generator
src = input_drug_df["SMILES"].apply(lambda x: clean_and_validate_smiles(x)).tolist()
smiles_df = [src,src]
smiles_df = pd.DataFrame(smiles_df)
smiles_df = pd.DataFrame.transpose(smiles_df)
smiles_df.columns = ['src','trg']
smiles_df.to_csv("../Data/SMILES_Autoencoder/test_smiles.csv", index=False)

# %%
command = "python ls_generator2.py --input test_smiles.csv --output test_LS.csv"
os.system(command)

# %%
#Read the SMILES presentation generated and write the same
smiles_embedding_df = pd.read_csv("../Data/SMILES_Autoencoder/test_LS.csv")
final_drug_df = pd.concat([input_drug_df,smiles_embedding_df],axis=1)
final_drug_df.to_csv("../Results/Drug_Full_SMILES_Embedding.csv",index=False)

# %%
#Get the drug's representation using the MolFormer model from huggingface
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

src = input_drug_df["SMILES"].apply(lambda x: clean_and_validate_smiles(x)).tolist()
inputs = tokenizer(src, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

outputs.pooler_output.shape

smiles_molformer_embeddings = outputs.pooler_output.numpy()
smiles_molformer_embeddings_df = pd.DataFrame(smiles_molformer_embeddings)

#Add mf to column names 
no_of_columns = smiles_molformer_embeddings_df.shape[1]
molformer_column_names = ["MF_"+str(i) for i in range(no_of_columns)]
smiles_molformer_embeddings_df.columns = molformer_column_names

final_drug_df = pd.concat([input_drug_df,smiles_molformer_embeddings_df],axis=1)
final_drug_df.to_csv("../Data/Drug_Full_MolFormer_Embedding.csv",index=False)

# %%
## Perform the same for ChemBERTa
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM", trust_remote_code=True)

src = input_drug_df["SMILES"].apply(lambda x: clean_and_validate_smiles(x)).tolist()
inputs = tokenizer(src, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

#From the output select the last hidden state and the end of sequence token embedding 
outputs.pooler_output.shape

smiles_chemberta_embeddings = outputs.pooler_output.numpy()
smiles_chemberta_embeddings_df = pd.DataFrame(smiles_chemberta_embeddings)

#Add mf to column names 
no_of_columns = smiles_chemberta_embeddings_df.shape[1]
chemberta_column_names = ["CBT_"+str(i) for i in range(no_of_columns)]
smiles_chemberta_embeddings_df.columns = chemberta_column_names

final_drug_df = pd.concat([input_drug_df,smiles_chemberta_embeddings_df],axis=1)
final_drug_df.to_csv("../Data/Drug_Full_ChemBERTa_Embedding.csv",index=False)
