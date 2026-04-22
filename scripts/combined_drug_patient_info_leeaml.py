# %%
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# %%
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# %%
# ─── Helpers ──────────────────────────────────────────────────────────────────

def remove_bracket_columns(df):
    """Strip '[...]' tags from column names and replace spaces with underscores."""
    new_columns = []
    for col in df.columns:
        new_col = pd.Series(col).str.replace(r'\[.*?\]', '', regex=True).values[0].strip()
        new_col = new_col.replace(' ', '_')
        new_columns.append(new_col)
    df.columns = new_columns
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LeeAML PATIENT FEATURE MATRIX
# Build the per-sample feature table from LeeAML_Set_with_Onco_Var_Expr_PA_CTS.csv
# mirroring the column-labelling logic in preprocess_patients.py.
#
# LeeAML column layout (1-based):
#   Col  1        : sample_id
#   Cols 2–792    : oncogene + high-variance gene expression (791 genes)
#   Cols 793–846  : pathway enrichment scores (54 pathways)
#   Cols 847–852  : AML cell-type ssGSEA scores (6 types)
#   Cols 853–866  : WGCNA module ssGSEA scores (14 modules)
# No clinical traits and no mutation data exist in LeeAML.
# ══════════════════════════════════════════════════════════════════════════════

# %%
leeaml_df = pd.read_csv("../Data/leeaml/LeeAML_Set_with_Onco_Var_Expr_PA_CTS.csv",
                         sep="\t", low_memory=False)
print(f"LeeAML patient feature matrix: {leeaml_df.shape}")

# %%
all_columns = list(leeaml_df.columns)

# Sample ID column
sample_names = all_columns[0]          # 'sample_id'

# Gene expression columns (cols 2–792, 0-based indices 1–791)
gene_names = all_columns[1:792]
gene_names = [g + "_Expr" for g in gene_names]   # add _Expr suffix to match BeatAML convention
all_columns[1:792] = gene_names

# Pathway enrichment columns (cols 793–846, 0-based indices 792–845)
pathway_names = all_columns[792:846]

# Cell-type + module ssGSEA columns (cols 847–866, 0-based indices 846–865)
cts_names = all_columns[846:866]

# Apply updated column names to the dataframe
leeaml_df.columns = all_columns

# Full set of patient features to carry forward (no clinical / mutation columns)
all_cols_of_interest = [sample_names] + gene_names + pathway_names + cts_names
print(f"Feature columns selected: {len(all_cols_of_interest)}")

# %%
leeaml_feature_df = leeaml_df[all_cols_of_interest].copy()
leeaml_feature_df = remove_bracket_columns(leeaml_feature_df)
print(f"LeeAML feature dataframe: {leeaml_feature_df.shape}")
print(leeaml_feature_df.columns.tolist()[:10], "...")

# Save clean feature matrix
leeaml_feature_df.to_pickle("../Data/leeaml/LeeAML_Set_Var_Mod.pkl", compression="zip")
leeaml_feature_df.to_csv("../Data/leeaml/LeeAML_Set_Var_Mod.csv", index=False, sep="\t")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMMON DRUGS (shared with BeatAML)
# Drug-response source : Common_LeeAML_Set_with_AUC.csv
# Drug-feature sources : LeeAML_Common_Drug_Full_{SMILES,Info,MolFormer,ChemBERTa}_Embedding.csv
# Merge key            : AUC file  'inhibitor' == drug file 'LeeAML_Drug_Name'
# ══════════════════════════════════════════════════════════════════════════════

# %%
common_auc_df = pd.read_csv("../Data/leeaml/Common_LeeAML_Set_with_AUC.csv",
                             sep="\t", header='infer')
common_auc_df = remove_bracket_columns(common_auc_df)
print(f"Common drug-patient AUC pairs: {common_auc_df.shape}")

plt.figure()
plt.hist(common_auc_df["auc"], bins=np.linspace(0, 1000, 100))
plt.title("Common drugs — AUC distribution")
plt.xlabel("AUC"); plt.ylabel("Count")
plt.savefig("../Data/leeaml/Common_LeeAML_AUC_distribution.pdf")
plt.close()

# %%
# Merge AUC table with patient features on sample_id
common_drug_patient_df = pd.merge(common_auc_df, leeaml_feature_df,
                                   on="sample_id", how="inner")
print(f"After merging with patient features: {common_drug_patient_df.shape}")

# %%
# ── SMILES autoencoder embedding ──────────────────────────────────────────────
common_smiles_df = pd.read_csv("../Data/leeaml/LeeAML_Common_Drug_Full_SMILES_Embedding.csv",
                                header='infer')
# Merge key: LeeAML_Drug_Name (drug file) == inhibitor (AUC/patient file)
common_smiles_df = common_smiles_df.rename(columns={"LeeAML_Drug_Name": "inhibitor"})

final_common_smiles_df = pd.merge(common_smiles_df, common_drug_patient_df,
                                   on="inhibitor", how="inner")
print(f"Common SMILES embedding + patient: {final_common_smiles_df.shape}")
final_common_smiles_df.to_pickle(
    "../Data/leeaml/LeeAML_Common_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    compression="zip")

# %%
# ── Physicochemical (PC) descriptors + fingerprints ───────────────────────────
common_pc_df = pd.read_csv("../Data/leeaml/LeeAML_Common_Drug_Full_Info.csv", header='infer')
common_pc_df = common_pc_df.rename(columns={"LeeAML_Drug_Name": "inhibitor"})

final_common_pc_df = pd.merge(common_pc_df, common_drug_patient_df,
                               on="inhibitor", how="inner")
print(f"Common PC features + patient: {final_common_pc_df.shape}")
final_common_pc_df.to_pickle(
    "../Data/leeaml/LeeAML_Common_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    compression="zip")

# %%
# ── MolFormer embedding ───────────────────────────────────────────────────────
common_molformer_df = pd.read_csv("../Data/leeaml/LeeAML_Common_Drug_Full_MolFormer_Embedding.csv",
                                   header='infer')
common_molformer_df = common_molformer_df.rename(columns={"LeeAML_Drug_Name": "inhibitor"})

final_common_molformer_df = pd.merge(common_molformer_df, common_drug_patient_df,
                                      on="inhibitor", how="inner")
print(f"Common MolFormer + patient: {final_common_molformer_df.shape}")
final_common_molformer_df.to_pickle(
    "../Data/leeaml/LeeAML_Common_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
    compression="zip")

# %%
# ── ChemBERTa embedding ───────────────────────────────────────────────────────
common_chemberta_df = pd.read_csv("../Data/leeaml/LeeAML_Common_Drug_Full_ChemBERTa_Embedding.csv",
                                   header='infer')
common_chemberta_df = common_chemberta_df.rename(columns={"LeeAML_Drug_Name": "inhibitor"})

final_common_chemberta_df = pd.merge(common_chemberta_df, common_drug_patient_df,
                                      on="inhibitor", how="inner")
print(f"Common ChemBERTa + patient: {final_common_chemberta_df.shape}")
final_common_chemberta_df.to_pickle(
    "../Data/leeaml/LeeAML_Common_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    compression="zip")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LEEAML-SPECIFIC DRUGS
# Drug-response source : Specific_LeeAML_Set_with_AUC.csv
# Drug-feature sources : LeeAML_Specific_Drug_Full_{SMILES,Info,MolFormer,ChemBERTa}_Embedding.csv
# Merge key            : both use 'Name' / 'inhibitor' as the specific drug name
#                        (no LeeAML_Drug_Name column in specific drug files)
# ══════════════════════════════════════════════════════════════════════════════

# %%
specific_auc_df = pd.read_csv("../Data/leeaml/Specific_LeeAML_Set_with_AUC.csv",
                               sep="\t", header='infer')
specific_auc_df = remove_bracket_columns(specific_auc_df)
print(f"Specific drug-patient AUC pairs: {specific_auc_df.shape}")

plt.figure()
plt.hist(specific_auc_df["auc"], bins=np.linspace(0, 1000, 100))
plt.title("Specific drugs — AUC distribution")
plt.xlabel("AUC"); plt.ylabel("Count")
plt.savefig("../Data/leeaml/Specific_LeeAML_AUC_distribution.pdf")
plt.close()

# %%
# Merge AUC table with patient features on sample_id
specific_drug_patient_df = pd.merge(specific_auc_df, leeaml_feature_df,
                                     on="sample_id", how="inner")
print(f"After merging with patient features: {specific_drug_patient_df.shape}")

# %%
# ── SMILES autoencoder embedding ──────────────────────────────────────────────
specific_smiles_df = pd.read_csv("../Data/leeaml/LeeAML_Specific_Drug_Full_SMILES_Embedding.csv",
                                  header='infer')
# In specific files 'Name' is the drug name, matching 'inhibitor' in the AUC file
specific_smiles_df = specific_smiles_df.rename(columns={"Name": "inhibitor"})

final_specific_smiles_df = pd.merge(specific_smiles_df, specific_drug_patient_df,
                                     on="inhibitor", how="inner")
print(f"Specific SMILES embedding + patient: {final_specific_smiles_df.shape}")
final_specific_smiles_df.to_pickle(
    "../Data/leeaml/LeeAML_Specific_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    compression="zip")

# %%
# ── Physicochemical (PC) descriptors + fingerprints ───────────────────────────
specific_pc_df = pd.read_csv("../Data/leeaml/LeeAML_Specific_Drug_Full_Info.csv", header='infer')
specific_pc_df = specific_pc_df.rename(columns={"Name": "inhibitor"})

final_specific_pc_df = pd.merge(specific_pc_df, specific_drug_patient_df,
                                 on="inhibitor", how="inner")
print(f"Specific PC features + patient: {final_specific_pc_df.shape}")
final_specific_pc_df.to_pickle(
    "../Data/leeaml/LeeAML_Specific_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    compression="zip")

# %%
# ── MolFormer embedding ───────────────────────────────────────────────────────
specific_molformer_df = pd.read_csv(
    "../Data/leeaml/LeeAML_Specific_Drug_Full_MolFormer_Embedding.csv", header='infer')
specific_molformer_df = specific_molformer_df.rename(columns={"Name": "inhibitor"})

final_specific_molformer_df = pd.merge(specific_molformer_df, specific_drug_patient_df,
                                        on="inhibitor", how="inner")
print(f"Specific MolFormer + patient: {final_specific_molformer_df.shape}")
final_specific_molformer_df.to_pickle(
    "../Data/leeaml/LeeAML_Specific_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
    compression="zip")

# %%
# ── ChemBERTa embedding ───────────────────────────────────────────────────────
specific_chemberta_df = pd.read_csv(
    "../Data/leeaml/LeeAML_Specific_Drug_Full_ChemBERTa_Embedding.csv", header='infer')
specific_chemberta_df = specific_chemberta_df.rename(columns={"Name": "inhibitor"})

final_specific_chemberta_df = pd.merge(specific_chemberta_df, specific_drug_patient_df,
                                        on="inhibitor", how="inner")
print(f"Specific ChemBERTa + patient: {final_specific_chemberta_df.shape}")
final_specific_chemberta_df.to_pickle(
    "../Data/leeaml/LeeAML_Specific_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    compression="zip")

# %%
# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Output summary ===")
for label, df in [
    ("Common  | SMILES embed",   final_common_smiles_df),
    ("Common  | PC / MFP",       final_common_pc_df),
    ("Common  | MolFormer",      final_common_molformer_df),
    ("Common  | ChemBERTa",      final_common_chemberta_df),
    ("Specific| SMILES embed",   final_specific_smiles_df),
    ("Specific| PC / MFP",       final_specific_pc_df),
    ("Specific| MolFormer",      final_specific_molformer_df),
    ("Specific| ChemBERTa",      final_specific_chemberta_df),
]:
    print(f"  {label}: {df.shape}")
