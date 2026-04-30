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
# SECTION 1 — FIMM-AML PATIENT FEATURE MATRIX
# Build the per-sample feature table from FIMMAML_Set_with_Onco_Var_Expr_PA_CTS.csv
# using FIMMAML_Metadata.csv to identify column domains.
#
# FIMM-AML column layout (0-based):
#   Col  0        : Sample_ID (Id)
#   Cols 1–817    : oncogene + high-variance gene expression (817 genes)
#   Cols 818–871  : pathway enrichment scores (54 pathways)
#   Cols 872–877  : AML cell-type ssGSEA scores (6 types)
#   Cols 878–891  : WGCNA module ssGSEA scores (14 modules)
#   Cols 892–948  : mutation indicators (57 genes, prefix MUT_)
# ══════════════════════════════════════════════════════════════════════════════

# %%
fimmaml_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Set_with_Onco_Var_Expr_PA_CTS.csv",
                          sep="\t", low_memory=False)
print(f"FIMM-AML patient feature matrix: {fimmaml_df.shape}")

# %%
# Use metadata to identify column domains
meta_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Metadata.csv")
meta_dict = dict(zip(meta_df["Column_Label"], meta_df["Type"]))

all_columns = list(fimmaml_df.columns)

# Sample ID column
sample_names = all_columns[0]          # 'Sample_ID'

# Gene expression columns (Gene_Expr type)
gene_names = [c for c in all_columns if meta_dict.get(c, "") == "Gene_Expr"]
gene_names_expr = [g + "_Expr" for g in gene_names]   # add _Expr suffix to match BeatAML convention
col_rename = {g: g + "_Expr" for g in gene_names}
fimmaml_df = fimmaml_df.rename(columns=col_rename)
all_columns = list(fimmaml_df.columns)

# Pathway enrichment columns
pathway_names = [c for c in all_columns if meta_dict.get(c, "") == "Pathway"]

# Cell-type ssGSEA columns
celltype_names = [c for c in all_columns if meta_dict.get(c, "") == "CellType"]

# WGCNA module ssGSEA columns
module_names = [c for c in all_columns if meta_dict.get(c, "") == "Module"]

# Mutation columns: MUT_GENE -> GENE_Mut to match BeatAML convention
mut_raw = [c for c in all_columns if c.startswith("MUT_")]
mut_rename = {c: c.replace("MUT_", "") + "_Mut" for c in mut_raw}
fimmaml_df = fimmaml_df.rename(columns=mut_rename)
all_columns = list(fimmaml_df.columns)
mut_names = list(mut_rename.values())

print(f"  Gene expression columns : {len(gene_names_expr)}")
print(f"  Pathway columns         : {len(pathway_names)}")
print(f"  Cell-type columns       : {len(celltype_names)}")
print(f"  Module columns          : {len(module_names)}")
print(f"  Mutation columns        : {len(mut_names)}")

# Full set of patient features
all_cols_of_interest = [sample_names] + gene_names_expr + pathway_names + celltype_names + module_names + mut_names
print(f"Feature columns selected: {len(all_cols_of_interest)}")

# %%
fimmaml_feature_df = fimmaml_df[all_cols_of_interest].copy()
fimmaml_feature_df = remove_bracket_columns(fimmaml_feature_df)
# Normalise sample ID column name to lowercase for consistent merge key
fimmaml_feature_df = fimmaml_feature_df.rename(columns={"Sample_ID": "sample_id"})
print(f"FIMM-AML feature dataframe: {fimmaml_feature_df.shape}")
print(fimmaml_feature_df.columns.tolist()[:10], "...")

# Save clean feature matrix
fimmaml_feature_df.to_pickle("../Data/FIMM-AML/FIMMAML_Set_Var_Mod.pkl", compression="zip")
fimmaml_feature_df.to_csv("../Data/FIMM-AML/FIMMAML_Set_Var_Mod.csv", index=False, sep="\t")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ALL DRUGS (all 78 FIMM-AML drugs are common with BeatAML)
# Drug-response source : FIMMAML_Set_with_AUC.csv   (DSS response variable)
# Drug-feature sources : FIMMAML_Drug_Full_{SMILES,Info,MolFormer,ChemBERTa}_Embedding.csv
# Merge key            : drug files 'FIMM_Name' == AUC file 'inhibitor'
# ══════════════════════════════════════════════════════════════════════════════

# %%
auc_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Set_with_AUC.csv",
                      sep="\t", header="infer")
auc_df = remove_bracket_columns(auc_df)
print(f"Drug-patient DSS pairs: {auc_df.shape}")

plt.figure()
plt.hist(auc_df["DSS"], bins=np.linspace(0, 50, 100))
plt.title("All drugs — DSS distribution")
plt.xlabel("DSS"); plt.ylabel("Count")
plt.savefig("../Data/FIMM-AML/FIMMAML_DSS_distribution.pdf")
plt.close()

# %%
# Merge DSS table with patient features on sample_id
drug_patient_df = pd.merge(auc_df, fimmaml_feature_df, on="sample_id", how="inner")
print(f"After merging with patient features: {drug_patient_df.shape}")

# %%
# ── SMILES autoencoder embedding ──────────────────────────────────────────────
smiles_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Drug_Full_SMILES_Embedding.csv",
                         header="infer")
# Merge key: FIMM_Name (drug file) == inhibitor (AUC/patient file)
smiles_df = smiles_df.rename(columns={"FIMM_Name": "inhibitor"})

final_smiles_df = pd.merge(smiles_df, drug_patient_df, on="inhibitor", how="inner")
print(f"SMILES embedding + patient: {final_smiles_df.shape}")
final_smiles_df.to_pickle(
    "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_Embedding_Patient_Info.pkl",
    compression="zip")

# %%
# ── Physicochemical (PC) descriptors + fingerprints ───────────────────────────
pc_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Drug_Full_Info.csv", header="infer")
pc_df = pc_df.rename(columns={"FIMM_Name": "inhibitor"})

final_pc_df = pd.merge(pc_df, drug_patient_df, on="inhibitor", how="inner")
print(f"PC features + patient: {final_pc_df.shape}")
final_pc_df.to_pickle(
    "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",
    compression="zip")

# %%
# ── MolFormer embedding ───────────────────────────────────────────────────────
molformer_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Drug_Full_MolFormer_Embedding.csv",
                            header="infer")
molformer_df = molformer_df.rename(columns={"FIMM_Name": "inhibitor"})

final_molformer_df = pd.merge(molformer_df, drug_patient_df, on="inhibitor", how="inner")
print(f"MolFormer + patient: {final_molformer_df.shape}")
final_molformer_df.to_pickle(
    "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",
    compression="zip")

# %%
# ── ChemBERTa embedding ───────────────────────────────────────────────────────
chemberta_df = pd.read_csv("../Data/FIMM-AML/FIMMAML_Drug_Full_ChemBERTa_Embedding.csv",
                            header="infer")
chemberta_df = chemberta_df.rename(columns={"FIMM_Name": "inhibitor"})

final_chemberta_df = pd.merge(chemberta_df, drug_patient_df, on="inhibitor", how="inner")
print(f"ChemBERTa + patient: {final_chemberta_df.shape}")
final_chemberta_df.to_pickle(
    "../Data/FIMM-AML/FIMMAML_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",
    compression="zip")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Summary
# ══════════════════════════════════════════════════════════════════════════════

# %%
print("\n=== Output summary ===")
for label, df in [
    ("SMILES embed",  final_smiles_df),
    ("PC / MFP",      final_pc_df),
    ("MolFormer",     final_molformer_df),
    ("ChemBERTa",     final_chemberta_df),
]:
    print(f"  {label}: {df.shape}")
