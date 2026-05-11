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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import os
import pickle

# %%
#Run the command to generate the train and test pkl files
command = "python preprocess_patients.py"
os.system(command)


# %%
#Create a function to take dataframe columns and remove strings with '[]' in any column name while retaining rest of the string using regex
def remove_bracket_columns(df):
    #Replace string within '[]' with empty string
    new_columns = []
    for col in df.columns:
        new_col = pd.Series(col).str.replace(r'\[.*?\]', '', regex=True).values[0].strip()
        new_col = new_col.replace(' ','_')
        new_columns.append(new_col)
    df.columns = new_columns
    return df


# %%
#Load the training and test set with gene expression, clinical traits, pathway activations, celltype and module activations, mutations
train_feature_df = pd.read_pickle("../Data/Training_Set_Var_Mod.pkl",compression="zip")
test_feature_df = pd.read_pickle("../Data/Test_Set_Var_Mod.pkl",compression="zip")

#Clearn the train and test features
train_feature_df = remove_bracket_columns(train_feature_df)
test_feature_df = remove_bracket_columns(test_feature_df)
        
print(train_feature_df.shape)
print(test_feature_df.shape)
print(train_feature_df.columns)

# %%
#Find columns related to pathways
pathway_columns = [col for col in train_feature_df.columns if 'Signaling' in col]
print(pathway_columns)

# %%
#Load the training and test drug, patient combination file
train_drug_patient_df = pd.read_csv("../Data/Revised_Training_Set_with_IC50.csv.gz",compression="gzip",header='infer',sep="\t")
test_drug_patient_df = pd.read_csv("../Data/Revised_Test_Set_with_IC50.csv.gz",compression="gzip",header="infer",sep="\t")
print(train_drug_patient_df.shape)
print(test_drug_patient_df.shape)

#This part of code was not looking at gene expression profiles
#rev_train_feature_df = train_feature_df.iloc[:,[0]+[i for i in range(22844,23322)]]
#rev_test_feature_df = test_feature_df.iloc[:,[0]+[i for i in range(22844,23322)]]

#We now focus on oncogenes, pathway enrichments, module enrichments, mutations in genes, mutation classes
rev_train_drug_patient_df = remove_bracket_columns(train_drug_patient_df)
rev_test_drug_patient_df = remove_bracket_columns(test_drug_patient_df)
print(rev_train_drug_patient_df.columns)

import matplotlib.pyplot as plt
plt.hist(rev_train_drug_patient_df["auc"],bins=np.linspace(0,300,100))
plt.hist(rev_test_drug_patient_df["auc"],bins=np.linspace(0,300,100))

# %%
#Merge the dataframes containing drug-cell info and cell line info df
train_drug_patient_feature_df = pd.merge(rev_train_drug_patient_df, train_feature_df, on="dbgap_rnaseq_sample")
print(train_drug_patient_feature_df.shape)
test_drug_patient_feature_df = pd.merge(rev_test_drug_patient_df, test_feature_df, on="dbgap_rnaseq_sample")
print(test_drug_patient_feature_df.shape)

# %%
#Get the drug embedding representation 
drug_embed_df = pd.read_csv("../Data/Drug_Full_SMILES_Embedding.csv",header='infer')
drug_embed_df.rename(columns={"Name":"inhibitor"},inplace=True)
drug_embed_df.head()

#Merge with the drug_patient_feature_df
final_train_drug_patient_feature_df = pd.merge(drug_embed_df, train_drug_patient_feature_df, on = "inhibitor")
print(final_train_drug_patient_feature_df.shape)
final_test_drug_patient_feature_df = pd.merge(drug_embed_df, test_drug_patient_feature_df, on = "inhibitor")
print(final_test_drug_patient_feature_df.shape)

#Write the pickle files
final_train_drug_patient_feature_df.to_pickle("../Data/Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl", compression="zip")
final_test_drug_patient_feature_df.to_pickle("../Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.pkl",compression="zip")

# %%
drug_morgan_df = pd.read_csv("../Data/Drug_Full_Info.csv",header='infer')
drug_morgan_df.rename(columns={"Name":"inhibitor"},inplace=True)
#drug_morgan_df.columns = ["CID","MolecularWeight","CanonicalSMILES","InChIKey","XlogP","inhibitor"]+["MFP"+str(i) for i in range(0,1024)]
drug_morgan_df.head()

#Merge with the drug_patient_feature_df
final_train_drug_mfp_patient_feature_df = pd.merge(drug_morgan_df, train_drug_patient_feature_df, on = "inhibitor")
print(final_train_drug_mfp_patient_feature_df.shape)
final_test_drug_mfp_patient_feature_df = pd.merge(drug_morgan_df, test_drug_patient_feature_df, on = "inhibitor")
print(final_test_drug_mfp_patient_feature_df.shape)

#Write the pickle files
final_train_drug_mfp_patient_feature_df.to_pickle("../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",compression="zip")
final_test_drug_mfp_patient_feature_df.to_pickle("../Data/Test_Set_Var_with_Drug_Only_PC_Patient_Info.pkl",compression="zip")
# %%
drug_chemberta_df = pd.read_csv("../Data/Drug_Full_ChemBERTa_Embedding.csv",header='infer')
drug_chemberta_df.rename(columns={"Name":"inhibitor"},inplace=True)
drug_chemberta_df.head()

#Merge with the drug_patient_feature_df
final_train_drug_chemberta_patient_feature_df = pd.merge(drug_chemberta_df, train_drug_patient_feature_df, on = "inhibitor")
print(final_train_drug_chemberta_patient_feature_df.shape)
final_test_drug_chemberta_patient_feature_df = pd.merge(drug_chemberta_df, test_drug_patient_feature_df, on = "inhibitor")
print(final_test_drug_chemberta_patient_feature_df.shape)

#Write the pickle files
final_train_drug_chemberta_patient_feature_df.to_pickle("../Data/Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",compression="zip")
final_test_drug_chemberta_patient_feature_df.to_pickle("../Data/Test_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl",compression="zip")

# %%
drug_molformer_df = pd.read_csv("../Data/Drug_Full_MolFormer_Embedding.csv",header='infer')
drug_molformer_df.rename(columns={"Name":"inhibitor"},inplace=True)
drug_molformer_df.head()

#Merge with drug_patient_feature_df
final_train_drug_molformer_patient_feature_df = pd.merge(drug_molformer_df, train_drug_patient_feature_df, on = "inhibitor")
print(final_train_drug_molformer_patient_feature_df.shape)
final_test_drug_molformer_patient_feature_df = pd.merge(drug_molformer_df, test_drug_patient_feature_df, on = "inhibitor")
print(final_test_drug_molformer_patient_feature_df.shape)   

#Write the pickle files
final_train_drug_molformer_patient_feature_df.to_pickle("../Data/Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",compression="zip")
final_test_drug_molformer_patient_feature_df.to_pickle("../Data/Test_Set_Var_with_Drug_MolFormer_Patient_Info.pkl",compression="zip")


# %%
