# PT-AML2.0 — Data Directory

This directory contains all datasets used for personalised treatment prediction in Acute Myeloid Leukaemia (AML), spanning three cohorts: **BeatAML** (primary training cohort), **LeeAML** (independent test cohort), and **FIMM-AML** (independent test cohort).

---

## Root Level

### Reference / Feature-Selection Lists

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `150_gene.csv` | 150 | 1 | List of 150 high-variance genes used for patient feature selection alongside oncogenes. |
| `Oncogenes.csv` | 736 | 20 | Curated oncogene list with annotations (gene symbol, cancer type, role, etc.). Used to select biologically relevant expression features. |
| `Drug_Names.csv` | 166 | 1 | Names of all 166 BeatAML drugs included in the study. |
| `vocab.txt` | — | — | Vocabulary file for the SMILES autoencoder; defines the character-level token set used during encoding/decoding. |

### BeatAML Patient Feature Matrices

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `Training_Set_Mod.csv` | 337 | 1,131 | BeatAML training-set patient feature matrix — full gene expression set plus pathway, cell-type, module, clinical and mutation features. |
| `Training_Set_Var_Mod.csv` | 337 | 1,272 | BeatAML training-set patient feature matrix — restricted to high-variance oncogene subset; used as primary training input. |
| `Training_Set_with_Onco_Var_Expr_Clin_PA_CTS.csv` | 460 | 965 | Extended BeatAML training set including oncogene expression, clinical variables, pathway activity, and cell-type scores (pre-mutation-merge). |
| `Test_Set_Mod.csv` | 183 | 1,131 | BeatAML test-set patient feature matrix — full gene expression set, matching columns of `Training_Set_Mod.csv`. |
| `Test_Set_Var_Mod.csv` | 183 | 1,272 | BeatAML test-set patient feature matrix — high-variance oncogene subset, matching columns of `Training_Set_Var_Mod.csv`. |
| `Test_Set_with_Onco_Var_Expr_Clin_PA_CTS.csv` | 211 | 965 | Extended BeatAML test set; same structure as the training extended set. |
| `Train_Test_Tsne_with_Annotations.csv` | 671 | 98 | t-SNE 2-D coordinates and sample annotations for the combined BeatAML train + test set. |
| `TRAINING_Metadata.csv` | 23,014 | 2 | Column-level metadata for the BeatAML training feature matrix: column name and domain type (Gene_Expr, Pathway, CellType, etc.). |
| `Patient_Metadata.csv` | 1,348 | 3 | Reference column metadata for BeatAML patient features: `Column_Label`, `Type`, `Keep` flag. Used to define canonical column-naming conventions across cohorts. |

### BeatAML Drug Feature Files

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `Drug_Full_SMILES_Embedding.csv` | 162 | 735 | BeatAML drugs — SMILES autoencoder latent embedding (512 dims) plus RDKit molecular descriptors and graph features. |
| `Drug_Full_Info.csv` | 162 | 479 | BeatAML drugs — physicochemical descriptors, Morgan fingerprints (1024 bits), and MACCS keys. |
| `Drug_Full_MolFormer_Embedding.csv` | 162 | 1,247 | BeatAML drugs — IBM MolFormer transformer embeddings (1,024 dims) plus metadata. |
| `Drug_Full_ChemBERTa_Embedding.csv` | 162 | 863 | BeatAML drugs — DeepChem ChemBERTa transformer embeddings (768 dims) plus metadata. |
| `target_gene_info.csv` | 166 | 7 | BeatAML drug properties: Name, CID, MolecularWeight, CanonicalSMILES, InChIKey, XLogP, and semicolon-separated target gene symbols. |
| `target_gene_info_overlapping_name_with_fimmaml.csv` | 166 | 7 | Same as `target_gene_info.csv` but with drug-name annotations highlighting overlap with FIMM-AML drug names. |

### BeatAML Combined Drug–Patient Pickles

| File | Description |
|------|-------------|
| `Training_Set_Var_with_Drug_Embedding_Patient_Info.pkl` | BeatAML training rows (33,803 drug–patient pairs × 2,086 features): patient features + SMILES autoencoder drug embedding. Label: `auc`. |
| `Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl` | Same pairs with physicochemical / fingerprint drug features (1,830 cols). |
| `Training_Set_Var_with_Drug_MolFormer_Patient_Info.pkl` | Same pairs with MolFormer drug embedding (2,598 cols). |
| `Training_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl` | Same pairs with ChemBERTa drug embedding (2,214 cols). |

### Feature Mappings

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `Feature_Mapping_LS_Feat_Var.tsv` | 1,565 | 2 | Feature name → domain mapping for the SMILES latent-space feature variant (LS_Feat_Var). |
| `Feature_Mapping_MFP_Feat_Var.tsv` | 2,333 | 2 | Feature name → domain mapping for the Morgan fingerprint feature variant (MFP_Feat_Var). |
| `ablation_feature_columns.csv` | 2,086 | 2 | Feature column names and their group assignments used in TabPFN ablation experiments. |

### Network & Pathway Data

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `String_PPI_Cutoff_0.7.csv` | 504,026 | 3 | STRING protein–protein interaction network filtered at combined score ≥ 0.7. Columns: gene1, gene2, combined_score. Used to build the PPI graph for Random Walk with Restart (RWR). |
| `Selected.pathways.3.4.RData` | — | — | R object containing selected Hallmark and curated pathway gene sets (MSigDB v3.4) used for ssGSEA and AUCell enrichment. |
| `Revised_Train_Test_Ids.Rdata.gz` | — | — | Compressed R object storing the BeatAML train/test patient ID split. |

---

## FIMM-AML/

Independent AML cohort from the Institute for Molecular Medicine Finland (FIMM). Used as an external test set. Drug sensitivity is measured as **Drug Sensitivity Score (DSS)**.

### Raw Data

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `FIMM-AML_data.xlsx` | — | — | Original FIMM-AML Excel file containing raw drug screening, clinical, and genomic data. |
| `RNASeq_Matrix_Data.csv` | 18,202 | 168 | Raw RNA-seq expression matrix: ENSEMBL gene IDs (rows) × sample IDs (columns, including healthy controls). |
| `Mutation_Matrix_Data.csv` | 57 | 226 | Binary somatic mutation matrix: mutated genes (rows) × AML sample IDs (columns). Values are 0/1. |
| `Clinical_Data.csv` | 187 | 89 | FIMM-AML patient clinical variables (age, diagnosis, cytogenetics, etc.) for 187 samples. |
| `Drug_Sensitivity_Scores.csv` | 75,295 | 19 | Full FIMM-AML drug sensitivity data for all 540 screened drugs. Columns include Sample_ID, Chemical_compound, DSS, sDSS, and dose-response curve parameters. |

### Drug Mapping & Sensitivity (Common Drugs)

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `Drug_Mapping_Data.csv` | 78 | 4 | Manual mapping of 78 FIMM-AML drugs to their BeatAML counterparts. Columns: Id, FIMM_Name, BeatAML_Name, Match_Type. |
| `Drug_Sensitivity_Scores_Common.csv` | 12,480 | 19 | Drug sensitivity scores filtered to the 78 drugs common with BeatAML. Same schema as the full scores file. |
| `fimmaml_drug_cids.csv` | 78 | 5 | PubChem CIDs for the 78 common drugs: Id, FIMM_Name, BeatAML_Name, Match_Type, CID. |

### Drug Feature Files

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `FIMMAML_Drug_Full_SMILES_Embedding.csv` | 78 | 736 | FIMM-AML drugs — SMILES autoencoder latent embedding plus RDKit descriptors and graph features. |
| `FIMMAML_Drug_Full_Info.csv` | 78 | 480 | FIMM-AML drugs — physicochemical descriptors, Morgan fingerprints, and MACCS keys. |
| `FIMMAML_Drug_Full_MolFormer_Embedding.csv` | 78 | 1,248 | FIMM-AML drugs — MolFormer transformer embeddings plus metadata. |
| `FIMMAML_Drug_Full_ChemBERTa_Embedding.csv` | 78 | 864 | FIMM-AML drugs — ChemBERTa transformer embeddings plus metadata. |

### Patient Feature Matrices

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `FIMMAML_Set_with_Expr_PA_CTS.csv` | 149 | 17,814 | Full FIMM-AML patient feature matrix: all HGNC gene expression values, pathway activity (ssGSEA), van Galen cell-type scores, WGCNA module scores, and mutation indicators. |
| `FIMMAML_Set_with_Onco_Var_Expr_PA_CTS.csv` | 149 | 949 | Reduced FIMM-AML patient feature matrix: oncogene + high-variance gene subset, pathway, cell-type, module, and mutation columns. Primary patient input for modelling. |
| `FIMMAML_Set_Var_Mod.csv` | 149 | 949 | Clean FIMM-AML patient feature matrix with standardised column names (`_Expr`, `_Mut` suffixes); generated by `combined_drug_patient_info_fimmaml.py`. |
| `FIMMAML_Set_with_AUC.csv` | 8,357 | 65 | FIMM-AML drug–patient pairs with DSS, sDSS, drug properties, and AUCell pathway enrichment scores computed from drug–patient affinity combinations. |
| `FIMMAML_Tsne_with_Annotations.csv` | 149 | 3 | t-SNE 2-D coordinates (Tsne1, Tsne2) for FIMM-AML samples plus Sample_ID. |
| `FIMMAML_Metadata.csv` | 17,814 | 2 | Column metadata for FIMM-AML feature matrices: `Column_Label` and `Type` (Id, Gene_Expr, Pathway, CellType, Module, Mutated_Gene). |

### Combined Drug–Patient Pickles

| File | Description |
|------|-------------|
| `FIMMAML_Set_Var_Mod.pkl` | Compressed pickle of `FIMMAML_Set_Var_Mod.csv` (149 samples × 949 features). |
| `FIMMAML_Set_Var_with_Drug_Embedding_Patient_Info.pkl` | FIMM-AML drug–patient pairs (8,357 × 1,748): patient features + SMILES autoencoder drug embedding. Label: `DSS`. |
| `FIMMAML_Set_Var_with_Drug_Only_PC_Patient_Info.pkl` | Same pairs with physicochemical / fingerprint drug features (1,492 cols). |
| `FIMMAML_Set_Var_with_Drug_MolFormer_Patient_Info.pkl` | Same pairs with MolFormer drug embedding (2,260 cols). |
| `FIMMAML_Set_Var_with_Drug_ChemBERTa_Patient_Info.pkl` | Same pairs with ChemBERTa drug embedding (1,876 cols). |

### Auxiliary Files

| File | Description |
|------|-------------|
| `FIMMAML_Celltype_Moduletype_info.Rdata` | R object with van Galen AML cell-type and WGCNA module gene lists used in ssGSEA enrichment. |
| `FIMMAML_Expr_T-SNE_plot.pdf` | t-SNE scatter of FIMM-AML samples based on RNA-seq expression. |
| `FIMMAML_Feature_Set_T-SNE_plot.pdf` | t-SNE scatter based on the full oncogene + pathway + cell-type feature set. |
| `FIMMAML_DSS_distribution.pdf` | Histogram of DSS values across all drug–patient pairs. |

---

## leeaml/

Independent AML cohort from Lee et al. Used as a second external test set. Drug sensitivity is measured as **AUC** (area under the dose-response curve), consistent with BeatAML.

### Raw / Source Data

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `leeaml_gene_expression.csv` | 17,280 | 31 | LeeAML gene expression matrix: genes (rows) × 30 patient samples (columns) in FPKM/TPM units. |
| `leeaml_drug_response_auc.csv` | 159 | 31 | LeeAML drug response AUC values: drugs (rows) × 30 patient samples (columns). |
| `leeaml_clinical.csv` | 37 | 23 | LeeAML patient clinical variables for 37 samples (age, FAB, cytogenetics, etc.). |
| `leeaml_common_drugs.csv` | 49 | 3 | List of 49 LeeAML drugs shared with BeatAML, with FIMM_Name and BeatAML_Name columns. |
| `leeaml_specific_drugs.csv` | 111 | 3 | List of 111 LeeAML-specific drugs not present in BeatAML. |
| `leeaml_common_drug_cids.csv` | 49 | 4 | PubChem CIDs for the 49 common LeeAML drugs. |
| `leeaml_specific_drug_cids.csv` | 111 | 4 | PubChem CIDs for the 111 LeeAML-specific drugs. |
| `leeaml_common_oncogenes.csv` | 615 | 31 | Oncogene expression subset for LeeAML samples with common drugs. |

### Drug Feature Files

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `LeeAML_Common_Drug_Full_SMILES_Embedding.csv` | 49 | 736 | Common LeeAML drugs — SMILES autoencoder embedding plus RDKit descriptors. |
| `LeeAML_Common_Drug_Full_Info.csv` | 49 | 480 | Common LeeAML drugs — physicochemical descriptors and fingerprints. |
| `LeeAML_Common_Drug_Full_MolFormer_Embedding.csv` | 49 | 1,248 | Common LeeAML drugs — MolFormer embeddings. |
| `LeeAML_Common_Drug_Full_ChemBERTa_Embedding.csv` | 49 | 864 | Common LeeAML drugs — ChemBERTa embeddings. |
| `LeeAML_Specific_Drug_Full_SMILES_Embedding.csv` | 102 | 735 | LeeAML-specific drugs — SMILES autoencoder embedding plus RDKit descriptors. |
| `LeeAML_Specific_Drug_Full_Info.csv` | 102 | 479 | LeeAML-specific drugs — physicochemical descriptors and fingerprints. |
| `LeeAML_Specific_Drug_Full_MolFormer_Embedding.csv` | 102 | 1,247 | LeeAML-specific drugs — MolFormer embeddings. |
| `LeeAML_Specific_Drug_Full_ChemBERTa_Embedding.csv` | 102 | 863 | LeeAML-specific drugs — ChemBERTa embeddings. |

### Patient Feature Matrices

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `LeeAML_Set_with_Expr_PA_CTS.csv` | 30 | 17,170 | Full LeeAML patient feature matrix: all gene expression values, pathway activity, van Galen cell-type scores, and WGCNA module scores. |
| `LeeAML_Set_with_Onco_Var_Expr_PA_CTS.csv` | 30 | 866 | Reduced LeeAML patient feature matrix: oncogene + high-variance gene subset, pathway, cell-type, and module columns. |
| `LeeAML_Set_Var_Mod.csv` | 30 | 866 | Clean LeeAML patient feature matrix with standardised column names. |
| `LeeAML_Tsne_with_Annotations.csv` | 30 | 3 | t-SNE 2-D coordinates for LeeAML samples plus Sample_ID. |
| `LeeAML_Metadata.csv` | 17,169 | 2 | Column metadata for LeeAML feature matrices: `Column_Label` and `Type`. |

### Combined Drug–Patient Datasets

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `Common_LeeAML_Set_with_AUC.csv` | 1,470 | 64 | LeeAML drug–patient pairs for 49 common drugs: drug properties, patient features, AUCell enrichment scores, and AUC label. |
| `Specific_LeeAML_Set_with_AUC.csv` | 2,790 | 63 | Same structure for 111 LeeAML-specific drugs and their AUC responses. |

### Auxiliary Files

| File | Description |
|------|-------------|
| `LeeAML_Expr_T-SNE_plot.pdf` | t-SNE scatter of LeeAML samples based on RNA-seq expression. |
| `LeeAML_Feature_Set_T-SNE_plot.pdf` | t-SNE scatter based on the oncogene + pathway + cell-type feature set. |
| `Common_LeeAML_AUC_distribution.pdf` | Histogram of AUC values for the 49 common-drug pairs. |
| `Specific_LeeAML_AUC_distribution.pdf` | Histogram of AUC values for the 111 LeeAML-specific drug pairs. |

---

## SMILES_Autoencoder/

| File | Rows | Cols | Description |
|------|-----:|-----:|-------------|
| `all_smiles_revised_final.csv.gz` | 2,454,663 | — | Large compressed SMILES database used to pre-train the autoencoder. |
| `test_smiles.csv` | 78 | — | SMILES strings for the 78/79 drugs used to generate autoencoder embeddings during testing. |
| `test_LS.csv` | 78 | — | Autoencoder latent-space (LS) representations for the test SMILES; used to validate embedding quality. |
