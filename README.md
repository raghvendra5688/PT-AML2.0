# PT-AML2.0: Personalized Treatment Recommendations for AML Patients

![PT-AML Workflow](Docs/Personalized_Medicine_RMall.png)

## Description

PT-AML2.0 contains the data, scripts, models, and results for personalized drug response prediction in Acute Myeloid Leukemia (AML) patients. The project uses multi-omics features (gene expression, mutation profiles, cell-type enrichments, pathway enrichments) combined with drug representations to predict patient-specific drug response (IC50/AUC).

**Primary dataset:** BeatAML (Waves 1+2 = training, Waves 3+4 = test)  
**Validation dataset:** LeeAML (external validation cohort)

---

## Directory Structure

```
PT-AML2.0/
├── Data/                    # Processed feature matrices and drug embeddings
│   ├── leeaml/              # LeeAML dataset files
│   └── SMILES_Autoencoder/  # SMILES encoding input/output data
├── Docs/                    # Project documentation and publications
├── Models/                  # Trained model artefacts
│   ├── catboost_models/
│   ├── cnn_models/
│   ├── gat_models/
│   ├── gcn_models/
│   ├── glr_models/
│   ├── lgbm_models/
│   ├── lstm_models/
│   ├── nn_models/
│   ├── rf_models/
│   ├── svr_models/
│   ├── tabpfn_models/
│   └── xgb_models/
├── Results/                 # Cross-validation, optuna, and ablation results
│   ├── catboost/
│   ├── cnn/
│   ├── gat/
│   ├── gcn/
│   ├── glr/
│   ├── lgbm/
│   ├── lstm/
│   ├── rf/
│   ├── svr/
│   ├── tabpfn/
│   ├── tabpfn_leeaml/
│   └── xgb/
├── scripts/                 # All analysis, preprocessing, and training scripts
├── environment.yml          # Conda/micromamba environment specification
└── README.md
```

---

## Environment Setup

```bash
micromamba env create -f environment.yml
micromamba activate PT-AML2.0
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate PT-AML2.0
```

---

## Objectives

1. Build models to learn the relationship between multi-omics features and drug response.
2. Predict the optimal drug for a patient given their omics and disease-state profile.
3. Identify patient cohorts for which a given drug will induce the best response.

---

## Pipeline Overview

### 1. Pre-processing

| Script | Description |
|---|---|
| `analyze_patient_rnaseq.R` | Loads BeatAML gene expression + clinical data; generates t-SNE; merges pathway, cell-type, and module enrichments; outputs training/test feature matrices |
| `analyze_patient_dnaseq.R` | Processes whole-exome sequencing data; outputs sample-by-mutation-type matrices for train and test |
| `analyze_patient_drug_combinations.R` | Combines RNAseq + mutation profiles; runs String PPI random-walk diffusion scores; estimates pathway–drug distances via AUCell; outputs final training/test sets |
| `preprocess_patients.py` | Cleans patient features; builds training/test pickle datasets |
| `preprocess_beataml_drugs.py` | Retrieves BeatAML drug metadata (PubChem CIDs, SMILES, MolWeight, XLogP) |
| `preprocess_leeaml_drugs.py` | Same pipeline for LeeAML dataset drugs |
| `combined_drug_patient_info.py` | Merges drug representations (SMILES-LS, MFP, ChemBERTa, MolFormer) with patient features to build unified training/test PKL files |
| `combined_drug_patient_info_leeaml.py` | Same merge pipeline for LeeAML patients |
| `EDA_plots.R` | Exploratory data analysis and visualisation |
| `all_patients_functions.R` | Shared R utility functions for patient-level analyses |
| `inhibitor_analysis.py` | Analyses inhibitor compound properties |

### 2. Drug Representations

| Script | Description |
|---|---|
| `morgan_fps.py` | Generates Morgan fingerprints (radius 2, 2048 bits) for all drugs |
| `ls_generator2.py` | Encodes SMILES strings to fixed-length vectors via the LS encoder |
| `seq2seq.py` | Sequence-to-sequence SMILES autoencoder model |
| `tokeniser.py` | Tokenisation utilities for SMILES and sequence models |

### 3. Machine Learning Models

| Script | Runner | Description |
|---|---|---|
| `glr_model.py` | — | Generalised linear regression with 5-fold CV |
| `svr_model.py` | — | Support vector regression (RBF kernel) |
| `neosvr_model.py` | — | Extended SVR variant |
| `LSSVM_implementation.py` | — | Least-squares SVM |
| `rf_model.py` | `run_rf.sh` | Random forest with Optuna hyperparameter tuning |
| `xgboost_model.py` | — | XGBoost with Optuna tuning |
| `lightgbm_model_refactored.py` | — | LightGBM with Optuna tuning |
| `catboost_model_refactored.py` | `run_catboost.sh` | CatBoost with Optuna tuning |
| `nn_model.py` | — | Feed-forward neural network |
| `tabpfn_model.py` | `run_tabpfn.sh` | TabPFN (transformer for tabular data) |
| `tabpfn_leeaml_model.py` | `run_tabpfn_leeaml.sh` | TabPFN trained on LeeAML cohort |

### 4. Deep Learning Models (GPU)

| Script | Runner | Description |
|---|---|---|
| `cnn_model.py` | `run_cnn.sh` | 1D-CNN over SMILES tokens + patient FFNN |
| `lstm_model.py` | `run_lstm.sh` | LSTM over SMILES tokens + patient FFNN |
| `gat_model_optuna.py` | `run_gat.sh` | Graph Attention Network (molecular graph) |
| `gcn_model_optuna.py` | `run_gcn.sh` | Graph Convolutional Network (molecular graph) |
| `dl_model_architecture.py` | — | Shared DL layer definitions |
| `dl_model_evaluations.py` | — | DL model evaluation utilities |

### 5. LeeAML-Specific Analyses

| Script | Description |
|---|---|
| `analyze_leeaml_rnaseq.R` | Processes LeeAML gene expression data |
| `analyze_leeaml_drug_combinations.R` | Builds LeeAML drug–patient feature matrices |

### 6. Evaluation & Cross-Validation

| Script | Runner | Description |
|---|---|---|
| `cv_evaluation.py` | `run_cv.sh` | 5-fold CV metrics for all ML models |
| `cv_tabpfn_evaluation.py` | — | TabPFN-specific CV evaluation |
| `dl_cv_metrics.py` | `run_dl_cv_metrics_cnn.sh` / `_gat.sh` / `_gcn.sh` / `_lstm.sh` | CV metrics for DL models |
| `dl_test_cv_metrics.py` | `run_dl_test_cv_metrics.sh` | Test-set metrics for DL models |
| `gs_cv_info.py` | — | Grid-search + CV summary across all models |
| `misc.py` | — | Shared utility functions |

### 7. TabPFN Ablation Studies

| Script | Runner | Description |
|---|---|---|
| `tabpfn_ablation_study.py` | `run_cv_ablation.sh` | Full feature ablation for TabPFN |
| `tabpfn_ablation_1group.py` | `run_tabpfn_1group.sh` | Ablation — group 1 feature subset |
| `tabpfn_ablation_2group.py` | `run_tabpfn_2group.sh` | Ablation — group 2 feature subset |
| `tabpfn_ablation_3group.py` | `run_tabpfn_3group.sh` | Ablation — group 3 feature subset |
| `tabpfn_ablation_summary.py` | `run_tabpfn_4group.sh` | Summarises all ablation results |

---

## Key Data Files

| File | Description |
|---|---|
| `Data/Training_Set_Var_Mod.pkl` | Training set (variance-filtered, modular features) |
| `Data/Test_Set_Var_Mod.pkl` | Test set |
| `Data/Revised_Training_Set_with_Onco_Var_Expr_Clin_PA_CTS_Mut.csv.gz` | Full feature matrix (oncogenes + variants + expression + clinical + pathway/celltype enrichments + mutations) |
| `Data/Drug_Full_ChemBERTa_Embedding.csv` | ChemBERTa drug embeddings |
| `Data/Drug_Full_MolFormer_Embedding.csv` | MolFormer drug embeddings |
| `Data/Drug_Full_SMILES_Embedding.csv` | LS-encoder drug embeddings |
| `Data/Feature_Mapping_LS_Feat_Var.tsv` | Feature index mapping for LS embeddings |
| `Data/Feature_Mapping_MFP_Feat_Var.tsv` | Feature index mapping for Morgan fingerprints |
| `Data/String_PPI_Cutoff_0.7.csv` | STRING protein–protein interaction network (confidence ≥ 0.7) |
| `Data/leeaml/` | All LeeAML processed files |
| `Data/SMILES_Autoencoder/` | SMILES autoencoder training data |
| `Data/vocab.txt` | SMILES vocabulary for tokeniser |

---

## HPC Job Submission (SLURM)

All shell scripts use SLURM and `micromamba` to activate the `PT-AML2.0` environment. Example:

```bash
sbatch scripts/run_tabpfn.sh
sbatch scripts/run_cnn.sh
sbatch scripts/run_cv.sh
```

GPU jobs require a `gpu-H200` partition with `--gres=gpu:1` or `gpu:2` depending on the model.

---

## Citation

If you use this work, please cite:

> Mall R et al. *Personalized Treatment Recommendations for AML patients using Multi-Omics Data*, under preparation (2026).
