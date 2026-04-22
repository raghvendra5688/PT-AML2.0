library(data.table)
library(ggplot2)
library(gplots)
library(ComplexHeatmap)
library(BiocManager)
library(GSVA)
library(BiocParallel)
library(doParallel)
library(stringr)
library(circlize)
library(RColorBrewer)
library(ggpubr)
library(GSA)
library(Matrix)
library(extrafont)
library(Rtsne)
library(Polychrome)
library(R.utils)
library(igraph)
library(dnet)
library(AUCell)
library(doMC)
loadfonts()
registerDoParallel(20)
ht_opt$message = FALSE

setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/")
source("scripts/all_patients_functions.R")

################################################################################
# Load LeeAML drug response AUC (wide: drugs x samples -> melt to long)
################################################################################
leeaml_auc_wide <- fread("Data/leeaml/leeaml_drug_response_auc.csv", header = T)
leeaml_auc_wide <- as.data.frame(leeaml_auc_wide)
colnames(leeaml_auc_wide)[1] <- "inhibitor"

# Sample IDs in drug response use dots (AML.017); normalise to match expression IDs (AML017)
orig_sample_cols <- colnames(leeaml_auc_wide)[-1]
norm_sample_cols <- gsub("\\.", "", orig_sample_cols)
colnames(leeaml_auc_wide)[-1] <- norm_sample_cols

# Melt to long format: inhibitor | sample_id | auc
leeaml_drug_df <- melt(as.data.table(leeaml_auc_wide),
                        id.vars      = "inhibitor",
                        variable.name = "sample_id",
                        value.name    = "auc")
leeaml_drug_df <- as.data.frame(leeaml_drug_df)
leeaml_drug_df <- leeaml_drug_df[complete.cases(leeaml_drug_df), ]

unique_drugs    <- unique(leeaml_drug_df$inhibitor)
unique_patients <- unique(leeaml_drug_df$sample_id)

################################################################################
# Build unified drug-target table combining:
#   - Common drugs (shared with BeatAML): Results/target_gene_info.csv
#     matched via Data/leeaml/leeaml_common_drug_cids.csv
#   - LeeAML-specific drugs: Data/leeaml/leeaml_specific_drug_cids.csv
#     (carries Target_Genes directly)
################################################################################
drug_targets_df <- fread("Results/target_gene_info.csv", header = T)
drug_targets_df <- as.data.frame(drug_targets_df)

# --- common drugs ---
common_cids_df <- fread("Data/leeaml/leeaml_common_drug_cids.csv", header = T)
common_cids_df <- as.data.frame(common_cids_df)

# common_cids_df columns: LeeAML_Drug_Name, Alternate_Name, BeatAML_Drug_Name, cid
colnames(common_cids_df) <- c("LeeAML_Drug_Name", "Alternate_Name", "Name", "cid")

# Join target info by CID and attach the LeeAML drug name
common_drug_info <- merge(common_cids_df[, c("Name", "LeeAML_Drug_Name","cid")],
                          drug_targets_df,
                          by.x = "Name", by.y = "Name", all.x = FALSE)

# Use the LeeAML drug name as the inhibitor key, keep Targets column
common_drug_info_rev <- common_drug_info[, c("Name", "LeeAML_Drug_Name","MolecularWeight",
                                              "CanonicalSMILES", "InChIKey",
                                              "XLogP", "Targets")]
colnames(common_drug_info_rev)[2] <- "inhibitor"

# --- LeeAML-specific drugs ---
specific_cids_df <- fread("Data/leeaml/leeaml_specific_drug_cids.csv", header = T)
specific_cids_df <- as.data.frame(specific_cids_df)
specific_cids_df <- specific_cids_df[!is.na(specific_cids_df$cid),]

# specific_cids_df columns: leeaml_specific, Alternate_Name, Target_Genes, cid
specific_cids_full_df <- fread("Data/leeaml/LeeAML_Specific_Drug_Full_Info.csv")
specific_cids_full_df <- as.data.frame(specific_cids_full_df) 

specific_drug_info_rev <- data.frame(
  inhibitor       = specific_cids_df$leeaml_specific,
  MolecularWeight = specific_cids_full_df$MolecularWeight,
  CanonicalSMILES = specific_cids_full_df$SMILES,
  InChIKey        = specific_cids_full_df$InChIKey,
  XLogP           = specific_cids_full_df$XLogP,
  Targets         = specific_cids_df$Target_Genes,
  stringsAsFactors = FALSE
)

cat("Drugs in response data:    ", length(unique_drugs), "\n")
cat("Drugs with target info:    ", sum(unique_drugs %in% common_drug_info_rev$inhibitor), "\n")

################################################################################
# Merge drug info into long drug-response table
################################################################################
common_leeaml_drug_df_with_info <- merge(leeaml_drug_df, common_drug_info_rev, by = "inhibitor", all = FALSE)
common_leeaml_drug_df_with_info <- common_leeaml_drug_df_with_info[complete.cases(common_leeaml_drug_df_with_info), ]

# Primary key: inhibitor_sampleid
common_leeaml_drug_df_with_info$primary_key <- paste0(common_leeaml_drug_df_with_info$inhibitor, "_",
                                                common_leeaml_drug_df_with_info$sample_id)

################################################################################
# Load LeeAML patient expression + pathway + cell-type feature matrix
# (generated by analyze_leeaml_rnaseq.R)
################################################################################
leeaml_patient_df <- fread("Data/leeaml/LeeAML_Set_with_Expr_PA_CTS.csv", header = T, sep = "\t")
leeaml_patient_df <- as.data.frame(leeaml_patient_df)

leeaml_onco_patient_df <- fread("Data/leeaml/LeeAML_Set_with_Onco_Var_Expr_PA_CTS.csv", header = T, sep = "\t")
leeaml_onco_patient_df <- as.data.frame(leeaml_onco_patient_df)

# Keep only samples present in both expression and drug response data
rev_patient_ids <- intersect(leeaml_patient_df$sample_id, unique_patients)
leeaml_patient_df      <- leeaml_patient_df[leeaml_patient_df$sample_id %in% rev_patient_ids, ]
leeaml_onco_patient_df <- leeaml_onco_patient_df[leeaml_onco_patient_df$sample_id %in% rev_patient_ids, ]
common_leeaml_drug_df_with_info <- common_leeaml_drug_df_with_info[common_leeaml_drug_df_with_info$sample_id %in% rev_patient_ids, ]

################################################################################
# t-SNE on LeeAML oncogene+pathway+cell-type feature set
################################################################################
set.seed(123)

# Identify numeric feature columns (exclude sample_id and any character columns)
str_col_ids <- which(sapply(leeaml_onco_patient_df, class) == "character")
na_col_ids  <- which(colSums(is.na(leeaml_onco_patient_df)) > 0)
sd_col_ids  <- which(apply(leeaml_onco_patient_df, 2, function(x) {
  xn <- suppressWarnings(as.numeric(x)); sd(xn, na.rm = TRUE) }) == 0)
excl_ids    <- union(union(str_col_ids, na_col_ids), sd_col_ids)
feat_df     <- leeaml_onco_patient_df[, -excl_ids, drop = FALSE]

perp <- min(5, floor((nrow(feat_df) - 1) / 3))
tsne_out <- Rtsne(X = feat_df, dims = 2, perplexity = perp,
                  pca_center = TRUE, pca_scale = TRUE, pca = TRUE)
tsne_df  <- as.data.frame(tsne_out[["Y"]])
colnames(tsne_df) <- c("Tsne1", "Tsne2")
tsne_df$sample_id <- leeaml_onco_patient_df$sample_id

P28 <- createPalette(28, c("#ff0000", "#00ff00", "#0000ff"))
palette(P28)

g_tsne <- ggplot(data = tsne_df, aes(x = Tsne1, y = Tsne2)) +
  geom_point(size = 2, color = "#2166AC") + theme_bw() +
  xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
  theme(axis.text.x  = element_text(color = "grey20", size = 16, angle = 0,  hjust = .5, vjust = .5, face = "plain"),
        strip.text   = element_text(color = "white",  size = 20, angle = 0,  hjust = 0.5, vjust = 0.5, face = "plain"),
        axis.text.y  = element_text(color = "grey20", size = 16, angle = 0,  hjust = 1,  vjust = 0,  face = "plain"),
        axis.title.x = element_text(color = "grey20", size = 20, angle = 0,  hjust = .5, vjust = 0,  face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text  = element_text(color = "grey20", size = 18, angle = 0,  face = "plain"),
        title        = element_text(color = "grey20", size = 20, face = "plain"))
ggsave(filename = "Data/leeaml/LeeAML_Feature_Set_T-SNE_plot.pdf",
       plot = g_tsne, device = pdf(), height = 8, width = 8, units = "in")
dev.off()

################################################################################
# Load STRING PPI graph
################################################################################
string_ppi_df <- read.table("Data/String_PPI_Cutoff_0.7.csv", header = T)
g_ppi    <- graph_from_edgelist(as.matrix(string_ppi_df[, c(1:2)]), directed = TRUE)
E(g_ppi)$weight <- string_ppi_df$combined_score
N_ppi    <- length(V(g_ppi))

# Drugs in the final merged set (those with both drug response and target info)
unique_drugs_final <- unique(common_leeaml_drug_df_with_info$inhibitor)

################################################################################
# Build drug seed matrix for Random Walk with Restart (RWR)
################################################################################
p0_matrix <- matrix(0, nrow = N_ppi, ncol = length(unique_drugs_final))
rownames(p0_matrix) <- V(g_ppi)$name
colnames(p0_matrix) <- unique_drugs_final

for (i in seq_along(unique_drugs_final)) {
  drug_name <- unique_drugs_final[i]
  targets   <- common_drug_info_rev[common_drug_info_rev$inhibitor == drug_name, ]$Targets
  targets   <- unlist(strsplit(targets, split = ";"))
  p0_vec    <- rep(0, N_ppi)
  names(p0_vec) <- V(g_ppi)$name
  p0_vec[names(p0_vec) %in% targets] <- 1
  p0_matrix[, i] <- p0_vec
}

# RWR stationary distribution (sqrt as per original)
pinf_matrix <- sqrt(dRWR(g = g_ppi, normalise = "laplacian",
                          setSeeds = p0_matrix, restart = 0.5, multicores = 8))
rownames(pinf_matrix) <- V(g_ppi)$name
colnames(pinf_matrix) <- unique_drugs_final
pinf_matrix <- as.matrix(pinf_matrix)

################################################################################
# Build drug-patient affinity matrix (gene expression x RWR affinity)
################################################################################
# Gene columns in leeaml_patient_df: column 2 onwards up to the last gene
# Gene names are rownames(leeaml_mat) — reload from expression file to get gene list
leeaml_expr_header <- fread("Data/leeaml/leeaml_gene_expression.csv", header = T, nrows = 0)
all_expr_genes     <- colnames(leeaml_expr_header)[-1]   # sample cols, transposed below
# Actually gene names are in the UID column (rows of the wide file):
leeaml_expr_genes_df <- fread("Data/leeaml/leeaml_gene_expression.csv", header = T,
                               select = "UID")
leeaml_gene_names  <- leeaml_expr_genes_df$UID

all_genes_in_ppi     <- rownames(pinf_matrix)
all_genes_in_patient <- colnames(leeaml_patient_df)[colnames(leeaml_patient_df) %in% leeaml_gene_names]
common_genes         <- intersect(all_genes_in_ppi, all_genes_in_patient)

# Subset affinity and expression matrices to common genes
revised_pinf_matrix <- t(pinf_matrix[common_genes, ])  # drugs x common_genes

revised_patient_expr_df <- leeaml_patient_df[, common_genes, drop = FALSE]
rownames(revised_patient_expr_df) <- leeaml_patient_df$sample_id

################################################################################
# Drug-patient affinity combinations matrix (common_genes x drug*patient)
################################################################################
drug_patient_affinity_combinations <- t(sapply(
  seq_along(common_genes),
  function(i) tcrossprod(revised_pinf_matrix[, i], revised_patient_expr_df[, i])
))
drug_patient_affinity_combinations_mat <- Matrix(drug_patient_affinity_combinations, sparse = TRUE)
rm(drug_patient_affinity_combinations)
gc()

all_drug_patient_combination <- as.data.frame(
  expand.grid(rownames(revised_pinf_matrix), rownames(revised_patient_expr_df))
)
all_drug_patient_combination <- paste0(all_drug_patient_combination$Var1, "_",
                                        all_drug_patient_combination$Var2)
colnames(drug_patient_affinity_combinations_mat) <- all_drug_patient_combination
rownames(drug_patient_affinity_combinations_mat) <- common_genes

################################################################################
# AUCell pathway enrichment on drug-patient affinity matrix
################################################################################
load("Data/Selected.pathways.3.4.RData")

N         <- ncol(drug_patient_affinity_combinations_mat)
blocks    <- min(50, N)   # cap blocks to N so no block is empty
blocksize <- ceiling(N / blocks)
auc_scores_df <- NULL

for (i in seq_len(blocks)) {
  start <- (i - 1) * blocksize + 1
  end   <- min(i * blocksize, N)
  if (start > N) next
  ids   <- start:end
  patient_rankings_temp <- AUCell_buildRankings(
    exprMat   = as.matrix(drug_patient_affinity_combinations_mat[, ids]),
    nCores    = 8,
    plotStats = FALSE
  )
  temp_auc_scores <- AUCell_calcAUC(
    geneSets   = Selected.pathways,
    rankings   = patient_rankings_temp,
    nCores     = 8,
    normAUC    = TRUE,
    aucMaxRank = ceiling(0.5 * nrow(patient_rankings_temp))
  )
  auc_scores_df <- AUCell::cbind(auc_scores_df,
                                  temp_auc_scores@assays@data@listData$AUC)
  rm(temp_auc_scores, patient_rankings_temp)
  gc()
}

auc_scores_df <- t(auc_scores_df)
colnames(auc_scores_df) <- paste0("AUC_", names(Selected.pathways))

################################################################################
# Subset AUC scores to observed drug-patient combinations and merge with response
################################################################################
combinations_to_keep <- common_leeaml_drug_df_with_info$primary_key

subset_auc_scores_df <- auc_scores_df[rownames(auc_scores_df) %in% combinations_to_keep, , drop = FALSE]
for (i in seq_len(ncol(subset_auc_scores_df))) {
  subset_auc_scores_df[, i] <- signif(subset_auc_scores_df[, i], digits = 3)
}
subset_auc_scores_df             <- as.data.frame(subset_auc_scores_df)
subset_auc_scores_df$primary_key <- rownames(subset_auc_scores_df)

leeaml_drug_response_final <- merge(common_leeaml_drug_df_with_info, subset_auc_scores_df,
                                     by = "primary_key", all = FALSE)

write.table(leeaml_drug_response_final,
            file      = "Data/leeaml/Common_LeeAML_Set_with_AUC.csv",
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")

################################################################################
# SECTION 2 — LeeAML-SPECIFIC DRUGS
################################################################################

################################################################################
# Merge specific drug info into long drug-response table
################################################################################
specific_leeaml_drug_df_with_info <- merge(leeaml_drug_df, specific_drug_info_rev,
                                            by = "inhibitor", all = FALSE)
specific_leeaml_drug_df_with_info <- specific_leeaml_drug_df_with_info[
  complete.cases(specific_leeaml_drug_df_with_info), ]
specific_leeaml_drug_df_with_info$primary_key <- paste0(
  specific_leeaml_drug_df_with_info$inhibitor, "_",
  specific_leeaml_drug_df_with_info$sample_id)

# Keep only samples present in the expression data
specific_leeaml_drug_df_with_info <- specific_leeaml_drug_df_with_info[
  specific_leeaml_drug_df_with_info$sample_id %in% rev_patient_ids, ]

cat("Specific drugs in response data:  ", length(unique(leeaml_drug_df$inhibitor[
  leeaml_drug_df$inhibitor %in% specific_drug_info_rev$inhibitor])), "\n")
cat("Specific drug-patient pairs kept: ", nrow(specific_leeaml_drug_df_with_info), "\n")

################################################################################
# Build drug seed matrix for RWR — specific drugs
################################################################################
unique_specific_drugs <- unique(specific_leeaml_drug_df_with_info$inhibitor)

p0_matrix_specific <- matrix(0, nrow = N_ppi, ncol = length(unique_specific_drugs))
rownames(p0_matrix_specific) <- V(g_ppi)$name
colnames(p0_matrix_specific) <- unique_specific_drugs

for (i in seq_along(unique_specific_drugs)) {
  drug_name <- unique_specific_drugs[i]
  targets   <- specific_drug_info_rev[specific_drug_info_rev$inhibitor == drug_name, ]$Targets
  targets   <- unlist(strsplit(targets, split = ";"))
  p0_vec    <- rep(0, N_ppi)
  names(p0_vec) <- V(g_ppi)$name
  p0_vec[names(p0_vec) %in% targets] <- 1
  p0_matrix_specific[, i] <- p0_vec
}

# RWR stationary distribution
pinf_matrix_specific <- sqrt(dRWR(g = g_ppi, normalise = "laplacian",
                                   setSeeds = p0_matrix_specific, restart = 0.5, multicores = 8))
rownames(pinf_matrix_specific) <- V(g_ppi)$name
colnames(pinf_matrix_specific) <- unique_specific_drugs
pinf_matrix_specific <- as.matrix(pinf_matrix_specific)

################################################################################
# Build drug-patient affinity matrix — specific drugs
# (reuse common_genes and revised_patient_expr_df from Section 1)
################################################################################
revised_pinf_matrix_specific <- t(pinf_matrix_specific[common_genes, , drop = FALSE])  # specific_drugs x common_genes

drug_patient_affinity_specific <- t(sapply(
  seq_along(common_genes),
  function(i) tcrossprod(revised_pinf_matrix_specific[, i], revised_patient_expr_df[, i])
))
drug_patient_affinity_specific_mat <- Matrix(drug_patient_affinity_specific, sparse = TRUE)
rm(drug_patient_affinity_specific)
gc()

all_drug_patient_combination_specific <- as.data.frame(
  expand.grid(rownames(revised_pinf_matrix_specific), rownames(revised_patient_expr_df))
)
all_drug_patient_combination_specific <- paste0(
  all_drug_patient_combination_specific$Var1, "_",
  all_drug_patient_combination_specific$Var2)
colnames(drug_patient_affinity_specific_mat) <- all_drug_patient_combination_specific
rownames(drug_patient_affinity_specific_mat) <- common_genes

################################################################################
# AUCell pathway enrichment — specific drugs
################################################################################
N_specific         <- ncol(drug_patient_affinity_specific_mat)
blocks_specific    <- min(50, N_specific)
blocksize_specific <- ceiling(N_specific / blocks_specific)
auc_scores_specific_df <- NULL

for (i in seq_len(blocks_specific)) {
  start <- (i - 1) * blocksize_specific + 1
  end   <- min(i * blocksize_specific, N_specific)
  if (start > N_specific) next
  ids   <- start:end
  patient_rankings_temp <- AUCell_buildRankings(
    exprMat   = as.matrix(drug_patient_affinity_specific_mat[, ids]),
    nCores    = 8,
    plotStats = FALSE
  )
  temp_auc_scores <- AUCell_calcAUC(
    geneSets   = Selected.pathways,
    rankings   = patient_rankings_temp,
    nCores     = 8,
    normAUC    = TRUE,
    aucMaxRank = ceiling(0.5 * nrow(patient_rankings_temp))
  )
  auc_scores_specific_df <- AUCell::cbind(auc_scores_specific_df,
                                           temp_auc_scores@assays@data@listData$AUC)
  rm(temp_auc_scores, patient_rankings_temp)
  gc()
}

auc_scores_specific_df <- t(auc_scores_specific_df)
colnames(auc_scores_specific_df) <- paste0("AUC_", names(Selected.pathways))

################################################################################
# Subset AUC scores to observed drug-patient combinations and merge
################################################################################
combinations_to_keep_specific <- specific_leeaml_drug_df_with_info$primary_key

subset_auc_specific_df <- auc_scores_specific_df[
  rownames(auc_scores_specific_df) %in% combinations_to_keep_specific, , drop = FALSE]
for (i in seq_len(ncol(subset_auc_specific_df))) {
  subset_auc_specific_df[, i] <- signif(subset_auc_specific_df[, i], digits = 3)
}
subset_auc_specific_df             <- as.data.frame(subset_auc_specific_df)
subset_auc_specific_df$primary_key <- rownames(subset_auc_specific_df)

specific_leeaml_drug_response_final <- merge(specific_leeaml_drug_df_with_info,
                                              subset_auc_specific_df,
                                              by = "primary_key", all = FALSE)

write.table(specific_leeaml_drug_response_final,
            file      = "Data/leeaml/Specific_LeeAML_Set_with_AUC.csv",
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")
