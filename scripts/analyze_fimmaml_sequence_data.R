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
library(biomaRt)
loadfonts()
registerDoParallel(20)
ht_opt$message = FALSE

setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/")
source("scripts/all_patients_functions.R")

################################################################################
# Load FIMM-AML RNASeq matrix (genes x samples, tab-separated, ENSEMBL row IDs)
fimmaml_expr_df <- fread("Data/FIMM-AML/RNASeq_Matrix_Data.csv", header = TRUE)
fimmaml_expr_df <- as.data.frame(fimmaml_expr_df)

ensembl_ids  <- fimmaml_expr_df[[1]]
all_sample_names <- colnames(fimmaml_expr_df)[-1]

# Remove healthy control samples
aml_sample_names <- all_sample_names[!grepl("Healthy", all_sample_names)]

fimmaml_mat <- as.matrix(fimmaml_expr_df[, aml_sample_names])
rownames(fimmaml_mat) <- ensembl_ids
rm(fimmaml_expr_df)
gc()

################################################################################
# Convert ENSEMBL gene IDs to HGNC gene symbols via biomaRt
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
gene_map <- getBM(
  attributes = c("ensembl_gene_id", "hgnc_symbol"),
  filters    = "ensembl_gene_id",
  values     = ensembl_ids,
  mart       = mart
)
# Keep only genes with non-empty symbol and unique mapping
gene_map <- gene_map[gene_map$hgnc_symbol != "", ]
gene_map <- gene_map[!duplicated(gene_map$ensembl_gene_id), ]

# Filter matrix to mapped genes and rename rows to gene symbols
mapped_ids   <- intersect(rownames(fimmaml_mat), gene_map$ensembl_gene_id)
fimmaml_mat  <- fimmaml_mat[mapped_ids, ]
gene_symbols <- gene_map$hgnc_symbol[match(mapped_ids, gene_map$ensembl_gene_id)]
rownames(fimmaml_mat) <- gene_symbols
unique_gene_names <- unique(gene_symbols)
final_fimmaml_mat <- NULL
for (i in 1:length(unique_gene_names))
{
  unique_gene_name <- unique_gene_names[i]
  indices <- which(gene_symbols==unique_gene_name)
  if (length(indices)>1)
  {
    index <- which.max(rowSums(fimmaml_mat[indices,]))
    temp <- fimmaml_mat[indices[index],]
  }else
  {
    temp <- fimmaml_mat[indices,]  
  }
  final_fimmaml_mat <- rbind(final_fimmaml_mat, temp)
}
rownames(final_fimmaml_mat) <- unique_gene_names
colnames(final_fimmaml_mat) <- colnames(fimmaml_mat)

rm(fimmaml_mat)
gc()
fimmaml_mat <- final_fimmaml_mat

cat(sprintf("Mapped %d / %d ENSEMBL IDs to gene symbols\n", length(mapped_ids), length(ensembl_ids)))

################################################################################
# Load FIMM-AML Mutation Matrix (genes x samples, tab-separated, already binary 0/1)
fimmaml_mut_df <- fread("Data/FIMM-AML/Mutation_Matrix_Data.csv", header = TRUE)
fimmaml_mut_df <- as.data.frame(fimmaml_mut_df)

mut_gene_names <- fimmaml_mut_df[[1]]
mut_sample_names <- colnames(fimmaml_mut_df)[-1]

fimmaml_mut_mat <- as.matrix(fimmaml_mut_df[, mut_sample_names])
rownames(fimmaml_mut_mat) <- mut_gene_names
rm(fimmaml_mut_df)
gc()

################################################################################
# Find common samples across RNASeq and Mutation data
# Sample IDs follow the pattern AML_XXX_YY; base patient ID = AML_XXX

rnaseq_patient_ids <- aml_sample_names
mut_patient_ids    <- mut_sample_names

common_patients <- intersect(unique(rnaseq_patient_ids), unique(mut_patient_ids))
cat(sprintf("Common patients (RNASeq ∩ Mutation): %d\n", length(common_patients)))

common_rnaseq_samples <- common_patients
common_mut_samples    <- common_patients

################################################################################
# Subset matrices to common samples; keep original sample IDs as column names
fimmaml_mat_common     <- fimmaml_mat[, common_rnaseq_samples]
fimmaml_mut_mat_common <- fimmaml_mut_mat[, common_mut_samples]

# Align mutation columns to RNASeq sample IDs for consistent merging
colnames(fimmaml_mut_mat_common) <- common_rnaseq_samples

################################################################################
# t-SNE on RNASeq expression matrix
set.seed(123)
perp <- min(30, floor((ncol(fimmaml_mat_common) - 1) / 3))
fimmaml_tsne_out <- Rtsne(X = t(fimmaml_mat_common), dims = 2,
                           perplexity = perp,
                           pca_center = FALSE, pca_scale = FALSE, pca = FALSE)
fimmaml_tsne_df <- as.data.frame(fimmaml_tsne_out[["Y"]])
colnames(fimmaml_tsne_df) <- c("Tsne1", "Tsne2")
fimmaml_tsne_df$Sample_ID <- common_rnaseq_samples

write.table(fimmaml_tsne_df,
            file = "Data/FIMM-AML/FIMMAML_Tsne_with_Annotations.csv",
            col.names = TRUE, row.names = FALSE, quote = FALSE, sep = ";")

# t-SNE plot
fimmaml_tsne_df$Cohort <- "FIMM-AML"

g_tsne <- ggplot(data = fimmaml_tsne_df, aes(x = Tsne1, y = Tsne2, color = Cohort)) +
  geom_point(size = 3) + theme_bw() +
  xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
  theme(legend.title.align = 0.0, legend.text.align = 0.0,
        axis.text.x  = element_text(color = "grey20", size = 16, angle = 0,  hjust = .5, vjust = .5, face = "plain"),
        axis.text.y  = element_text(color = "grey20", size = 16, angle = 0,  hjust = 1,  vjust = 0,  face = "plain"),
        axis.title.x = element_text(color = "grey20", size = 20, angle = 0,  hjust = .5, vjust = 0,  face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text  = element_text(color = "grey20", size = 18, angle = 0,  face = "plain"),
        title        = element_text(color = "grey20", size = 20, face = "plain"))
ggsave(filename = "Data/FIMM-AML/FIMMAML_Expr_T-SNE_plot.pdf",
       plot = g_tsne, device = pdf(), height = 8, width = 8, units = "in")
dev.off()

################################################################################
# Build samples x genes expression data frame
fimmaml_expr_samples_df <- as.data.frame(t(fimmaml_mat_common))
fimmaml_expr_samples_df$Sample_ID <- rownames(fimmaml_expr_samples_df)
na_col_ids <- which(is.na(colnames(fimmaml_expr_samples_df)))
if (length(na_col_ids) > 0) {
  fimmaml_expr_samples_df <- fimmaml_expr_samples_df[, -na_col_ids]
}

################################################################################
# Pathway activity via ssGSEA (Selected pathways)
load("Data/Selected.pathways.3.4.RData")
rev_hallmark_pathways_of_interest <- Selected.pathways

rev_ssgsea_results <- as.data.frame(t(gsva(ssgseaParam(
  exprData  = fimmaml_mat_common,
  geneSets  = rev_hallmark_pathways_of_interest,
  normalize = TRUE))))
rev_ssgsea_results$Sample_ID <- rownames(rev_ssgsea_results)

fimmaml_mat_with_expr_pa <- merge(fimmaml_expr_samples_df, rev_ssgsea_results,
                                   by = "Sample_ID", all = TRUE)

################################################################################
# Cell-type enrichment (van Galen AML cell types) via ssGSEA
celltype_sig <- parse.van.galen.supp("BeatAML/Data/1-s2.0-S0092867419300947-mmc3.xlsx")
celltypes_of_interest <- c("cDC-like", "GMP-like", "HSC-like",
                            "Monocyte-like", "Progenitor-like", "Promono-like")
rev_celltype_sig  <- celltype_sig[celltype_sig$vg_type %in% celltypes_of_interest, ]
celltype_genelist <- NULL
unique_celltypes  <- unique(rev_celltype_sig$vg_type)
for (i in seq_along(unique_celltypes)) {
  ct_name <- unique_celltypes[i]
  celltype_genelist[[i]] <- rev_celltype_sig[rev_celltype_sig$vg_type == ct_name, ]$display_label
}
names(celltype_genelist) <- unique_celltypes

ssgsea_celltype_results <- as.data.frame(t(gsva(ssgseaParam(
  exprData  = fimmaml_mat_common,
  geneSets  = celltype_genelist,
  normalize = TRUE))))

################################################################################
# Module enrichment (WGCNA modules) via ssGSEA
load("BeatAML/Data/merged_older_wgcna_kme.RData")
unique_modules  <- unique(cur.map$cur_labels)
module_genelist <- NULL
for (i in seq_along(unique_modules)) {
  module_name <- unique_modules[i]
  module_genelist[[i]] <- cur.map[cur.map$cur_labels == module_name, ]$display_label
}
names(module_genelist) <- unique_modules

ssgsea_module_results <- as.data.frame(t(gsva(ssgseaParam(
  exprData  = fimmaml_mat_common,
  geneSets  = module_genelist,
  normalize = TRUE))))

# Combine cell-type and module ssGSEA scores
ssgsea_celltype_module_combinations <- as.data.frame(
  cbind(ssgsea_celltype_results, ssgsea_module_results))
ssgsea_celltype_module_combinations$Sample_ID <- rownames(ssgsea_celltype_module_combinations)

fimmaml_mat_with_expr_pa_cts <- merge(fimmaml_mat_with_expr_pa,
                                       ssgsea_celltype_module_combinations,
                                       by = "Sample_ID", all = TRUE)

save(celltype_genelist, module_genelist,
     file = "Data/FIMM-AML/FIMMAML_Celltype_Moduletype_info.Rdata")

################################################################################
# Merge mutation matrix (samples x genes) into combined data frame
fimmaml_mut_samples_df <- as.data.frame(t(fimmaml_mut_mat_common))
colnames(fimmaml_mut_samples_df) <- paste0("MUT_", colnames(fimmaml_mut_samples_df))
fimmaml_mut_samples_df$Sample_ID <- rownames(fimmaml_mut_samples_df)

fimmaml_full_df <- merge(fimmaml_mat_with_expr_pa_cts, fimmaml_mut_samples_df,
                          by = "Sample_ID", all = TRUE)

################################################################################
################################################################################
# Save full combined FIMM-AML dataset
write.table(fimmaml_full_df,
            file = "Data/FIMM-AML/FIMMAML_Set_with_Expr_PA_CTS.csv",
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")

################################################################################
# Oncogene + most-varying-gene subset
list_oncogenes   <- fread("Data/Oncogenes.csv", header = TRUE)
oncogenes_symbol <- list_oncogenes$`Gene Symbol`
list_varygenes   <- fread("Data/150_gene.csv", header = TRUE)
varygenes_symbol <- list_varygenes$X

all_colnames   <- colnames(fimmaml_full_df)
gene_col_ids   <- which(all_colnames %in% unique_gene_names)
oncogene_ids   <- which(all_colnames %in% oncogenes_symbol)
varygene_ids   <- which(all_colnames %in% varygenes_symbol)
finalgene_ids  <- union(oncogene_ids, varygene_ids)
annot_col_ids  <- setdiff(seq_along(all_colnames), gene_col_ids)

fimmaml_onco_df <- fimmaml_full_df[,
  c(1, finalgene_ids, annot_col_ids[annot_col_ids != 1])]

write.table(fimmaml_onco_df,
            file = "Data/FIMM-AML/FIMMAML_Set_with_Onco_Var_Expr_PA_CTS.csv",
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")

################################################################################
# Write column metadata (index, name, domain type — matching Patient_Metadata.csv convention)
pathway_col_names  <- names(rev_hallmark_pathways_of_interest)
celltype_col_names <- names(celltype_genelist)
module_col_names   <- names(module_genelist)

assign_col_type <- function(col_name) {
  if (col_name == "Sample_ID")             return("Id")
  if (col_name %in% unique_gene_names)            return("Gene_Expr")
  if (col_name %in% pathway_col_names)     return("Pathway")
  if (col_name %in% celltype_col_names)    return("CellType")
  if (col_name %in% module_col_names)      return("Module")
  if (grepl("^MUT_", col_name))            return("Mutated_Gene")
  return("Other")
}

fimmaml_metadata_df <- data.frame(
  Column_Label = colnames(fimmaml_full_df),
  Type         = sapply(colnames(fimmaml_full_df), assign_col_type),
  stringsAsFactors = FALSE
)
write.table(fimmaml_metadata_df,
            file = "Data/FIMM-AML/FIMMAML_Metadata.csv",
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = ",")
