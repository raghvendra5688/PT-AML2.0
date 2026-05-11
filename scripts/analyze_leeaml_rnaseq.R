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
loadfonts()
registerDoParallel(20)
ht_opt$message = FALSE

setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/")
source("scripts/all_patients_functions.R")

################################################################################
# Load LeeAML gene expression data
# Format: rows = genes (UID column), columns = samples
leeaml_expr_df <- fread("Data/leeaml/leeaml_gene_expression.csv", header = T)
leeaml_expr_df <- as.data.frame(leeaml_expr_df)

gene_names   <- leeaml_expr_df$UID
sample_names <- colnames(leeaml_expr_df)[-1]   # drop "UID" column

# Build genes x samples matrix
leeaml_mat <- as.matrix(leeaml_expr_df[, -1])
rownames(leeaml_mat) <- gene_names
colnames(leeaml_mat) <- sample_names
rm(leeaml_expr_df)
gc()

################################################################################
# Load LeeAML clinical metadata
leeaml_metadata <- fread("Data/leeaml/leeaml_clinical.csv", header = T, skip = 1)
leeaml_metadata <- as.data.frame(leeaml_metadata)
# Rename first column to match downstream merge key
colnames(leeaml_metadata)[1] <- "sample_id"
# Keep only rows that have matching expression data
leeaml_metadata <- leeaml_metadata[leeaml_metadata$sample_id %in% sample_names, ]

################################################################################
# t-SNE on full LeeAML expression matrix
set.seed(123)
leeaml_tsne_out <- Rtsne(X = t(leeaml_mat), dims = 2, perplexity = min(5, floor((ncol(leeaml_mat) - 1) / 3)),
                          pca_center = FALSE, pca_scale = FALSE, pca = FALSE)
leeaml_tsne_df <- as.data.frame(leeaml_tsne_out[["Y"]])
colnames(leeaml_tsne_df) <- c("Tsne1", "Tsne2")
leeaml_tsne_df$sample_id <- sample_names

# Merge t-SNE coordinates with clinical annotations
#leeaml_tsne_with_annotations_df <- merge(leeaml_tsne_df, leeaml_metadata, by = "sample_id", all = TRUE)
write.table(#leeaml_tsne_with_annotations_df,
            leeaml_tsne_df,
            file = "Data/leeaml/LeeAML_Tsne_with_Annotations.csv",
            col.names = T, row.names = F, quote = F, sep = ";")

# t-SNE plot coloured by ELN cytogenetic risk group
eln_col <- "European Leukemia Net (ELN) cyto risk (Döhner et al 2010)"
if (eln_col %in% colnames(leeaml_tsne_df)) {
  leeaml_tsne_df$ELN_Risk <- leeaml_tsne_df[[eln_col]]
} else {
  leeaml_tsne_df$ELN_Risk <- "LeeAML"
}
unique_phenotypes <- unique(leeaml_tsne_df$ELN_Risk)
leeaml_tsne_df$ELN_Risk <- factor(leeaml_tsne_df$ELN_Risk,
                                                     levels = unique_phenotypes)

P28 <- createPalette(max(28, length(unique_phenotypes)), c("#ff0000", "#00ff00", "#0000ff"))
palette(P28)

g_tsne <- ggplot(data = leeaml_tsne_df,
                 aes(x = Tsne1, y = Tsne2, color = ELN_Risk)) +
  geom_point(size = 3) + theme_bw() +
  xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
  theme(legend.title.align = 0.0, legend.text.align = 0.0,
        axis.text.x  = element_text(color = "grey20", size = 16, angle = 0,  hjust = .5, vjust = .5, face = "plain"),
        axis.text.y  = element_text(color = "grey20", size = 16, angle = 0,  hjust = 1,  vjust = 0,  face = "plain"),
        axis.title.x = element_text(color = "grey20", size = 20, angle = 0,  hjust = .5, vjust = 0,  face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text  = element_text(color = "grey20", size = 18, angle = 0,  face = "plain"),
        title        = element_text(color = "grey20", size = 20, face = "plain"))
ggsave(filename = "Data/leeaml/LeeAML_Expr_T-SNE_plot.pdf",
       plot = g_tsne, device = pdf(), height = 8, width = 8, units = "in")
dev.off()

################################################################################
# Build combined expression + annotation data frame (samples x genes)
leeaml_mat <- leeaml_mat[-which(is.na(rownames(leeaml_mat))),]
leeaml_expr_samples_df <- as.data.frame(t(leeaml_mat))
leeaml_expr_samples_df$sample_id <- rownames(leeaml_expr_samples_df)
#Remove columns with na
na_col_ids <- which(is.na(colnames(leeaml_expr_samples_df)))
if (length(na_col_ids)>0)
{
  leeaml_expr_samples_df <- leeaml_expr_samples_df[,-na_col_ids]
}

#final_leeaml_mat <- merge(leeaml_expr_samples_df, leeaml_tsne_with_annotations_df,
#                           by = "sample_id", all = TRUE)
final_leeaml_mat <- leeaml_expr_samples_df

################################################################################
# Pathway activity via ssGSEA (Selected pathways)
load("Data/Selected.pathways.3.4.RData")
rev_hallmark_pathways_of_interest <- Selected.pathways

rev_ssgsea_results <- as.data.frame(t(gsva(ssgseaParam(
  exprData  = leeaml_mat,
  geneSets  = rev_hallmark_pathways_of_interest,
  normalize = TRUE))))
rev_ssgsea_results$sample_id <- rownames(rev_ssgsea_results)

# Merge pathway activity with full matrix
leeaml_mat_with_expr_pa <- merge(final_leeaml_mat, rev_ssgsea_results,
                                       by = "sample_id", all = TRUE)

################################################################################
# Cell-type enrichment (van Galen AML cell types) via ssGSEA
celltype_sig <- parse.van.galen.supp("BeatAML/Data/1-s2.0-S0092867419300947-mmc3.xlsx")
celltypes_of_interest <- c("cDC-like", "GMP-like", "HSC-like", "Monocyte-like",
                            "Progenitor-like", "Promono-like")
rev_celltype_sig <- celltype_sig[celltype_sig$vg_type %in% celltypes_of_interest, ]
celltype_genelist <- NULL
unique_celltypes  <- unique(rev_celltype_sig$vg_type)
for (i in seq_along(unique_celltypes)) {
  ct_name            <- unique_celltypes[i]
  celltype_genelist[[i]] <- rev_celltype_sig[rev_celltype_sig$vg_type == ct_name, ]$display_label
}
names(celltype_genelist) <- unique_celltypes

ssgsea_celltype_results <- as.data.frame(t(gsva(ssgseaParam(
  exprData  = leeaml_mat,
  geneSets  = celltype_genelist,
  normalize = TRUE))))

################################################################################
# Module enrichment (WGCNA modules) via ssGSEA
load("BeatAML/Data/merged_older_wgcna_kme.RData")
unique_modules  <- unique(cur.map$cur_labels)
module_genelist <- NULL
for (i in seq_along(unique_modules)) {
  module_name        <- unique_modules[i]
  module_genelist[[i]] <- cur.map[cur.map$cur_labels == module_name, ]$display_label
}
names(module_genelist) <- unique_modules

ssgsea_module_results <- as.data.frame(t(gsva(ssgseaParam(
  exprData  = leeaml_mat,
  geneSets  = module_genelist,
  normalize = TRUE))))

# Combine cell-type and module ssGSEA scores
ssgsea_celltype_module_combinations <- as.data.frame(cbind(ssgsea_celltype_results, ssgsea_module_results))
ssgsea_celltype_module_combinations$sample_id <- rownames(ssgsea_celltype_module_combinations)

# Merge into full LeeAML data frame
leeaml_mat_with_expr_pa_cts <- merge(leeaml_mat_with_expr_pa,
                                            ssgsea_celltype_module_combinations,
                                            by = "sample_id", all = TRUE)

save(celltype_genelist, module_genelist,
     file = "Data/leeaml/LeeAML_Celltype_Moduletype_info.Rdata")

################################################################################
# Save full combined LeeAML dataset
write.table(leeaml_mat_with_expr_pa_cts,
            file = "Data/leeaml/LeeAML_Set_with_Expr_PA_CTS.csv",
            row.names = F, col.names = T, quote = F, sep = "\t")

################################################################################
# Oncogene + most-varying-gene subset
list_oncogenes  <- fread("Data/Oncogenes.csv", header = T)
oncogenes_symbol <- list_oncogenes$`Gene Symbol`
list_varygenes  <- fread("Data/150_gene.csv", header = T)
varygenes_symbol <- list_varygenes$X

# Identify gene columns (all columns that are gene names, i.e. before annotation columns)
# Gene columns start at column 2 (after sample_id) up to the first non-gene annotation column.
# We detect them by matching against the expression matrix row names.
all_colnames    <- colnames(leeaml_mat_with_expr_pa_cts)
gene_col_ids    <- which(all_colnames %in% gene_names)
oncogene_ids    <- which(all_colnames %in% oncogenes_symbol)
varygene_ids    <- which(all_colnames %in% varygenes_symbol)
finalgene_ids   <- union(oncogene_ids, varygene_ids)
annot_col_ids   <- setdiff(seq_along(all_colnames), gene_col_ids)  # non-gene columns

leeaml_mat_with_onco_expr_pa_cts <- leeaml_mat_with_expr_pa_cts[,
  c(1, finalgene_ids, annot_col_ids[annot_col_ids != 1])]

write.table(leeaml_mat_with_onco_expr_pa_cts,
            file = "Data/leeaml/LeeAML_Set_with_Onco_Var_Expr_PA_CTS.csv",
            row.names = F, col.names = T, quote = F, sep = "\t")

################################################################################
# Write column metadata
leeaml_column_names <- colnames(leeaml_mat_with_expr_pa_cts)
write.table(leeaml_column_names,
            file = "Data/leeaml/LeeAML_Metadata.csv",
            row.names = T, col.names = F, quote = F, sep = "\t")
