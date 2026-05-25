# =========================================================
# LOAD REQUIRED LIBRARIES
# =========================================================

library(dplyr)
library(ComplexHeatmap)
library(data.table)
library(circlize)
library(Matrix)
library(grid)

setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/scripts/")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

identify_problematic_combs <- function(mat, min_shared_fields = 1) {

  exclude_rows <- NULL
  exclude_cols <- NULL

  stopifnot(is.matrix(mat))

  for (k in 1:nrow(mat)) {

    candidate_rows <- setdiff(1:nrow(mat), exclude_rows)
    problem_row_combs <- NULL

    for (i in candidate_rows) {
      i_idx <- which(candidate_rows == i)
      for (j in candidate_rows[i_idx:length(candidate_rows)]) {
        if (sum(!is.na(mat[i, ]) & !is.na(mat[j, ])) <= min_shared_fields) {
          problem_row_combs <- rbind(problem_row_combs, c(i, j))
        }
      }
    }

    if (is.null(problem_row_combs)) break

    exclude_rows <- c(
      exclude_rows,
      as.integer(names(which.max(table(problem_row_combs))))
    )
  }

  for (k in 1:ncol(mat)) {

    candidate_cols <- setdiff(1:ncol(mat), exclude_cols)
    problem_col_combs <- NULL

    for (i in candidate_cols) {
      i_idx <- which(candidate_cols == i)
      for (j in candidate_cols[i_idx:length(candidate_cols)]) {
        if (sum(!is.na(mat[, i]) & !is.na(mat[, j])) <= min_shared_fields) {
          problem_col_combs <- rbind(problem_col_combs, c(i, j))
        }
      }
    }

    if (is.null(problem_col_combs)) break

    exclude_cols <- c(
      exclude_cols,
      as.integer(names(which.max(table(problem_col_combs))))
    )
  }

  return(list(row = exclude_rows, column = exclude_cols))
}

remove_problematic_combs <- function(mat, min_shared_fields = 1) {

  problematic_combs <- identify_problematic_combs(
    mat = mat,
    min_shared_fields = min_shared_fields
  )

  if (!is.null(problematic_combs$row)) {
    mat <- mat[-problematic_combs$row, ]
  }

  if (!is.null(problematic_combs$column)) {
    mat <- mat[, -problematic_combs$column]
  }

  return(mat)
}

# Build drug-patient matrix from a data frame
build_pred_matrix <- function(df, sample_col, inhibitor_col, pred_col) {

  df[[sample_col]]   <- as.character(df[[sample_col]])
  df[[inhibitor_col]] <- as.character(df[[inhibitor_col]])

  mat_counts <- as.matrix(Matrix(table(df[[sample_col]], df[[inhibitor_col]])))
  mat_counts[mat_counts == 0] <- NA

  for (i in seq_len(nrow(df))) {
    mat_counts[df[[sample_col]][i], df[[inhibitor_col]][i]] <- df[[pred_col]][i]
  }

  return(mat_counts)
}

# Compute per-drug Pearson correlation between predictions and labels
compute_correlations <- function(df, inhibitor_col, pred_col, label_col) {

  split(df, df[[inhibitor_col]]) %>%
    lapply(function(d) cor(d[[pred_col]], d[[label_col]], method = "pearson")) %>%
    { data.frame(drug = names(.), correlation = unlist(.), stringsAsFactors = FALSE) }
}

# Main plotting function
plot_heatmap <- function(mat, correlation_vec, out_pdf, title,
                         row_km = 3, col_km = 3,
                         pdf_width = 14, pdf_height = 10) {

  col_fun <- colorRamp2(c(0, 0.5, 1),   c("blue", "white", "red"))
  col_fun2 <- colorRamp2(c(0, 0.4, 0.8), c("blue", "white", "red"))

  row_ha <- rowAnnotation(
    r = correlation_vec,
    col = list(r = col_fun2)
  )

  ht <- Heatmap(
    mat,
    name               = "Normalized AUC",
    na_col             = "grey",
    rect_gp            = gpar(col = "white", lwd = 1),
    row_km             = row_km,
    column_km          = col_km,
    row_title          = "Drug Clusters",
    column_title       = title,
    cluster_columns    = FALSE,
    cluster_rows       = TRUE,
    show_column_dend   = FALSE,
    show_row_dend      = FALSE,
    show_column_names  = FALSE,
    clustering_distance_rows = "pearson",
    clustering_method_rows   = "centroid",
    row_labels         = rownames(mat),
    row_names_gp       = gpar(fontsize = 9, fontface = "bold"),
    right_annotation   = row_ha,
    col                = col_fun,
    border             = TRUE,
    row_gap            = unit(2, "mm"),
    column_gap         = unit(2, "mm")
  )

  pdf(out_pdf, width = pdf_width, height = pdf_height)
  ComplexHeatmap::draw(ht)
  dev.off()

  message("Saved: ", out_pdf)
}

# =========================================================
# LeeAML HEATMAP
# =========================================================

message("=== Processing LeeAML ===")

lee_df <- read.csv(
  "../Results/tabpfn_leeaml/optuna/Embed_Feat_Var_dbgap_rnaseq_sample_optuna_predictions.csv",
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)

# All 49 drugs in this file already overlap with BeatAML
# Normalize predictions to [0, 1] using max AUC = 300
lee_df$norm_auc <- lee_df$labels / 800.0

# Build matrix (rows = patients, cols = drugs) then transpose
lee_mat_raw <- build_pred_matrix(
  df           = lee_df,
  sample_col   = "sample_id",
  inhibitor_col = "inhibitor",
  pred_col     = "norm_auc"
)

lee_mat_raw <- remove_problematic_combs(lee_mat_raw, min_shared_fields = 10)

# Transpose: rows = drugs, cols = patients
lee_mat <- t(lee_mat_raw)

# Impute NA with row mean
for (i in seq_len(nrow(lee_mat))) {
  row_mean <- mean(lee_mat[i, ], na.rm = TRUE)
  lee_mat[i, is.na(lee_mat[i, ])] <- row_mean
}

# Remove zero-variance rows/cols
lee_mat <- lee_mat[
  apply(lee_mat, 1, sd, na.rm = TRUE) > 0,
  apply(lee_mat, 2, sd, na.rm = TRUE) > 0
]

# Per-drug Pearson correlation (predictions vs measured labels)
lee_cor_df <- compute_correlations(lee_df, "inhibitor", "predictions", "labels")
lee_cor_df <- lee_cor_df[lee_cor_df$drug %in% rownames(lee_mat), ]
lee_cor_df <- lee_cor_df[match(rownames(lee_mat), lee_cor_df$drug), ]
lee_cor_vec <- setNames(lee_cor_df$correlation, lee_cor_df$drug)

# Order patients: high mean prediction -> left
lee_mat <- lee_mat[, order(colMeans(lee_mat, na.rm = TRUE), decreasing = TRUE)]

plot_heatmap(
  mat            = lee_mat,
  correlation_vec = lee_cor_vec,
  out_pdf        = "../Results/Figures/Drug_vs_Patient_NormalizedAUC_Heatmap_LeeAML.pdf",
  title          = "Patient Clusters (LeeAML — BeatAML-overlapping drugs)",
  row_km         = 3,
  col_km         = 3,
  pdf_width      = 14,
  pdf_height     = 10
)

# =========================================================
# FIMM-AML HEATMAP
# =========================================================

message("=== Processing FIMM-AML ===")

fimm_df <- read.csv(
  "../Results/tabpfn_fimmaml/optuna/Embed_Feat_Var_optuna_predictions.csv",
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)

# All 75 drugs in this file already overlap with BeatAML
# Normalize predictions to [0, 1] using max AUC = 300
fimm_df$norm_DSS <- fimm_df$DSS / 50

# Build matrix (rows = patients, cols = drugs) then transpose
fimm_mat_raw <- build_pred_matrix(
  df            = fimm_df,
  sample_col    = "sample_id",
  inhibitor_col = "inhibitor",
  pred_col      = "norm_DSS"
)

fimm_mat_raw <- remove_problematic_combs(fimm_mat_raw, min_shared_fields = 10)

# Transpose: rows = drugs, cols = patients
fimm_mat <- t(fimm_mat_raw)

# Impute NA with row mean
for (i in seq_len(nrow(fimm_mat))) {
  row_mean <- mean(fimm_mat[i, ], na.rm = TRUE)
  fimm_mat[i, is.na(fimm_mat[i, ])] <- row_mean
}

# Remove zero-variance rows/cols
fimm_mat <- fimm_mat[
  apply(fimm_mat, 1, sd, na.rm = TRUE) > 0,
  apply(fimm_mat, 2, sd, na.rm = TRUE) > 0
]

# Per-drug Pearson correlation (predicted_auc vs DSS)
fimm_cor_df <- compute_correlations(fimm_df, "inhibitor", "predicted_auc", "DSS")
fimm_cor_df <- fimm_cor_df[fimm_cor_df$drug %in% rownames(fimm_mat), ]
fimm_cor_df <- fimm_cor_df[match(rownames(fimm_mat), fimm_cor_df$drug), ]
fimm_cor_vec <- setNames(fimm_cor_df$correlation, fimm_cor_df$drug)

# Order patients: high mean prediction -> left
fimm_mat <- fimm_mat[, order(colMeans(fimm_mat, na.rm = TRUE), decreasing = TRUE)]

plot_heatmap(
  mat            = fimm_mat,
  correlation_vec = fimm_cor_vec,
  out_pdf        = "../Results/Figures/Drug_vs_Patient_NormalizedPrediction_Heatmap_FIMMAML.pdf",
  title          = "Patient Clusters (FIMM-AML — BeatAML-overlapping drugs)",
  row_km         = 3,
  col_km         = 3,
  pdf_width      = 16,
  pdf_height     = 12
)

message("=== Done ===")
