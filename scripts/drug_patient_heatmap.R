# =========================================================
# LOAD REQUIRED LIBRARIES
# =========================================================

library(dplyr)
library(ggplot2)
library(ComplexHeatmap)
library(data.table)
library(circlize)
library(colorspace)
library(RColorBrewer)
library(Matrix)
library(grid)

# =========================================================
# FUNCTION TO IDENTIFY PROBLEMATIC ROW/COLUMN COMBINATIONS
# 
# =========================================================
setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/scripts/")

identify_problematic_combs <- function(mat, min_shared_fields = 1) {
  
  exclude_rows <- NULL
  exclude_cols <- NULL
  
  stopifnot(is.matrix(mat))
  
  # -------------------------------------------------------
  # Identify problematic rows
  # -------------------------------------------------------
  
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
  
  # -------------------------------------------------------
  # Identify problematic columns
  # -------------------------------------------------------
  
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
  
  return(list(
    row = exclude_rows,
    column = exclude_cols
  ))
}

# =========================================================
# FUNCTION TO REMOVE PROBLEMATIC COMBINATIONS
# =========================================================

remove_problematic_combs <- function() {
  
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

formals(remove_problematic_combs) <- formals(identify_problematic_combs)

# =========================================================
# LOAD PREDICTION DATA
# =========================================================

data_df <- read.csv(
  "../Results/tabpfn/best_model_analysis/test_predictions_with_CI.csv",
  header = TRUE,
  sep = ","
)

data_df <- as.data.frame(data_df)
data_df$dbgap_subject_id <- as.character(as.vector(data_df$dbgap_subject_id))

# =========================================================
# CALCULATE MAE AND NORMALIZED MAE and NORMALIZED LABELS
# =========================================================

data_df$mae <- abs(
  data_df$label - data_df$pred_mean
)

data_df$nmae <- 1.0*data_df$mae / 300.0

data_df$nlabel <- 1.0*data_df$label/300.0

# =========================================================
# CREATE EMPTY MATRIX
# rows = patients
# cols = inhibitors
# =========================================================

mae_matrix <- as.matrix(
  Matrix(
    table(
      data_df$dbgap_subject_id,
      data_df$inhibitor
    )
  )
)

mae_matrix[mae_matrix == 0] <- NA

nlabel_matrix <- mae_matrix

rownames(mae_matrix) <- unique(data_df$dbgap_subject_id)
colnames(mae_matrix) <- unique(data_df$inhibitor)

rownames(nlabel_matrix) <- unique(data_df$dbgap_subject_id)
colnames(nlabel_matrix) <- unique(data_df$inhibitor)

# =========================================================
# FILL MATRIX WITH NMAE VALUES
# =========================================================

for (i in 1:nrow(data_df)) {
  
  sample_id <- data_df$dbgap_subject_id[i]
  
  inhibitor_id <- data_df$inhibitor[i]
  
  nmae_val <- data_df$nmae[i]
  
  nlabel_val <- data_df$nlabel[i]
  
  #mae_matrix[sample_id, inhibitor_id] <- nmae_val
  mae_matrix[sample_id, inhibitor_id] <- nmae_val
  
  nlabel_matrix[sample_id, inhibitor_id] <- nlabel_val
}

# =========================================================
# REMOVE PROBLEMATIC ROWS/COLUMNS
# =========================================================

rev_mae_matrix <- remove_problematic_combs(
  mae_matrix,
  min_shared_fields = 50
)

# =========================================================
# CALCULATE PEARSON CORRELATION FOR EACH DRUG
# =========================================================

correlations <- split(
  data_df,
  data_df$inhibitor
) %>%
  lapply(function(data)
    cor(
      data$pred_mean,
      data$label,
      method = "pearson"
    )
  )

correlations_df <- data.frame(
  drug = names(correlations),
  correlation = unlist(correlations)
)

subset_correlations_df <- correlations_df[
  correlations_df$drug %in% colnames(rev_mae_matrix),
]

correlation_vec <- subset_correlations_df$correlation

names(correlation_vec) <- subset_correlations_df$drug

# =========================================================
# DEFINE COLOR FUNCTIONS
# =========================================================

col_fun <- colorRamp2(
  c(0, 0.125, 0.25),
  c("blue", "white", "red")
)

col_fun2 <- colorRamp2(
  c(0, 0.4, 0.8),
  c("blue", "white", "red")
)

# =========================================================
# CREATE CORRELATION ANNOTATION
# =========================================================

row_ha <- rowAnnotation(
  r = correlation_vec,
  col = list(r = col_fun2)
)

# =========================================================
# TRANSPOSE MATRIX
# rows = drugs
# cols = patients
# =========================================================

mat <- t(rev_mae_matrix)

# =========================================================
# REPLACE NA VALUES WITH ROW MEAN
# =========================================================

for(i in 1:nrow(mat)) {
  
  row_mean <- mean(
    mat[i, ],
    na.rm = TRUE
  )
  
  mat[i, is.na(mat[i, ])] <- row_mean
}

# =========================================================
# REMOVE ZERO VARIANCE ROWS/COLUMNS
# =========================================================

mat <- mat[
  apply(mat, 1, sd, na.rm = TRUE) > 0,
  apply(mat, 2, sd, na.rm = TRUE) > 0
]

# =========================================================
# SELECT IMPORTANT DRUG LABELS
# =========================================================

selected_drugs <- c(
  "Venetoclax",
  "Elesclomol",
  "Selumetinib (AZD6244)",
  "JNJ-28312141",
  "Trametinib (GSK1120212)",
  "Rapamycin",
  "Motesanib (AMG-706)",
  "Sorafenib",
  "Panobinostat",
  "Dasatinib",
  "BMS-754807",
  "Birinapant",
  "Perhexiline maleate",
  "Ranolazine",
  "Indisulam",
  "Selinexor",
  "GSK-2879552",
  "AT-101",
  "Lovastatin",
  "A-674563"
)

# =========================================================
# ADD EXTRA DRUG LABELS
# =========================================================

extra_drugs <- setdiff(
  rownames(mat),
  selected_drugs
)

extra_drugs <- head(extra_drugs, 20)

final_drugs <- c(
  selected_drugs,
  extra_drugs
)

# =========================================================
# SHOW LABELS ONLY FOR SELECTED DRUGS
# =========================================================

custom_labels <- ifelse(
  rownames(mat) %in% final_drugs,
  rownames(mat),
  ""
)

# =========================================================
# ORDER PATIENTS
# High NMAE -> left
# Low NMAE -> right
# =========================================================

patient_order <- order(
  colMeans(mat, na.rm = TRUE),
  decreasing = TRUE
)

mat <- mat[, patient_order]

# =========================================================
# SAVE NMAE HEATMAP TO PDF
# =========================================================

pdf(
  "../Results/Figures/Drug_vs_Patient_Heatmap_NMAE.pdf",
  width = 16,
  height = 12
)

# =========================================================
# CREATE NMAE HEATMAP
# =========================================================

ht <- Heatmap(
  
  mat,
  
  name = "NMAE",
  
  na_col = "grey",
  
  rect_gp = gpar(
    col = "white",
    lwd = 1
  ),
  
  row_km = 3,
  column_km = 3,
  
  row_title = "Drug Clusters",
  column_title = "Patient Clusters",
  
  cluster_columns = FALSE,
  cluster_rows = TRUE,
  
  show_column_dend = FALSE,
  show_row_dend = FALSE,
  
  show_column_names = FALSE,
  
  clustering_distance_rows = "pearson",
  
  clustering_method_rows = "centroid",
  
  row_labels = custom_labels,
  
  row_names_gp = gpar(
    fontsize = 8,
    fontface = "bold"
  ),
  
  right_annotation = row_ha,
  
  col = col_fun,
  
  border = TRUE,
  
  row_gap = unit(2, "mm"),
  column_gap = unit(2, "mm")
)

ComplexHeatmap::draw(ht)

dev.off()

# =========================================================
# NORMALIZE PREDICTIONS
# =========================================================

data_df$norm_prediction <- data_df$pred_mean / 300

# =========================================================
# CREATE EMPTY MATRIX FOR NORMALIZED PREDICTIONS
# =========================================================
pred_matrix <- as.matrix(
  Matrix(
    table(
      data_df$dbgap_subject_id,
      data_df$inhibitor
    )
  )
)

pred_matrix[pred_matrix == 0] <- NA

# =========================================================
# FILL MATRIX WITH NORMALIZED PREDICTIONS
# =========================================================

for(i in 1:nrow(data_df)) {
  
  sample_id <- data_df$dbgap_subject_id[i]
  
  inhibitor_id <- data_df$inhibitor[i]
  
  pred_val <- data_df$norm_prediction[i]
  
  pred_matrix[sample_id, inhibitor_id] <- pred_val
}

# =========================================================
# TRANSPOSE MATRIX
# =========================================================

mat <- t(pred_matrix)

# =========================================================
# REPLACE NA VALUES WITH ROW MEAN
# =========================================================

for(i in 1:nrow(mat)) {
  
  row_mean <- mean(
    mat[i, ],
    na.rm = TRUE
  )
  
  mat[i, is.na(mat[i, ])] <- row_mean
}

# =========================================================
# REMOVE ZERO VARIANCE ROWS/COLUMNS
# =========================================================

mat <- mat[
  apply(mat, 1, sd, na.rm = TRUE) > 0,
  apply(mat, 2, sd, na.rm = TRUE) > 0
]

# =========================================================
# ORDER PATIENTS
# High prediction -> left
# Low prediction -> right
# =========================================================

patient_order <- order(
  colMeans(mat, na.rm = TRUE),
  decreasing = TRUE
)

mat <- mat[, patient_order]

# =========================================================
# COLOR FUNCTION FOR NORMALIZED PREDICTIONS
# =========================================================

col_fun <- colorRamp2(
  c(0, 0.5, 1),
  c("blue", "white", "red")
)

rev_correlations_df <- correlations_df[correlations_df$drug %in% rownames(mat), ]

# Fix: match order to rownames(mat) and use a named vector
rev_correlations_df <- rev_correlations_df[match(rownames(mat), rev_correlations_df$drug), ]
rev_correlation_vec <- rev_correlations_df$correlation
names(rev_correlation_vec) <- rev_correlations_df$drug

row_ha <- rowAnnotation(
  r = rev_correlation_vec,
  col = list(r = col_fun2)
)

# =========================================================
# SHOW LABELS ONLY FOR SELECTED DRUGS
# =========================================================

custom_labels <- ifelse(
  rownames(mat) %in% final_drugs,
  rownames(mat),
  ""
)

# =========================================================
# SAVE NORMALIZED PREDICTION HEATMAP
# =========================================================

pdf(
  "../Results/Figures/Drug_vs_Patient_NormalizedPrediction_Heatmap.pdf",
  width = 16,
  height = 12
)

# =========================================================
# CREATE NORMALIZED PREDICTION HEATMAP
# =========================================================

ht <- Heatmap(
  
  mat,
  
  name = "Normalized Prediction",
  
  na_col = "grey",
  
  rect_gp = gpar(
    col = "white",
    lwd = 1
  ),
  
  row_km = 3,
  column_km = 3,
  
  row_title = "Drug Clusters",
  column_title = "Patient Clusters",
  
  cluster_columns = FALSE,
  cluster_rows = TRUE,
  
  show_column_dend = FALSE,
  show_row_dend = FALSE,
  
  show_column_names = FALSE,
  
  clustering_distance_rows = "pearson",
  
  clustering_method_rows = "centroid",
  
  row_labels = custom_labels,
  
  row_names_gp = gpar(
    fontsize = 8,
    fontface = "bold"
  ),
  
  right_annotation = row_ha,
  
  col = col_fun,
  
  border = TRUE,
  
  row_gap = unit(2, "mm"),
  column_gap = unit(2, "mm")
)

ComplexHeatmap::draw(ht)

dev.off()

# =========================================================
# SAVE NORMALIZED PREDICTION HEATMAP
# WITHOUT CORRELATION ANNOTATION
# =========================================================

pdf(
  "Drug_vs_Patient_NormalizedPrediction_Heatmap_NoCorrelation.pdf",
  width = 16,
  height = 12
)

# =========================================================
# CREATE HEATMAP WITHOUT CORRELATION
# =========================================================

ht <- Heatmap(
  
  mat,
  
  name = "Normalized Prediction",
  
  na_col = "grey",
  
  rect_gp = gpar(
    col = "white",
    lwd = 1
  ),
  
  row_km = 3,
  column_km = 3,
  
  row_title = "Drug Clusters",
  column_title = "Patient Clusters",
  
  cluster_columns = FALSE,
  cluster_rows = TRUE,
  
  show_column_dend = FALSE,
  show_row_dend = FALSE,
  
  show_column_names = FALSE,
  
  clustering_distance_rows = "pearson",
  
  clustering_method_rows = "centroid",
  
  row_labels = custom_labels,
  
  row_names_gp = gpar(
    fontsize = 8,
    fontface = "bold"
  ),
  
  right_annotation = NULL,
  
  col = col_fun,
  
  border = TRUE,
  
  row_gap = unit(2, "mm"),
  column_gap = unit(2, "mm")
)

draw(ht)

dev.off()