suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})

#.libPaths(c("/export/cse/rmall/R/x86_64-redhat-linux-gnu-library/3.5", .libPaths()))

BASE <- "/export/qcai-omics/Raghvendra/PT-AML2.0"
OUT  <- file.path(BASE, "Results/Tables/best_model_correlation_interpretation")
ASSO <- file.path(OUT, "top10_drug_features_asso")
dir.create(ASSO, showWarnings = FALSE, recursive = TRUE)

# =========================================================
# SECTION 1: Load Data
# =========================================================

predictions <- read.csv(file.path(OUT, "prediction_data.csv"), stringsAsFactors = FALSE)
feat_df      <- read.csv(file.path(OUT, "feature_subset.csv"), stringsAsFactors = FALSE)
data         <- read.csv(file.path(BASE, "Data/Test_Set_Var_with_Drug_Embedding_Patient_Info.csv"),
                         stringsAsFactors = FALSE, check.names = FALSE)

# =========================================================
# SECTION 2: Identify Top-10 Drugs (Pearson r, >=100 patients)
# =========================================================

drug_stats <- predictions %>%
  group_by(inhibitor) %>%
  filter(n_distinct(dbgap_rnaseq_sample) >= 100) %>%
  summarise(pearson = cor(predictions, labels, method = "pearson", use = "complete.obs"),
            n_pat   = n_distinct(dbgap_rnaseq_sample), .groups = "drop") %>%
  arrange(desc(pearson))

top10_drugs      <- drug_stats$inhibitor[1:10]
top10_drugs_safe <- gsub("[^A-Za-z0-9_]", "_", top10_drugs)

cat("Top-10 drugs:\n")
print(data.frame(drug    = top10_drugs,
                 safe    = top10_drugs_safe,
                 pearson = round(drug_stats$pearson[1:10], 3),
                 n_pat   = drug_stats$n_pat[1:10]))

# =========================================================
# SECTION 3: Merge Predictions with Feature Data
# =========================================================

keep_feats <- feat_df$feature[!grepl("^LS_", feat_df$feature)]
keep_feats <- intersect(keep_feats, colnames(data))

cols_to_keep  <- intersect(c("dbgap_rnaseq_sample", "inhibitor", keep_feats), colnames(data))
filtered_data <- data[, cols_to_keep, drop = FALSE]

filtered_df <- predictions %>%
  filter(inhibitor %in% top10_drugs) %>%
  inner_join(filtered_data, by = c("dbgap_rnaseq_sample", "inhibitor"))

cat(sprintf("\nMerged data: %d rows, %d cols\n", nrow(filtered_df), ncol(filtered_df)))

# =========================================================
# SECTION 4: Feature Association Analysis per Drug
#            Wilcoxon for binary | Linear regression for continuous
# =========================================================

all_significant_results <- list()

for (i in seq_along(top10_drugs)) {

  drug      <- top10_drugs[i]
  drug_safe <- top10_drugs_safe[i]

  cat("\n====================================\n")
  cat("Processing:", drug, "\n")

  drug_df <- filtered_df[filtered_df$inhibitor == drug, ]
  cat("Samples:", nrow(drug_df), "\n")

  assoc_results <- data.frame(
    feature   = character(),
    p_value   = numeric(),
    q_value   = numeric(),
    method    = character(),
    beta      = numeric(),
    r_squared = numeric(),
    stringsAsFactors = FALSE
  )

  for (col in keep_feats) {

    if (!col %in% colnames(drug_df)) next

    feature_data <- drug_df[[col]]
    auc          <- drug_df$predictions / 300

    if (all(is.na(feature_data)) || length(unique(na.omit(feature_data))) <= 1) next

    unique_vals <- unique(na.omit(feature_data))

    if (all(unique_vals %in% c(0, 1))) {
      # Binary → Wilcoxon
      group0 <- auc[feature_data == 0]
      group1 <- auc[feature_data == 1]

      if (length(group0) > 2 && length(group1) > 2) {
        test <- tryCatch(wilcox.test(group0, group1), error = function(e) NULL)
        if (!is.null(test)) {
          assoc_results <- rbind(assoc_results, data.frame(
            feature   = col,
            p_value   = test$p.value,
            q_value   = NA_real_,
            method    = "wilcox_test",
            beta      = mean(group1, na.rm = TRUE) - mean(group0, na.rm = TRUE),
            r_squared = NA_real_,
            stringsAsFactors = FALSE
          ))
        }
      }

    } else {
      # Continuous → Linear regression
      df_tmp <- data.frame(auc = auc, x = feature_data)
      df_tmp <- df_tmp[complete.cases(df_tmp), ]

      if (nrow(df_tmp) > 3) {
        fit <- tryCatch(lm(auc ~ x, data = df_tmp), error = function(e) NULL)
        if (!is.null(fit)) {
          s <- summary(fit)
          if ("x" %in% rownames(s$coefficients)) {
            assoc_results <- rbind(assoc_results, data.frame(
              feature   = col,
              p_value   = s$coefficients["x", "Pr(>|t|)"],
              q_value   = NA_real_,
              method    = "linear_regression",
              beta      = s$coefficients["x", "Estimate"],
              r_squared = s$r.squared,
              stringsAsFactors = FALSE
            ))
          }
        }
      }
    }
  }

  if (nrow(assoc_results) == 0) {
    cat("No valid features tested\n")
    all_significant_results[[drug]] <- character(0)
    next
  }

  assoc_results$q_value <- p.adjust(assoc_results$p_value, method = "fdr")

  sig <- assoc_results[
    assoc_results$q_value < 0.05 &
    (is.na(assoc_results$r_squared) | assoc_results$r_squared > 0.15), ]

  cat("Total tested:", nrow(assoc_results), "| Significant:", nrow(sig), "\n")

  write.csv(sig,
    file.path(ASSO, paste0(drug_safe, "_LR_significant_features.csv")),
    row.names = FALSE)

  if (nrow(sig) > 0) {
    top100 <- sig %>% arrange(q_value) %>% head(100) %>% pull(feature)
    all_significant_results[[drug]] <- top100
  } else {
    all_significant_results[[drug]] <- character(0)
  }
}

cat("\nALL DRUGS COMPLETED\n")

# =========================================================
# SECTION 5: Feature Counts (top-100 per drug)
# =========================================================

feat_counts_top100 <- table(unlist(all_significant_results))
feat_count_df <- data.frame(
  CommonFeature = names(feat_counts_top100),
  Drugs         = as.integer(feat_counts_top100),
  stringsAsFactors = FALSE
) %>% arrange(desc(Drugs))

write.csv(feat_count_df,
  file.path(ASSO, "feature_counts_across_drugs_top100.csv"), row.names = FALSE)

# =========================================================
# SECTION 6: Reload All Significant Features and Count
# =========================================================

all_sig_features <- c()

for (i in seq_along(top10_drugs)) {
  f <- file.path(ASSO, paste0(top10_drugs_safe[i], "_LR_significant_features.csv"))
  if (file.exists(f)) {
    tmp <- read.csv(f, stringsAsFactors = FALSE)
    all_sig_features <- c(all_sig_features, tmp$feature)
  }
}

feat_count_all_df <- data.frame(
  CommonFeature = names(table(all_sig_features)),
  Drugs         = as.integer(table(all_sig_features)),
  stringsAsFactors = FALSE
) %>% arrange(desc(Drugs))

write.csv(feat_count_all_df,
  file.path(ASSO, "feature_counts_across_all_drugs.csv"), row.names = FALSE)

# =========================================================
# SECTION 7: Beta Matrix Across Drugs
# =========================================================

beta_matrix <- data.frame(Feature = feat_count_all_df$CommonFeature, stringsAsFactors = FALSE)
r2_matrix   <- data.frame(Feature = feat_count_all_df$CommonFeature, stringsAsFactors = FALSE)
q_matrix    <- data.frame(Feature = feat_count_all_df$CommonFeature, stringsAsFactors = FALSE)

for (i in seq_along(top10_drugs)) {
  f <- file.path(ASSO, paste0(top10_drugs_safe[i], "_LR_significant_features.csv"))
  if (!file.exists(f)) {
    beta_matrix[[top10_drugs[i]]] <- NA_real_
    r2_matrix[[top10_drugs[i]]]   <- NA_real_
    q_matrix[[top10_drugs[i]]]    <- NA_real_
    next
  }
  drug_res <- read.csv(f, stringsAsFactors = FALSE)
  beta_matrix[[top10_drugs[i]]] <- sapply(feat_count_all_df$CommonFeature, function(feat) {
    row <- drug_res[drug_res$feature == feat, ]
    if (nrow(row) > 0) row$beta[1] else NA_real_
  })
  r2_matrix[[top10_drugs[i]]] <- sapply(feat_count_all_df$CommonFeature, function(feat) {
    row <- drug_res[drug_res$feature == feat, ]
    if (nrow(row) > 0) row$r_squared[1] else NA_real_
  })
  q_matrix[[top10_drugs[i]]] <- sapply(feat_count_all_df$CommonFeature, function(feat) {
    row <- drug_res[drug_res$feature == feat, ]
    if (nrow(row) > 0) row$q_value[1] else NA_real_
  })
}

write.csv(beta_matrix,
  file.path(ASSO, "betas_for_selected_features_across_drugs.csv"), row.names = FALSE)

# =========================================================
# SECTION 8: ComplexHeatmap — features significant in >=9 drugs
# =========================================================

core_feats <- feat_count_all_df$CommonFeature[feat_count_all_df$Drugs >= 9]
cat(sprintf("\n%d features significant in >8 drugs\n", length(core_feats)))

if (length(core_feats) > 0) {

  extract_mat <- function(df, feats, drugs) {
    sub <- df[df$Feature %in% feats, ]
    rownames(sub) <- sub$Feature
    sub$Feature   <- NULL
    as.matrix(sub[, drugs, drop = FALSE])
  }

  beta_mat <- extract_mat(beta_matrix, core_feats, top10_drugs)
  r2_mat   <- extract_mat(r2_matrix,   core_feats, top10_drugs)
  q_mat    <- extract_mat(q_matrix,    core_feats, top10_drugs)

  # Sort rows by mean R² descending
  mean_r2 <- rowMeans(r2_mat, na.rm = TRUE)
  ord     <- order(mean_r2, decreasing = TRUE)
  beta_mat <- beta_mat[ord, ]
  r2_mat   <- r2_mat[ord, ]
  q_mat    <- q_mat[ord, ]
  mean_r2  <- mean_r2[rownames(beta_mat)]

  # Significance stars from q-value
  sig_stars <- matrix("", nrow = nrow(q_mat), ncol = ncol(q_mat),
                      dimnames = dimnames(q_mat))
  sig_stars[!is.na(q_mat) & q_mat < 0.001] <- "***"
  sig_stars[!is.na(q_mat) & q_mat >= 0.001 & q_mat < 0.01] <- "**"
  sig_stars[!is.na(q_mat) & q_mat >= 0.01  & q_mat < 0.05] <- "*"

  ra <- rowAnnotation(
    `Mean R²` = anno_barplot(
      mean_r2,
      bar_width  = 0.8,
      gp         = gpar(fill = "grey40", col = NA),
      width      = unit(3, "cm"),
      axis_param = list(gp = gpar(fontsize = 13))
    ),
    annotation_name_gp  = gpar(fontsize = 15, fontface = "bold"),
    annotation_name_rot = 0
  )

  lim     <- max(abs(beta_mat), na.rm = TRUE)
  col_fun <- colorRamp2(c(-lim, 0, lim), c("blue", "white", "red"))

  ht <- Heatmap(
    beta_mat,
    name              = "Beta",
    col               = col_fun,
    width             = unit(ncol(beta_mat) * 18, "mm"),
    rect_gp           = gpar(col = "white", lwd = 3),
    cluster_rows      = FALSE,
    cluster_columns   = FALSE,
    show_row_names    = TRUE,
    show_column_names = TRUE,
    row_names_gp      = gpar(fontsize = 15),
    row_names_max_width = max_text_width(rownames(beta_mat), gp = gpar(fontsize = 15)) + unit(4, "mm"),
    column_names_gp   = gpar(fontsize = 15),
    column_names_rot  = 45,
    right_annotation  = ra,
    heatmap_legend_param = list(
      title     = "Beta",
      title_gp  = gpar(fontsize = 15, fontface = "bold"),
      labels_gp = gpar(fontsize = 13)
    ),
    cell_fun = function(j, i, x, y, width, height, fill) {
      b <- beta_mat[i, j]
      s <- sig_stars[i, j]
      if (!is.na(b)) {
        grid.text(sprintf("%.2f%s", b, s), x, y, gp = gpar(fontsize = 12))
      }
    }
  )

  pdf(file.path(OUT, "top10_drug_feature_heatmap.pdf"),
      width = 15, height = max(9, length(core_feats) * 0.3 + 3))
  draw(ht, padding = unit(c(4, 4, 4, 4), "mm"))
  dev.off()
  cat("Heatmap saved\n")

} else {
  cat("No features meet the >=8 drug threshold\n")
}

# # =========================================================
# # SECTION 9: LM scatter plots for features in >=5 drugs
# # =========================================================
# 
# plot_feats <- feat_count_df$CommonFeature[feat_count_df$Drugs >= 5]
# 
# make_lm_plot <- function(df, column) {
#   col_sym     <- rlang::sym(column)
#   df$norm_auc <- df$predictions / 300
#   ggplot(df, aes(x = !!col_sym, y = norm_auc, color = inhibitor, group = inhibitor)) +
#     geom_point(size = 0.8) +
#     geom_smooth(method = "lm", lwd = 1, color = "black") +
#     stat_cor(aes(label = paste(..rr.label.., "*\",\"~", ..p.label..)),
#              parse = TRUE, label.x.npc = "left", label.y = 1, size = 3.5) +
#     facet_wrap(~ inhibitor, ncol = 5, scales = "free") +
#     labs(x = column, y = "Normalised Predicted AUC",
#          title = paste0(column, " — test set")) +
#     theme_classic() +
#     theme(legend.position = "none",
#           axis.text  = element_text(size = 10, face = "bold"),
#           axis.title = element_text(size = 12, face = "bold"),
#           strip.text = element_text(size = 9,  face = "bold"),
#           plot.title = element_text(hjust = 0.5, size = 13, face = "bold"))
# }
# 
# pdf(file.path(OUT, "All_CommonFeature_Associations_test_set.pdf"), height = 8, width = 12)
# for (feat in plot_feats) {
#   if (!feat %in% colnames(filtered_df)) next
#   col_data <- filtered_df[[feat]]
#   if (!is.numeric(col_data) || sum(!is.na(col_data)) < 3 ||
#       length(unique(col_data[!is.na(col_data)])) <= 1) next
#   p <- tryCatch(make_lm_plot(filtered_df, feat), error = function(e) NULL)
#   if (!is.null(p)) print(p)
# }
# dev.off()
# 
# cat("Association plots saved\n")
