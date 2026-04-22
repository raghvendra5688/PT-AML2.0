library(readr)
library(data.table)
library(dplyr)
library(ggplot2)
library(factoextra)
library(ggfortify)
library(tidyverse)
library(cluster)
library(fpc)
library(data.table)
library(ggplot2)
library(gridExtra)
data1 <- fread("Revised_Training_Set_with_Onco_Var_Expr_Clin_PA_CTS_Mut.csv",header=T)
#data2 <- fread("Revised_Training_Set_with_Expr_Clin_PA_CTS_P2.csv.gz",header=F)
data1 <- as.data.frame(data1)
data2 <- as.data.frame(data2)
colnames(data2) <- colnames(data1)
train_data <- data1
data3 <- fread("Revised_Test_Set_with_Onco_Var_Expr_Clin_PA_CTS_Mut.csv",header = T)
data3 <- as.data.frame(data3)
test_data <- data3
train_data$set <- "train"
test_data$set  <- "test"

data_all <- rbind(train_data, test_data)
group <- data_all$set
exclude_cols <- c("dbgap_rnaseq_sample","dbgap_subject_id","dbgap_dnaseq_sample",
                  "cohort","used_manuscript_analyses","manuscript_dnaseq",
                  "manuscript_rnaseq","manuscript_inhibitor","set")

feature_cols <- setdiff(colnames(data_all), exclude_cols)
outdir <- "EDA_train_vs_test"
dir.create(outdir, showWarnings = FALSE)


evaluate_feature <- function(df, feature, group){
  values <- df[[feature]]
  
  # NUMERIC → Wilcoxon
  if (is.numeric(values)) {
    pval <- tryCatch({
      wilcox.test(values ~ group, exact = FALSE)$p.value
    }, error = function(e) NA)
    
    plot_obj <- ggplot(df, aes(x = group, y = values, fill = group)) +
      geom_boxplot() +
      theme_bw(base_size = 8) +
      ggtitle(paste0(feature, "\nP = ", signif(pval, 3))) +
      theme(legend.position = "none")
    
    return(list(feature = feature, p = pval, type = "numeric", plot = plot_obj))
  }
  
  # CATEGORICAL → Chi-square
  tbl <- table(values, group)
  
  if (min(dim(tbl)) < 2) {
    pval <- NA
  } else {
    pval <- tryCatch(chisq.test(tbl)$p.value, error = function(e) NA)
  }
  
  plot_obj <- ggplot(df, aes(x = group, fill = values)) +
    geom_bar(position = "fill") +
    theme_bw(base_size = 8) +
    ggtitle(paste0(feature, "\nP = ", signif(pval, 3))) +
    ylab("Proportion") +
    theme(legend.position = "none")
  
  return(list(feature = feature, p = pval, type = "categorical", plot = plot_obj))
}

# -----------------------------
# 3. Select 5 features for test
# -----------------------------
phenos_test <- c("consensus_sex", "ageAtDiagnosis", "ALT", "AST", "albumin")

# -----------------------------
# 4. Run EDA
# -----------------------------
results_list <- list()
for (feat in phenos_test) {
  cat("Processing:", feat, "\n")
  results_list[[feat]] <- evaluate_feature(data_all, feat, group)
}

# -----------------------------
# 5. Save p-values to CSV
# -----------------------------
results_df <- do.call(rbind, lapply(results_list, function(x)
  data.frame(feature = x$feature, p = x$p, type = x$type)
))

dir.create("EDA_train_vs_test", showWarnings = FALSE)
write.csv(results_df, "EDA_train_vs_test/results_5feats.csv", row.names = FALSE)

# -----------------------------
# 6. Save plots in 5x5 grid PDF
# -----------------------------
plots <- lapply(results_list, function(x) x$plot)

pdf("EDA_train_vs_test/EDA_first5_features.pdf", width = 12, height = 12)
do.call(grid.arrange, c(plots, ncol = 5))
dev.off()

######################
#pathways
######################
train_color <- "#d62728"   # red
test_color  <- "#1f77b4"   # blue

cat_palette_numeric <- c(train_color, test_color)

cat_palette_categorical <- c(
  "train" = train_color,
  "test"  = test_color
)

# -----------------------------
# Function to evaluate features
# -----------------------------
evaluate_numeric_only <- function(df, feature, group){
  
  values <- as.numeric(df[[feature]])  # FORCE numeric
  
  pval <- tryCatch({
    wilcox.test(values ~ group, exact = FALSE)$p.value
  }, error = function(e) NA)
  
  plot_obj <- ggplot(df, aes(x = group, y = values, fill = group)) +
    geom_boxplot() +
    scale_fill_manual(values = c("train" = train_color, "test" = test_color)) +
    theme_bw(base_size = 8) +
    ggtitle(paste0(feature, "\nP = ", signif(pval, 3))) +
    theme(legend.position = "none")
  
  return(list(feature = feature, p = pval, plot = plot_obj))
}

# -----------------------------
# Pathway list (all numeric)
# -----------------------------
pathway_feats <- c(
  "[HM] Hypoxia","[HM] Mitotic spindle","[HM] Wnt beta catenin signaling",
  "[HM] TGF beta signaling","[HM] DNA repair","[HM] G2M checkpoint",
  "[HM] Apoptosis","[HM] Notch signaling","[HM] Hedgehog signaling",
  "[HM] PI3K Akt mTOR signaling","[HM] mTORC1 signaling","[HM] E2F targets",
  "[HM] Epithelial mesenchymal transition","[HM] Oxidative phosphorylation",
  "[HM] Glycolysis","[HM] Reactive oxigen species pathway","[HM] p53 pathway",
  "[HM] UV response up","[HM] UV response down","[HM] Angiogenesis",
  "[HM] KRAS signaling up","[HM] KRAS signaling down","[HM] Myc targets",
  "[HM] Estrogen response","[IPA] ERK MAPK Signaling","[IPA] HMGB1 Signaling",
  "[IPA] ErbB Signaling","[IPA] PTEN Signaling","[IPA] PI3K AKT Signaling",
  "[IPA] HER 2 Signaling in Breast Cancer","[IPA] mTOR Signaling",
  "[IPA] AMPK Signaling","[IPA] p38 MAPK Signaling",
  "[IPA] Myc Mediated Apoptosis Signaling","[IPA] EGF Signaling",
  "[IPA] VEGF Signaling","[IPA] Estrogen Dependent Breast Cancer Signaling",
  "[IPA] TNFR1 Signaling","[IPA] ErbB2 ErbB3 Signaling",
  "[IPA] UVC Induced MAPK Signaling","[IPA] UVA Induced MAPK Signaling",
  "[IPA] UVB Induced MAPK Signaling","[IPA] ERK5 Signaling",
  "[IPA] Mismatch Repair in Eukaryotes",
  "[IPA] Telomere Extension by Telomerase",
  "[TBI] Barrier genes","[TBI] MAPK up genes","[TBI] Phopholipase",
  "[LM] Proliferation","[TPW] PI3Kgamma Signature","[TPW] NOS1 Signature",
  "[TPW] SHC1/pSTAT3 Signature","[TPW] Hypoxia/Adenosine Immune Cell Suppression",
  "[TPW] Immunogenic Cell Death (ICD)","HSC-like","Progenitor-like",
  "GMP-like","Promono-like","Monocyte-like","cDC-like"
)

# -----------------------------
# Run analysis for pathways
# -----------------------------
results_list <- list()
for (feat in pathway_feats) {
  cat("Processing:", feat, "\n")
  results_list[[feat]] <- evaluate_numeric_only(data_all, feat, group)
}

# -----------------------------
# Save p-values
# -----------------------------
results_df <- do.call(rbind, lapply(results_list, function(x)
  data.frame(feature = x$feature, p = x$p)
))

write.csv(results_df,
          file = file.path(outdir, "pathway_EDA_pvalues.csv"),
          row.names = FALSE)

# -----------------------------
# Save all plots in 5×5 grid, one PDF page
# -----------------------------
plots <- lapply(results_list, function(x) x$plot)
n <- length(plots)
plots_per_page <- 25
num_pages <- ceiling(n / plots_per_page)

pdf(file.path(outdir, "pathway_EDA.pdf"), width = 15, height = 15)

for (i in 1:num_pages) {
  start <- (i-1)*plots_per_page + 1
  end <- min(i*plots_per_page, n)
  cat("Writing page", i, "plots", start, "to", end, "\n")
  do.call(grid.arrange, c(plots[start:end], ncol = 5))
}

dev.off()

# -----------------------------
# Module list
# -----------------------------
module_feats <- c("M0", "M1", "M11", "M12", "M14", "M16",
                  "M21", "M23", "M3", "M4", "M6", "M7", "M8", "M9")

# -----------------------------
# Run module analysis
# -----------------------------
results_mod_list <- list()

for (feat in module_feats) {
  cat("Processing:", feat, "\n")
  results_mod_list[[feat]] <- evaluate_numeric_only(data_all, feat, group)
}

# -----------------------------
# Save p-values
# -----------------------------
results_mod_df <- do.call(rbind, lapply(results_mod_list, function(x)
  data.frame(feature = x$feature, p = x$p)
))

write.csv(results_mod_df,
          file = file.path("EDA_train_vs_test", "module_EDA_pvalues.csv"),
          row.names = FALSE)

# -----------------------------
# Save all module plots in 5×5 grid PDF
# -----------------------------
plots <- lapply(results_mod_list, function(x) x$plot)
n <- length(plots)
plots_per_page <- 25
num_pages <- ceiling(n / plots_per_page)

pdf(file.path("EDA_train_vs_test", "module_EDA.pdf"), width = 15, height = 15)

for (i in 1:num_pages) {
  start <- (i-1)*plots_per_page + 1
  end <- min(i*plots_per_page, n)
  cat("Writing page", i, "plots", start, "to", end, "\n")
  do.call(grid.arrange, c(plots[start:end], ncol = 5))
}

dev.off()
########################
#clinical
########################
data <- data_all                          # your dataframe
split_column <- "set"                     # train/test column
outdir <- "EDA_train_vs_test"
dir.create(outdir, showWarnings = FALSE)

# -----------------------------
# Variables
# -----------------------------
exclude_vars <- c("cumulativeTreatmentRegimens", "typeInductionTx", 
                  "cumulativeTreatmentStages","cumulativeTreatmentTypes")

clinical_vars <- c(
  "manuscript_dnaseq","manuscript_rnaseq","manuscript_inhibitor","consensus_sex",
  "inferred_sex","reportedRace","reportedEthnicity","inferred_ethnicity","centerID",
  "CEBPA_Biallelic","consensusAMLFusions","ageAtDiagnosis","isRelapse","isDenovo",
  "isTransformed","specificDxAtAcquisition_MDSMPN","nonAML_MDSMPN_specificDxAtAcquisition",
  "priorMalignancyNonMyeloid","priorMalignancyType","cumulativeChemo","priorMalignancyRadiationTx",
  "priorMDS","priorMDSMoreThanTwoMths","priorMDSMPN","priorMDSMPNMoreThanTwoMths","priorMPN",
  "priorMPNMoreThanTwoMths","dxAtInclusion","specificDxAtInclusion","ELN2017",
  "dxAtSpecimenAcquisition","specificDxAtAcquisition","ageAtSpecimenAcquisition",
  "timeOfSampleCollectionRelativeToInclusion","specimenGroups","diseaseStageAtSpecimenCollection",
  "specimenType","rnaSeq","exomeSeq","totalDrug","analysisRnaSeq","analysisExomeSeq",
  "analysisDrug","cumulativeTreatmentTypeCount","cumulativeTreatmentRegimenCount",
  "cumulativeTreatmentStageCount","responseToInductionTx","mostRecentTreatmentType",
  "currentRegimen","currentStage","mostRecentTreatmentDuration","vitalStatus",
  "overallSurvival","causeOfDeath"
)

# Remove excluded vars for main plots
main_vars <- setdiff(clinical_vars, exclude_vars)
main_vars <- main_vars[main_vars %in% colnames(data_all)]
cat("Using", length(main_vars), "clinical features for main EDA\n")

# -----------------------------
# Function to generate plot
# -----------------------------
generate_plot <- function(df, var_name, is_cat){
  if (!is_cat) {
    df$Value <- as.numeric(df$Value)
    test_res <- try(wilcox.test(Value ~ Group, data=df), silent=TRUE)
    if (inherits(test_res, "try-error")) pval <- NA else pval <- test_res$p.value
    
    plt <- ggplot(df, aes(x=Group, y=Value, fill=Group)) +
      geom_boxplot() +
      scale_fill_manual(values=c("train"="#E41A1C","test"="#377EB8")) +
      ggtitle(str_wrap(paste(var_name, "(Wilcoxon p=", signif(pval,3), ")"), width=30)) +
      theme_bw(base_size=10) +
      theme(plot.title = element_text(size=9, hjust=0.5),
            axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
            legend.position = "bottom", legend.title = element_blank())
    
  } else {
    df$Value <- factor(df$Value)
    tab <- table(df$Group, df$Value)
    test_res <- try({
      if (any(tab < 5) || prod(dim(tab)) > 100) {
        fisher.test(tab, simulate.p.value = TRUE, B = 10000)
      } else chisq.test(tab)
    }, silent=TRUE)
    pval <- ifelse(inherits(test_res, "try-error"), NA, test_res$p.value)
    
    plt <- ggplot(df, aes(x=Value, fill=Group)) +
      geom_bar(position="dodge") +
      scale_fill_manual(values=c("train"="#E41A1C","test"="#377EB8")) +
      ggtitle(str_wrap(paste(var_name, "(ChiSq/Fisher p=", signif(pval,3), ")"), width=30)) +
      theme_bw(base_size=10) +
      theme(plot.title = element_text(size=9, hjust=0.5),
            axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
            legend.position = "bottom", legend.title = element_blank())
  }
  return(list(plot=plt, pval=pval))
}

# -----------------------------
# Generate main plots
# -----------------------------
results <- list()
plot_list <- list()
plot_counter <- 1

for (v in main_vars) {
  df <- data_all[, c(split_column, v)]
  colnames(df) <- c("Group", "Value")
  df <- df[!is.na(df$Value) & !is.na(df$Group), ]
  if (nrow(df) < 3) next
  
  unique_vals <- length(unique(df$Value))
  is_cat <- unique_vals <= 5 || is.character(df$Value) || is.factor(df$Value)
  
  res <- generate_plot(df, v, is_cat)
  
  results[[v]] <- res$pval
  plot_list[[plot_counter]] <- res$plot
  plot_counter <- plot_counter + 1
}

# -----------------------------
# Generate excluded vars plots (1 per page)
# -----------------------------
excluded_plot_list <- list()
for (v in exclude_vars) {
  if (!(v %in% colnames(data_all))) next
  df <- data_all[, c(split_column, v)]
  colnames(df) <- c("Group", "Value")
  df <- df[!is.na(df$Value) & !is.na(df$Group), ]
  if (nrow(df) < 3) next
  unique_vals <- length(unique(df$Value))
  is_cat <- unique_vals <= 5 || is.character(df$Value) || is.factor(df$Value)
  res <- generate_plot(df, v, is_cat)
  results[[v]] <- res$pval
  excluded_plot_list[[v]] <- res$plot
}

# -----------------------------
# Save all plots to one PDF
# -----------------------------
pdf(file.path(outdir, "clinical_EDA_all.pdf"), width=14, height=14)

# Main plots (25 per page)
plots_per_page <- 4
n_pages <- ceiling(length(plot_list)/plots_per_page)
for (i in 1:n_pages) {
  start_idx <- (i-1)*plots_per_page + 1
  end_idx <- min(i*plots_per_page, length(plot_list))
  do.call(grid.arrange, c(plot_list[start_idx:end_idx], ncol=2))
}

# Excluded vars (1 per page)
for (v in names(excluded_plot_list)) {
  grid.arrange(excluded_plot_list[[v]])
}

dev.off()

# -----------------------------
# Save p-values CSV
# -----------------------------
results_df <- data.frame(
  feature = names(results),
  p = unlist(results),
  stringsAsFactors = FALSE
)
write.csv(results_df, file=file.path(outdir, "clinical_EDA_all_pvalues.csv"), row.names=FALSE)

cat("Clinical EDA completed. Plots and p-values saved in", outdir, "\n")


####################################
#gene expression and mutation
####################################
# -----------------------------
# Load gene list
# -----------------------------
gene_file <- "genesexp_and_mutation.txt"
gene_lines <- readLines(gene_file)

# Split by space or tab and remove quotes
genes <- unlist(strsplit(gene_lines, "[ \t]+"))
genes <- gsub('^"|"$', '', genes)
cat("Total genes loaded:", length(genes), "\n")

# ------------------------
# 2. Identify categorical (.y = mutation) vs numeric (expression)
# ------------------------
cat_genes <- genes[grepl("\\.y$", genes)]
num_genes <- setdiff(genes, cat_genes)

# ------------------------
# 3. Initialize
# ------------------------
split_column <- "set"   # train/test column
results <- list()
plot_list <- list()
plot_counter <- 1

# ------------------------
# 4. Loop over all genes
# ------------------------
for (g in genes) {
  
  if (!(g %in% colnames(data_all))) next  # skip missing columns
  
  df <- data_all[, c(split_column, g)]
  colnames(df) <- c("Group", "Value")
  df <- df[!is.na(df$Value) & !is.na(df$Group), ]
  if (nrow(df) < 3) next  # too few rows
  
  is_cat <- g %in% cat_genes
  
  if (!is_cat) {
    # Numeric → Wilcoxon
    df$Value <- as.numeric(df$Value)
    test_res <- try(wilcox.test(Value ~ Group, data=df), silent=TRUE)
    if (inherits(test_res, "try-error")) next
    pval <- test_res$p.value
    
    plt <- ggplot(df, aes(x=Group, y=Value, fill=Group)) +
      geom_boxplot() +
      scale_fill_manual(values=c("train"="#E41A1C", "test"="#377EB8")) +
      ggtitle(str_wrap(paste(g, "(Wilcoxon p=", signif(pval,3), ")"), width=30)) +
      theme_bw(base_size=10) +
      theme(
        plot.title = element_text(size=9, hjust=0.5),
        axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
        legend.position = "bottom",
        legend.title = element_blank()
      )
    
  } else {
    # Categorical → Chi-square / Fisher
    df$Value <- factor(df$Value)
    tab <- table(df$Group, df$Value)
    test_res <- try(if(any(tab < 5)) fisher.test(tab) else chisq.test(tab), silent=TRUE)
    if (inherits(test_res, "try-error")) next
    pval <- test_res$p.value
    
    plt <- ggplot(df, aes(x=Value, fill=Group)) +
      geom_bar(position="dodge") +
      scale_fill_manual(values=c("train"="#E41A1C","test"="#377EB8")) +
      ggtitle(str_wrap(paste(g, "(ChiSq/Fisher p=", signif(pval,3), ")"), width=30)) +
      theme_bw(base_size=10) +
      theme(
        plot.title = element_text(size=9, hjust=0.5),
        axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
        legend.position = "bottom",
        legend.title = element_blank()
      )
  }
  
  results[[g]] <- pval
  plot_list[[plot_counter]] <- plt
  plot_counter <- plot_counter + 1
}

# ------------------------
# 5. Save PDF (25 plots per page)
# ------------------------
pdf("EDA_train_vs_test/gene_EDA.pdf", width=14, height=14)
plots_per_page <- 25
n_pages <- ceiling(length(plot_list)/plots_per_page)

for (i in 1:n_pages) {
  start_idx <- (i-1)*plots_per_page + 1
  end_idx <- min(i*plots_per_page, length(plot_list))
  do.call(grid.arrange, c(plot_list[start_idx:end_idx], ncol=5))
}
dev.off()

# ------------------------
# 6. Save p-values
# ------------------------
results_df <- data.frame(
  feature = names(results),
  p = unlist(results),
  stringsAsFactors = FALSE
)
write.csv(results_df, "EDA_train_vs_test/gene_EDA_pvalues.csv", row.names=FALSE)

###################
#cell type
###################
cell_type_vars <- c(
  "%.Basophils.in.PB", "%.Blasts.in.BM", "%.Blasts.in.PB",
  "%.Eosinophils.in.PB", "%.Immature.Granulocytes.in.PB", "%.Lymphocytes.in.PB",
  "%.Monocytes.in.PB", "%.Neutrophils.in.PB", "%.Nucleated.RBCs.in.PB",
  "ALT", "AST", "albumin", "creatinine", "fabBlastMorphology",
  "hematocrit", "hemoglobin", "karyotype", "LDH", "MCV",
  "plateletCount", "surfaceAntigensImmunohistochemicalStains", "totalProtein",
  "wbcCount", "FLT3-ITD"
)

cell_type_vars <- cell_type_vars[cell_type_vars %in% colnames(data_all)]
cat("Total variables for cell_type_EDA:", length(cell_type_vars), "\n")

split_column <- "set"
results <- list()
plot_list <- list()
plot_counter <- 1

cat_vars <- c("FLT3-ITD","fabBlastMorphology","karyotype","otherCytogenetics")

# ------------------------
# Loop over variables
# ------------------------
for (v in cell_type_vars) {
  
  df <- data_all[, c(split_column, v)]
  colnames(df) <- c("Group", "Value")
  df <- df[!is.na(df$Value) & !is.na(df$Group), ]
  if (nrow(df) < 3) next
  
  is_cat <- v %in% cat_vars
  
  if (!is_cat) {
    df$Value <- as.numeric(df$Value)
    test_res <- try(wilcox.test(Value ~ Group, data=df), silent=TRUE)
    if (inherits(test_res, "try-error")) next
    pval <- test_res$p.value
    
    plt <- ggplot(df, aes(x=Group, y=Value, fill=Group)) +
      geom_boxplot() +
      scale_fill_manual(values=c("train"="#E41A1C", "test"="#377EB8")) +
      ggtitle(str_wrap(paste(v, "(Wilcoxon p=", signif(pval,3), ")"), width=30)) +
      theme_bw(base_size=10) +
      theme(
        plot.title = element_text(size=9, hjust=0.5),
        axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
        legend.position = "bottom",
        legend.title = element_blank()
      )
    
  } else {
    df$Value <- factor(df$Value)
    tab <- table(df$Group, df$Value)
    test_res <- try(if(any(tab < 5)) fisher.test(tab) else chisq.test(tab), silent=TRUE)
    if (inherits(test_res, "try-error")) next
    pval <- test_res$p.value
    
    plt <- ggplot(df, aes(x=Value, fill=Group)) +
      geom_bar(position="dodge") +
      scale_fill_manual(values=c("train"="#E41A1C", "test"="#377EB8")) +
      ggtitle(str_wrap(paste(v, "(ChiSq/Fisher p=", signif(pval,3), ")"), width=30)) +
      theme_bw(base_size=10) +
      theme(
        plot.title = element_text(size=9, hjust=0.5),
        axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
        legend.position = "bottom",
        legend.title = element_blank()
      )
  }
  
  results[[v]] <- pval
  plot_list[[plot_counter]] <- plt
  plot_counter <- plot_counter + 1
}

# ------------------------
# Save PDF with 25 plots per page
# ------------------------
pdf("EDA_train_vs_test/cell_type_EDA.pdf", width=14, height=14)
plots_per_page <- 25
n_pages <- ceiling(length(plot_list)/plots_per_page)

for (i in 1:n_pages) {
  start_idx <- (i-1)*plots_per_page + 1
  end_idx <- min(i*plots_per_page, length(plot_list))
  do.call(grid.arrange, c(plot_list[start_idx:end_idx], ncol=5))
}
dev.off()

# ------------------------
# Save p-values CSV
# ------------------------
results_df <- data.frame(
  feature = names(results),
  p = unlist(results),
  stringsAsFactors = FALSE
)
write.csv(results_df, "EDA_train_vs_test/cell_type_EDA_pvalues.csv", row.names=FALSE)

####################
# ------------------------
# Mutation consequence columns (all categorical)
# ------------------------
mutation_vars <- c(
  "frameshift_variant", "missense_variant", "stop_gained", "inframe_deletion",
  "protein_altering_variant", "splice_acceptor_variant", "splice_donor_variant",
  "start_lost", "inframe_insertion", "stop_lost"
)
mutation_vars <- mutation_vars[mutation_vars %in% colnames(data_all)]
cat("Total mutation features:", length(mutation_vars), "\n")

split_column <- "set"
results <- list()
plot_list <- list()
plot_counter <- 1

# ------------------------
# Loop over variables
# ------------------------
for (v in mutation_vars) {
  
  df <- data_all[, c(split_column, v)]
  colnames(df) <- c("Group", "Value")
  df <- df[!is.na(df$Value) & !is.na(df$Group), ]
  
  if (nrow(df) < 3) next
  
  # All categorical
  df$Value <- factor(df$Value)
  tab <- table(df$Group, df$Value)
  
  # Chi-square or Fisher
  test_res <- try(if(any(tab < 5)) fisher.test(tab) else chisq.test(tab), silent=TRUE)
  if (inherits(test_res, "try-error")) next
  pval <- test_res$p.value
  
  # Wrap long titles to ~30 chars per line
  plot_title <- str_wrap(paste(v, "(ChiSq/Fisher p=", signif(pval,3), ")"), width=30)
  
  # Plot
  plt <- ggplot(df, aes(x=Value, fill=Group)) +
    geom_bar(position="dodge") +
    scale_fill_manual(values=c("train"="#E41A1C", "test"="#377EB8")) +
    ggtitle(plot_title) +
    theme_bw(base_size=10) +
    theme(
      plot.title = element_text(size=9, hjust=0.5),  # smaller title
      axis.text.x = element_text(angle=90, vjust=0.5, hjust=1, size=8),
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  results[[v]] <- pval
  plot_list[[plot_counter]] <- plt
  plot_counter <- plot_counter + 1
}

# ------------------------
# Save PDF with 25 plots per page
# ------------------------
pdf("EDA_train_vs_test/mutation_consequence_EDA.pdf", width=14, height=14)
plots_per_page <- 25
n_pages <- ceiling(length(plot_list)/plots_per_page)

for (i in 1:n_pages) {
  start_idx <- (i-1)*plots_per_page + 1
  end_idx <- min(i*plots_per_page, length(plot_list))
  do.call(grid.arrange, c(plot_list[start_idx:end_idx], ncol=5))
}
dev.off()

# ------------------------
# Save p-values CSV
# ------------------------
results_df <- data.frame(
  feature = names(results),
  p = unlist(results),
  stringsAsFactors = FALSE
)
write.csv(results_df, "EDA_train_vs_test/mutation_consequence_EDA_pvalues.csv", row.names=FALSE)
