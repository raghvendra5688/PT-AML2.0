
########################
library(dplyr)
library(ggplot2)
library(tidyr)
library(gridExtra)

# =========================================================
# Load data
# =========================================================
setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/Results/Tables/best_model_correlation_interpretation/correlation results/")

data <- read.csv(
  "../prediction_data.csv",
  header = TRUE,
  sep = ","
)

# =========================================================
# MAE calculations
# =========================================================

mae_by_drug <- aggregate(
  abs(data$predictions - data$labels),
  by = list(data$inhibitor),
  FUN = mean
)

mae_by_samples <- aggregate(
  abs(data$predictions - data$labels),
  by = list(data$dbgap_rnaseq_sample),
  FUN = mean
)


library(dplyr)

# Remove inhibitors with <10 samples
data_filtered <- data %>%
  group_by(inhibitor) %>%
  filter(n_distinct(dbgap_rnaseq_sample) >= 100) %>%
  ungroup()

# Remove patients with <10 inhibitors
data_filtered2 <- data_filtered %>%
  group_by(dbgap_rnaseq_sample) %>%
  filter(n_distinct(inhibitor) >= 100) %>%
  ungroup()

# Check result
dim(data_filtered2)

data <- data_filtered
data2 <- data_filtered2

# =========================================================
# PEARSON CORRELATION BY DRUG
# =========================================================

pearson_drug <- split(data2, data2$inhibitor) %>%
  lapply(function(x)
    cor(
      x$predictions,
      x$labels,
      method = "pearson",
      use = "complete.obs"
    )
  )

pearson_drug_df <- data.frame(
  drug = names(pearson_drug),
  pearson_corr = as.numeric(unlist(pearson_drug))
)

# =========================================================
# SPEARMAN CORRELATION BY DRUG
# =========================================================

spearman_drug <- split(data, data$inhibitor) %>%
  lapply(function(x)
    cor(
      x$predictions,
      x$labels,
      method = "spearman",
      use = "complete.obs"
    )
  )

spearman_drug_df <- data.frame(
  drug = names(spearman_drug),
  spearman_corr = as.numeric(unlist(spearman_drug))
)

# =========================================================
# MERGE DRUG CORRELATIONS
# =========================================================

data_drug <- merge(
  pearson_drug_df,
  spearman_drug_df,
  by = "drug"
)

data_drug <- na.omit(data_drug)

write.table(
  data_drug,
  file = "drug_correlations.txt",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

# =========================================================
# PEARSON CORRELATION BY PATIENT
# =========================================================

pearson_patient <- split(data, data$dbgap_rnaseq_sample) %>%
  lapply(function(x)
    cor(
      x$predictions,
      x$labels,
      method = "pearson",
      use = "complete.obs"
    )
  )

pearson_patient_df <- data.frame(
  patients = names(pearson_patient),
  pearson_corr = as.numeric(unlist(pearson_patient))
)

# =========================================================
# SPEARMAN CORRELATION BY PATIENT
# =========================================================

spearman_patient <- split(data, data$dbgap_rnaseq_sample) %>%
  lapply(function(x)
    cor(
      x$predictions,
      x$labels,
      method = "spearman",
      use = "complete.obs"
    )
  )

spearman_patient_df <- data.frame(
  patients = names(spearman_patient),
  spearman_corr = as.numeric(unlist(spearman_patient))
)

# =========================================================
# MERGE PATIENT CORRELATIONS
# =========================================================

data_sam <- merge(
  pearson_patient_df,
  spearman_patient_df,
  by = "patients"
)

data_sam <- na.omit(data_sam)

write.table(
  data_sam,
  file = "sample_correlations.txt",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

# # =========================================================
# # TOP 10 PATIENTS
# # =========================================================
# data_sam <- merge(
#   pearson_patient_df,
#   spearman_patient_df,
#   by = "patients"
# )
# 
# # =====================================================
# # FORCE CLEAN TYPES
# # =====================================================
# 
# data_sam$pearson_corr <- as.numeric(data_sam$pearson_corr)
# data_sam$spearman_corr <- as.numeric(data_sam$spearman_corr)
# data_sam$patients <- as.character(data_sam$patients)
# 
# # remove NA rows
# data_sam <- na.omit(data_sam)
# 
# # verify structure
# str(data_sam)
# top_10_sam <- data_sam %>%
#   dplyr::arrange(desc(pearson_corr)) %>%
#   dplyr::slice(1:10)
# 
# library(dplyr)
# 
# # Count number of inhibitors per patient
# patient_counts <- data %>%
#   group_by(dbgap_rnaseq_sample) %>%
#   summarise(n_inhibitors = n_distinct(inhibitor))
# 
# # Merge counts with top_10_sam
# top_10_sam <- top_10_sam %>%
#   left_join(
#     patient_counts,
#     by = c("patients" = "dbgap_rnaseq_sample")
#   )
# 
# # Add counts to patient names
# top_10_sam$patients <- paste0(
#   top_10_sam$patients,
#   "(",
#   top_10_sam$n_inhibitors,
#   ")"
# )
# 
# # Optional: remove helper column
# top_10_sam$n_inhibitors <- NULL
# 
# head(top_10_sam)
# 
# write.csv(top_10_sam, "top_10_patients.csv")
# top_10_sam_long <- top_10_sam %>%
#   pivot_longer(
#     cols = c(pearson_corr, spearman_corr),
#     names_to = "Correlation_Type",
#     values_to = "Correlation"
#   )
# 
# sam_plot_1 <- ggplot(
#   top_10_sam_long,
#   aes(
#     x = reorder(patients, Correlation),
#     y = Correlation,
#     fill = Correlation_Type
#   )
# ) +
#   geom_bar(
#     stat = "identity",
#     position = position_dodge(width = 0.8)
#   ) +
#   coord_flip() +
#   labs(
#     x = "Patients",
#     y = "Correlation",
#     title = "Top 10 Patients"
#   ) +
#   scale_fill_manual(
#     values = c(
#       "pearson_corr" = "blue",
#       "spearman_corr" = "red"
#     ),
#     labels = c(
#       "Pearson",
#       "Spearman"
#     )
#   ) +
#   theme_minimal()+
#   theme(
#     
#     # title
#     plot.title = element_text(
#       size = 20,
#       face = "bold",
#       hjust = 0.5
#     ),
#     
#     # axis titles
#     axis.title.x = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     axis.title.y = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     # axis text
#     axis.text.x = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     axis.text.y = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     # legend
#     legend.title = element_text(
#       size = 16,
#       face = "bold"
#     ),
#     
#     legend.text = element_text(
#       size = 14,
#       face = "bold"
#     )
#   )
# 
# # =========================================================
# # BOTTOM 10 PATIENTS
# # =========================================================
# 
# bottom_10_sam <- data_sam %>%
#   dplyr::arrange(pearson_corr) %>%
#   dplyr::slice(1:10)
# library(dplyr)
# 
# # Count number of inhibitors per patient
# patient_counts <- data %>%
#   group_by(dbgap_rnaseq_sample) %>%
#   summarise(n_inhibitors = n_distinct(inhibitor))
# 
# # Merge counts with bottom_10_sam
# bottom_10_sam <- bottom_10_sam %>%
#   left_join(
#     patient_counts,
#     by = c("patients" = "dbgap_rnaseq_sample")
#   )
# 
# # Add counts to patient names
# bottom_10_sam$patients <- paste0(
#   bottom_10_sam$patients,
#   "(",
#   bottom_10_sam$n_inhibitors,
#   ")"
# )
# 
# # Optional: remove helper column
# bottom_10_sam$n_inhibitors <- NULL
# 
# head(bottom_10_sam)
# write.csv(bottom_10_sam, "bottom_10_sam.csv")
# bottom_10_sam_long <- bottom_10_sam %>%
#   pivot_longer(
#     cols = c(pearson_corr, spearman_corr),
#     names_to = "Correlation_Type",
#     values_to = "Correlation"
#   )
# 
# sam_plot_2 <- ggplot(
#   bottom_10_sam_long,
#   aes(
#     x = reorder(patients, Correlation),
#     y = Correlation,
#     fill = Correlation_Type
#   )
# ) +
#   geom_bar(
#     stat = "identity",
#     position = position_dodge(width = 0.8)
#   ) +
#   coord_flip() +
#   labs(
#     x = "Patients",
#     y = "Correlation",
#     title = "Bottom 10 Patients"
#   ) +
#   scale_fill_manual(
#     values = c(
#       "pearson_corr" = "blue",
#       "spearman_corr" = "red"
#     ),
#     labels = c(
#       "Pearson",
#       "Spearman"
#     )
#   ) +
#   theme_minimal()+
#   theme(
#     
#     # title
#     plot.title = element_text(
#       size = 20,
#       face = "bold",
#       hjust = 0.5
#     ),
#     
#     # axis titles
#     axis.title.x = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     axis.title.y = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     # axis text
#     axis.text.x = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     axis.text.y = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     # legend
#     legend.title = element_text(
#       size = 16,
#       face = "bold"
#     ),
#     
#     legend.text = element_text(
#       size = 14,
#       face = "bold"
#     )
#   )
# 
# # =========================================================
# # COMBINE PATIENT PLOTS
# # =========================================================
# 
# sam <- grid.arrange(
#   sam_plot_1,
#   sam_plot_2,
#   ncol = 2,
#   top = "Pearson vs Spearman Correlation for Patients"
# )
# 
# ggsave(
#   "combined_plot_patients.png",
#   sam,
#   width = 14,
#   height = 6,
#   dpi = 300
# )
# 
# # =========================================================
# # TOP 10 DRUGS
# # =========================================================
# 
# top_10_drug <- data_drug %>%
#   dplyr::arrange(desc(pearson_corr)) %>%
#   dplyr::slice(1:10)
# library(dplyr)
# 
# # Count number of samples per inhibitor
# drug_counts <- data %>%
#   group_by(inhibitor) %>%
#   summarise(n_samples = n_distinct(dbgap_rnaseq_sample))
# 
# # Merge counts with top_10_drug
# top_10_drug <- top_10_drug %>%
#   left_join(
#     drug_counts,
#     by = c("drug" = "inhibitor")
#   )
# 
# # Add counts to drug names
# top_10_drug$drug <- paste0(
#   top_10_drug$drug,
#   "(",
#   top_10_drug$n_samples,
#   ")"
# )
# 
# # Optional: remove helper column
# top_10_drug$n_samples <- NULL
# 
# head(top_10_drug)
# write.csv(top_10_drug, "top_10_drugs.csv")
# top_10_drug_long <- top_10_drug %>%
#   pivot_longer(
#     cols = c(pearson_corr, spearman_corr),
#     names_to = "Correlation_Type",
#     values_to = "Correlation"
#   )
# 
# drug_plot_1 <- ggplot(
#   top_10_drug_long,
#   aes(
#     x = reorder(drug, Correlation),
#     y = Correlation,
#     fill = Correlation_Type
#   )
# ) +
#   geom_bar(
#     stat = "identity",
#     position = position_dodge(width = 0.8)
#   ) +
#   coord_flip() +
#   labs(
#     x = "Drug",
#     y = "Correlation",
#     title = "Top 10 Drugs"
#   ) +
#   scale_fill_manual(
#     values = c(
#       "pearson_corr" = "blue",
#       "spearman_corr" = "red"
#     ),
#     labels = c(
#       "Pearson",
#       "Spearman"
#     )
#   ) +
#   theme_minimal()+
#   theme(
#     
#     # title
#     plot.title = element_text(
#       size = 20,
#       face = "bold",
#       hjust = 0.5
#     ),
#     
#     # axis titles
#     axis.title.x = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     axis.title.y = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     # axis text
#     axis.text.x = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     axis.text.y = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     # legend
#     legend.title = element_text(
#       size = 16,
#       face = "bold"
#     ),
#     
#     legend.text = element_text(
#       size = 14,
#       face = "bold"
#     )
#   )
# 
# # =========================================================
# # BOTTOM 10 DRUGS
# # =========================================================
# 
# bottom_10_drug <- data_drug %>%
#   dplyr::arrange(pearson_corr) %>%
#   dplyr::slice(1:10)
# library(dplyr)
# 
# # Count number of samples per inhibitor
# drug_counts <- data %>%
#   group_by(inhibitor) %>%
#   summarise(n_samples = n_distinct(dbgap_rnaseq_sample))
# 
# # Merge counts with bottom_10_drug
# bottom_10_drug <- bottom_10_drug %>%
#   left_join(
#     drug_counts,
#     by = c("drug" = "inhibitor")
#   )
# 
# # Add counts to drug names
# bottom_10_drug$drug <- paste0(
#   bottom_10_drug$drug,
#   "(",
#   bottom_10_drug$n_samples,
#   ")"
# )
# 
# # Optional: remove helper column
# bottom_10_drug$n_samples <- NULL
# 
# head(bottom_10_drug)
# write.csv(bottom_10_drug, "bottom_10_drugs.csv")
# bottom_10_drug_long <- bottom_10_drug %>%
#   pivot_longer(
#     cols = c(pearson_corr, spearman_corr),
#     names_to = "Correlation_Type",
#     values_to = "Correlation"
#   )
# 
# drug_plot_2 <- ggplot(
#   bottom_10_drug_long,
#   aes(
#     x = reorder(drug, Correlation),
#     y = Correlation,
#     fill = Correlation_Type
#   )
# ) +
#   geom_bar(
#     stat = "identity",
#     position = position_dodge(width = 0.8)
#   ) +
#   coord_flip() +
#   labs(
#     x = "Drug",
#     y = "Correlation",
#     title = "Bottom 10 Drugs"
#   ) +
#   scale_fill_manual(
#     values = c(
#       "pearson_corr" = "blue",
#       "spearman_corr" = "red"
#     ),
#     labels = c(
#       "Pearson",
#       "Spearman"
#     )
#   ) +
#   theme_minimal()+
#   theme(
#     
#     # title
#     plot.title = element_text(
#       size = 20,
#       face = "bold",
#       hjust = 0.5
#     ),
#     
#     # axis titles
#     axis.title.x = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     axis.title.y = element_text(
#       size = 18,
#       face = "bold"
#     ),
#     
#     # axis text
#     axis.text.x = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     axis.text.y = element_text(
#       size = 14,
#       face = "bold"
#     ),
#     
#     # legend
#     legend.title = element_text(
#       size = 16,
#       face = "bold"
#     ),
#     
#     legend.text = element_text(
#       size = 14,
#       face = "bold"
#     )
#   )
# 
# # =========================================================
# # COMBINE DRUG PLOTS
# # =========================================================
# 
# drug <- grid.arrange(
#   drug_plot_1,
#   drug_plot_2,
#   ncol = 2,
#   top = "Pearson vs Spearman Correlation for Drugs"
# )
# 
# ggsave(
#   "combined_plot_drug.png",
#   drug,
#   width = 14,
#   height = 6,
#   dpi = 300
# )
# 
# # =========================================================
# # FINAL COMBINED PLOT
# # =========================================================
# 
# all_plot <- grid.arrange(
#   sam,
#   drug,
#   nrow = 2
# )
# 
# ggsave(
#   "all_correlations.png",
#   all_plot,
#   width = 14,
#   height = 12,
#   dpi = 300
# )
# 
# ggsave(
#   "all_correlations.pdf",
#   all_plot,
#   width = 14,
#   height = 12,
#   dpi = 300,
#   device = "pdf"
# )
# 
# ###Patient performace trend
# library(dplyr)
# library(ggplot2)
# 
# patient_trend <- data_sam %>%
#   arrange(desc(pearson_corr)) %>%
#   mutate(
#     Rank = 1:n(),
#     Top10 = Rank <= 10
#   )
# 
# ggplot(
#   patient_trend,
#   aes(
#     x = Rank,
#     y = pearson_corr,
#     fill = Top10
#   )
# ) +
#   geom_col() +
#   scale_fill_manual(
#     values = c(
#       "TRUE" = "red",
#       "FALSE" = "grey80"
#     ),
#     labels = c(
#       "Other Patients",
#       "Top 10 Patients"
#     ),
#     name = ""
#   ) +
#   labs(
#     title = "Patient Performance Trend",
#     x = "Patients (Ranked by Pearson Correlation)",
#     y = "Pearson Correlation"
#   ) +
#   theme_classic(base_size = 16)

#####pearson and spearman both overlap
library(dplyr)
library(tidyr)
library(ggplot2)

patient_trend <- data_sam %>%
  arrange(desc(pearson_corr)) %>%
  mutate(Rank = 1:n())

patient_long <- patient_trend %>%
  pivot_longer(
    cols = c(pearson_corr, spearman_corr),
    names_to = "Correlation_Type",
    values_to = "Correlation"
  )

ggplot(
  patient_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = "identity",
    alpha = 0.5,
    width = 0.9
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = c(
      "Pearson",
      "Spearman"
    )
  ) +
  labs(
    title = "Patient Performance Trend",
    x = "Patients (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  theme_classic(base_size = 16) +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    legend.position = "top"
  )

#### pearson and spearman side by side
ggplot(
  patient_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.8
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = c(
      "Pearson",
      "Spearman"
    )
  ) +
  labs(
    title = "Patient Performance Trend",
    x = "Patients (Ranked by Pearson Correlation)",
    y = "Correlation"
  ) +
  theme_classic(base_size = 16)
### some changes
# Maximum correlation value
ymax <- max(patient_long$Correlation, na.rm = TRUE) * 1.10
xmax <- max(length(patient_long$patients), na.rm = TRUE) * 1.10
common_theme <- theme_classic(base_size = 18) +
  theme(
    plot.title = element_text(
      size = 22,
      face = "bold",
      hjust = 0.5
    ),
    axis.title.x = element_text(
      size = 20,
      face = "bold"
    ),
    axis.title.y = element_text(
      size = 20,
      face = "bold"
    ),
    axis.text.x = element_text(
      size = 16,
      face = "bold"
    ),
    axis.text.y = element_text(
      size = 16,
      face = "bold"
    ),
    legend.title = element_text(
      size = 18,
      face = "bold"
    ),
    legend.text = element_text(
      size = 16,
      face = "bold"
    ),
    legend.position = "top"
  )
p1 <- ggplot(
  patient_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = "identity",
    alpha = 0.5,
    width = 0.9
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = c(
      "Pearson",
      "Spearman"
    )
  ) +
  scale_y_continuous(
    limits = c(0, ymax),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Patient Performance Trend",
    x = "Patients (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

p1
ggsave(
  "Patient_Performance_Trend_Overlay.pdf",
  p1,
  width = 10,
  height = 6
)

p2 <- ggplot(
  patient_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.8
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = c(
      "Pearson",
      "Spearman"
    )
  ) +
  scale_y_continuous(
    limits = c(0, ymax),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Patient Performance Trend",
    x = "Patients (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

p2
ggsave(
  "Patient_Performance_Trend_Dodge.pdf",
  p2,
  width = 10,
  height = 6
)

#########for drugs
library(dplyr)
library(tidyr)

drug_trend <- data_drug %>%
  arrange(desc(pearson_corr)) %>%
  mutate(
    Rank = 1:n()
  )

drug_long <- drug_trend %>%
  pivot_longer(
    cols = c(pearson_corr, spearman_corr),
    names_to = "Correlation_Type",
    values_to = "Correlation"
  )

ymax_drug <- max(drug_long$Correlation, na.rm = TRUE) * 1.10
drug_p1 <- ggplot(
  drug_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = "identity",
    alpha = 0.5,
    width = 0.9
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = c(
      "Pearson",
      "Spearman"
    )
  ) +
  scale_x_continuous(
    limits = c(0, max(drug_trend$Rank) + 5)
  ) +
  scale_y_continuous(
    limits = c(0, ymax_drug),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Drug Performance Trend",
    x = "Drugs (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

drug_p1
ggsave(
  "Drug_Performance_Trend_Overlay.pdf",
  drug_p1,
  width = 10,
  height = 6
)
drug_p2 <- ggplot(
  drug_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.8
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = c(
      "Pearson",
      "Spearman"
    )
  ) +
  scale_x_continuous(
    limits = c(0, max(drug_trend$Rank) + 5)
  ) +
  scale_y_continuous(
    limits = c(0, ymax_drug),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Drug Performance Trend",
    x = "Drugs (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

drug_p2
ggsave(
  "Drug_Performance_Trend_Dodge.pdf",
  drug_p2,
  width = 10,
  height = 6
)

#########with mean sd 
pearson_mean <- mean(data_drug$pearson_corr, na.rm = TRUE)
pearson_sd   <- sd(data_drug$pearson_corr, na.rm = TRUE)

spearman_mean <- mean(data_drug$spearman_corr, na.rm = TRUE)
spearman_sd   <- sd(data_drug$spearman_corr, na.rm = TRUE)

legend_labels <- c(
  pearson_corr = paste0(
    "Pearson (",
    round(pearson_mean, 3),
    " \u00B1 ",
    round(pearson_sd, 3),
    ")"
  ),
  spearman_corr = paste0(
    "Spearman (",
    round(spearman_mean, 3),
    " \u00B1 ",
    round(spearman_sd, 3),
    ")"
  )
)
max_rank <- max(drug_trend$Rank)

drug_p2 <- ggplot(
  drug_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.8
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = legend_labels
  ) +
  scale_x_continuous(
    limits = c(1, max_rank),
    breaks = c(1, seq(20, max_rank, by = 20), max_rank)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2)
  ) +
  labs(
    title = "Drug Performance Trend",
    x = "Drugs (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme
ggsave(
  "Final_Drug_Performance_Trend_sidebyside.pdf",
  drug_p2,
  width = 10,
  height = 6,
  device = cairo_pdf
)


max_rank <- max(drug_trend$Rank)

drug_p1 <- ggplot(
  drug_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = "identity",
    alpha = 0.5,
    width = 0.9
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = legend_labels
  ) +
  scale_x_continuous(
    limits = c(1, max_rank),
    breaks = c(1, seq(20, max_rank, by = 20), max_rank)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2)
  ) +
  labs(
    title = "Drug Performance Trend",
    x = "Drugs (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

drug_p1
ggsave(
  "Final_Drug_Performance_Trend_Overlap.pdf",
  drug_p1,
  width = 10,
  height = 6,
  device = cairo_pdf
)


#########patients plot
pearson_mean_pat <- mean(data_sam$pearson_corr, na.rm = TRUE)
pearson_sd_pat   <- sd(data_sam$pearson_corr, na.rm = TRUE)

spearman_mean_pat <- mean(data_sam$spearman_corr, na.rm = TRUE)
spearman_sd_pat   <- sd(data_sam$spearman_corr, na.rm = TRUE)
legend_labels_pat <- c(
  pearson_corr = paste0(
    "Pearson (",
    round(pearson_mean_pat, 3),
    " ± ",
    round(pearson_sd_pat, 3),
    ")"
  ),
  spearman_corr = paste0(
    "Spearman (",
    round(spearman_mean_pat, 3),
    " ± ",
    round(spearman_sd_pat, 3),
    ")"
  )
)
library(dplyr)
library(tidyr)

patient_trend <- data_sam %>%
  arrange(desc(pearson_corr)) %>%
  mutate(Rank = 1:n())

patient_long <- patient_trend %>%
  pivot_longer(
    cols = c(pearson_corr, spearman_corr),
    names_to = "Correlation_Type",
    values_to = "Correlation"
  )

max_rank_pat <- max(patient_trend$Rank)
patient_overlap <- ggplot(
  patient_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = "identity",
    alpha = 0.5,
    width = 0.9
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = legend_labels_pat
  ) +
  scale_x_continuous(
    limits = c(1, max_rank_pat),
    breaks = c(1, seq(20, max_rank_pat, by = 20), max_rank_pat)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2)
  ) +
  labs(
    title = "Patient Performance Trend",
    x = "Patients (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

patient_overlap
ggsave(
  "Final_Patient_Performance_Trend_Overlap.pdf",
  patient_overlap,
  width = 10,
  height = 6,
  device = cairo_pdf
)

patient_dodge <- ggplot(
  patient_long,
  aes(
    x = Rank,
    y = Correlation,
    fill = Correlation_Type
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.8
  ) +
  scale_fill_manual(
    values = c(
      "pearson_corr" = "red",
      "spearman_corr" = "blue"
    ),
    labels = legend_labels_pat
  ) +
  scale_x_continuous(
    limits = c(1, max_rank_pat),
    breaks = c(1, seq(20, max_rank_pat, by = 20), max_rank_pat)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2)
  ) +
  labs(
    title = "Patient Performance Trend",
    x = "Patients (Ranked by Pearson Correlation)",
    y = "Correlation",
    fill = ""
  ) +
  common_theme

patient_dodge
ggsave(
  "Final_Patient_Performance_Trend_Dodge.pdf",
  patient_dodge,
  width = 10,
  height = 6,
  device = cairo_pdf
)
