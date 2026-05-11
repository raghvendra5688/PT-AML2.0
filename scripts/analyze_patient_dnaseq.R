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
library(stringr)
library(GSA)
library(Matrix)
library(extrafont)
library(Rtsne)
#library(immunedeconv)
library(Polychrome)
library(R.utils)
loadfonts()
registerDoParallel(20)
ht_opt$message = FALSE

setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/")
source("scripts/all_cells_functions.R")

#Get the DNAseq file
dnaseq_df <- fread("BeatAML/Data/beataml_wes_wv1to4_mutations_dbgap.txt",header=T)
dnaseq_df <- as.data.frame(dnaseq_df)

mutation_df <- as.data.frame(cbind(dnaseq_df$dbgap_sample_id, dnaseq_df$symbol, dnaseq_df$variant_classification))
colnames(mutation_df) <- c("dbgap_sample_id","symbol","varclass")
#Convert DNA-seq sample (D) to RNA-seq sample (R)
mutation_df$dbgap_sample_id <- str_replace(mutation_df$dbgap_sample_id, 'D','R')

#Convert into mutation-by-gene matrix
unique_genes <- unique(mutation_df$symbol)
unique_samples <- unique(mutation_df$dbgap_sample_id)
unique_samples <- str_replace(unique_samples, 'D','R')
mutation_mat <- Matrix(0, nrow=length(unique_samples), ncol=length(unique_genes))
colnames(mutation_mat) <- unique_genes
rownames(mutation_mat) <- unique_samples
for (i in 1:nrow(mutation_df))
{
  sample_id <- mutation_df$dbgap_sample_id[i]
  gene <- mutation_df$symbol[i]
  mutation_mat[sample_id,gene] <- mutation_mat[sample_id,gene]+1
}

#Get the train, test ids and create train+test mutation matrix
load("Data/Train_Test_Ids.Rdata")
all_sample_ids <- c(train_sample_ids, test_sample_ids)
all_mut_mat <- Matrix(0, nrow=length(all_sample_ids), ncol=length(unique_genes))
colnames(all_mut_mat) <- unique_genes
rownames(all_mut_mat) <- all_sample_ids
for (i in 1:length(all_sample_ids))
{
  sample_id <- all_sample_ids[i]
  if (sample_id %in% unique_samples)
  {
    all_mut_mat[sample_id,] <- mutation_mat[sample_id,]
  }
}

#Select genes with maximum standard deviation (sd>0.1) 
sd_genes <- names(which(sort(apply(all_mut_mat, 2, sd),decreasing = T)>0.1))
rev_mut_mat <- all_mut_mat[,sd_genes]

#Get train mut-by-gene matrix and test mut-by-gene matrix
train_mut_mat <- rev_mut_mat[train_sample_ids,]
test_mut_mat <- rev_mut_mat[test_sample_ids,]

################################################################################
#Convert into mutation-by-mut-type matrix
unique_variants <- unique(mutation_df$varclass)
mutation_var_mat <- Matrix(0, nrow=length(unique_samples), ncol=length(unique_variants))
colnames(mutation_var_mat) <- unique_variants
rownames(mutation_var_mat) <- unique_samples
for (i in 1:nrow(mutation_df))
{
  sample_id <- mutation_df$dbgap_sample_id[i]
  variant <- mutation_df$varclass[i]
  mutation_var_mat[sample_id, variant] <- mutation_var_mat[sample_id, variant]+1
}

all_mut_var_mat <- Matrix(0, nrow=length(all_sample_ids), ncol=length(unique_variants))
colnames(all_mut_var_mat) <- unique_variants
rownames(all_mut_var_mat) <- all_sample_ids
for (i in 1:length(all_sample_ids))
{
  sample_id <- all_sample_ids[i]
  if (sample_id %in% unique_samples)
  {
    all_mut_var_mat[sample_id,] <- mutation_var_mat[sample_id,]
  }
}

#Get the train mut-by-mut-type matrix and same for test set
train_mut_var_mat <- all_mut_var_mat[train_sample_ids,]
test_mut_var_mat <- all_mut_var_mat[test_sample_ids,]

train_mut_mat <- as.data.frame(as.matrix(train_mut_mat))
test_mut_mat <- as.data.frame(as.matrix(test_mut_mat))
train_mut_var_mat <- as.data.frame(as.matrix(train_mut_var_mat))
test_mut_var_mat <- as.data.frame(as.matrix(test_mut_var_mat))
train_mut_mat$dbgap_rnaseq_sample <- rownames(train_mut_mat)
test_mut_mat$dbgap_rnaseq_sample <- rownames(test_mut_mat)
train_mut_var_mat$dbgap_rnaseq_sample <- rownames(train_mut_var_mat)
test_mut_var_mat$dbgap_rnaseq_sample <- rownames(test_mut_var_mat)
rownames(train_mut_mat) <- NULL
rownames(test_mut_mat) <- NULL
rownames(train_mut_var_mat) <- NULL
rownames(test_mut_var_mat) <- NULL

save(train_mut_mat,test_mut_mat,train_mut_var_mat, test_mut_var_mat, file="Data/Train_Test_Mutation_Matrices.Rdata")