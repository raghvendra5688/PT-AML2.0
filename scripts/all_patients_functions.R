library(data.table)
library(ggplot2)
#library(oligo)
library(preprocessCore)
library(rstatix)
library(ggpubr)
library(extrafont)
library(openxlsx)
loadfonts()

setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/")

plot_umap <- function(df, plot_title, expression_range, linetype_name)
{
  g_final <- ggplot(data = df, aes(x=umap1, y=umap2, fill=PANoptosis))+
    geom_point(aes(size=PANoptosis), pch=21)+
    stat_ellipse(aes(linetype=cell_types, color=cell_types), level=0.99, alpha=1)+
    scale_size_continuous(name="Activity", range=expression_range)+
    scale_fill_viridis_c(name="Activity", option="plasma")+
    scale_linetype_manual(name=linetype_name,values=c("twodash","solid","longdash","dotted","dashed","dotdash","solid"))+  
    scale_color_manual(name=linetype_name,values=c("Red","Blue","Green","Orange","Pink","Purple","Brown"))+
    xlab("UMAP_1")+ylab("UMAP_2")+
    theme_bw()+
    theme(legend.text=element_text(size=11, family="Arial"),
          legend.title=element_text(size=11, family="Arial"))+ 
    ggtitle(plot_title)+
    theme(
    text = element_text(size=11, family = "Arial", colour="black" ),
    axis.text.x = element_text(size=11, family = "Arial", colour = "black", angle=90, hjust=0.5, vjust=0.5),
    axis.text.y = element_text(size=11, family = "Arial", colour = "black" ),
    axis.title.x = element_text(size=11, family = "Arial", colour = "black"),
    axis.title.y = element_text(size=11, family = "Arial", colour = "black", angle=90, hjust=0.5, vjust=0.5),
    plot.title = element_text(size = 11, family = "Arial", colour="black"),
    axis.text = element_text(size=11, family="Arial", colour="black"))
    # theme(axis.text.x = element_text(color = "grey20", size = 1, angle = 0, hjust = .5, vjust = .5, face = "plain"),
     #       strip.text = element_text(color = "grey20", size=20, angle=0, hjust = 0.5, vjust = 0.5, face = "plain"),
    #       axis.text.y = element_text(color = "grey20", size = 16, angle = 0, hjust = 1, vjust = 0, face = "plain"),
    #       axis.title.x = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = 0, face = "plain"),
    #       axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
    #       title=element_text(color="grey20", size=20, face="plain"))
  return(g_final)
}

get_mean_se_stat_info <- function(df,output_path="Results/")
{
  
  #Perform the stat test using wilcox test
  ###############################################################################
  stat.test <- df %>%
    group_by(immune) %>%
    wilcox_test(value~Group) %>%
    adjust_pvalue(method = "bonferroni") %>%
    add_significance("p.adj")
  stat.test
  out_table <- desc_statby(df, "value", grps=c("immune","Group"))
  
  #Write out the mean and standard error of mean of expression of ZBP1 across cell types (for PRISM)
  #############################################################################
  write.table(out_table, file=paste0(output_path,"Mean_Se_Stat.csv"), row.names=F, col.names=T, sep="\t", quote=F)
  write.table(as.data.frame(stat.test), file=paste0(output_path,"P_Value_Stats.csv"), row.names=F, col.names=T, sep="\t", quote=F)
  
  return(stat.test)
}

#Make the plot comparing mean and se of mean for all cell types w.r.t. phenotype
make_mean_se_stat_plot <- function(df, stat.test, plot_title, xlabel, ylabel)
{
  #Make the plot of a particular gene expression across cell types (mean + sd of mean) and pvalues for the disease status
  ###########################################################################################3
  bp <- ggbarplot(df, x = "immune", y = "value", add = "mean_se", 
                  fill= "Group", 
                  position = position_dodge(0.8), width = 0.80)
  
  # Add p-values onto the bar plots
  stat.test <- stat.test %>%
    add_xy_position(fun = "mean_se", x = "immune", dodge = 0.8) 
  
  bp <- bp + stat_pvalue_manual(
    stat.test,  label = "p.adj.signif", tip.length = 0.01, y.position = stat.test$y.position+0.2,
  )
  
  # Move down the brackets using `bracket.nudge.y`
  bp + stat_pvalue_manual(
    stat.test,  label = "p.adj.signif", tip.length = 0.0,
    bracket.nudge.y = -2
  )
  
  bp <- bp+xlab(xlabel) + ylab(ylabel) + scale_fill_discrete(name="Pathway")+
    theme_light()+
    theme(legend.text=element_text(size=11),
          legend.title=element_text(size=11))+ 
    ggtitle(plot_title)+
    theme(
      text = element_text(size=11, family = "Arial", colour="black" ),
      axis.text.x = element_text(size=11, family = "Arial", colour = "black", angle=90, hjust=0.5, vjust=0.5),
      axis.text.y = element_text(size=11, family = "Arial", colour = "black" ),
      axis.title.x = element_text(size=11, family = "Arial", colour = "black"),
      axis.title.y = element_text(size=11, family = "Arial", colour = "black", angle=90, hjust=0.5, vjust=0.5),
      plot.title = element_text(size = 11, family = "Arial", colour="black"),
      axis.text = element_text(size=11, family="Arial", colour="black"))
    #theme(axis.text.x = element_text(color = "grey20", size = 16, angle = 90, hjust = .5, vjust = .5, face = "plain"),
    #      strip.text = element_text(color = "white", size=20, angle=0, hjust = 0.5, vjust = 0.5, face = "plain"),
    #      axis.text.y = element_text(color = "grey20", size = 16, angle = 0, hjust = 1, vjust = 0, face = "plain"),
    #      axis.title.x = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = 0, face = "plain"),
    #      axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
    #      title=element_text(color="grey20", size=20, face="plain"))
  return(bp)
}

get_cell_lines_drugs_of_interest <- function(cell_lines_for_intersect, drug_cell_line_response_matrix, named_cancer_info, topn=20)
{
  subset_drug_cell_line_response_matrix <- drug_cell_line_response_matrix[,cell_lines_for_intersect]
  
  #Get number of drugs which have negative z-score or sensitive to cell lines and keep only those cell lines which have at least 20 sensitive drugs
  cell_lines_with_sensitive_drugs <- colSums(subset_drug_cell_line_response_matrix < -1, na.rm=T)
  cell_lines_to_remove <- as.numeric(which(cell_lines_with_sensitive_drugs<topn))
  rev_subset_drug_cell_line_response_matrix <- subset_drug_cell_line_response_matrix[,-cell_lines_to_remove]
  
  #Get the drugs which are more sensitive across cell lines than those drugs which are more resistant across the cell lines of interest
  sensitive_drugs_across_cell_lines <- rowSums(rev_subset_drug_cell_line_response_matrix < -1, na.rm=T)
  resistant_drugs_across_cell_lines <- rowSums(rev_subset_drug_cell_line_response_matrix > 1, na.rm=T)
  drugs_of_interset <- names(which(sensitive_drugs_across_cell_lines>resistant_drugs_across_cell_lines))
  subset_sensitive_drugs_across_cell_lines <- sensitive_drugs_across_cell_lines[drugs_of_interset]
  
  #Top 25 drugs which have more sensitive cell lines than resistant cell lines
  subset_sensitive_drugs_across_cell_lines <- names(sort(subset_sensitive_drugs_across_cell_lines, decreasing = T))[c(1:25)]
  
  refined_drug_cell_response_matrix <- rev_subset_drug_cell_line_response_matrix[subset_sensitive_drugs_across_cell_lines,]
  final_cell_line_names <- colnames(refined_drug_cell_response_matrix)
  subset_named_cancer_info <- named_cancer_info[final_cell_line_names]
  return(list(refined_drug_cell_response_matrix, subset_named_cancer_info, final_cell_line_names))
}

rename_cell_lines <- function(refined_drug_cell_response_matrix, cell_lines_moi_COSMIC)
{
  cell_line_orig_names_mapping <- NULL
  cell_line_depmap_ids <- colnames(refined_drug_cell_response_matrix)
  for (i in 1:length(cell_line_depmap_ids))
  {
    cell_line_id <- cell_line_depmap_ids[i]
    cell_line_name <- cell_lines_moi_COSMIC[cell_lines_moi_COSMIC$DepMap_ID==cell_line_id,]$cell_line_name
    cell_line_orig_names_mapping <- c(cell_line_orig_names_mapping, cell_line_name)
  }
  colnames(refined_drug_cell_response_matrix) <- cell_line_orig_names_mapping
  return(refined_drug_cell_response_matrix) 
}

enrichIt_rmall <- function (obj, gene.sets = NULL, groups = 1000, cores = 2) 
{
  if (is.null(gene.sets)) {
    stop("Please provide the gene.sets you would like to use for \n            the enrichment analysis")
  }
  else {
    egc <- gene.sets
  }
  if (inherits(x = obj, what = "Seurat")) {
    cnts <- obj@assays[["RNA"]]@data
    cnts <- cnts[tabulate(summary(cnts)$i) != 0, , drop = FALSE]
    cnts <- as.matrix(cnts)
  }
  else {
    cnts <- obj
  }
  if (attr(class(egc), "package") == "GSEABase") {
    egc <- GSEABase::geneIds(egc)
  }
  cnts <- cnts[rowSums(cnts > 0) != 0, ]
  scores <- list()
  wind <- seq(1, ncol(cnts), by = groups)
  print(paste("Using sets of", groups, "cells. Running", length(wind), 
              "times."))
  for (i in wind) {
    last <- min(ncol(cnts), i + groups - 1)
    a <- suppressWarnings(gsva(cnts[, i:last], egc, method = "gsva", 
                               kcdf = "Gaussian", parallel.sz = cores, 
                               BPPARAM = SnowParam()))
    scores[[i]] <- a
  }
  scores <- do.call(cbind, scores)
  output <- data.frame(t(scores))
  return(output)
}

make_heatmap <- function(scaled_matrix){
  ht <- Heatmap(scaled_matrix, 
                cluster_rows = T,
                cluster_columns = T,
                name = "Z-score",
                width = unit(5, "cm"),
                height = unit(25, "cm"),
                cell_fun = function(j, i, x, y, width, height, fill) {
                  grid.rect(x = x, y = y, width = width, height = height, 
                            gp = gpar(col = "grey", fill = NA))
                  height = convertHeight(height, "mm")
                },
                row_names_centered = F,
                row_names_max_width = max_text_width(rownames(scaled_matrix), gp = gpar(fontsize = 10, family="Arial")),
                column_names_centered = F,
                show_column_names = T,
                column_title_rot = 0,
                column_title_side = "top",
                column_title_gp = gpar(fontsize=10, family="Aria", col=RColorBrewer::brewer.pal(8,"Set2")),
                column_gap = unit(2, "mm"),
                border_gp = gpar(col = "black", lty = 1),
                border = T,
                col = colorRamp2(c(-2,0,2),c("blue","white","red")),
                use_raster = T,
                heatmap_legend_param = list(direction = "horizontal")
  )
  return(ht)
}

parse.van.galen.supp <- function(vg.supp.file){
  
  vg <- data.table(read.xlsx(vg.supp.file))
  vg <- vg[,-c(1:4, 14),with=F]
  names(vg) <- unlist(vg[1,])
  names(vg)[1:3] <- paste(names(vg)[1:3], "Combined", sep="-")
  vg <- vg[-1,]
  vg <- vg[-51,]
  vg[,rank:=.I]
  
  vg[,`GMP-like`:=`GMP-like-Combined`]
  
  melt.vg <- melt(vg, id.vars="rank", measure.vars=setdiff(colnames(vg), "rank"), variable.factor = F)
  
  names(melt.vg) <- c("vg_rank", "vg_type", "display_label")
  
  melt.vg
  return(melt.vg)
}

