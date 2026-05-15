## ============================================================
##  Multi-Omics Heatmap  |  Colorblind-safe  |  Manuscript-ready
##
##  Requires:
##    BiocManager::install("ComplexHeatmap")
##    install.packages(c("circlize", "viridisLite"))
## ============================================================

library(ComplexHeatmap)
library(circlize)
library(viridisLite)

set.seed(42)
setwd("/export/cse/rmall/Raghvendra/PT-AML2.0/scripts/")

## ============================================================
## STEP 1 — YOUR DATA
##
## Supply only the matrices you need. Set any matrix to NULL
## to exclude it entirely — no other changes required.
##
## All non-NULL matrices must share the same columns (samples).
## ============================================================

N <- 15   # number of samples
S <- paste0("S", seq_len(N))

# Set any of these to NULL to drop that block
mat_clin <- matrix(rnorm(2 * N),           nrow = 2,
                   dimnames = list(paste0("ClinTrait_", 1:2), S))

mat_mut  <- matrix(rbeta(10 * N, 0.3, 5),   nrow = 10,
                   dimnames = list(paste0("Gene_", 1:10), S))

mat_expr <- matrix(rnorm(10 * N, sd = 2),   nrow = 10,
                   dimnames = list(paste0("GeneExpr_", 1:10), S))

# All three below are combined into one "functional enrichments" block.
# Set all three to NULL to drop that block entirely.
mat_path <- matrix(rnorm(5 * N, sd = 0.4),  nrow = 5,
                   dimnames = list(paste0("Pathway_", 1:5), S))

mat_cell <- matrix(runif(3 * N),             nrow = 3,
                   dimnames = list(c("Progenitor-like", "Monocyte-like", "cDC-like"), S))

mat_mod  <- matrix(rnorm(4 * N, sd = 0.3),  nrow = 4,
                   dimnames = list(paste0("Module_", 1:4), S))

## ============================================================
## STEP 2 — COLOUR SCALES  (all colorblind-safe)
## ============================================================

# Brown → White → Teal  (diverging, for signed data)
col_clin <- colorRamp2(
  seq(-2, 2, len = 11),
  c("#543005","#8C510A","#BF812D","#DFC27D","#F6E8C3","white",
    "#C7EAE5","#80CDC1","#35978F","#01665E","#003C30")
)

# Yellow → Purple  (sequential, for 0–1 frequencies)
col_mut <- colorRamp2(seq(0, 1, len = 128), viridis(128))

# Blue → White → Red  (diverging, for signed expression)
col_expr <- colorRamp2(
  seq(-5, 5, len = 11),
  c("#053061","#2166AC","#4393C3","#92C5DE","#D1E5F0","white",
    "#FDDBC7","#F4A582","#D6604D","#B2182B","#67001F")
)

# Cream → Deep purple  (sequential, shared across all enrichment rows)
col_enrich <- colorRamp2(
  seq(-0.5, 1, len = 128),
  magma(128, begin = 0.05, end = 0.92)
)

## ============================================================
## STEP 3 — BUILD HEATMAPS
## (only builds blocks whose data matrices are non-NULL)
## ============================================================

# Height scales automatically with number of rows
cm_per_row <- 0.3
min_cm     <- 2.0
block_h    <- function(mat) unit(max(min_cm, nrow(mat) * cm_per_row), "cm")

# Options applied to every heatmap block
SHARED <- list(
  cluster_columns     = FALSE,
  show_column_names   = FALSE,
  show_row_names      = FALSE,
  show_row_dend       = FALSE,
  show_heatmap_legend = FALSE,
  top_annotation      = NULL,
  border              = TRUE,
  border_gp           = gpar(col = "grey40", lwd = 0.4),
  row_title_side      = "right",
  row_title_rot       = 0,
  row_title_gp        = gpar(fontsize = 8, fontface = "bold")
)

ht <- function(...) do.call(Heatmap, modifyList(SHARED, list(...)))

# ── Build each block only if its matrix is not NULL ───────────

BLOCKS <- list()

if (!is.null(mat_clin))
  BLOCKS[["clinical annotation"]] <- ht(
    matrix    = mat_clin,  name = "Clinical",
    col       = col_clin,  row_title = "Clinical\nTraits",
    height    = block_h(mat_clin)
  )

if (!is.null(mat_mut))
  BLOCKS[["mutation frequencies"]] <- ht(
    matrix    = mat_mut,   name = "Mutation",
    col       = col_mut,   row_title = "Mutation\nFrequency",
    height    = block_h(mat_mut)
  )

if (!is.null(mat_expr))
  BLOCKS[["gene expression"]] <- ht(
    matrix    = mat_expr,  name = "Expression",
    col       = col_expr,  row_title = "Gene\nExpression",
    height    = block_h(mat_expr)
  )

# Enrichment block: built only if at least one sub-matrix is non-NULL
enrich_mats  <- Filter(Negate(is.null), list(
  path = mat_path, cell = mat_cell, mod = mat_mod
))

if (length(enrich_mats) > 0) {
  mat_enrich   <- do.call(rbind, enrich_mats)
  enrich_split <- factor(
    rep(
      c("Pathway\nActivities", "Cell Type\nEnrichment", "Module\nEnrichments")[
        c(!is.null(mat_path), !is.null(mat_cell), !is.null(mat_mod))
      ],
      sapply(enrich_mats, nrow)
    ),
    levels = c("Pathway\nActivities", "Cell Type\nEnrichment",
               "Module\nEnrichments")
  )
  
  BLOCKS[["functional enrichments"]] <- ht(
    matrix             = mat_enrich,  name = "Enrichment",
    col                = col_enrich,
    row_split          = enrich_split,
    cluster_row_slices = FALSE,
    row_gap            = unit(1, "mm"),
    height             = block_h(mat_enrich)
  )
}

if (length(BLOCKS) == 0) stop("No data matrices supplied — please set at least one matrix.")

## ============================================================
## STEP 4 — COMBINE AND SAVE
## ============================================================

save_heatmap <- function(
    blocks     = names(BLOCKS),   # defaults to all available blocks
    out_prefix = "Fig1",
    width_in   = 4,
    height_in  = NULL,            # NULL = auto from row counts
    res_dpi    = 300
) {
  chosen <- blocks[blocks %in% names(BLOCKS)]
  if (length(chosen) == 0)
    stop("None of the requested blocks are available: ",
         paste(blocks, collapse = ", "), "\nAvailable: ",
         paste(names(BLOCKS), collapse = ", "))
  
  ht_list <- if (length(chosen) == 1) BLOCKS[[chosen]] else Reduce(`%v%`, BLOCKS[chosen])
  
  if (is.null(height_in)) {
    cm_total  <- sum(sapply(chosen, function(b) as.numeric(BLOCKS[[b]]@matrix_param$height)))
    height_in <- cm_total / 2.54 + 1
  }
  
  draw_it <- function() {
    ComplexHeatmap::draw(
      ht_list,
      show_heatmap_legend    = FALSE,
      show_annotation_legend = FALSE,
      ht_gap                 = unit(3, "mm"),
      padding                = unit(c(2, 2, 2, 2), "mm"),
      background             = "white"
    )
  }
  
  pdf(paste0(out_prefix, ".pdf"), width = width_in, height = height_in,
      useDingbats = FALSE)
  draw_it(); dev.off()
  message("Saved: ", out_prefix, ".pdf  (",
          round(width_in, 1), " x ", round(height_in, 1), " in)")
  
  tiff(paste0(out_prefix, ".tiff"), width = width_in, height = height_in,
       units = "in", res = res_dpi, compression = "lzw")
  draw_it(); dev.off()
  message("Saved: ", out_prefix, ".tiff")
}

## ── Run ───────────────────────────────────────────────────────

# All available blocks (whatever is non-NULL above)
# save_heatmap(out_prefix = "BeatAML_full", width=5, height=11)

# Any subset — just name the ones you want
save_heatmap(blocks = c("clinical annotation", "mutation frequencies", "gene expression","functional enrichments"), out_prefix = "FIMM-AML_full", height=6, width=5)
# save_heatmap(blocks = c("mutation frequencies", "functional enrichments"), out_prefix = "BeatAML_mut_enrich")
# save_heatmap(blocks = "functional enrichments",                            out_prefix = "BeatAML_enrichment")