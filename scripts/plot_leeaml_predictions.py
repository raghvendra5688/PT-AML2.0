# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: BeatAML2.0
#     language: python
#     name: python3
# ---

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

RESULTS_DIR = "../Results/tabpfn_leeaml/optuna"
PLOT_DIR    = "../Results/tabpfn_leeaml/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Pretty labels for each data-type key
LABEL_MAP = {
    "Embed_Feat_Var":    "SMILES Embedding",
    "PC_Feat_Var":       "PC / Fingerprints",
    "MolFormer_Feat_Var":"MolFormer",
    "ChemBERTa_Feat_Var":"ChemBERTa",
}

POINT_COLOR = "#2166AC"
LINE_COLOR  = "#D73027"
ALPHA       = 0.25
POINT_SIZE  = 8


def scatter_one(ax, x, y, title, xlabel="Actual AUC", ylabel="Predicted AUC"):
    """Draw a tight-bounding-box scatter with regression line and correlation stats."""
    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    ax.scatter(x, y, s=POINT_SIZE, color=POINT_COLOR, alpha=ALPHA, linewidths=0)

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color=LINE_COLOR, linewidth=1.5, zorder=3)

    pad_x = (x.max() - x.min()) * 0.03
    pad_y = (y.max() - y.min()) * 0.03
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    stats_text = (f"Pearson  $r$ = {pearson_r:.3f}  ($p$={pearson_p:.2e})\n"
                  f"Spearman $r$ = {spearman_r:.3f}  ($p$={spearman_p:.2e})\n"
                  f"$n$ = {len(x):,}")
    ax.text(0.97, 0.03, stats_text,
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="grey", alpha=0.85))

    ax.set_title(title, fontsize=15, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return pearson_r, spearman_r


# ── Discover available prediction files ──────────────────────────────────────
pred_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_optuna_predictions.csv")))
if not pred_files:
    raise FileNotFoundError(f"No prediction CSVs found in {RESULTS_DIR}")

print(f"Found {len(pred_files)} prediction file(s):")
for f in pred_files:
    print(f"  {os.path.basename(f)}")

# ── Individual plots ──────────────────────────────────────────────────────────
summary_rows = []

for fpath in pred_files:
    basename  = os.path.basename(fpath).replace("_optuna_predictions.csv", "")
    # Strip stratify_by suffix (e.g. _dbgap_rnaseq_sample) to get data_type key
    data_type = basename.replace("_dbgap_rnaseq_sample", "").replace("_inhibitor", "")
    title     = LABEL_MAP.get(data_type, data_type)

    df = pd.read_csv(fpath, sep="\t")
    df = df.dropna(subset=["predictions", "labels"])
    x  = df["labels"].to_numpy(dtype=float)       # actual AUC
    y  = df["predictions"].to_numpy(dtype=float)  # predicted AUC

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    pearson_r, spearman_r = scatter_one(ax, x, y, title)
    fig.tight_layout(pad=1.0)

    out_path = os.path.join(PLOT_DIR, f"{basename}_scatter.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    summary_rows.append({
        "data_type":  data_type,
        "n":          len(df),
        "pearson_r":  round(pearson_r,  4),
        "spearman_r": round(spearman_r, 4),
    })

# ── Combined multi-panel figure ───────────────────────────────────────────────
n_plots = len(pred_files)
n_cols  = min(n_plots, 2)
n_rows  = int(np.ceil(n_plots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 4.5 * n_rows),
                         squeeze=False)
axes_flat = axes.flatten()

for idx, fpath in enumerate(pred_files):
    basename  = os.path.basename(fpath).replace("_optuna_predictions.csv", "")
    data_type = basename.replace("_dbgap_rnaseq_sample", "").replace("_inhibitor", "")
    title     = LABEL_MAP.get(data_type, data_type)

    df = pd.read_csv(fpath, sep="\t").dropna(subset=["predictions", "labels"])
    x  = df["labels"].to_numpy(dtype=float)
    y  = df["predictions"].to_numpy(dtype=float)

    scatter_one(axes_flat[idx], x, y, title)

for idx in range(n_plots, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("LeeAML: Predicted AUC vs Actual AUC (BeatAML-trained model)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout(pad=1.5)

combined_path = os.path.join(PLOT_DIR, "LeeAML_all_configs_scatter.pdf")
fig.savefig(combined_path, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"\nCombined plot saved: {combined_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
print("\n=== Correlation summary ===")
print(summary_df.to_string(index=False))

summary_path = os.path.join(RESULTS_DIR, "Embed_Feat_Var_test_metrics_summary.csv")
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    print("\n=== Metrics from training run ===")
    print(existing.to_string(index=False))
