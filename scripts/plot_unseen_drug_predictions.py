import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PRED_FILE = "../Results/tabpfn/best_model_analysis/test_predictions_with_CI.csv"
OUT_FILE  = "../Results/Figures/Unseen_Drugs_Scatter.pdf"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

UNSEEN_DRUGS = {
    'AT-101', 'BLZ945', 'BMS-754807', 'Birinapant', 'Etomoxir',
    'GSK-2879552', 'Indisulam', 'Metformin', 'NVP-AEW541', 'PH-797804',
    'Perhexiline maleate', 'Ralimetinib (LY2228820)', 'Ranolazine', 'XMD 8-87'
}

# ── load predictions ─────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with open(PRED_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['inhibitor'] not in UNSEEN_DRUGS:
            continue
        all_preds.append(float(row['pred_mean']))
        all_labels.append(float(row['label']))

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

pr, pp = stats.pearsonr(all_preds, all_labels)
sr, sp = stats.spearmanr(all_preds, all_labels)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(all_labels, all_preds, color="#2166AC",
           s=14, alpha=0.45, linewidths=0, zorder=2)

# regression line on pooled data
m, b = np.polyfit(all_labels, all_preds, 1)
x_line = np.linspace(all_labels.min(), all_labels.max(), 300)
ax.plot(x_line, m * x_line + b, color="#D73027", linewidth=1.8,
        linestyle="--", zorder=3, label="Regression line")

# diagonal (perfect prediction)
lims = [min(all_labels.min(), all_preds.min()),
        max(all_labels.max(), all_preds.max())]
ax.plot(lims, lims, color="grey", linewidth=1.0, linestyle=":", zorder=1,
        label="Perfect prediction")

# stats box
stats_text = (
    f"$r_{{pc}}$ = {pr:.3f}  ($p$ = {pp:.2e})\n"
    f"$r_{{sc}}$ = {sr:.3f}  ($p$ = {sp:.2e})\n"
    f"$n$ = {len(all_preds)} pairs  |  14 drugs"
)
ax.text(0.97, 0.04, stats_text,
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="grey", alpha=0.9))

ax.set_xlabel("Observed AUC", fontsize=11)
ax.set_ylabel("Predicted AUC", fontsize=11)
ax.set_title(
    "TabPFN (KD-Embed, PS-CV): Predictions on 14 Novel Test Drugs\n"
    "(drugs absent from BeatAML training set; Baiclein excluded — no embedding)",
    fontsize=10, fontweight="bold"
)
ax.tick_params(labelsize=9)
ax.spines[['top', 'right']].set_visible(False)


plt.tight_layout()
fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_FILE}")
print(f"\nPooled ({len(all_preds)} pairs, 14 drugs):")
print(f"  Pearson  r_pc = {pr:.4f}  (p = {pp:.2e})")
print(f"  Spearman r_sc = {sr:.4f}  (p = {sp:.2e})")
