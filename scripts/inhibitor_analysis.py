"""
Analyse unique inhibitors and patients in the training and test sets,
and produce a publication-ready grouped bar plot of unique patients
per inhibitor (Train vs. Test).

Also produces a paper-ready scatter plot of MDREAM predictions for the
14 non-overlapping (test-only) drugs, annotated with MAE, RMSE,
Pearson r, and Spearman rho.

Identifiers sourced from cv_evaluation.py:
  - Patient  : dbgap_rnaseq_sample
  - Inhibitor: inhibitor

Outputs
-------
../Results/PreProcess/patients_per_inhibitor.pdf / .png
../Results/PreProcess/mdream_scatter.pdf / .png
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_FILE    = "../Data/Training_Set_Var_with_Drug_Only_PC_Patient_Info.pkl"
TEST_FILE     = "../Data/Test_Set_Var_with_Drug_Only_PC_Patient_Info.pkl"
PATIENT_COL   = "dbgap_rnaseq_sample"
INHIBITOR_COL = "inhibitor"
OUT_DIR       = "../Results/PreProcess/"

# ColorBrewer RdBu — colorblind-safe and print-safe
TRAIN_COLOR = "#2166AC"   # blue
TEST_COLOR  = "#D6604D"   # red
EDGE_COLOR  = "white"
BAR_WIDTH   = 0.38

# Typography — consistent with most journal styles
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "pdf.fonttype":      42,   # embed fonts as TrueType in PDF
    "ps.fonttype":       42,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading training set ...")
train_df = pd.read_pickle(TRAIN_FILE, compression="zip")
print(f"  Shape: {train_df.shape}")

print("Loading test set ...")
test_df = pd.read_pickle(TEST_FILE, compression="zip")
print(f"  Shape: {test_df.shape}")

# ---------------------------------------------------------------------------
# Inhibitor analysis
# ---------------------------------------------------------------------------
train_inhibitors = set(train_df[INHIBITOR_COL].dropna().unique())
test_inhibitors  = set(test_df[INHIBITOR_COL].dropna().unique())

only_in_test  = test_inhibitors - train_inhibitors
only_in_train = train_inhibitors - test_inhibitors
in_both       = train_inhibitors & test_inhibitors

print("\n" + "=" * 60)
print("INHIBITOR SUMMARY")
print("=" * 60)
print(f"  Unique inhibitors in TRAIN : {len(train_inhibitors)}")
print(f"  Unique inhibitors in TEST  : {len(test_inhibitors)}")
print(f"  Inhibitors in both         : {len(in_both)}")
print(f"  Only in TRAIN (not in test): {len(only_in_train)}")
print(f"  Only in TEST  (not in train): {len(only_in_test)}")

if only_in_test:
    print(f"\n  Inhibitors present in TEST but absent from TRAIN ({len(only_in_test)}):")
    for inh in sorted(only_in_test):
        print(f"    - {inh}")

if only_in_train:
    print(f"\n  Inhibitors present in TRAIN but absent from TEST ({len(only_in_train)}):")
    for inh in sorted(only_in_train):
        print(f"    - {inh}")

# ---------------------------------------------------------------------------
# Patient analysis
# ---------------------------------------------------------------------------
train_patients = set(train_df[PATIENT_COL].dropna().unique())
test_patients  = set(test_df[PATIENT_COL].dropna().unique())

only_patients_in_test  = test_patients - train_patients
only_patients_in_train = train_patients - test_patients
patients_in_both       = train_patients & test_patients

print("\n" + "=" * 60)
print("PATIENT SUMMARY  (dbgap_rnaseq_sample)")
print("=" * 60)
print(f"  Unique patients in TRAIN : {len(train_patients)}")
print(f"  Unique patients in TEST  : {len(test_patients)}")
print(f"  Patients in both         : {len(patients_in_both)}")
print(f"  Only in TRAIN            : {len(only_patients_in_train)}")
print(f"  Only in TEST             : {len(only_patients_in_test)}")

# ---------------------------------------------------------------------------
# Count unique patients per inhibitor
# ---------------------------------------------------------------------------
train_counts = (
    train_df.groupby(INHIBITOR_COL)[PATIENT_COL]
    .nunique()
    .rename("Train")
)
test_counts = (
    test_df.groupby(INHIBITOR_COL)[PATIENT_COL]
    .nunique()
    .rename("Test")
)

# Union of all inhibitors; fill 0 where an inhibitor is absent from one split
counts = (
    pd.concat([train_counts, test_counts], axis=1)
    .fillna(0)
    .astype(int)
    .sort_values("Train", ascending=False)
)

print(f"\nUnique patients per inhibitor:")
print(counts.to_string())
print(f"\nTotal inhibitors: {len(counts)}")

# ---------------------------------------------------------------------------
# Figure — stacked bar plot: unique patients per inhibitor (Train vs. Test)
# Common inhibitors: stacked bar (Train bottom, Test top)
# Train-only: single Train-coloured bar
# Test-only : single Test-coloured bar
# ---------------------------------------------------------------------------

# Sort: common inhibitors first (by Train count desc), then train-only, then test-only
mask_common     = (counts["Train"] > 0) & (counts["Test"] > 0)
mask_train_only = (counts["Train"] > 0) & (counts["Test"] == 0)
mask_test_only  = (counts["Train"] == 0) & (counts["Test"] > 0)

counts = pd.concat([
    counts[mask_common].sort_values("Train", ascending=False),
    counts[mask_train_only].sort_values("Train", ascending=False),
    counts[mask_test_only].sort_values("Test", ascending=False),
])

n = len(counts)
BAR_WIDTH_STACK = 0.70

# Compact fixed size — independent of inhibitor count
fig_w = 7.0
fig_h = 3.5

# Assign D1, D2, … labels in sorted order; save mapping for reference
drug_labels = [f"D{i+1}" for i in range(n)]
label_map   = dict(zip(drug_labels, counts.index))   # D1 → inhibitor name
print("\nDrug label mapping:")
for lbl, name in label_map.items():
    print(f"  {lbl}: {name}")

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

x = np.arange(n)

# Train portion (bottom of stack for common; full bar for train-only)
ax.bar(
    x, counts["Train"], BAR_WIDTH_STACK,
    color=TRAIN_COLOR, edgecolor=EDGE_COLOR, linewidth=0.4,
    label="Train", zorder=3,
)
# Test portion (stacked on top of Train for common; full bar for test-only)
ax.bar(
    x, counts["Test"], BAR_WIDTH_STACK,
    bottom=counts["Train"],
    color=TEST_COLOR, edgecolor=EDGE_COLOR, linewidth=0.4,
    label="Test", zorder=3,
)

# Show every 5th label to avoid crowding
tick_step = 5
ax.set_xticks(x[::tick_step])
ax.set_xticklabels(drug_labels[::tick_step], rotation=90, ha="center", va="top", fontsize=7)
ax.set_ylabel("Unique patients", fontsize=9, labelpad=4)
ax.set_xlabel("Drug", fontsize=9, labelpad=4)
ax.set_title(
    "Unique patients per inhibitor — Train vs. Test",
    fontsize=10, fontweight="bold", pad=6,
)

# Vertical separator between common and train-only sections
n_common = mask_common.sum()
n_train_only = mask_train_only.sum()
if n_common > 0 and n_train_only > 0:
    ax.axvline(n_common - 0.5, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=2)
if (n_common + n_train_only) > 0 and mask_test_only.sum() > 0:
    ax.axvline(n_common + n_train_only - 0.5, color="#AAAAAA", linewidth=0.8,
               linestyle=":", zorder=2)

ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
ax.grid(axis="y", linestyle="--", linewidth=0.4, color="#CCCCCC", zorder=0)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)

ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="x", length=0)

ax.legend(
    frameon=False, fontsize=8,
    loc="upper right",
    handlelength=1.0, handleheight=0.7,
    borderpad=0.3, labelspacing=0.2,
)

fig.tight_layout(pad=0.4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, "patients_per_inhibitor.pdf")
png_path = os.path.join(OUT_DIR, "patients_per_inhibitor.png")

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")

# ---------------------------------------------------------------------------
# MDREAM helpers — shared constants
# ---------------------------------------------------------------------------
MDREAM_FILE = "../Results/MDREAM/predictions_all_ablations.csv"
ABLATIONS   = {"A1": "pred_auc_A1", "A2": "pred_auc_A2", "A3": "pred_auc_A3"}
ABL_COLORS  = {"A1": "#1B9E77", "A2": "#D95F02", "A3": "#7570B3"}  # ColorBrewer Dark2
MIN_PEARSON = 0.2   # only drugs whose mean Pearson r (across ablations) exceeds this are plotted


def _pearson(y_true, y_pred):
    """Return Pearson r, or NaN if not computable."""
    if len(y_true) >= 3 and y_true.std() > 0 and y_pred.std() > 0:
        r, _ = stats.pearsonr(y_true, y_pred)
        return r
    return np.nan


def _drug_mean_r(grp):
    """Mean Pearson r across the three ablations for a single drug group."""
    rs = [_pearson(grp["true_auc"].values, grp[col].values)
          for col in ABLATIONS.values()]
    return np.nanmean(rs)


def _select_drugs(df, train_df, test_df, min_r=MIN_PEARSON):
    """
    Return sorted list of non-overlapping (test-only) drugs whose mean
    Pearson r across ablations exceeds *min_r*.
    """
    train_drugs = set(train_df[INHIBITOR_COL].dropna().unique())
    test_drugs  = set(test_df[INHIBITOR_COL].dropna().unique())
    non_overlap = test_drugs - train_drugs
    df_nl = df[df["drug"].isin(non_overlap)]
    kept = [
        drug for drug, grp in df_nl.groupby("drug")
        if _drug_mean_r(grp) > min_r
    ]
    return sorted(kept)


# ---------------------------------------------------------------------------
# MDREAM metrics — save full CSV (all 14 drugs)
# ---------------------------------------------------------------------------

def compute_mdream_metrics(train_df, test_df, out_dir=OUT_DIR):
    """
    Compute per-ablation metrics (MAE, RMSE, Pearson r, Spearman ρ) for the
    14 non-overlapping (test-only) drugs across MDREAM ablations A1, A2, A3.
    Saves results to mdream_metrics.csv (all 14 drugs, for completeness).
    """
    df = pd.read_csv(MDREAM_FILE)
    train_drugs = set(train_df[INHIBITOR_COL].dropna().unique())
    test_drugs  = set(test_df[INHIBITOR_COL].dropna().unique())
    non_overlap = test_drugs - train_drugs
    df_hl       = df[df["drug"].isin(non_overlap)]

    def _full_metrics(grp, col):
        y_true = grp["true_auc"].values
        y_pred = grp[col].values
        n      = len(y_true)
        mae    = np.mean(np.abs(y_true - y_pred))
        rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r_p    = _pearson(y_true, y_pred)
        if n >= 3 and y_true.std() > 0 and y_pred.std() > 0:
            r_s, _ = stats.spearmanr(y_true, y_pred)
        else:
            r_s = np.nan
        return n, mae, rmse, r_p, r_s

    metrics_rows = []
    for label, col in ABLATIONS.items():
        for drug, grp in df_hl.groupby("drug"):
            n, mae, rmse, r_p, r_s = _full_metrics(grp, col)
            metrics_rows.append({
                "drug":         drug,
                "ablation":     label,
                "n_samples":    n,
                "MAE":          round(mae,  4),
                "RMSE":         round(rmse, 4),
                "Pearson_r":    round(r_p,  4) if not np.isnan(r_p) else np.nan,
                "Spearman_rho": round(r_s,  4) if not np.isnan(r_s) else np.nan,
            })

    os.makedirs(out_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["drug", "ablation"])
    csv_path   = os.path.join(out_dir, "mdream_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(metrics_df.to_string(index=False))
    return metrics_df


# ---------------------------------------------------------------------------
# MDREAM combined scatter — all 6 qualifying drugs in one plot
# ---------------------------------------------------------------------------

# ColorBrewer Set2 — 6 qualitative colors, colorblind-safe, print-safe
_DRUG_PALETTE = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F"]


def plot_mdream_nonoverlap_scatter(train_df, test_df, out_dir=OUT_DIR):
    """
    1×3 panel figure — one Spearman scatter per ablation (A1, A2, A3).
    Each panel pools all patients from the 6 non-overlapping drugs with
    mean Pearson r > MIN_PEARSON. Annotates Spearman ρ, Pearson r, and MAE.
    Axis limits are shared across panels for direct visual comparison.
    """
    df    = pd.read_csv(MDREAM_FILE)
    drugs = _select_drugs(df, train_df, test_df)
    df_nl = df[df["drug"].isin(drugs)].copy()

    # Global axis limits (shared across all 3 panels)
    all_vals = np.concatenate(
        [df_nl["true_auc"].values] +
        [df_nl[col].values for col in ABLATIONS.values()]
    )
    lo = all_vals.min(); hi = all_vals.max()
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))

    for ax, (label, col) in zip(axes, ABLATIONS.items()):
        y_true = df_nl["true_auc"].values
        y_pred = df_nl[col].values
        n      = len(y_true)

        # Metrics
        r_s, p_s = stats.spearmanr(y_true, y_pred)
        r_p      = _pearson(y_true, y_pred)
        mae      = np.mean(np.abs(y_true - y_pred))

        # Scatter
        ax.scatter(y_true, y_pred, s=14, alpha=0.45, color=TRAIN_COLOR,
                   linewidths=0, zorder=3)

        # Identity line
        ax.plot([lo, hi], [lo, hi], color="#AAAAAA", lw=1.0, ls="--",
                zorder=2, alpha=0.8, label="y = x")

        # Annotation
        p_str = f"{p_s:.2e}" if p_s < 0.001 else f"{p_s:.3f}"
        ann = (
            f"Spearman ρ = {r_s:.2f}  (p = {p_str})\n"
            f"Pearson r  = {r_p:.2f}\n"
            f"MAE = {mae:.1f}\n"
            f"n = {n}"
        )
        ax.text(
            0.03, 0.97, ann,
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC",
                      lw=0.5, alpha=0.93),
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlabel("True AUC", fontsize=8.5, labelpad=3)
        if ax is axes[0]:
            ax.set_ylabel("Predicted AUC", fontsize=8.5, labelpad=3)
        ax.tick_params(labelsize=7.5, length=3, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.grid(linestyle=":", linewidth=0.4, color="#DDDDDD", zorder=0)
        ax.legend(fontsize=7, frameon=False, loc="upper left",
                  handlelength=1.2, labelspacing=0.25)

    fig.suptitle(
        f"MDREAM predictions — {len(drugs)} non-overlapping drugs "
        f"(mean Pearson r > {MIN_PEARSON})",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout(pad=0.5, w_pad=1.0)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"mdream_nonoverlap_scatter.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MDREAM metrics heatmap — drugs with mean Pearson r > MIN_PEARSON only
# ---------------------------------------------------------------------------

def plot_mdream_metrics_heatmap(metrics_df, train_df, test_df, out_dir=OUT_DIR):
    """
    Side-by-side heatmap of MAE (left) and Pearson r (right) for the
    non-overlapping drugs that pass the MIN_PEARSON threshold.
    Drugs are sorted by mean Pearson r descending (best predictable at top).
    """
    df   = pd.read_csv(MDREAM_FILE)
    kept = _select_drugs(df, train_df, test_df)

    sub = metrics_df[metrics_df["drug"].isin(kept)].copy()

    mae_piv = sub.pivot(index="drug", columns="ablation", values="MAE")
    r_piv   = sub.pivot(index="drug", columns="ablation", values="Pearson_r")

    # Sort by mean Pearson r descending (most predictable drug first)
    order   = r_piv.mean(axis=1).sort_values(ascending=False).index
    mae_piv = mae_piv.loc[order]
    r_piv   = r_piv.loc[order]

    n_drugs = len(mae_piv)
    fig_h   = max(3.5, n_drugs * 0.45 + 1.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, fig_h))

    def _draw(ax, data, cmap, vmin, vmax, fmt, title, cbar_label):
        im = ax.imshow(
            data.values, aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns, fontsize=9, fontweight="bold")
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index, fontsize=8)
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Cell annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, fmt.format(val),
                        ha="center", va="center", fontsize=8.5, color="black",
                        bbox=dict(fc="white", ec="none", alpha=0.5, pad=0.4),
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, shrink=0.75)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label(cbar_label, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    _draw(ax1, mae_piv, "YlOrRd",
          vmin=0, vmax=mae_piv.values.max(),
          fmt="{:.1f}", title="MAE  (lower = better)", cbar_label="MAE")

    _draw(ax2, r_piv, "RdYlGn",
          vmin=-0.5, vmax=0.5,
          fmt="{:.2f}", title="Pearson r  (higher = better)", cbar_label="Pearson r")
    ax2.set_yticks([])   # drug names already on left panel

    fig.suptitle(
        f"MDREAM metrics — non-overlapping drugs (mean Pearson r > {MIN_PEARSON})",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=0.7, w_pad=1.2)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"mdream_nonoverlap_metrics.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run MDREAM analysis
# ---------------------------------------------------------------------------
metrics_df = compute_mdream_metrics(train_df, test_df)
plot_mdream_nonoverlap_scatter(train_df, test_df)
plot_mdream_metrics_heatmap(metrics_df, train_df, test_df)
