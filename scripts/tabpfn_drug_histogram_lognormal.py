"""
tabpfn_drug_histogram_lognormal.py

For each drug with >= MIN_PATIENTS test predictions, plot a histogram of
predicted AUC values and test whether the distribution is log-normal.

Log-normality test:
  1. Shapiro-Wilk on log(pred_mean)   — primary test
  2. D'Agostino-Pearson on log(pred_mean) — secondary (n > 20)
  A drug is called log-normal if the Shapiro-Wilk p-value > ALPHA.

Outputs (all written to OUT_DIR):
  drug_pred_histograms_lognormal.pdf   — grid of per-drug histograms
  drug_lognormal_summary.csv           — per-drug test statistics
  drug_lognormal_overview.pdf          — summary bar chart (pass/fail fractions)
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PRED_FILE   = "../Results/tabpfn/best_model_analysis/test_predictions_with_CI.csv"
OUT_DIR     = "../Results/tabpfn/best_model_analysis/"
MIN_PATIENTS = 20      # minimum number of patients per drug
ALPHA        = 0.05    # significance level for log-normal test
N_COLS       = 4       # histograms per row in the grid PDF
BINS         = 20      # histogram bins per drug
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def test_lognormal(values: np.ndarray, alpha: float = ALPHA):
    """
    Test whether `values` follow a log-normal distribution.

    Returns a dict with:
      sw_stat, sw_pval   — Shapiro-Wilk on log(values)
      dp_stat, dp_pval   — D'Agostino-Pearson on log(values)  (NaN if n <= 20)
      ks_stat, ks_pval   — K-S test against fitted lognorm
      is_lognormal       — True if Shapiro-Wilk p > alpha
      lognorm_mu         — mean of log(values)
      lognorm_sigma      — std of log(values)
    """
    log_vals = np.log(values)
    sw_stat, sw_pval = stats.shapiro(log_vals)

    if len(values) > 20:
        dp_stat, dp_pval = stats.normaltest(log_vals)
    else:
        dp_stat, dp_pval = float("nan"), float("nan")

    # Fit log-normal and run K-S
    shape, loc, scale = stats.lognorm.fit(values, floc=0)
    ks_stat, ks_pval = stats.kstest(values, "lognorm", args=(shape, loc, scale))

    mu    = log_vals.mean()
    sigma = log_vals.std(ddof=1)

    return dict(
        sw_stat=sw_stat, sw_pval=sw_pval,
        dp_stat=dp_stat, dp_pval=dp_pval,
        ks_stat=ks_stat, ks_pval=ks_pval,
        is_lognormal=(sw_pval > alpha),
        lognorm_mu=mu, lognorm_sigma=sigma,
    )


def plot_drug_histogram(ax, drug_name: str, values: np.ndarray, result: dict):
    """Draw one histogram with fitted log-normal overlay onto `ax`."""
    n = len(values)
    is_ln = result["is_lognormal"]

    color_hist = "#4C8BF5" if is_ln else "#E34234"
    edge_color  = "#1A5296" if is_ln else "#8B1A0E"
    label_color = "#155724" if is_ln else "#721c24"
    bg_color    = "#d4edda" if is_ln else "#f8d7da"

    ax.hist(values, bins=BINS, density=True,
            color=color_hist, edgecolor=edge_color, alpha=0.75, linewidth=0.5)

    # Fitted log-normal overlay
    x = np.linspace(max(values.min() * 0.9, 1e-3), values.max() * 1.05, 300)
    shape, loc, scale = stats.lognorm.fit(values, floc=0)
    ax.plot(x, stats.lognorm.pdf(x, shape, loc, scale),
            color="black", linewidth=1.4, label="Fitted log-normal")

    # Mean and median lines
    ax.axvline(values.mean(),   color="navy",  linewidth=1.0, linestyle="--", alpha=0.8, label=f"Mean={values.mean():.1f}")
    ax.axvline(np.median(values), color="darkorange", linewidth=1.0, linestyle=":", alpha=0.8, label=f"Median={np.median(values):.1f}")

    # Title and annotation
    status = "Log-normal ✓" if is_ln else "Not log-normal ✗"
    short_name = drug_name.split("(")[0].strip()
    if len(short_name) > 28:
        short_name = short_name[:26] + "…"
    ax.set_title(f"{short_name}\n$n$={n}", fontsize=7.5, fontweight="bold", pad=3)

    # p-value box
    pval_str = f"SW p={result['sw_pval']:.3f}"
    ax.text(0.97, 0.97, f"{status}\n{pval_str}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_color,
                      edgecolor=edge_color, alpha=0.9),
            color=label_color)

    ax.set_xlabel("Predicted AUC", fontsize=6.5)
    ax.set_ylabel("Density",       fontsize=6.5)
    ax.tick_params(axis="both", labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading predictions …")
df = pd.read_csv(PRED_FILE)
print(f"  {len(df):,} rows, {df['inhibitor'].nunique()} unique drugs")

# Drop rows with non-positive predictions (cannot log-transform)
df = df[df["pred_mean"] > 0].copy()

drug_counts = df.groupby("inhibitor").size()
qualifying  = drug_counts[drug_counts >= MIN_PATIENTS].index.sort_values().tolist()
print(f"  {len(qualifying)} drugs with >= {MIN_PATIENTS} patients")

# ── Per-drug log-normal tests ─────────────────────────────────────────────────
print("Running log-normality tests …")
records = []
for drug in qualifying:
    vals   = df.loc[df["inhibitor"] == drug, "pred_mean"].values
    result = test_lognormal(vals)
    records.append(dict(
        drug=drug,
        n=len(vals),
        mean_pred=vals.mean(),
        median_pred=np.median(vals),
        std_pred=vals.std(ddof=1),
        lognorm_mu=result["lognorm_mu"],
        lognorm_sigma=result["lognorm_sigma"],
        sw_stat=result["sw_stat"],
        sw_pval=result["sw_pval"],
        dp_stat=result["dp_stat"],
        dp_pval=result["dp_pval"],
        ks_stat=result["ks_stat"],
        ks_pval=result["ks_pval"],
        is_lognormal=result["is_lognormal"],
    ))

summary = pd.DataFrame(records).sort_values("drug")
n_pass = summary["is_lognormal"].sum()
n_fail = len(summary) - n_pass
print(f"  Log-normal (SW p > {ALPHA}): {n_pass}/{len(summary)} drugs")

# Save CSV
csv_path = os.path.join(OUT_DIR, "drug_lognormal_summary.csv")
summary.to_csv(csv_path, index=False, float_format="%.4f")
print(f"  Summary saved → {csv_path}")

# ── Grid histogram PDF ────────────────────────────────────────────────────────
n_drugs  = len(qualifying)
n_cols   = N_COLS
n_rows   = math.ceil(n_drugs / n_cols)
fig_w    = n_cols * 3.5
fig_h    = n_rows * 2.8

print(f"Plotting {n_drugs} histograms ({n_rows} rows × {n_cols} cols) …")

# Split across multiple A4-ish pages (max 5 rows per page)
MAX_ROWS_PER_PAGE = 5
n_pages = math.ceil(n_rows / MAX_ROWS_PER_PAGE)

hist_pdf_path = os.path.join(OUT_DIR, "drug_pred_histograms_lognormal.pdf")
with pdf_backend.PdfPages(hist_pdf_path) as pdf:
    drug_idx = 0
    for page in range(n_pages):
        drugs_on_page = qualifying[drug_idx: drug_idx + MAX_ROWS_PER_PAGE * n_cols]
        rows_on_page  = math.ceil(len(drugs_on_page) / n_cols)

        fig, axes = plt.subplots(
            rows_on_page, n_cols,
            figsize=(n_cols * 3.5, rows_on_page * 2.8),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)  # flatten

        for i, drug in enumerate(drugs_on_page):
            vals   = df.loc[df["inhibitor"] == drug, "pred_mean"].values
            result = summary[summary["drug"] == drug].iloc[0].to_dict()
            result["is_lognormal"] = bool(result["is_lognormal"])
            plot_drug_histogram(axes[i], drug, vals, result)

        # Hide unused axes on last page
        for j in range(len(drugs_on_page), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Per-drug predicted AUC distributions — TabPFN (KD-Embed) test set\n"
            f"Page {page + 1}/{n_pages}  |  "
            f"Blue = log-normal (SW p > {ALPHA}), Red = not log-normal",
            fontsize=9, fontweight="bold", y=1.01,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        drug_idx += len(drugs_on_page)

print(f"  Histogram PDF saved → {hist_pdf_path}")

# ── Overview bar chart ────────────────────────────────────────────────────────
print("Plotting overview …")

# Sort by log-normal status then by drug name for visual grouping
summary_sorted = summary.sort_values(["is_lognormal", "mean_pred"], ascending=[False, True])
colors = ["#4C8BF5" if v else "#E34234" for v in summary_sorted["is_lognormal"]]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Left: Shapiro-Wilk p-values per drug
ax_pval = axes2[0]
short_names = [d.split("(")[0].strip()[:30] for d in summary_sorted["drug"]]
ax_pval.barh(range(len(summary_sorted)), summary_sorted["sw_pval"], color=colors, edgecolor="white", linewidth=0.3)
ax_pval.axvline(ALPHA, color="black", linewidth=1.2, linestyle="--", label=f"α = {ALPHA}")
ax_pval.set_yticks(range(len(summary_sorted)))
ax_pval.set_yticklabels(short_names, fontsize=5.5)
ax_pval.set_xlabel("Shapiro-Wilk p-value (on log-transformed predictions)", fontsize=8)
ax_pval.set_title(
    f"Log-normality test per drug\n"
    f"{n_pass} pass (blue) / {n_fail} fail (red)  |  n ≥ {MIN_PATIENTS}",
    fontsize=9, fontweight="bold",
)
ax_pval.legend(fontsize=8)
ax_pval.spines[["top", "right"]].set_visible(False)

# Right: Pie chart
ax_pie = axes2[1]
wedge_colors = ["#4C8BF5", "#E34234"]
counts = [n_pass, n_fail]
labels = [f"Log-normal\n({n_pass} drugs, {100*n_pass/len(summary):.1f}%)",
          f"Not log-normal\n({n_fail} drugs, {100*n_fail/len(summary):.1f}%)"]
wedges, texts, autotexts = ax_pie.pie(
    counts, labels=labels, colors=wedge_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax_pie.set_title(
    f"Fraction of drugs with log-normal predicted AUC\n"
    f"(Shapiro-Wilk on log-predictions, α = {ALPHA})",
    fontsize=9, fontweight="bold",
)

overview_path = os.path.join(OUT_DIR, "drug_lognormal_overview.pdf")
fig2.savefig(overview_path, bbox_inches="tight")
plt.close(fig2)
print(f"  Overview saved → {overview_path}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"SUMMARY  (MIN_PATIENTS={MIN_PATIENTS}, ALPHA={ALPHA})")
print("=" * 60)
print(f"  Drugs tested      : {len(summary)}")
print(f"  Log-normal pass   : {n_pass} ({100*n_pass/len(summary):.1f}%)")
print(f"  Log-normal fail   : {n_fail} ({100*n_fail/len(summary):.1f}%)")
print(f"\n  Top 5 most log-normal (highest SW p-value):")
print(summary.nlargest(5, "sw_pval")[["drug", "n", "sw_pval", "mean_pred"]].to_string(index=False))
print(f"\n  Top 5 least log-normal (lowest SW p-value):")
print(summary.nsmallest(5, "sw_pval")[["drug", "n", "sw_pval", "mean_pred"]].to_string(index=False))
print("=" * 60)
