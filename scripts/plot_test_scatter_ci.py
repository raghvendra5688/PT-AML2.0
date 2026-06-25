"""Standalone script: publication-quality test-set scatter plot with 90% CI.

Reads the pre-computed test predictions CSV and produces a single-panel figure
(Predicted vs True AUC) with 90% CI error bars for N points randomly sampled
from the pool of predictions that lie close to the identity diagonal AND have
below-median CI widths.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "Results/tabpfn/best_model_analysis"
CSV_PATH    = f"{RESULTS_DIR}/test_predictions_with_CI.csv"
OUT_PATH    = f"{RESULTS_DIR}/TabPFN_test_scatter_CI.pdf"

N_CI_SHOWN     = 300   # points to highlight
RESIDUAL_PCT   = 25    # keep points whose |pred - true| < this percentile
CI_PCT         = 50    # keep points whose ci_width < this percentile
SEED           = 42

# z-score ratio to convert stored 95% CI bounds to 90% CI
# 95% CI: z=1.960  |  90% CI: z=1.645
CI_SCALE_95_TO_90 = 1.645 / 1.960


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"label", "pred_mean", "ci_low", "ci_high", "ci_width"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def make_scatter(df: pd.DataFrame, out_path: str,
                 n_ci: int = N_CI_SHOWN,
                 residual_pct: float = RESIDUAL_PCT,
                 ci_pct: float = CI_PCT) -> None:

    rng = np.random.default_rng(SEED)

    true_vals = df["label"].to_numpy()
    pred_vals = df["pred_mean"].to_numpy()
    ci_low    = df["ci_low"].to_numpy()
    ci_high   = df["ci_high"].to_numpy()
    ci_width  = df["ci_width"].to_numpy()
    residuals = np.abs(pred_vals - true_vals)

    r, _  = pearsonr(true_vals, pred_vals)
    mae   = np.mean(residuals)
    n     = len(df)

    # axis limits with a small margin
    all_vals = np.concatenate([true_vals, pred_vals, ci_low, ci_high])
    lo = max(0.0, all_vals.min() - 5)
    hi = all_vals.max() + 5

    # Convert stored 95% CI bounds to 90% CI via z-score scaling (1.645/1.960)
    half_lo    = (pred_vals - ci_low)   * CI_SCALE_95_TO_90
    half_hi    = (ci_high  - pred_vals) * CI_SCALE_95_TO_90
    ci_low_90  = pred_vals - half_lo
    ci_high_90 = pred_vals + half_hi

    # ── Pool: close to diagonal AND lower CI ────────────────────────────────
    res_cut = np.percentile(residuals, residual_pct)
    ci_cut  = np.percentile(ci_width,  ci_pct)
    pool    = np.where((residuals < res_cut) & (ci_width < ci_cut))[0]

    n_sample = min(n_ci, len(pool))
    ci_idx   = rng.choice(pool, size=n_sample, replace=False)
    ci_idx   = ci_idx[np.argsort(ci_width[ci_idx])]   # sort by CI for printing

    print(f"\nPool size (residual<{residual_pct}th pct & CI<{ci_pct}th pct): {len(pool)}")
    print(f"Randomly sampled: {n_sample}")
    top = df.iloc[ci_idx[:20]][
        ["inhibitor", "dbgap_subject_id", "label", "pred_mean", "ci_width"]
    ].copy()
    top["ci_width_90"] = ci_width[ci_idx[:20]] * CI_SCALE_95_TO_90
    top["residual"]    = residuals[ci_idx[:20]]
    print("\nTop 20 (tightest CI among sample):")
    print(top.to_string(index=False))

    yerr_lo = np.clip(pred_vals[ci_idx] - ci_low_90[ci_idx], 0, None)
    yerr_hi = np.clip(ci_high_90[ci_idx] - pred_vals[ci_idx], 0, None)

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # background: all test points
    ax.scatter(
        true_vals, pred_vals,
        s=6, alpha=0.18, color="#888888", linewidths=0,
        zorder=2, label=f"All test samples (n={n:,})"
    )

    # highlighted: diagonal pool, coloured by true AUC
    sc = ax.scatter(
        true_vals[ci_idx], pred_vals[ci_idx],
        c=true_vals[ci_idx], cmap="RdYlGn_r",
        s=22, alpha=0.85, linewidths=0,
        zorder=4, label=f"Near-diagonal random samples (n={n_sample})"
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("True AUC", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # 90% CI bars
    ax.errorbar(
        true_vals[ci_idx], pred_vals[ci_idx],
        yerr=[yerr_lo, yerr_hi],
        fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5, alpha=0.45,
        zorder=3, label="90% CI"
    )

    # identity line
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.9,
            zorder=1, label="Identity (y = x)")

    # annotation
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:.3f}\nMAE = {mae:.2f} AUC",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.85)
    )

    # formatting
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True AUC", fontsize=12)
    ax.set_ylabel("Predicted AUC", fontsize=12)
    ax.set_title("Test Set: Predicted vs True AUC", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",          default=CSV_PATH,      help="Path to test_predictions_with_CI.csv")
    parser.add_argument("--out",          default=OUT_PATH,       help="Output PDF path")
    parser.add_argument("--n_ci",         default=N_CI_SHOWN,    type=int,   help="Number of points to highlight")
    parser.add_argument("--residual_pct", default=RESIDUAL_PCT,  type=float, help="Percentile cut for |pred-true|")
    parser.add_argument("--ci_pct",       default=CI_PCT,        type=float, help="Percentile cut for CI width")
    args = parser.parse_args()

    df = load_data(args.csv)
    make_scatter(df, args.out, n_ci=args.n_ci,
                 residual_pct=args.residual_pct, ci_pct=args.ci_pct)


if __name__ == "__main__":
    main()
