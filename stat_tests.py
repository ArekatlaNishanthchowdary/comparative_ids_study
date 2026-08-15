"""
stat_tests.py
=============
Statistical Significance Analysis for NIDS Data-Balancing Comparison Paper.

Performs:
  1. Friedman Test across all 10 balancing strategies.
  2. Nemenyi Post-Hoc Test & Average Rank calculation.
  3. Critical Difference (CD) Diagram generation at alpha = 0.05.
  4. Exports results/stat_tests.csv and images/cd_diagram.png (300 DPI).

Usage:
  python stat_tests.py
"""

import os
import sys
import io
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_CSV = "results/results.csv"
RESULTS_DIR = "results"
IMAGES_DIR  = "images"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def run_statistical_tests(csv_path: str = RESULTS_CSV):
    df = pd.read_csv(csv_path)

    # 1. Pivot matrix: Scenario (Dataset x Classifier) x Balancer -> F1 Score
    # Scenario is a distinct dataset-classifier pair (e.g. nslkdd_RF, ids2018_LGBM, etc.)
    df["scenario"] = df["dataset"] + "_" + df["classifier"]
    pivot = df.pivot(index="scenario", columns="balancer", values="f1").dropna()

    scenarios = pivot.index.tolist()
    balancers = pivot.columns.tolist()
    N = len(scenarios)  # number of evaluation blocks
    k = len(balancers)  # number of balancing techniques

    print("=" * 65)
    print("  NIDS Data-Balancing — Statistical Significance Analysis")
    print("=" * 65)
    print(f"  Scenarios (N): {N}  (Dataset x Classifier combinations)")
    print(f"  Balancers (k): {k}  {balancers}")
    print("-" * 65)

    # 2. Rank each balancer per scenario (1 = best/highest F1)
    # Higher F1 is better -> ascending=False -> rank(1 = highest)
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()

    print("\n--- Average Ranks (1 = Best F1 Performance) ---")
    for b_name, rank_val in avg_ranks.items():
        print(f"  {b_name:16s} : {rank_val:.3f}")

    # 3. Friedman Test
    # SciPy friedmanchisquare takes vectors of scores per treatment
    treatment_data = [pivot[b].values for b in balancers]
    stat, p_val = stats.friedmanchisquare(*treatment_data)

    print("\n--- Friedman Test Results ---")
    print(f"  Friedman Chi-Square Statistic : {stat:.4f}")
    print(f"  Degrees of Freedom (df)       : {k - 1}")
    print(f"  p-value                       : {p_val:.4e}")
    print(f"  Statistically Significant (p < 0.05)? : {'YES' if p_val < 0.05 else 'NO'}")

    # 4. Nemenyi Critical Difference (CD) Calculation
    # Formula: CD = q_alpha * sqrt( (k * (k + 1)) / (6 * N) )
    # Studentized range statistic q_alpha at alpha = 0.05 for k = 10 is approx 3.164 / sqrt(2) * sqrt(2) -> q_alpha = 3.164
    # Critical value table for Nemenyi test (alpha = 0.05):
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                     7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q_alpha = q_alpha_table.get(k, 3.164)
    cd = q_alpha * math.sqrt((k * (k + 1)) / (6.0 * N))

    print("\n--- Nemenyi Post-Hoc Test ---")
    print(f"  Critical Value q_alpha (k={k}, alpha=0.05) : {q_alpha}")
    print(f"  Critical Difference (CD)                   : {cd:.4f}")

    # 5. Export Summary CSV
    summary_df = pd.DataFrame({
        "balancer": avg_ranks.index,
        "avg_rank": avg_ranks.values,
        "friedman_stat": stat,
        "friedman_pvalue": p_val,
        "critical_difference": cd,
    })
    out_csv = os.path.join(RESULTS_DIR, "stat_tests.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n  [OK] Saved statistical test table -> {out_csv}")

    # 6. Plot Critical Difference (CD) Diagram
    plot_cd_diagram(avg_ranks, cd, k, N, os.path.join(IMAGES_DIR, "cd_diagram.png"))


def _find_cliques(sorted_ranks, cd):
    """Maximal groups of consecutive (rank-sorted) methods whose full rank
    span is <= CD -- i.e. groups that are NOT significantly different from
    each other by the Nemenyi test. Sub-groups already covered by a larger
    clique are dropped so only maximal cliques are returned."""
    n = len(sorted_ranks)
    cliques = []
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    return [c for c in cliques
            if not any(c != o and o[0] <= c[0] and c[1] <= o[1] for o in cliques)]


def plot_cd_diagram(avg_ranks: pd.Series, cd: float, k: int, N: int, out_path: str):
    """Draw publication-quality Critical Difference (CD) Diagram."""
    sorted_vals = avg_ranks.values
    cliques = _find_cliques(list(sorted_vals), cd)
    clique_rows = max(len(cliques), 1) if cliques else 0

    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * clique_rows))

    # X-axis scale representing ranks from 1 to k
    low_rank, high_rank = 1, k
    ax.set_xlim(low_rank - 0.5, high_rank + 0.5)
    ax.set_ylim(-1 - 0.5 * clique_rows, len(avg_ranks) + 1)

    # Main rank axis
    ax.hlines(0, low_rank, high_rank, color="black", linewidth=1.5)
    for r in range(low_rank, high_rank + 1):
        ax.vlines(r, -0.15, 0.15, color="black", linewidth=1.0)
        ax.text(r, 0.3, str(r), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.text((low_rank + high_rank) / 2, 0.7, "Average Rank (1 = Best)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    # CD Bar in top-left corner
    ax.hlines(len(avg_ranks) - 0.5, low_rank, low_rank + cd, color="crimson", linewidth=2.5)
    ax.vlines(low_rank, len(avg_ranks) - 0.7, len(avg_ranks) - 0.3, color="crimson", linewidth=2.0)
    ax.vlines(low_rank + cd, len(avg_ranks) - 0.7, len(avg_ranks) - 0.3, color="crimson", linewidth=2.0)
    ax.text(low_rank + cd / 2, len(avg_ranks) - 0.2, f"CD = {cd:.2f}",
            ha="center", va="bottom", color="crimson", fontsize=10, fontweight="bold")

    # Clique bars: thick bars connecting the group(s) of methods that are
    # NOT significantly different from each other (rank span <= CD)
    for depth, (i, j) in enumerate(cliques):
        y = -0.4 - depth * 0.5
        ax.hlines(y, sorted_vals[i], sorted_vals[j],
                  color="black", linewidth=4.0, alpha=0.85, capstyle="butt")
    if cliques:
        ax.text(low_rank - 0.5, -0.4 - (len(cliques) - 1) * 0.5 - 0.35,
                "Thick bar = not significantly different (CD-connected)",
                ha="left", va="top", fontsize=8, color="dimgrey", fontstyle="italic")

    # Plot each method's rank line and text label
    y_step = 0.8
    sorted_methods = avg_ranks.index.tolist()

    # Left side vs Right side positioning for clean visualization
    mid = len(sorted_methods) // 2
    left_methods = sorted_methods[:mid]
    right_methods = sorted_methods[mid:]

    colors = plt.cm.Set2(np.linspace(0, 1, k))

    for idx, (method, r_val) in enumerate(avg_ranks.items()):
        y_pos = idx + 1
        # Point on rank line
        ax.scatter(r_val, 0, color="navy", s=35, zorder=5)
        # Line from rank axis to method label
        ax.plot([r_val, r_val], [0, y_pos], color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        # Horizontal extension to label
        if idx < mid:
            ax.plot([r_val, low_rank - 0.3], [y_pos, y_pos], color="grey", linestyle="-", linewidth=0.8)
            ax.text(low_rank - 0.4, y_pos, f"{method} ({r_val:.2f})",
                    ha="right", va="center", fontsize=9.5, fontweight="bold")
        else:
            ax.plot([r_val, high_rank + 0.3], [y_pos, y_pos], color="grey", linestyle="-", linewidth=0.8)
            ax.text(high_rank + 0.4, y_pos, f"({r_val:.2f}) {method}",
                    ha="left", va="center", fontsize=9.5, fontweight="bold")

    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved Critical Difference Diagram -> {out_path}")


if __name__ == "__main__":
    run_statistical_tests()
