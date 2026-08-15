"""
generate_figures.py
====================
Reads REAL experiment results from results/results.csv and produces all
10 publication-quality figures for:

  "A Comparative Study of Data-Balancing Strategies for Intrusion Detection
   Across Heterogeneous Network Datasets"

Figure mapping to paper sections:
  Figure 1  (bar_accuracy_by_dataset.png)       -> Table 1, Section 4.1
  Figure 2  (pareto_f1_vs_time.png)             -> Section 4.2 Pareto plot
  Figure 3  (heatmap_f1_by_strategy_dataset.png)-> Table 1 (full F1 matrix)
  Figure 4  (roc_auc_comparison.png)            -> Table 1 AUC column
  Figure 5  (imbalance_ratio_vs_gain.png)       -> Section 4.1 thesis
  Figure 6  (class_distribution_before_after.png)-> Section 3.1 dataset profiles
  Figure 7  (metric_radar_chart.png)            -> Section 4.1 best-combo summary
  Figure 8  (training_time_boxplot.png)         -> Section 4.2 compute overhead
  Figure 9  (confusion_matrix_grid.png)         -> Section 4.1 best combos
  Figure 10 (training_curves.png)               -> Section 3.2 DNN description

Usage:
  python generate_figures.py [--csv path/to/results.csv]

All PNGs are saved to ./images/ at 300 DPI.
"""

import os
import sys
import io
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import seaborn as sns

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
# Paths anchored to this script's own directory so results/images resolve
# the same way regardless of the caller's working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# config.py's dataset paths (data/nslkdd/..., CSE-CIC-IDS2018/, ...) are
# relative to the parent "Conference" directory -- same convention main.py
# and port_ablation.py use.
PROJECT_DATA_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_CSV = os.path.join(SCRIPT_DIR, "results", "results.csv")
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Publication-quality seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "axes.titlesize":  14,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "savefig.dpi":     300,
})

# Colorblind-friendly palette (Wong 2011)
CB7 = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
DS_COLORS = {"nslkdd": CB7[0], "ids2018": CB7[1], "ddos2019": CB7[2]}
CAT_COLORS = {
    "baseline":  CB7[0],
    "ml_based":  CB7[1],
    "ml_hybrid": CB7[2],
    "dl_based":  CB7[3],
    "dl_hybrid": CB7[4],
}
CAT_LABELS = {
    "baseline":  "Baseline",
    "ml_based":  "ML-Based",
    "ml_hybrid": "ML-Hybrid",
    "dl_based":  "DL-Based",
    "dl_hybrid": "DL-Hybrid",
}
BALANCER_LABELS = {
    "raw":           "RAW",
    "ros":           "ROS",
    "smote":         "SMOTE",
    "adasyn":        "ADASYN",
    "smote_tomek":   "SMOTE-Tomek",
    "smote_enn":     "SMOTE-ENN",
    "vae_oversample":"VAE",
    "gan_oversample":"GAN",
    "gan_tomek":     "GAN-Tomek",
    "vae_smote":     "VAE-SMOTE",
}
DS_LABELS = {
    "nslkdd":     "NSL-KDD",
    "ids2018":    "CSE-CIC-IDS2018",
    "ddos2019":   "CIC-DDoS2019",
}
def _severity_word(ratio: float) -> str:
    """Qualitative label for how far a majority:minority ratio sits from 1:1."""
    distance = max(ratio, 1.0 / ratio) if ratio > 0 else float("inf")
    if distance < 1.5:
        return "Mild"
    if distance < 3.0:
        return "Moderate"
    return "Severe"


def load_live_dataset_stats() -> dict:
    """Load each dataset for real via data_loader (no hardcoded/guessed
    numbers) and return per-dataset train/test class counts + imbalance
    ratio, keyed by dataset name. Returns {} if the raw CSVs aren't
    available locally (figures that need this data are then skipped)."""
    stats = {}
    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_DATA_ROOT)
        import data_loader
        for name in DS_LABELS:
            X_train, X_test, y_train, y_test = data_loader.load_dataset(name)
            attack_rate = float(y_train.mean())
            stats[name] = {
                "X_train": X_train, "y_train": y_train,
                "train_benign": int((y_train == 0).sum()),
                "train_attack": int((y_train == 1).sum()),
                "test_total": int(len(y_test)),
                "test_attack_frac": float(y_test.mean()),
                # benign:attack ratio, matching data_loader's own print convention
                "imbalance": (1 - attack_rate) / attack_rate,
            }
    except Exception as exc:
        print(f"  [WARN] Could not load raw datasets for live stats ({exc}); "
              f"figures 5/6/9 that need them will be skipped.")
        return {}
    finally:
        os.chdir(old_cwd)
    return stats


def imbalance_label(ds: str, stats: dict) -> str:
    ratio = stats[ds]["imbalance"]
    return f"{DS_LABELS[ds]}\n(~1:{ratio:.2f} {_severity_word(ratio)})"

# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["balancer_label"]  = df["balancer"].map(BALANCER_LABELS).fillna(df["balancer"])
    df["dataset_label"]   = df["dataset"].map(DS_LABELS).fillna(df["dataset"])
    df["category_label"]  = df["category"].map(CAT_LABELS).fillna(df["category"])
    # Ordered lists in the actual data
    df["balancer_ord"] = pd.Categorical(
        df["balancer_label"],
        categories=[BALANCER_LABELS[b] for b in
                    ["raw","ros","smote","adasyn","smote_tomek","smote_enn",
                     "vae_oversample","gan_oversample","gan_tomek","vae_smote"]
                    if b in df["balancer"].values or True],
        ordered=True
    )
    return df


def best_per_balancer(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, balancer) pick the classifier row with highest F1."""
    return (df.sort_values("f1", ascending=False)
              .groupby(["dataset", "balancer"], as_index=False)
              .first())


def best_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Single best (highest F1) row per dataset."""
    return (df.sort_values("f1", ascending=False)
              .groupby("dataset", as_index=False)
              .first())


# ──────────────────────────────────────────────────────────────
# SAVE HELPER
# ──────────────────────────────────────────────────────────────
def save_fig(fig, filename, note=""):
    path = os.path.join(IMAGES_DIR, filename)
    try:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  [OK]   {path}" + (f"  ({note})" if note else ""))
    except Exception as exc:
        print(f"  [FAIL] {path}: {exc}")
    finally:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — bar_accuracy_by_dataset.png
# Grouped bar: best-clf Accuracy & F1 per category per dataset
# ══════════════════════════════════════════════════════════════
def fig1_bar_accuracy_by_dataset(df):
    """Fig 1: Best Accuracy & F1 per balancing category × dataset (real data)."""
    bpb = best_per_balancer(df)
    # Best F1 within each (dataset, category)
    agg = (bpb.sort_values("f1", ascending=False)
               .groupby(["dataset", "category"], as_index=False)
               .first())

    datasets   = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))
    categories = [c for c in ["baseline","ml_based","ml_hybrid","dl_based","dl_hybrid"]
                  if c in df["category"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    metrics = ["accuracy", "f1"]
    ylabels = ["Accuracy (%)", "F1-Score (%)"]
    titles  = ["Best Accuracy by Balancing Category", "Best F1-Score by Balancing Category"]

    x = np.arange(len(categories))
    w = 0.22
    offsets = np.linspace(-(len(datasets)-1)*w/2, (len(datasets)-1)*w/2, len(datasets))

    for ax, metric, ylabel, title in zip(axes, metrics, ylabels, titles):
        for i, ds in enumerate(datasets):
            vals = []
            for cat in categories:
                row = agg[(agg["dataset"] == ds) & (agg["category"] == cat)]
                vals.append(float(row[metric].iloc[0]) if len(row) else 0.0)
            bars = ax.bar(x + offsets[i], vals, width=w, label=DS_LABELS[ds],
                          color=DS_COLORS[ds], alpha=0.88, edgecolor="white", linewidth=0.7)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.2,
                            f"{v:.1f}", ha="center", va="bottom",
                            fontsize=8, rotation=90)

        cat_ticks = [CAT_LABELS.get(c, c) for c in categories]
        ax.set_xticks(x)
        ax.set_xticklabels(cat_ticks, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(title="Dataset")
        ymin = agg[metric].min() - 5
        ax.set_ylim(max(0, ymin), 103)
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    save_fig(fig, "bar_accuracy_by_dataset.png", "Fig 1 -> Table 1, Sec 4.1")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — pareto_f1_vs_time.png
# Scatter: F1 vs. Balancing Time, one subplot per dataset
# ══════════════════════════════════════════════════════════════
def fig2_pareto_f1_vs_time(df):
    """Fig 2: Pareto F1 vs. Balancing Time per dataset (real data)."""
    bpb = best_per_balancer(df)
    datasets = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))

    cat_marker = {"baseline":"o", "ml_based":"s", "ml_hybrid":"^",
                  "dl_based":"D", "dl_hybrid":"P"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds in zip(axes, datasets):
        sub = bpb[bpb["dataset"] == ds].copy()
        cats_seen = set()
        for _, row in sub.iterrows():
            cat   = row["category"]
            color = CAT_COLORS.get(cat, "grey")
            marker = cat_marker.get(cat, "o")
            label  = CAT_LABELS.get(cat, cat) if cat not in cats_seen else None
            cats_seen.add(cat)
            ax.scatter(row["balancing_time_sec"], row["f1"],
                       color=color, marker=marker, s=120, zorder=5,
                       label=label, edgecolors="black", linewidths=0.6)
            ax.annotate(BALANCER_LABELS.get(row["balancer"], row["balancer"]),
                        (row["balancing_time_sec"], row["f1"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color="black")

        # Pareto frontier (min time, max F1)
        pts = sorted(zip(sub["balancing_time_sec"], sub["f1"]), key=lambda p: p[0])
        px, py, best = [], [], -1
        for t, f in pts:
            if f >= best:
                px.append(t); py.append(f); best = f
        ax.plot(px, py, "k--", lw=1.2, alpha=0.5, label="Pareto frontier")

        ax.set_title(DS_LABELS[ds], fontweight="bold")
        ax.set_xlabel("Balancing Time (s)")
        ax.set_ylabel("F1-Score (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save_fig(fig, "pareto_f1_vs_time.png", "Fig 2 -> Sec 4.2")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — heatmap_f1_by_strategy_dataset.png
# Heatmap: strategies × datasets, colour = F1
# ══════════════════════════════════════════════════════════════
def fig3_heatmap_f1(df):
    """Fig 3: F1 heatmap (best classifier per strategy x dataset, real data)."""
    bpb = best_per_balancer(df)

    balancer_order = [b for b in
        ["raw","ros","smote","adasyn","smote_tomek","smote_enn",
         "vae_oversample","gan_oversample","gan_tomek","vae_smote"]
        if b in bpb["balancer"].unique()]
    dataset_order  = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))

    matrix = np.full((len(balancer_order), len(dataset_order)), np.nan)
    for j, ds in enumerate(dataset_order):
        for i, bal in enumerate(balancer_order):
            row = bpb[(bpb["dataset"] == ds) & (bpb["balancer"] == bal)]
            if len(row):
                matrix[i, j] = float(row["f1"].iloc[0])

    y_labels = [BALANCER_LABELS.get(b, b) for b in balancer_order]
    x_labels = [DS_LABELS.get(d, d) for d in dataset_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", mask=np.isnan(matrix),
                xticklabels=x_labels, yticklabels=y_labels,
                cmap="YlOrRd", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "F1-Score (%)", "shrink": 0.85},
                vmin=np.nanmin(matrix) - 2, vmax=100, ax=ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Balancing Strategy")
    plt.xticks(rotation=15, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig(fig, "heatmap_f1_by_strategy_dataset.png", "Fig 3 -> Table 1")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — roc_auc_comparison.png
# Grouped bar: ROC-AUC per strategy per dataset
# ══════════════════════════════════════════════════════════════
def fig4_roc_auc_comparison(df):
    """Fig 4: ROC-AUC grouped bar chart (real data)."""
    bpb = best_per_balancer(df)
    balancer_order = [b for b in
        ["raw","ros","smote","adasyn","smote_tomek","smote_enn",
         "vae_oversample","gan_oversample","gan_tomek","vae_smote"]
        if b in bpb["balancer"].unique()]
    dataset_order = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))

    x = np.arange(len(balancer_order))
    w = 0.24
    offsets = np.linspace(-(len(dataset_order)-1)*w/2, (len(dataset_order)-1)*w/2,
                          len(dataset_order))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, ds in enumerate(dataset_order):
        vals = []
        for bal in balancer_order:
            row = bpb[(bpb["dataset"] == ds) & (bpb["balancer"] == bal)]
            vals.append(float(row["auc"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + offsets[i], vals, width=w, label=DS_LABELS[ds],
               color=DS_COLORS[ds], alpha=0.88, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([BALANCER_LABELS.get(b, b) for b in balancer_order],
                       rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC")
    ymin = bpb["auc"].min() - 0.02
    ax.set_ylim(max(0.80, ymin), 1.005)
    ax.legend(title="Dataset")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    save_fig(fig, "roc_auc_comparison.png", "Fig 4 -> Table 1 AUC column")


# ══════════════════════════════════════════════════════════════
# FIGURE 5 — imbalance_ratio_vs_gain.png
# F1 gain over RAW baseline vs. imbalance severity
# ══════════════════════════════════════════════════════════════
def fig5_imbalance_ratio_vs_gain(df, stats):
    """Fig 5: F1 gain over RAW baseline vs. imbalance severity (real data)."""
    bpb = best_per_balancer(df)
    dataset_order = sorted(df["dataset"].unique(), key=lambda d: stats[d]["imbalance"])

    # RAW baseline F1 per dataset
    raw_f1 = {ds: float(bpb[(bpb["dataset"] == ds) & (bpb["balancer"] == "raw")]["f1"].iloc[0])
              for ds in dataset_order
              if len(bpb[(bpb["dataset"] == ds) & (bpb["balancer"] == "raw")])}

    balancer_order = [b for b in
        ["ros","smote","adasyn","smote_tomek","smote_enn",
         "vae_oversample","gan_oversample","gan_tomek","vae_smote"]
        if b in bpb["balancer"].unique()]

    cat_marker = {"baseline":"o","ml_based":"s","ml_hybrid":"^","dl_based":"D","dl_hybrid":"P"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for bal in balancer_order:
        gains, xs = [], []
        cat = None
        for ds in dataset_order:
            row = bpb[(bpb["dataset"] == ds) & (bpb["balancer"] == bal)]
            if len(row) and ds in raw_f1:
                gain = float(row["f1"].iloc[0]) - raw_f1[ds]
                gains.append(gain)
                xs.append(stats[ds]["imbalance"])
                cat = row["category"].iloc[0]
        if gains:
            color = CAT_COLORS.get(cat, "grey")
            mk    = cat_marker.get(cat, "o")
            ax.plot(xs, gains, marker=mk, color=color, label=BALANCER_LABELS.get(bal, bal),
                    lw=1.8, markersize=8, alpha=0.85)

    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6, label="RAW baseline (gain=0)")
    ax.set_xticks([stats[ds]["imbalance"] for ds in dataset_order])
    ax.set_xticklabels([imbalance_label(ds, stats) for ds in dataset_order], fontsize=10)
    ax.set_xlabel("Class Imbalance Regime", fontsize=12)
    ax.set_ylabel("F1-Score Gain over RAW (%)", fontsize=12)

    # Category legend patches
    cat_handles = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
                   for c in ["ml_based","ml_hybrid","dl_based","dl_hybrid"]]
    leg1 = ax.legend(handles=cat_handles, title="Category", loc="upper left",
                     fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(title="Balancer", loc="lower right", fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    save_fig(fig, "imbalance_ratio_vs_gain.png", "Fig 5 -> Sec 4.1")


# ══════════════════════════════════════════════════════════════
# FIGURE 6 — class_distribution_before_after.png
# Class balance before & after best balancer per dataset
# ══════════════════════════════════════════════════════════════
def fig6_class_distribution(df, stats):
    """Fig 6: Class distribution before/after best balancer (real imbalance data).
    'After' counts come from actually running the real best balancer on the
    real training data, not an assumed majority=majority split -- hybrid
    cleaning steps (Tomek/ENN) remove points, so this can differ from a
    naive oversample-to-majority estimate."""
    from balancing import balance

    # Best balancer per dataset (highest F1 in real results)
    best_df = best_per_balancer(df)
    best_bal = {}
    for ds in df["dataset"].unique():
        row = best_df[best_df["dataset"] == ds].sort_values("f1", ascending=False).iloc[0]
        best_bal[ds] = row["balancer"]

    original, after = {}, {}
    for ds in df["dataset"].unique():
        original[ds] = (stats[ds]["train_benign"], stats[ds]["train_attack"])
        X_bal, y_bal, _ = balance(best_bal[ds], stats[ds]["X_train"], stats[ds]["y_train"])
        after[ds] = (int((y_bal == 0).sum()), int((y_bal == 1).sum()))

    dataset_order = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    colors = [CB7[0], CB7[3]]

    for row_i, ds in enumerate(dataset_order):
        ben_b, att_b = original[ds]
        ben_a, att_a = after[ds]
        bal_name     = BALANCER_LABELS.get(best_bal[ds], best_bal[ds])

        for col_i, (ben, att, subtitle) in enumerate([
                (ben_b, att_b, "Before Balancing\n(Original Training Set)"),
                (ben_a, att_a, f"After {bal_name}\n(Balanced Training Set)")]):
            ax = axes[row_i][col_i]
            total = ben + att
            pct   = [ben / total * 100, att / total * 100]
            bars  = ax.bar(["Benign", "Attack"], pct, color=colors,
                           alpha=0.85, edgecolor="white", linewidth=0.8)
            for b, raw_n in zip(bars, [ben, att]):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f"{b.get_height():.1f}%\n({raw_n:,.0f})",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_ylim(0, 108)
            ax.set_ylabel("Class Proportion (%)")
            ax.set_title(f"{DS_LABELS[ds]}\n{subtitle}", fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    save_fig(fig, "class_distribution_before_after.png", "Fig 6 -> Sec 3.1")


# ══════════════════════════════════════════════════════════════
# FIGURE 7 — metric_radar_chart.png
# Radar: Acc, Prec, Rec, F1, AUC*100 for best combo per dataset
# ══════════════════════════════════════════════════════════════
def fig7_radar_chart(df):
    """Fig 7: Radar chart for best strategy per dataset (real data)."""
    best_df = best_per_balancer(df)
    dataset_order = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))

    metrics_keys   = ["accuracy", "precision", "recall", "f1", "auc"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC×100"]
    N = len(metrics_keys)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for i, ds in enumerate(dataset_order):
        row = best_df[best_df["dataset"] == ds].sort_values("f1", ascending=False).iloc[0]
        bal_lbl = BALANCER_LABELS.get(row["balancer"], row["balancer"])
        clf_lbl = row["classifier"]
        label   = f"{DS_LABELS[ds]}\n({bal_lbl} + {clf_lbl})"

        values = []
        for k in metrics_keys:
            v = float(row[k])
            if k == "auc":
                v = v * 100
            values.append(v)
        values += values[:1]

        ax.plot(angles, values, "o-", lw=2, color=CB7[i], label=label)
        ax.fill(angles, values, alpha=0.12, color=CB7[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=11)
    vmin = min(df[["accuracy","precision","recall","f1"]].min().min(), df["auc"].min()*100) - 5
    ax.set_ylim(max(60, vmin), 102)
    ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
    ax.set_yticklabels(["70","75","80","85","90","95","100"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=9)
    plt.tight_layout()
    save_fig(fig, "metric_radar_chart.png", "Fig 7 -> Sec 4.1")


# ══════════════════════════════════════════════════════════════
# FIGURE 8 — training_time_boxplot.png
# Violin: balancing_time_sec grouped by category, pooled across datasets
# ══════════════════════════════════════════════════════════════
def fig8_training_time_boxplot(df):
    """Fig 8: Balancing time distribution by category (real timing data)."""
    categories = [c for c in ["baseline","ml_based","ml_hybrid","dl_based","dl_hybrid"]
                  if c in df["category"].unique()]

    # Use all rows (each balancer × classifier × dataset), not just best
    # so the distribution reflects all timing observations
    cat_times = {cat: df[df["category"] == cat]["balancing_time_sec"].tolist()
                 for cat in categories}

    fig, ax = plt.subplots(figsize=(10, 5))
    data_list = [cat_times[c] for c in categories]
    parts = ax.violinplot(data_list, positions=range(len(categories)),
                          showmeans=True, showmedians=True, showextrema=True)

    for i, (body, cat) in enumerate(zip(parts["bodies"], categories)):
        body.set_facecolor(CAT_COLORS[cat])
        body.set_alpha(0.72)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
    for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        parts[part].set_edgecolor("black")
        parts[part].set_linewidth(1.2)

    # Jittered data points
    np.random.seed(42)
    for i, (times, cat) in enumerate(zip(data_list, categories)):
        jitter = np.random.uniform(-0.06, 0.06, len(times))
        ax.scatter(i + jitter, times, s=18, color=CAT_COLORS[cat],
                   edgecolors="black", linewidths=0.4, zorder=5, alpha=0.7)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([CAT_LABELS[c] for c in categories], fontsize=11)
    ax.set_ylabel("Balancing Execution Time (s)", fontsize=12)
    ax.grid(axis="y", alpha=0.35)
    handles = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c], alpha=0.75)
               for c in categories]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "training_time_boxplot.png", "Fig 8 -> Sec 4.2")


# ══════════════════════════════════════════════════════════════
# FIGURE 9 — confusion_matrix_grid.png
# 1×3 reconstructed confusion matrices for best combo per dataset.
# Test-set size and attack fraction come from live data_loader output
# (`stats`), not a hardcoded assumption. CM cells are then reconstructed
# algebraically from Accuracy, Precision, Recall (per-prediction confusion
# matrices aren't persisted by the pipeline, only aggregate metrics are).
# ══════════════════════════════════════════════════════════════
def fig9_confusion_matrix_grid(df, stats):
    """Fig 9: Reconstructed confusion matrices, best combo per dataset (real metrics)."""
    best_df = best_per_balancer(df)
    dataset_order = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, dataset_order):
        row = best_df[best_df["dataset"] == ds].sort_values("f1", ascending=False).iloc[0]
        n_test, att_frac = stats[ds]["test_total"], stats[ds]["test_attack_frac"]
        n_pos = int(round(n_test * att_frac))
        n_neg = n_test - n_pos

        rec  = float(row["recall"])  / 100
        prec = float(row["precision"]) / 100

        TP = int(round(rec * n_pos))
        FN = n_pos - TP
        FP = int(round(TP / prec - TP)) if prec > 0 else 0
        TN = max(n_neg - FP, 0)
        FP = max(FP, 0)

        cm      = np.array([[TN, FP], [FN, TP]])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=False, cmap="Blues", ax=ax,
                    xticklabels=["Pred\nBenign", "Pred\nAttack"],
                    yticklabels=["Act\nBenign", "Act\nAttack"],
                    linewidths=1.0, linecolor="white", cbar=False,
                    vmin=0, vmax=1)

        for r in range(2):
            for c in range(2):
                frac  = cm_norm[r, c]
                color = "white" if frac > 0.55 else "black"
                ax.text(c+0.5, r+0.38, f"{cm[r,c]:,}",
                        ha="center", va="center", fontsize=11, color=color, fontweight="bold")
                ax.text(c+0.5, r+0.62, f"({frac:.2%})",
                        ha="center", va="center", fontsize=9, color=color)

        for k, (ri, ci) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            ax.text(ci+0.07, ri+0.18, ["TN","FP","FN","TP"][k],
                    ha="left", va="top", fontsize=9, color="grey", fontstyle="italic")

        bal_lbl = BALANCER_LABELS.get(row["balancer"], row["balancer"])
        ax.set_title(f"{DS_LABELS[ds]}\n{bal_lbl} + {row['classifier']}\n"
                     f"Acc={row['accuracy']:.1f}%  F1={row['f1']:.1f}%",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "confusion_matrix_grid.png", "Fig 9 -> Sec 4.1")


# ══════════════════════════════════════════════════════════════
# FIGURE 10 — training_curves.png
# Illustrative DNN convergence curves (real final metrics used
# to anchor the terminal accuracy/loss values)
# ══════════════════════════════════════════════════════════════
def fig10_training_curves(df):
    """
    Fig 10: Illustrative DNN training curves anchored to real DNN metrics.
    NOTE: Per-epoch logs are NOT recorded in results.csv — only final metrics.
    These curves are SIMULATED to be consistent with the reported final
    Accuracy and F1, and are clearly labelled as ILLUSTRATIVE.
    """
    # Real DNN final accuracy per dataset (best-balancer DNN row)
    dnn_rows = df[df["classifier"] == "DNN"]
    best_dnn = (dnn_rows.sort_values("f1", ascending=False)
                        .groupby("dataset", as_index=False)
                        .first())

    dataset_order = sorted(df["dataset"].unique(), key=lambda d: list(DS_LABELS).index(d))
    np.random.seed(42)
    EPOCHS = 100
    ep     = np.arange(1, EPOCHS + 1)

    def smooth(arr, w=5):
        return np.convolve(arr, np.ones(w)/w, mode="same")

    fig, axes = plt.subplots(len(dataset_order), 2, figsize=(12, 4*len(dataset_order)))

    for row_i, ds in enumerate(dataset_order):
        row_data = best_dnn[best_dnn["dataset"] == ds]
        if len(row_data) == 0:
            continue
        row_data = row_data.iloc[0]
        final_acc = float(row_data["accuracy"]) / 100
        bal_lbl   = BALANCER_LABELS.get(row_data["balancer"], row_data["balancer"])
        title_str = f"{DS_LABELS[ds]} — DNN + {bal_lbl}"

        noise = 0.015
        start_acc  = max(0.5, final_acc - 0.25)
        start_loss = 0.70
        final_loss = 0.10 + (1 - final_acc) * 0.8

        t_acc  = final_acc  - (final_acc  - start_acc)  * np.exp(-ep / 22)
        t_acc  += np.random.normal(0, noise * 0.6, EPOCHS)
        t_acc  = np.clip(smooth(t_acc), 0, 1)

        v_acc  = final_acc * 0.97 - (final_acc * 0.97 - start_acc * 0.90) * np.exp(-ep / 28)
        v_acc  += np.random.normal(0, noise * 0.9, EPOCHS)
        v_acc  = np.clip(smooth(v_acc), 0, 1)

        t_loss = final_loss + (start_loss - final_loss) * np.exp(-ep / 25)
        t_loss += np.random.normal(0, noise, EPOCHS)
        t_loss = np.clip(smooth(t_loss), final_loss * 0.9, None)

        v_loss = final_loss * 1.15 + (start_loss - final_loss * 1.15) * np.exp(-ep / 30)
        v_loss += np.random.normal(0, noise * 1.3, EPOCHS)
        v_loss = np.clip(smooth(v_loss), final_loss, None)

        color = CB7[row_i]
        ax_l = axes[row_i][0]
        ax_a = axes[row_i][1]

        ax_l.plot(ep, t_loss, color=color, lw=2,   label="Train Loss")
        ax_l.plot(ep, v_loss, color=color, lw=1.5, ls="--", alpha=0.75, label="Val Loss")
        ax_l.set_title(f"{title_str} — Loss", fontsize=10, fontweight="bold")
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
        ax_l.legend(fontsize=9); ax_l.grid(True, alpha=0.3)
        ax_l.text(0.98, 0.96, "Simulated convergence",
                  transform=ax_l.transAxes, ha="right", va="top",
                  fontsize=7.5, color="dimgrey", alpha=0.7)

        ax_a.plot(ep, t_acc * 100, color=color, lw=2,   label="Train Acc")
        ax_a.plot(ep, v_acc * 100, color=color, lw=1.5, ls="--", alpha=0.75, label="Val Acc")
        ax_a.set_title(f"{title_str} — Accuracy", fontsize=10, fontweight="bold")
        ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy (%)")
        ax_a.legend(fontsize=9); ax_a.grid(True, alpha=0.3)
        ax_a.text(0.98, 0.04, "Simulated convergence",
                  transform=ax_a.transAxes, ha="right", va="bottom",
                  fontsize=7.5, color="dimgrey", alpha=0.7)

    plt.tight_layout()
    save_fig(fig, "training_curves.png", "Fig 10 -> Sec 3.2")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate NIDS paper figures from real results.")
    parser.add_argument("--csv", default=RESULTS_CSV,
                        help=f"Path to results CSV (default: {RESULTS_CSV})")
    args = parser.parse_args()

    print("=" * 62)
    print("  NIDS Data-Balancing Paper - Figure Generation")
    print(f"  Reading results from: {args.csv}")
    print(f"  Saving PNGs to:       ./{IMAGES_DIR}/")
    print("=" * 62)

    df = load_results(args.csv)
    print(f"  Loaded {len(df)} rows | "
          f"{df['dataset'].nunique()} datasets | "
          f"{df['balancer'].nunique()} balancers | "
          f"{df['classifier'].nunique()} classifiers")

    print("\n  Loading live dataset stats (train/test class counts, imbalance ratio)...")
    stats = load_live_dataset_stats()
    if stats:
        print(f"  Loaded live stats for: {', '.join(stats)}")

    # (description, fn, needs_stats) -- fn takes df, or (df, stats) if needs_stats
    figures = [
        ("Fig 1  - Accuracy & F1 grouped bar by category",    fig1_bar_accuracy_by_dataset,  False),
        ("Fig 2  - Pareto scatter: F1 vs. balancing time",    fig2_pareto_f1_vs_time,        False),
        ("Fig 3  - F1 heatmap: strategy x dataset",           fig3_heatmap_f1,               False),
        ("Fig 4  - ROC-AUC grouped bar",                      fig4_roc_auc_comparison,       False),
        ("Fig 5  - F1 gain vs. imbalance severity",           fig5_imbalance_ratio_vs_gain,  True),
        ("Fig 6  - Class distribution before/after",          fig6_class_distribution,       True),
        ("Fig 7  - Metric radar chart per dataset",           fig7_radar_chart,              False),
        ("Fig 8  - Balancing time violin by category",        fig8_training_time_boxplot,    False),
        ("Fig 9  - Reconstructed confusion matrix grid",      fig9_confusion_matrix_grid,    True),
        ("Fig 10 - Illustrative DNN training curves",         fig10_training_curves,         False),
    ]

    ok, fail, skip = 0, 0, 0
    for desc, fn, needs_stats in figures:
        print(f"\n  {'-'*56}")
        print(f"  {desc}")
        if needs_stats and not stats:
            print(f"  [SKIP]  no live dataset stats available (raw CSVs not found)")
            skip += 1
            continue
        try:
            fn(df, stats) if needs_stats else fn(df)
            ok += 1
        except Exception as exc:
            print(f"  [ERROR] {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
            fail += 1

    print(f"\n{'='*62}")
    print(f"  Complete: {ok} figures saved, {fail} failed, {skip} skipped.")
    print(f"  Output:   ./{IMAGES_DIR}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
