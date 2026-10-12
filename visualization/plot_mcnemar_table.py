"""
McNemar Paired Contingency Table Visualization
==============================================
Visualizes the 2x2 contingency matrix comparing discordant predictions 
between the Optimized Hybrid VQC and the Classical MLP baseline on the 
held-out test set (N=624).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Experimental Data from Test Set Evaluation
n11_both_correct = 290
n10_vqc_right_mlp_wrong = 234
n01_vqc_wrong_mlp_right = 23
n00_both_wrong = 77

chi2_statistic = 171.6
p_value = 3.317e-39
odds_ratio = 10.17

mcnemar_matrix = np.array([
    [n11_both_correct, n10_vqc_right_mlp_wrong],
    [n01_vqc_wrong_mlp_right, n00_both_wrong]
])

_S = {
    "bg": "#0F172A",
    "grid": "#1E293B",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "accent_blue": "#3B82F6"
}

plt.rcParams.update({
    "figure.facecolor": _S["bg"],
    "axes.facecolor": _S["bg"],
    "axes.edgecolor": _S["grid"],
    "axes.labelcolor": _S["text"],
    "axes.titlecolor": _S["text"],
    "xtick.color": _S["muted"],
    "ytick.color": _S["muted"],
    "text.color": _S["text"],
    "font.family": "monospace",
})


def plot_mcnemar_table():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mcnemar_matrix, cmap="Blues", alpha=0.2, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["MLP Correct", "MLP Wrong"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["VQC Correct", "VQC Wrong"], fontsize=11, fontweight="bold")

    ax.set_title(
        f"McNemar Paired Contingency Table (N=624)\n"
        f"Discordant Pairs: n\u2081\u2080={n10_vqc_right_mlp_wrong}, n\u2080\u2081={n01_vqc_wrong_mlp_right}\n"
        f"\u03c7\u00b2 = {chi2_statistic:.1f} | p = {p_value:.3e} | Odds Ratio = {odds_ratio:.2f}x",
        fontsize=11,
        pad=20,
        lineheight=1.5
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{mcnemar_matrix[i, j]}",
                ha="center", va="center",
                color=_S["text"],
                fontsize=16,
                fontweight="bold"
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=_S["grid"], linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "mcnemar_contingency_table.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Figure Saved] {out_path}")


if __name__ == "__main__":
    plot_mcnemar_table()
