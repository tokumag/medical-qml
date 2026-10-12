"""
Publication-Quality Plotting and Styling Utilities
==================================================
Clean academic styling with standardized typography, palettes, and vector export.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from typing import Optional, List

# Unified Academic Color Palette
C_NAVY = "#002060"    # Primary Academic Core / Normal Class
C_TEAL = "#008080"    # Quantum Dimension
C_SLATE = "#5A5A5A"   # Secondary / Baseline Accent
C_RED = "#CC0000"     # Pneumonia Class / Alert
C_LIGHT = "#F4F6F9"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "savefig.bbox_inches": "tight",
    "savefig.dpi": 300
})


def plot_confusion_matrix(matrix_data: np.ndarray, title: str, save_path: str):
    """Generates an academic white-background confusion matrix."""
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    tn, fp = int(matrix_data[0, 0]), int(matrix_data[0, 1])
    fn, tp = int(matrix_data[1, 0]), int(matrix_data[1, 1])

    labels = np.array([
        [f"TN\n\n{tn}", f"FP\n\n{fp}"],
        [f"FN\n\n{fn}", f"TP\n\n{tp}"]
    ])

    sns.heatmap(
        matrix_data,
        annot=labels,
        fmt="",
        cmap=sns.light_palette(C_NAVY, as_cmap=True),
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
        ax=ax,
        annot_kws={"fontsize": 11, "weight": "bold"}
    )

    ax.set_xlabel("Predicted Condition Status", labelpad=8)
    ax.set_ylabel("True Confirmed Status", labelpad=8)
    ax.set_title(title, pad=12, weight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   save_path: str, label: str = "Model",
                   auc: Optional[float] = None):
    """Generates an annotated ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_str = f" (AUC = {auc:.4f})" if auc is not None else ""

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=C_NAVY, linewidth=2.0, label=f"{label}{auc_str}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Chance Level")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic", fontweight="bold")
    ax.legend(loc="lower right")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
