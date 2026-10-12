"""
Statistical and Evaluation Metrics Suite
========================================
Comprehensive evaluation utilities for medical image classification:
- Accuracy, AUC-ROC, Sensitivity, Specificity, G-mean
- Hanley-McNeil analytical standard error and confidence intervals
- McNemar paired test for discordant prediction contingency analysis
- Training history tracker and checkpoint management
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2 as chi2_dist, norm
from typing import Optional

# Optional import for sklearn metrics with pure-numpy fallbacks
try:
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, classification_report, confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> dict:
    """
    Computes accuracy, AUC-ROC, sensitivity (recall), specificity, and G-mean.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if HAS_SKLEARN:
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Pneumonia"],
            output_dict=True,
            zero_division=0
        )
    else:
        acc = float(np.mean(y_true == y_pred))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        cm = np.array([[tn, fp], [fn, tp]])
        report = {}

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    gmean = float(np.sqrt(sensitivity * specificity))

    auc = None
    if y_prob is not None:
        if HAS_SKLEARN:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                auc = None
        else:
            # Rank-sum approximation (Mann-Whitney U) for AUC
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos > 0 and n_neg > 0:
                ranks = np.argsort(np.argsort(y_prob)) + 1
                pos_ranks = np.sum(ranks[y_true == 1])
                auc = float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    return {
        "accuracy": float(acc),
        "auc_roc": auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "gmean": float(gmean),
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
        "classification_report": report
    }


def compute_hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """
    Analytical Standard Error of the AUC estimate via Hanley & McNeil (1982):
        Q1 = AUC / (2 - AUC)
        Q2 = 2 * AUC^2 / (1 + AUC)
        SE = sqrt( (AUC*(1-AUC) + (n_pos-1)*(Q1 - AUC^2) + (n_neg-1)*(Q2 - AUC^2)) / (n_pos*n_neg) )
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * (auc ** 2) / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc ** 2)
        + (n_neg - 1) * (q2 - auc ** 2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(variance, 1e-12)))


def compute_mcnemar_test(pred_a_correct: np.ndarray, pred_b_correct: np.ndarray) -> dict:
    """
    Calculates McNemar paired test statistic with Edwards (1948) continuity correction:
        chi^2 = (|n10 - n01| - 1)^2 / (n10 + n01)
    """
    n11 = int(np.sum(pred_a_correct & pred_b_correct))
    n10 = int(np.sum(pred_a_correct & ~pred_b_correct))  # A right, B wrong
    n01 = int(np.sum(~pred_a_correct & pred_b_correct))  # A wrong, B right
    n00 = int(np.sum(~pred_a_correct & ~pred_b_correct))

    discordant = n10 + n01
    if discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "odds_ratio": 1.0, "n11": n11, "n10": 0, "n01": 0, "n00": n00}

    chi2 = float(((abs(n10 - n01) - 1.0) ** 2) / discordant)
    p_val = float(chi2_dist.sf(chi2, df=1))
    odds_ratio = float(n10 / n01) if n01 > 0 else float("inf")

    return {
        "chi2": chi2,
        "p_value": p_val,
        "odds_ratio": odds_ratio,
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00
    }


class TrainingHistory:
    """Tracks training and validation metrics per epoch."""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epoch_times = []

    def record(self, train_loss, val_loss, train_acc, val_acc, epoch_time=None):
        self.train_loss.append(float(train_loss))
        self.val_loss.append(float(val_loss))
        self.train_acc.append(float(train_acc))
        self.val_acc.append(float(val_acc))
        if epoch_time is not None:
            self.epoch_times.append(float(epoch_time))

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "epoch_times": self.epoch_times
        }


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
