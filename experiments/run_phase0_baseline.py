"""
Phase 0: Classical Baseline and PCA Dimensionality Reduction
============================================================
Establishes the classical benchmark ceilings for PneumoniaMNIST:
  1. Classical CNN on full 28x28 grayscale images (784-dimensional Euclidean space)
  2. Classical MLP on 8 principal component analysis (PCA) dimensions
  3. Fitting and serialization of PCAReducer (saved to checkpoints/pca_reducer.pkl)

Outputs saved to results/phase0/ and checkpoints/phase0/.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add repository root to system path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_utils import get_medmnist_loaders, get_flat_arrays, PCAReducer, build_tensor_dataset
from src.classical_models import ClassicalCNN, ClassicalMLP
from src.metrics import TrainingHistory, evaluate_torch_model, save_json, compute_metrics
from src.plot_utils import plot_confusion_matrix, plot_roc_curve


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x_b, y_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.squeeze().long().to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_b.size(0)
        correct += (out.argmax(1) == y_b).sum().item()
        n += x_b.size(0)
    return total_loss / n, correct / n


def run_phase0(quick_mode: bool = False):
    print("=" * 70)
    print("  Phase 0: Classical Baselines & PCA Subspace Extraction")
    print("=" * 70)

    with open(os.path.join(REPO_ROOT, "configs", "config.json"), "r") as f:
        config = json.load(f)

    res_dir = os.path.join(REPO_ROOT, "results", "phase0")
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "phase0")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed = config["data"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Random Seed: {seed}")

    pca_dim = config["phase0"]["pca_components"]
    cnn_epochs = 2 if quick_mode else config["phase0"]["cnn_epochs"]
    mlp_epochs = 3 if quick_mode else config["phase0"]["mlp_epochs"]
    batch_size = config["data"]["batch_size"]

    # ── Step 0-A: PCA Feature Extraction ─────────────────────────────────────────
    print(f"\n[Step 0-A] Extracting {pca_dim} Principal Components from PneumoniaMNIST...")
    data_root = os.path.join(REPO_ROOT, config["paths"]["data_dir"])
    flat_data = get_flat_arrays(data_root=data_root, image_size=28)

    reducer = PCAReducer(n_components=pca_dim, random_state=seed)
    x_tr_pca = reducer.fit_transform(flat_data["X_train"])
    x_va_pca = reducer.transform(flat_data["X_val"])
    x_te_pca = reducer.transform(flat_data["X_test"])

    evr_info = reducer.explained_variance_summary()
    print(f"  Variance explained by {pca_dim} PCs: {evr_info['total_explained'] * 100:.2f}%")
    print(f"  Per-component ratios: {np.round(evr_info['per_component'], 4).tolist()}")

    pca_ckpt_path = os.path.join(ckpt_dir, "pca_reducer.pkl")
    reducer.save(pca_ckpt_path)
    print(f"  Saved fitted PCAReducer to {pca_ckpt_path}")

    # ── Step 0-B: Classical CNN on 784-px images ─────────────────────────────────
    print(f"\n[Step 0-B] Training Classical CNN on Full Pixels ({cnn_epochs} epochs)...")
    loaders = get_medmnist_loaders(batch_size=batch_size, data_root=data_root)
    cnn_model = ClassicalCNN(dropout_p=config["phase0"]["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=config["phase0"]["cnn_lr"])

    best_cnn_acc = 0.0
    for ep in range(1, cnn_epochs + 1):
        tr_loss, tr_acc = train_epoch(cnn_model, loaders["train"], optimizer_cnn, criterion, device)
        val_eval = evaluate_torch_model(cnn_model, loaders["val"], device, criterion)
        print(f"  Epoch {ep:2d}/{cnn_epochs} | Train Loss: {tr_loss:.4f} | Val Acc: {val_eval['accuracy']:.4f}")
        if val_eval["accuracy"] > best_cnn_acc:
            best_cnn_acc = val_eval["accuracy"]
            torch.save(cnn_model.state_dict(), os.path.join(ckpt_dir, "best_cnn.pt"))

    cnn_test_res = evaluate_torch_model(cnn_model, loaders["test"], device, criterion)
    print("\n  CNN Test Set Evaluation:")
    print(f"    Accuracy   : {cnn_test_res['accuracy']:.4f}")
    print(f"    AUC-ROC    : {cnn_test_res['auc_roc']:.4f}")
    print(f"    Sensitivity: {cnn_test_res['sensitivity']:.4f}")
    print(f"    Specificity: {cnn_test_res['specificity']:.4f}")

    plot_confusion_matrix(
        np.array(cnn_test_res["confusion_matrix"]),
        "Phase 0: CNN Confusion Matrix",
        os.path.join(res_dir, "cnn_confusion_matrix.png")
    )

    # ── Step 0-C: Classical MLP on PCA-8 Features ────────────────────────────────
    print(f"\n[Step 0-C] Training Classical MLP on PCA-8 Features ({mlp_epochs} epochs)...")
    tr_loader_mlp = build_tensor_dataset(x_tr_pca, flat_data["y_train"], batch_size)
    va_loader_mlp = build_tensor_dataset(x_va_pca, flat_data["y_val"], batch_size, shuffle=False)
    te_loader_mlp = build_tensor_dataset(x_te_pca, flat_data["y_test"], batch_size, shuffle=False)

    mlp_model = ClassicalMLP(input_dim=pca_dim, hidden=config["phase0"]["mlp_hidden"]).to(device)
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)

    best_mlp_acc = 0.0
    for ep in range(1, mlp_epochs + 1):
        tr_loss, tr_acc = train_epoch(mlp_model, tr_loader_mlp, optimizer_mlp, criterion, device)
        val_eval = evaluate_torch_model(mlp_model, va_loader_mlp, device, criterion)
        print(f"  Epoch {ep:2d}/{mlp_epochs} | Train Loss: {tr_loss:.4f} | Val Acc: {val_eval['accuracy']:.4f}")
        if val_eval["accuracy"] > best_mlp_acc:
            best_mlp_acc = val_eval["accuracy"]
            torch.save(mlp_model.state_dict(), os.path.join(ckpt_dir, "best_mlp.pt"))

    mlp_test_res = evaluate_torch_model(mlp_model, te_loader_mlp, device, criterion)
    print("\n  MLP Test Set Evaluation:")
    print(f"    Accuracy   : {mlp_test_res['accuracy']:.4f}")
    print(f"    AUC-ROC    : {mlp_test_res['auc_roc']:.4f}")
    print(f"    Sensitivity: {mlp_test_res['sensitivity']:.4f}")
    print(f"    Specificity: {mlp_test_res['specificity']:.4f}")

    plot_confusion_matrix(
        np.array(mlp_test_res["confusion_matrix"]),
        "Phase 0: MLP Confusion Matrix",
        os.path.join(res_dir, "mlp_confusion_matrix.png")
    )

    # Summary record
    summary = {
        "evr": evr_info,
        "cnn": {k: cnn_test_res[k] for k in ["accuracy", "auc_roc", "sensitivity", "specificity", "confusion_matrix"]},
        "mlp": {k: mlp_test_res[k] for k in ["accuracy", "auc_roc", "sensitivity", "specificity", "confusion_matrix"]}
    }
    save_json(summary, os.path.join(res_dir, "phase0_summary.json"))
    print(f"\n[Completed] Phase 0 summary saved to {os.path.join(res_dir, 'phase0_summary.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 0 classical baselines.")
    parser.add_argument("--quick", action="store_true", help="Run in fast smoke-test mode.")
    args = parser.parse_args()
    run_phase0(quick_mode=args.quick)
