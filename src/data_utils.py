"""
Data Preprocessing and Dimensionality Reduction Utilities
=========================================================
Handles MedMNIST loading, normalization, and PCA dimensionality reduction.
Features are projected from 784-dimensional Euclidean image space into an 
8-dimensional subspace before quantum state preparation.
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import medmnist
from medmnist import PneumoniaMNIST
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_medmnist_loaders(batch_size: int = 32,
                         image_size: int = 28,
                         download: bool = True,
                         data_root: str = "./data") -> dict:
    """
    Load train, validation, and test DataLoaders for PneumoniaMNIST.
    Labels: 0 = Normal, 1 = Pneumonia.
    """
    os.makedirs(data_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Map to [-1, 1]
    ])

    train_ds = PneumoniaMNIST(split="train", transform=transform, download=download, root=data_root)
    val_ds = PneumoniaMNIST(split="val", transform=transform, download=download, root=data_root)
    test_ds = PneumoniaMNIST(split="test", transform=transform, download=download, root=data_root)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
        "info": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "classes": ["Normal", "Pneumonia"]
        }
    }
    return loaders


def get_flat_arrays(data_root: str = "./data",
                    image_size: int = 28,
                    max_train: int = None,
                    max_test: int = None) -> dict:
    """
    Extract normalized flat numpy arrays for classical and quantum pipelines.
    Pixels normalized to [-1, 1].
    """
    os.makedirs(data_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def _extract(ds, max_n):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.view(x.size(0), -1).numpy())
            ys.append(y.squeeze().numpy())
        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        if max_n is not None:
            x_all, y_all = x_all[:max_n], y_all[:max_n]
        return x_all, y_all

    train_ds = PneumoniaMNIST(split="train", transform=transform, download=True, root=data_root)
    val_ds = PneumoniaMNIST(split="val", transform=transform, download=True, root=data_root)
    test_ds = PneumoniaMNIST(split="test", transform=transform, download=True, root=data_root)

    x_tr, y_tr = _extract(train_ds, max_train)
    x_va, y_va = _extract(val_ds, None)
    x_te, y_te = _extract(test_ds, max_test)

    return {
        "X_train": x_tr, "y_train": y_tr,
        "X_val": x_va, "y_val": y_va,
        "X_test": x_te, "y_test": y_te
    }


class PCAReducer:
    """
    StandardScaler + Principal Component Analysis wrapper.
    Projects 784-dimensional pixel arrays into the primary n_components subspace.
    """

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x_s = self.scaler.fit_transform(x)
        x_r = self.pca.fit_transform(x_s)
        self._fitted = True
        return x_r

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self._fitted, "PCAReducer must be fitted before transforming."
        x_s = self.scaler.transform(x)
        x_r = self.pca.transform(x_s)
        return x_r

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "n_components": self.n_components,
                "_fitted": self._fitted
            }, f)

    @classmethod
    def load(cls, path: str) -> "PCAReducer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(n_components=d["n_components"])
        obj.scaler = d["scaler"]
        obj.pca = d["pca"]
        obj._fitted = d["_fitted"]
        return obj

    def explained_variance_summary(self) -> dict:
        assert self._fitted, "PCAReducer must be fitted first."
        ev = self.pca.explained_variance_ratio_
        return {
            "per_component": ev.tolist(),
            "cumulative": np.cumsum(ev).tolist(),
            "total_explained": float(ev.sum())
        }


def build_tensor_dataset(x: np.ndarray,
                         y: np.ndarray,
                         batch_size: int = 16,
                         shuffle: bool = True) -> DataLoader:
    """Wraps numpy feature arrays and labels into a PyTorch DataLoader."""
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
