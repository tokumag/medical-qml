"""
Quantum Angle Encoding and ZZ Feature Map Formulation
=====================================================
Encodes 8-dimensional PCA feature vectors into a 256-dimensional complex Hilbert space 
(H = C^(2^8)) using an EVR-weighted Ising ZZ Hamiltonian:

    H_ZZ = sum_j (x_j Z_j) + sum_{j < k} (2 w_j w_k x_j x_k Z_j Z_k)
    where w_j = sqrt(EVR_j / EVR_max)

Angle scale is strictly bounded to [0, pi/4] to prevent phase over-rotation and 
kernel matrix collapse.
"""

import os
import pickle
import numpy as np
import pennylane as qml

# ── Immutable Experimental Physical Constants ─────────────────────────────────
ANGLE_SCALE = np.pi / 4  # [0, pi/4] radians prevents state orthogonality collapse
N_ZZ_LAYERS = 2
N_QUBITS = 8

# PCA Explained Variance Ratios (Fitted on Training Set)
EVR_RAW = np.array([
    0.41413912, 0.10647308, 0.05541586, 0.04525760,
    0.03899284, 0.03691639, 0.02805536, 0.02088385
], dtype=np.float64)

# Pairwise coupling weights: w_j = sqrt(EVR_j / EVR_max)
EVR_WEIGHTS = np.sqrt(EVR_RAW / EVR_RAW.max()).astype(np.float64)

# Multi-qubit observable Hamiltonian weights: w_obs_j = EVR_j / sum(EVR)
OBS_WEIGHTS = (EVR_RAW / EVR_RAW.sum()).astype(np.float64)


class QuantumAngleEncoder:
    """
    Normalizes PCA features into quantum rotation angles [0, pi/4].
    Bounds are strictly fitted on the TRAINING split to avoid data leakage.

    x_enc = (clip(x, p_lo, p_hi) - p_lo) / (p_hi - p_lo) * (pi / 4)
    """

    def __init__(self, scale: float = ANGLE_SCALE, clip_percentiles: tuple = (1.0, 99.0)):
        self.scale = scale
        self.clip_percentiles = clip_percentiles
        self.p_lo_ = None
        self.p_hi_ = None
        self.denom_ = None
        self.evr_weights = EVR_WEIGHTS.copy()
        self._fitted = False

    def fit(self, x: np.ndarray):
        """Fit quantile bounds exclusively on training features."""
        p_lo = np.percentile(x, self.clip_percentiles[0], axis=0)
        p_hi = np.percentile(x, self.clip_percentiles[1], axis=0)
        denom = p_hi - p_lo
        denom[denom == 0.0] = 1.0  # Guard against constant features

        self.p_lo_ = p_lo
        self.p_hi_ = p_hi
        self.denom_ = denom
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply frozen transformation bounds."""
        assert self._fitted, "QuantumAngleEncoder must be fitted prior to transform."
        x_clipped = np.clip(x, self.p_lo_, self.p_hi_)
        x_norm = (x_clipped - self.p_lo_) / self.denom_
        return (x_norm * self.scale).astype(np.float64)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"encoder": self}, f)

    @classmethod
    def load(cls, path: str) -> "QuantumAngleEncoder":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["encoder"]


def apply_zz_feature_map(x: np.ndarray, n_qubits: int = N_QUBITS,
                         n_layers: int = N_ZZ_LAYERS,
                         weights: np.ndarray = EVR_WEIGHTS):
    """
    PennyLane circuit routine applying the weighted ZZ Feature Map U_Phi(x).
    
    1. Hadamard transform: H^(x n)
    2. Repeated (n_layers):
       a. Single-qubit rotations: Rz(x_j)
       b. Two-qubit Ising ZZ couplings: exp(-i * 2 w_j w_k x_j x_k Z_j Z_k)
    """
    for j in range(n_qubits):
        qml.Hadamard(wires=j)

    for _ in range(n_layers):
        for j in range(n_qubits):
            qml.RZ(float(x[j]), wires=j)

        for j in range(n_qubits):
            for k in range(j + 1, n_qubits):
                coupling = 2.0 * float(weights[j]) * float(weights[k]) * float(x[j]) * float(x[k])
                qml.IsingZZ(coupling, wires=[j, k])


def build_kernel_qnode(dev=None, n_qubits: int = N_QUBITS):
    """
    Constructs a PennyLane QNode to compute transition fidelity between states:
        K(x1, x2) = |<psi(x1) | psi(x2)>|^2 = |<0| U_Phi(x1)^dagger U_Phi(x2) |0>|^2
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_eval(x1, x2):
        # Forward state preparation for x1
        apply_zz_feature_map(x1, n_qubits=n_qubits)
        # Adjoint state preparation for x2
        qml.adjoint(apply_zz_feature_map)(x2, n_qubits=n_qubits)
        # Transition probability to |0...0>
        return qml.probs(wires=range(n_qubits))

    def evaluate_pair(x1, x2):
        probs = kernel_eval(x1, x2)
        return float(probs[0])

    return evaluate_pair
