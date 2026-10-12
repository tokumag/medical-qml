"""
Variational Quantum Circuit (VQC) and Hybrid Trainer
====================================================
Implements the 2-layer Ry+Rz parameterized ansatz with nearest-neighbor CNOT 
entanglement coupled to the EVR-weighted Hamiltonian observable:

    |psi(x, theta)> = U_VQC(theta) U_Phi(x) |0>^(x n)
    O = sum_j (w_obs_j Z_j)
    y_hat = a * <psi(x, theta)| O |psi(x, theta)> + b

Optimization uses the parameter-shift rule for quantum parameters theta (32 parameters)
and analytical gradients for classical post-layer coefficients (a, b) (2 parameters).
"""

import time
import numpy as np
import pennylane as qml
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

from src.quantum_encoding import (
    N_QUBITS, N_ZZ_LAYERS, ANGLE_SCALE, EVR_WEIGHTS, OBS_WEIGHTS, apply_zz_feature_map
)


def build_vqc_circuit(n_qubits: int = N_QUBITS,
                      n_zz_layers: int = N_ZZ_LAYERS,
                      n_vqc_layers: int = 2,
                      evr_w: np.ndarray = EVR_WEIGHTS,
                      obs_w: np.ndarray = OBS_WEIGHTS,
                      device_name: str = "default.qubit",
                      diff_method: str = "parameter-shift"):
    """
    Constructs the hybrid QNode evaluating <O> under ZZ encoding and VQC evolution.
    """
    dev = qml.device(device_name, wires=n_qubits)
    w_zz = evr_w.astype(np.float64)
    w_obs = obs_w.astype(np.float64)

    obs = qml.Hamiltonian(
        [float(w_obs[j]) for j in range(n_qubits)],
        [qml.PauliZ(j) for j in range(n_qubits)],
    )

    @qml.qnode(dev, diff_method=diff_method)
    def quantum_circuit(x, theta):
        # Stage 1: ZZ Feature Map (Frozen Data Encoding)
        apply_zz_feature_map(x, n_qubits=n_qubits, n_layers=n_zz_layers, weights=w_zz)

        # Stage 2: Parameterized Variational Unitary (U_VQC)
        for ell in range(n_vqc_layers):
            # Full SU(2) coverage per qubit tilts states off the Bloch equator
            for j in range(n_qubits):
                qml.RY(float(theta[ell, j, 0]), wires=j)
                qml.RZ(float(theta[ell, j, 1]), wires=j)
            # Nearest-neighbor linear entangling cascade
            for j in range(n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])

        # Stage 3: Multi-qubit Hamiltonian expectation
        return qml.expval(obs)

    return quantum_circuit


def optimise_threshold(probs_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Finds the optimal decision threshold maximizing the geometric mean:
        G-mean = sqrt(Sensitivity * Specificity)
    Calibrated exclusively on validation probabilities to protect the Normal class.
    """
    fpr, tpr, thr = roc_curve(y_val, probs_val)
    spec = 1.0 - fpr
    gmean = np.sqrt(tpr * spec)
    best_idx = np.argmax(gmean)
    optimal_t = float(thr[best_idx])
    return optimal_t


class HybridVQCTrainer:
    """
    Joint optimizer for quantum parameters theta and classical affine post-layer (a, b).
    
    Model: y_hat = sigmoid(a * <O>(x, theta) + b)
    Loss : Weighted binary cross-entropy accounting for class imbalance.
    """

    def __init__(self, circuit, config: dict, n_qubits: int = N_QUBITS, n_vqc_layers: int = 2):
        self.circuit = circuit
        self.config = config
        self.n_qubits = n_qubits
        self.n_layers = n_vqc_layers
        self.pos_w = float(config.get("pos_weight", 390.0 / 234.0))

        # Adam hyper-parameters
        self.lr = float(config.get("lr", 0.02))
        self.b1 = float(config.get("beta1", 0.9))
        self.b2 = float(config.get("beta2", 0.999))
        self.eps_a = 1e-8

        # First and second moment vectors
        self.mq = np.zeros((n_vqc_layers, n_qubits, 2))
        self.vq = np.zeros((n_vqc_layers, n_qubits, 2))
        self.mc = np.zeros(2)
        self.vc = np.zeros(2)
        self.t = 0

        self.history = {
            "train_loss": [], "val_auc": [], "val_acc": [],
            "val_sens": [], "val_spec": [], "lr": [], "a": [], "b": []
        }

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def forward(self, x: np.ndarray, theta: np.ndarray, a: float, b: float):
        """Batch forward evaluation returning logits, probabilities, and observable values."""
        obs_vals = np.array([float(self.circuit(x[i], theta)) for i in range(len(x))])
        logits = a * obs_vals + b
        return logits, self.sigmoid(logits), obs_vals

    def bce_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-8
        p = np.clip(probs, eps, 1.0 - eps)
        loss = -np.mean(self.pos_w * y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(loss)

    def compute_gradients(self, x_batch: np.ndarray, y_batch: np.ndarray,
                          theta: np.ndarray, a: float, b: float):
        """
        Parameter-shift rule gradients for quantum parameters;
        analytical chain-rule gradients for classical post-layer.
        """
        n = len(x_batch)
        shift = np.pi / 2.0

        obs_vals = np.array([float(self.circuit(x_batch[i], theta)) for i in range(n)])
        logits = a * obs_vals + b
        probs = self.sigmoid(logits)

        # Weighted residuals
        residuals = (probs - y_batch) * (1.0 + (self.pos_w - 1.0) * y_batch) / n

        # Classical post-layer gradients
        grad_a = float(np.dot(residuals, obs_vals))
        grad_b = float(np.sum(residuals))

        # Quantum gradients via parameter-shift
        grad_q = np.zeros_like(theta)
        for ell in range(self.n_layers):
            for j in range(self.n_qubits):
                for r in range(2):  # 0: RY, 1: RZ
                    tp = theta.copy(); tp[ell, j, r] += shift
                    tm = theta.copy(); tm[ell, j, r] -= shift

                    f_p = np.array([float(self.circuit(x_batch[i], tp)) for i in range(n)])
                    f_m = np.array([float(self.circuit(x_batch[i], tm)) for i in range(n)])

                    grad_q[ell, j, r] = a * float(np.dot(residuals, (f_p - f_m) / 2.0))

        # Optional L2 regularization
        l2_lambda = float(self.config.get("l2_lambda", 1e-4))
        grad_q += l2_lambda * theta

        return grad_q, grad_a, grad_b, obs_vals, probs

    def adam_step(self, grad_q: np.ndarray, grad_a: float, grad_b: float):
        self.t += 1

        # Quantum update
        self.mq = self.b1 * self.mq + (1.0 - self.b1) * grad_q
        self.vq = self.b2 * self.vq + (1.0 - self.b2) * (grad_q ** 2)
        mq_hat = self.mq / (1.0 - self.b1 ** self.t)
        vq_hat = self.vq / (1.0 - self.b2 ** self.t)
        dq = self.lr * mq_hat / (np.sqrt(vq_hat) + self.eps_a)

        # Classical update
        g_c = np.array([grad_a, grad_b])
        self.mc = self.b1 * self.mc + (1.0 - self.b1) * g_c
        self.vc = self.b2 * self.vc + (1.0 - self.b2) * (g_c ** 2)
        mc_hat = self.mc / (1.0 - self.b1 ** self.t)
        vc_hat = self.vc / (1.0 - self.b2 ** self.t)
        dc = self.lr * mc_hat / (np.sqrt(vc_hat) + self.eps_a)

        return dq, float(dc[0]), float(dc[1])

    def evaluate(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                 a: float, b: float, threshold: float = 0.5) -> dict:
        _, probs, _ = self.forward(x, theta, a, b)
        preds = (probs >= threshold).astype(int)
        auc = roc_auc_score(y, probs)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        sens = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
        spec = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
        loss = self.bce_loss(probs, y)
        return dict(auc=float(auc), acc=float(acc), sens=float(sens),
                    spec=float(spec), loss=float(loss), probs=probs, preds=preds, cm=cm)
