# Medical QML: Hilbert Space Topology and Feature Mapping in Medical Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![MedMNIST](https://img.shields.io/badge/Dataset-MedMNIST%20v2-green.svg)](https://medmnist.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Quantum%20Machine%20Learning-blueviolet.svg)](https://github.com/topics/quantum-machine-learning)

> **Official Research Codebase**  
> **Title:** *Hilbert Space Topology and Feature Mapping in Medical Image Classification: A Quantum-Classical Hybrid Approach*  
> **Author:** Tokuma Gebissa  
> **Institution:** Addis Ababa University, Department of Physics  
> **Supervisor:** Mr. Tesfu Kassaye  

---

## 📌 Repository Overview

**Medical QML (`medical-qml`)** is an open-source research framework investigating quantum Hilbert-space state embeddings for medical radiograph analysis (**PneumoniaMNIST**, MedMNIST v2).

While conventional deep learning convolutional models suffer from a severe **specificity paradox** on imbalanced clinical datasets—sacrificing false-alarm specificity to maximize pneumonia recall—this hybrid Quantum Machine Learning (QML) architecture maps low-dimensional principal features into a 256-dimensional complex Hilbert space ($\mathcal{H} = \mathbb{C}^{2^8}$). This achieves balanced clinical decision boundaries with **$98\times$ fewer features** and **$1,470\times$ fewer parameters**.

<p align="center">
  <img src="figures/specificity_paradox.png" width="72%" alt="Clinical Specificity Paradox" />
</p>

---

## 🔍 Key Research Highlights

- **Clinical Specificity Breakthrough:** The hybrid Variational Quantum Circuit (VQC) achieves **$82.91\%$ specificity** on the held-out test set ($N=624$), outperforming full-resolution classical CNNs ($58.12\%$) by **$+24.79$ percentage points** and resolving the false-alarm dilemma.
- **Extreme Parameter Efficiency:** Achieves **$0.9132$ AUC** with only **34 parameters** (32 quantum gates + 2 classical affine parameters), compared to $\sim 50,000$ parameters in the classical CNN.
- **EVR-Weighted Ising ZZ Feature Map:** Incorporates explained variance ratios directly into pairwise coupling Hamiltonians $H_{ZZ}$, preventing the quantum kernel orthogonality collapse observed in unweighted embeddings ($\Delta = +0.0648$ vs. $+0.00002$).
- **Quadratic Noise Resilience $\mathcal{O}(\sigma^2)$:** Demonstrates intrinsic geometric robustness under Gaussian feature perturbations scaling as $\text{AUC}(\sigma) = 0.9266 - 2.89\sigma^2$, maintaining specificity $\ge 0.800$ across all noise levels where classical linear models degrade linearly $\mathcal{O}(\sigma)$.
- **Empirical Statistical Rigor:**
  - **McNemar Paired Test:** $\chi^2(1) = 171.60, p = 3.317 \times 10^{-39}, \text{Odds Ratio} = 10.17\times$ (VQC vs. classical MLP).
  - **Hanley-McNeil Analytical CI:** $\Delta\text{AUC} = 0.1330 \pm 0.0416, 95\%\text{ CI: } [0.0914, 0.1746], Z = 6.27, p = 3.61 \times 10^{-10}$.
  - **Non-Parametric Bootstrap:** $10,000$ resamples: balanced $\Delta\text{AUC} = 0.2514, 95\%\text{ CI: } [0.2201, 0.2813], P(\Delta \le 0) = 0.000$.

---

## 📊 Benchmark Results Summary

Evaluated on the independent, held-out test set ($N=624$, Normal = 234, Pneumonia = 390):

| Model | Architecture / Space | Trainable Params | Test AUC | Accuracy | Sensitivity | Specificity | G-Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical CNN** | Full image ($\mathbb{R}^{784}$) | $\sim 50,000$ | **0.9648** | 0.8429 | **1.0000** | 0.5812 | 0.7624 |
| **Classical MLP** | PCA components ($\mathbb{R}^8$) | $\sim 2,000$ | 0.7802 | 0.5016 | 0.2462 | **0.9274** | 0.4778 |
| **QSVM (ZZ Kernel)** | Quantum State ($\mathbb{C}^{256}$) | **0** | 0.9171 | **0.8622** | 0.9205 | 0.7650 | 0.8391 |
| **Hybrid VQC (Ours)**| Parameterized ($\mathbb{C}^{256}$) | **34** | 0.9132 | 0.8397 | 0.8462 | **0.8291** | **0.8376** |

<p align="center">
  <img src="figures/phase1_kernel_comparison.png" width="48%" alt="Kernel Matrix Topography" />
  <img src="figures/phase2_optimized_training.png" width="48%" alt="VQC Optimization Dynamics" />
</p>

---

## 🧮 Mathematical Architecture

```
Raw X-Ray Image (28×28 Grayscale) ─────────────────> ℝ⁷⁸⁴
  │
  ▼ [StandardScaler + PCA (Fit on Train Only)]
8 Principal Components ────────────────────────────> ℝ⁸   (74.6% Cumulative Variance)
  │
  ▼ [QuantumAngleEncoder (Scale = π/4, No Leakage)]
Angle-Encoded Features ────────────────────────────> [0, π/4]⁸
  │
  ▼ [Ising ZZ Feature Map Unitary U_Φ(x) (2 layers)]
Quantum State |ψ(x)⟩ ──────────────────────────────> ℂ²⁵⁶  (2⁸ Hilbert Space)
  │
  ▼ [Variational Ansatz U_VQC(θ) (Ry + Rz + CNOT)]
Rotated State |ψ(x, θ)⟩ ───────────────────────────> ℂ²⁵⁶
  │
  ▼ [EVR Hamiltonian Observable ⟨Ô⟩ = Σ w_j Z_j]
Expectation Value ─────────────────────────────────> ℝ
  │
  ▼ [Affine Post-Layer: ŷ = a⟨Ô⟩ + b (a=3.7266, b=1.0216)]
Calibrated Probabilities & G-Mean Threshold ───────> {0: Normal, 1: Pneumonia}
```

### 1. Quantum Angle Encoding
$$x_j^{\text{enc}} = \frac{\text{clip}(x_j, p_{\text{lo},j}, p_{\text{hi},j}) - p_{\text{lo},j}}{p_{\text{hi},j} - p_{\text{lo},j}} \cdot \frac{\pi}{4}$$

### 2. EVR-Weighted Ising ZZ Hamiltonian
$$H_{ZZ} = \sum_{j=0}^{7} x_j Z_j + \sum_{j < k} 2 w_j w_k x_j x_k Z_j Z_k, \quad w_j = \sqrt{\frac{\text{EVR}_j}{\text{EVR}_{\max}}}$$

### 3. Variational Ansatz & Observables
$$U_{\text{VQC}}(\theta) = \prod_{\ell=0}^{1} \left[ \left(\prod_{j=0}^{6} \text{CNOT}_{j \to j+1}\right) \left(\prod_{j=0}^{7} R_y(\theta_{\ell,j,0}) R_z(\theta_{\ell,j,1})\right) \right]$$
$$\hat{O} = \sum_{j=0}^{7} \frac{\text{EVR}_j}{\sum_k \text{EVR}_k} Z_j, \quad \hat{y} = a \langle\hat{O}\rangle + b$$

---

## 📈 Statistical Audit & Out-of-Distribution Noise

<p align="center">
  <img src="figures/mcnemar_contingency_table.png" width="48%" alt="McNemar Table" />
  <img src="figures/phase3_ood_robustness.png" width="48%" alt="OOD Robustness" />
</p>

- **Contingency Matrix ($N=624$):** $n_{11}=290, n_{10}=234, n_{01}=23, n_{00}=77$.
- **Noise Resilience Mechanism:** Perturbations $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ on quantum phase couplings produce zero-mean, quadratic variance shifts ($\text{Var}[\Delta J_{jk}] \propto \mathcal{O}(\sigma^2)$), whereas classical decision boundaries shift linearly ($\mathcal{O}(\sigma)$).

---

## 📂 Repository Organization

```
medical-qml/
├── configs/
│   └── config.json                 # Hyperparameters and pipeline settings
├── src/                            # Modular production library
│   ├── data_utils.py               # MedMNIST data loaders and PCAReducer
│   ├── classical_models.py         # Classical CNN and MLP models
│   ├── quantum_encoding.py         # QuantumAngleEncoder & ZZ Feature Map
│   ├── vqc_model.py                # Parameter-shift VQC circuit & trainer
│   ├── metrics.py                  # Hanley-McNeil SE, McNemar test, metrics
│   └── plot_utils.py               # Standardized academic plotting
├── experiments/                    # Runnable benchmark scripts
│   ├── run_phase0_baseline.py      # Classical baselines & PCA fitting
│   ├── run_phase1_qsvm.py          # Optimized QSVM kernel evaluation
│   ├── run_phase2_vqc.py           # Parameter-shift VQC evaluation/training
│   ├── run_phase3_statistical_audit.py # McNemar, Hanley-McNeil, and Bootstrap
│   ├── run_phase3_ablation.py      # Component contribution ablation
│   └── run_phase3_ood_robustness.py# OOD Gaussian noise perturbation audit
├── visualization/                  # Publication figure generation
│   ├── generate_thesis_figures.py  # Generates all publication-ready figures
│   └── plot_mcnemar_table.py       # McNemar paired contingency table plot
├── pretrained/                     # Exact serialized parameters & metrics
│   ├── theta_best.npy              # Optimal 32 quantum parameters (shape: 2, 8, 2)
│   ├── ab_best.npy                 # Optimal classical post-layer coefficients [a, b]
│   └── summary_metrics.json        # Complete consolidated experimental records
├── figures/                        # High-resolution generated PNG figures
├── web/
│   └── thesis_diagrams.jsx         # Interactive React / SVG diagrams component
├── requirements.txt                # Python environment requirements
├── LICENSE                         # MIT License
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/medical-qml.git
cd medical-qml

# Virtual environment setup
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Immediate Statistical Verification (No Retraining Required)
Run the verified evaluation pipelines using the frozen parameters in `pretrained/`:

```bash
# 1. Statistical Validity Audit (McNemar, Hanley-McNeil, Bootstrap 10k)
python experiments/run_phase3_statistical_audit.py

# 2. Component Ablation Study (ZZ Feature Map vs. VQC)
python experiments/run_phase3_ablation.py

# 3. Out-of-Distribution Noise Robustness Audit
python experiments/run_phase3_ood_robustness.py

# 4. Evaluate Pretrained Hybrid VQC
python experiments/run_phase2_vqc.py
```

### 3. Re-train Models from Scratch
```bash
# Phase 0: Classical CNN & MLP baselines
python experiments/run_phase0_baseline.py

# Phase 1: Quantum Kernel SVM
python experiments/run_phase1_qsvm.py --quick

# Phase 2: Hybrid VQC with parameter-shift gradients
python experiments/run_phase2_vqc.py --train --epochs 5

# Regenerate all publication figures
python visualization/generate_thesis_figures.py
python visualization/plot_mcnemar_table.py
```

---

## 🏷️ GitHub SEO & Discovery Metadata

When creating or configuring the repository on GitHub, configure the following settings:

| Setting | Recommended Value |
| :--- | :--- |
| **Repository Name** | `medical-qml` |
| **Short Description** | *Hilbert space topology and EVR-weighted ZZ feature mapping for medical image classification (PneumoniaMNIST). Hybrid quantum-classical VQC achieving 82.9% specificity with 98x fewer features.* |
| **Topics / Tags** | `medical-qml`, `quantum-machine-learning`, `medical-imaging`, `pennylane`, `pytorch`, `hilbert-space`, `feature-mapping`, `vqc`, `qsvm`, `pneumoniamnist`, `medmnist`, `quantum-computing`, `chest-xray`, `healthcare-ai` |

---

## 📖 Citation

```bibtex
@thesis{gebissa2026medicalqml,
  title     = {Hilbert Space Topology and Feature Mapping in Medical Image Classification: A Quantum-Classical Hybrid Approach},
  author    = {Gebissa, Tokuma},
  year      = {2026},
  school    = {Addis Ababa University},
  department= {Department of Physics},
  note      = {Supervisor: Mr. Tesfu Kassaye}
}
```
