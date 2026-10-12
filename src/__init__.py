"""
Hilbert Space Topology and Feature Mapping in Medical Image Classification
==========================================================================
Quantum-Classical Hybrid Framework for PneumoniaMNIST Analysis.
"""

__version__ = "1.0.0"

# Guarded imports to allow package introspection before dependencies are installed
try:
    from src.data_utils import get_medmnist_loaders, get_flat_arrays, PCAReducer, build_tensor_dataset
except ImportError:
    pass

try:
    from src.classical_models import ClassicalCNN, ClassicalMLP
except ImportError:
    pass

try:
    from src.quantum_encoding import (
        QuantumAngleEncoder, EVR_RAW, EVR_WEIGHTS, OBS_WEIGHTS, ANGLE_SCALE
    )
except ImportError:
    pass

try:
    from src.vqc_model import build_vqc_circuit, HybridVQCTrainer
except ImportError:
    pass

try:
    from src.metrics import compute_metrics, TrainingHistory, compute_hanley_mcneil_se
except ImportError:
    pass
