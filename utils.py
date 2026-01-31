import os
import random
import torch
import numpy as np
from dataclasses import dataclass


def set_device(device_str: str) -> torch.device:
    """Set computation device with fallback."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def seed_everything(seed: int, deterministic: bool = True):
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    Call this before any RNG-dependent work (data gen, model init, etc.).
    
    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


@dataclass
class DHOParams:
    """Damped Harmonic Oscillator parameters."""
    omega: float = 2.0
    zeta: float = 0.15
    dt: float = 0.01
    steps: int = 500
    process_noise: float = 0.0


@dataclass
class Lorenz63Params:
    """Lorenz-63 system parameters."""
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    steps: int = 500
    process_noise: float = 0.0

# ----------------------------------------------------------------------
# CFM hyperparameters 
#   - sigma_min: σ_min > 0 
#   - sigma:     σ >= 0   
# ----------------------------------------------------------------------
@dataclass
class HyperParams:
    """Hyperparameters for the closed-form model"""
    sigma_min: float = 0.05   
    sigma: float = 0.05       
