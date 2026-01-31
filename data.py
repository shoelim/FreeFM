import numpy as np
from typing import List, Tuple
from utils import DHOParams, Lorenz63Params


# -------------------------------------------------------------
# Simulate trajectories X^{(\tau,n)} for DHO and Lorenz-63
# -------------------------------------------------------------
def simulate_dho_sequence_np(params: DHOParams, x0: float = 1.0, v0: float = 0.0) -> np.ndarray:
    """
    Simulate Damped Harmonic Oscillator trajectory using Euler integration.
    
    Args:
        params: DHO system parameters
        x0: Initial position
        v0: Initial velocity
    
    Returns:
        Array of shape (steps, 2) containing [position, velocity] at each timestep
    """
    traj = np.zeros((params.steps, 2), dtype=np.float32)
    x, v = x0, v0
    
    for t in range(params.steps):
        traj[t] = [x, v]

        # Euler integration: dx/dt = v, dv/dt = -2ζωv - ω²x
        a = v
        b = -2.0 * params.zeta * params.omega * v - (params.omega ** 2) * x
        x = x + params.dt * a
        v = v + params.dt * b
        
        # Add process noise if specified
        if params.process_noise > 0:
            x += np.random.randn() * params.process_noise
            v += np.random.randn() * params.process_noise
            
    return traj


def simulate_lorenz63_sequence_np(params: Lorenz63Params, 
                                   x0: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Simulate Lorenz-63 trajectory using RK4 integration.
    
    Args:
        params: Lorenz-63 system parameters
        x0: Initial condition (x, y, z)
    
    Returns:
        Array of shape (steps, 3) containing [x, y, z] at each timestep
    """
    
    def lorenz_deriv(x, y, z, sigma, rho, beta):
        """Lorenz-63 differential equations."""
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    sigma, rho, beta = params.sigma, params.rho, params.beta
                                       
    x, y, z = x0
    dt = params.dt
    
    traj = np.zeros((params.steps, 3), dtype=np.float32)
    
    for t in range(params.steps):
        traj[t] = [x, y, z]
        
        # RK4 Integration
        # k1
        k1x, k1y, k1z = lorenz_deriv(x, y, z, sigma, rho, beta)
        
        # k2
        k2x, k2y, k2z = lorenz_deriv(
            x + 0.5 * dt * k1x,
            y + 0.5 * dt * k1y,
            z + 0.5 * dt * k1z,
            sigma, rho, beta
        )
        
        # k3
        k3x, k3y, k3z = lorenz_deriv(
            x + 0.5 * dt * k2x,
            y + 0.5 * dt * k2y,
            z + 0.5 * dt * k2z,
            sigma, rho, beta
        )
        
        # k4
        k4x, k4y, k4z = lorenz_deriv(
            x + dt * k3x,
            y + dt * k3y,
            z + dt * k3z,
            sigma, rho, beta
        )
        
        # Update using weighted average of k's
        x_new = x + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        y_new = y + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        z_new = z + (dt / 6.0) * (k1z + 2*k2z + 2*k3z + k4z)

        x, y, z = x_new, y_new, z_new
        
        # Add process noise if specified
        if params.process_noise > 0:
            x += np.random.randn() * params.process_noise
            y += np.random.randn() * params.process_noise
            z += np.random.randn() * params.process_noise
            
    return traj


def make_dho_dataset_np(N: int, params: DHOParams, 
                        x0_range: Tuple[float, float] = (0.5, 1.5), 
                        v0_range: Tuple[float, float] = (-0.5, 0.5)) -> List[np.ndarray]:
    """
    Generate N DHO sequences with random initial conditions.
    
    Args:
        N: Number of sequences to generate
        params: DHO system parameters
        x0_range: Range for initial position sampling
        v0_range: Range for initial velocity sampling
    
    Returns:
        List of N trajectory arrays, each of shape (steps, 2)
    """
    seqs = []
    for _ in range(N):
        x0 = np.random.uniform(*x0_range)
        v0 = np.random.uniform(*v0_range)
        seqs.append(simulate_dho_sequence_np(params, x0, v0))
    return seqs


def make_lorenz_dataset_np(N: int, params: Lorenz63Params, 
                           init_box: Tuple[float, float] = (-15, 15)) -> List[np.ndarray]:
    """
    Generate N Lorenz-63 sequences with random initial conditions.
    
    Args:
        N: Number of sequences to generate
        params: Lorenz-63 system parameters
        init_box: Range for random initial condition sampling in all dimensions
    
    Returns:
        List of N trajectory arrays, each of shape (steps, 3)
    """
    seqs = []
    for _ in range(N):
        # Random initial conditions to ensure chaotic behavior
        x0 = np.random.uniform(*init_box, size=3)
        seqs.append(simulate_lorenz63_sequence_np(params, tuple(x0)))
    return seqs


def build_memory_bank_np(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build memory bank from sequences: A[t] -> B[t+1].
    
    Args:
        seqs: List of trajectory arrays
    
    Returns:
        Tuple of (A, B) where:
            A: States at time t, shape (M, d)
            B: States at time t+1, shape (M, d)
            M = total number of transitions across all sequences
    """
    A, B = [], []
    for Z in seqs:
        A.append(Z[:-1])  # All states except last
        B.append(Z[1:])   # All states except first
    return np.concatenate(A, axis=0), np.concatenate(B, axis=0)


def split_train_val(seqs: List[np.ndarray], 
                    val_frac: float = 0.25) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split sequences into train and validation sets.
    
    Args:
        seqs: List of trajectory sequences
        val_frac: Fraction of data to use for validation
    
    Returns:
        Tuple of (train_seqs, val_seqs)
    """
    N = len(seqs)
    idx = np.random.permutation(N)
    Nv = max(1, int(N * val_frac))
    return [seqs[i] for i in idx[Nv:]], [seqs[i] for i in idx[:Nv]]
