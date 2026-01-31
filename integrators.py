import torch
import numpy as np


def euler_integrate_batched(f, z0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Forward Euler integration of dz/dt = f(z,t) on t ∈ [0,1].
    
    First-order method: z_{n+1} = z_n + h*f(z_n, t_n)
    
    Args:
        f: Drift function f(z, t) returning shape (B, d)
        z0: Initial state, shape (B, d)
        steps: Number of integration steps
    
    Returns:
        Final state z(1), shape (B, d)
    """
    z = z0.clone()
    t = torch.zeros((), device=z0.device)
    h = 1.0 / steps
    
    for _ in range(steps):
        z = z + h * f(z, t)
        t = t + h
    
    return z


def rk4_integrate_batched(f, z0: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Runge-Kutta 4th order integration of dz/dt = f(z,t) on t ∈ [0,1].
    
    Fourth-order method with four stages per step.
    
    Args:
        f: Drift function f(z, t) returning shape (B, d)
        z0: Initial state, shape (B, d)
        steps: Number of integration steps
    
    Returns:
        Final state z(1), shape (B, d)
    """
    z = z0.clone()
    t = torch.zeros((), device=z0.device)
    h = 1.0 / steps
    
    for _ in range(steps):
        k1 = f(z, t)
        k2 = f(z + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(z + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(z + h * k3, t + h)
        z = z + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
    
    return z

