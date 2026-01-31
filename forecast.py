from typing import Tuple, List

import torch
import numpy as np

from integrators import (
    euler_integrate_batched,
    rk4_integrate_batched,
)


def forecast_distribution_batched(
    model,
    x_tau: torch.Tensor,
    S: int = 50,
    steps: int = 12,
    integrator: str = "euler",
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-step predictive distribution via CFM memory ODE.

    Implements:
        Z_0 ∼ N(x_τ, σ_min^2 I_d),
        dZ_t/dt = v(t,Z_t),
        x_{τ+1} ≈ Z_1.

    Args:
        model: CFM forecaster (DenseCFMForecaster or TopKCFMForecaster),
               with model.hp.sigma_min and .drift(z,t).
        x_tau: (d,) current state.
        S:     number of Monte Carlo samples.
        steps: ODE integration steps for t ∈ [0,1].
        integrator: "euler" or "rk4".
        device: computation device.

    Returns:
        mu:       (d,) empirical mean of Z_1 samples.
        cov_est:  (d,d) empirical covariance of Z_1 samples.
        samples:  (S,d) all Z_1 samples.
    """
    x_tau = x_tau.to(device=device, dtype=torch.float32)
    d = x_tau.shape[0]

    sigma_min = float(model.hp.sigma_min)
    cov0 = (sigma_min ** 2) * torch.eye(d, device=device)
    dist0 = torch.distributions.MultivariateNormal(x_tau, covariance_matrix=cov0)

    # Z_0 samples: (S, d)
    z0 = dist0.sample((S,))

    f = lambda z, t: model.drift(z, t)

    if integrator == "euler":
        z1 = euler_integrate_batched(f, z0, steps=steps)
    elif integrator == "rk4":
        z1 = rk4_integrate_batched(f, z0, steps=steps)
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    # Empirical mean and covariance
    mu = z1.mean(dim=0)  # (d,)
    cov_est = torch.cov(z1.T) if S > 1 else torch.zeros(d, d, device=device)

    return mu, cov_est, z1


def multi_step_forecast_torch(
    model,
    seq_np: np.ndarray,
    horizon: int,
    steps: int = 12,
    S: int = 50,
    integrator: str = "euler",
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Autoregressive multi-step predictor with Particle Propagation
    
    Instead of collapsing to the mean at each step, we propagate the entire 
    ensemble of S particles. 

    Args:
        model: CFM forecaster.
        seq_np: observed prefix of one trajectory, shape (T_obs, d).
        horizon: number of steps to forecast.
        steps: ODE integration steps per forecast.
        S: number of MC samples.
        integrator: "euler" or "rk4".
        device: torch device.

    Returns:
        pred_mu:      (horizon, d) predicted mean at each future step (for metrics).
        pred_samples: list of length horizon, 
                      each element is a (S, d) tensor of samples for that step.
    """
    seq = torch.tensor(seq_np, dtype=torch.float32, device=device)
    x_last = seq[-1]  # The last observed state x_tau
    d = x_last.shape[0]

    sigma_min = float(model.hp.sigma_min)
    cov0 = (sigma_min ** 2) * torch.eye(d, device=device)
    dist0 = torch.distributions.MultivariateNormal(x_last, covariance_matrix=cov0)
    
    # Current state of the ensemble Z: (S, d)
    z_particles = dist0.sample((S,))

    pred_mu = []
    pred_samples: List[torch.Tensor] = []
    
    f = lambda z, t: model.drift(z, t)

    for _ in range(horizon):
        if integrator == "euler":
            z_next = euler_integrate_batched(f, z_particles, steps=steps)
        elif integrator == "rk4":
            z_next = rk4_integrate_batched(f, z_particles, steps=steps)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")
        
        mu_step = z_next.mean(dim=0)  # (d,)
        
        pred_mu.append(mu_step.unsqueeze(0))
        pred_samples.append(z_next)
        
        z_particles = z_next

    pred_mu = torch.cat(pred_mu, dim=0)  # (horizon, d)
    return pred_mu, pred_samples


def generate_trajectories_torch(
    model,
    z0_np: np.ndarray,
    horizon: int,
    dt: float,
    S: int = 50,
    steps: int = 100,
    integrator: str = "euler",
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an ensemble of unconditional trajectories from the CFM memory ODE.

    This differs from multi-step forecasting in that we:
      - start from a Gaussian cloud around a single initial condition z0,
      - evolve each particle forward for 'horizon' steps,
      - and do NOT condition on an observed prefix (it's purely model dynamics).

    Mathematically, for each ensemble member s:

      Z_0^{(s)} ∼ N(z0, σ_min^2 I_d),
      Z_{k+1}^{(s)} = Φ(Z_k^{(s)}),  k = 0,...,horizon-1,

    where Φ is the flow over t ∈ [0,1] induced by dZ_t/dt = v(t, Z_t).

    Args
    ----
    model : CFM forecaster
    z0_np : np.ndarray, shape (d,)
        Initial condition around which we sample the ensemble.
    horizon : int
        Number of discrete steps to generate (so we return horizon+1 time points).
    dt : float
        Physical time step size for the output time array (for plotting).
    S : int
        Number of ensemble members (trajectories).
    steps : int
        ODE integration steps per unit interval [0,1].
    integrator : {"euler", "rk4"}
        ODE integrator.
    device : torch.device

    Returns
    -------
    time : np.ndarray, shape (horizon+1,)
        Discrete time points 0, dt, 2dt, ..., horizon*dt.
    traj : np.ndarray, shape (S, horizon+1, d)
        Generated trajectories. traj[s, k, :] is Z_k^{(s)}.
    """
    z0_np = np.asarray(z0_np, dtype=np.float32)
    d = z0_np.shape[0]

    z0 = torch.tensor(z0_np, dtype=torch.float32, device=device)

    sigma_min = float(model.hp.sigma_min)
    cov0 = (sigma_min ** 2) * torch.eye(d, device=device)
    dist0 = torch.distributions.MultivariateNormal(z0, covariance_matrix=cov0)

    # Initial ensemble Z_0^{(s)}
    Z = dist0.sample((S,))  # (S, d)

    traj = torch.empty(S, horizon + 1, d, device=device)
    traj[:, 0, :] = Z

    f = lambda z, t: model.drift(z, t)

    for t_idx in range(1, horizon + 1):
        if integrator == "euler":
            Z = euler_integrate_batched(f, Z, steps=steps)
        elif integrator == "rk4":
            Z = rk4_integrate_batched(f, Z, steps=steps)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        traj[:, t_idx, :] = Z

    time = torch.arange(horizon + 1, device=device).float() * dt
    return time.cpu().numpy(), traj.detach().cpu().numpy()
