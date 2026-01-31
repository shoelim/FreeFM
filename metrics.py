import torch
import numpy as np
from typing import List, Tuple


def mse_torch(y: torch.Tensor, yhat: torch.Tensor) -> float:
    """
    Mean squared error between predictions and targets.
    
    Args:
        y: True values, shape (N, d) or (d,)
        yhat: Predicted values, shape (N, d) or (d,)
    
    Returns:
        MSE value (scalar)
    """
    return float(((y - yhat) ** 2).mean().item())


def crps_torch(samples: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for a set of forecast samples.
    
    CRPS measures the accuracy of probabilistic forecasts. It generalizes
    the mean absolute error to probabilistic predictions.
    
    Formula: CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent samples from the forecast distribution.
    
    Args:
        samples: Forecast samples, shape (S, d)
        y_true: True value, shape (d,)
    
    Returns:
        CRPS value averaged over all dimensions (scalar)
    """
    # samples: (S, d), y_true: (d,)
    S = samples.shape[0]
    
    # Term 1: E[|Z - y|]
    term1_abs_diff = torch.abs(samples - y_true.unsqueeze(0))
    term1 = term1_abs_diff.mean(dim=0)  # Mean over samples: (d,)
    
    # Term 2: 0.5 * E[|Z - Z'|]
    # Compute pairwise absolute differences between all samples
    abs_diffs = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0))
    term2 = 0.5 * abs_diffs.mean(dim=[0, 1])  # Mean over both sample dimensions: (d,)
    
    # CRPS for each dimension
    crps_per_dim = term1 - term2
    
    # Return the average CRPS over all dimensions
    return float(crps_per_dim.mean().item())


def normalized_acf_torch(X: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute normalized Autocorrelation Function (ACF) for a 1D time series.
    
    The ACF measures the correlation of a signal with a delayed copy of itself.
    
    Args:
        X: Time series, shape (T,)
        max_lag: Maximum lag to compute
    
    Returns:
        ACF values at lags 0 to max_lag, shape (max_lag+1,)
    """
    T = X.shape[0]
    if T <= max_lag:
        # Handle case where sequence is too short
        return torch.zeros(max_lag + 1, device=X.device)

    # Center the time series
    X_mean = X - X.mean()
    
    # Compute autocovariance for each lag
    acov = torch.zeros(max_lag + 1, device=X.device)
    for lag in range(max_lag + 1):
        # Covariance between X[0:T-lag] and X[lag:T]
        acov[lag] = (X_mean[:T-lag] * X_mean[lag:]).sum() / (T - lag)
        
    # Normalize by variance (lag 0)
    variance = acov[0].clamp_min(1e-9)
    acf = acov / variance
    return acf


def evaluate_acf_similarity(true_trajs: List[np.ndarray], 
                            pred_trajs: np.ndarray, 
                            max_lag: int = 50, 
                            n_components: int = 1) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate similarity between mean ACF of true and predicted trajectory ensembles.
    
    This metric assesses whether the model captures the temporal correlation
    structure of the dynamical system.
    
    Args:
        true_trajs: List of true trajectory arrays, each shape (T, d)
        pred_trajs: Predicted trajectory ensemble, shape (S, T, d)
        max_lag: Maximum lag for ACF computation
        n_components: Number of state components to evaluate
    
    Returns:
        acf_mae: Mean absolute error between true and predicted mean ACFs
        mean_true_acf: Mean ACF from true trajectories, shape (max_lag+1,)
        mean_pred_acf: Mean ACF from predicted trajectories, shape (max_lag+1,)
    """
    true_acfs = []
    pred_acfs = []
    
    # 1. Calculate ACF for true trajectories
    for traj in true_trajs:
        for k in range(n_components):
            X_true = torch.tensor(traj[:, k], dtype=torch.float32)
            true_acfs.append(normalized_acf_torch(X_true, max_lag).cpu().numpy())
    
    # 2. Calculate ACF for predicted trajectories
    for s in range(pred_trajs.shape[0]):
        for k in range(n_components):
            X_pred = torch.tensor(pred_trajs[s, :, k], dtype=torch.float32)
            pred_acfs.append(normalized_acf_torch(X_pred, max_lag).cpu().numpy())
    
    # Convert to arrays: (N_trajs * d, max_lag + 1)
    true_acfs_np = np.stack(true_acfs)
    pred_acfs_np = np.stack(pred_acfs)
    
    # 3. Calculate mean ACF across all trajectories
    mean_true_acf = np.mean(true_acfs_np, axis=0)
    mean_pred_acf = np.mean(pred_acfs_np, axis=0)
    
    # 4. Compute similarity (MAE between mean ACFs)
    acf_mae = np.mean(np.abs(mean_true_acf - mean_pred_acf))
    
    return acf_mae, mean_true_acf, mean_pred_acf
