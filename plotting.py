import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

from models import DenseCFMForecaster, TopKCFMForecaster


def plot_dho_conditional_full(test_seq, future_true, pred_mu, params, horizon, mode_tag, outdir):
    """
    DHO: Plot x(t), v(t) and phase portrait for conditional forecast.
    
    Args:
        test_seq: Observed part of test sequence
        future_true: True future values
        pred_mu: Predicted mean trajectory
        params: DHO parameters
        horizon: Forecast horizon
        mode_tag: Model mode tag (for filename)
        outdir: Output directory
    """
    T_obs = test_seq.shape[0]
    t_axis = np.arange(T_obs + horizon) * params.dt

    # x(t)
    x_full = np.concatenate([test_seq[:,0], future_true[:,0]])
    x_pred = np.concatenate([test_seq[:,0], pred_mu[:,0].detach().cpu().numpy()])
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, x_full, label="True x", linewidth=2)
    plt.plot(t_axis, x_pred, "--", label="Pred x", linewidth=2)
    plt.xlabel("Time"); plt.ylabel("x"); plt.title(f"DHO Conditional x(t) [{mode_tag}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cond_dho_x_{mode_tag}.png"), dpi=150); plt.close()

    # v(t)
    v_full = np.concatenate([test_seq[:,1], future_true[:,1]])
    v_pred = np.concatenate([test_seq[:,1], pred_mu[:,1].detach().cpu().numpy()])
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, v_full, label="True v", linewidth=2)
    plt.plot(t_axis, v_pred, "--", label="Pred v", linewidth=2)
    plt.xlabel("Time"); plt.ylabel("v"); plt.title(f"DHO Conditional v(t) [{mode_tag}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cond_dho_v_{mode_tag}.png"), dpi=150); plt.close()

    # Phase portrait x–v (future segment only)
    plt.figure(figsize=(6, 6))
    plt.plot(future_true[:,0], future_true[:,1], label="True future", linewidth=2)
    plt.plot(pred_mu[:,0].detach().cpu().numpy(),
             pred_mu[:,1].detach().cpu().numpy(), "--", label="Pred mean", linewidth=2)
    plt.xlabel("x"); plt.ylabel("v"); plt.title(f"DHO Conditional Phase Portrait [{mode_tag}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cond_dho_phase_{mode_tag}.png"), dpi=150); plt.close()


def plot_lorenz_conditional_coords(test_seq, future_true, pred_mu, params, horizon, mode_tag, outdir):
    """
    Lorenz: Plot x(t), y(t), z(t) and 3D phase portrait for conditional forecast.
    
    Args:
        test_seq: Observed part of test sequence
        future_true: True future values
        pred_mu: Predicted mean trajectory
        params: Lorenz-63 parameters
        horizon: Forecast horizon
        mode_tag: Model mode tag (for filename)
        outdir: Output directory
    """
    T_obs = test_seq.shape[0]
    t_axis = np.arange(T_obs + horizon) * params.dt
    labels = ["x", "y", "z"]
    
    # Time series for each coordinate
    for k, lab in enumerate(labels):
        true_full = np.concatenate([test_seq[:,k], future_true[:,k]])
        pred_full = np.concatenate([test_seq[:,k], pred_mu[:,k].detach().cpu().numpy()])
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, true_full, label=f"True {lab}", linewidth=2)
        plt.plot(t_axis, pred_full, "--", label=f"Pred {lab}", linewidth=2)
        plt.xlabel("Time"); plt.ylabel(lab); plt.title(f"Lorenz-63 Conditional {lab}(t) [{mode_tag}]")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cond_l63_{lab}_{mode_tag}.png"), dpi=150); plt.close()
    
    # 3D Phase portrait
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(future_true[:,0], future_true[:,1], future_true[:,2], 
            label="True future", linewidth=2)
    ax.plot(pred_mu[:,0].detach().cpu().numpy(),
            pred_mu[:,1].detach().cpu().numpy(),
            pred_mu[:,2].detach().cpu().numpy(),
            label="Pred mean", linestyle="--", linewidth=2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Lorenz-63 Conditional 3D Phase Portrait [{mode_tag}]")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cond_l63_phase_{mode_tag}.png"), dpi=150); plt.close()


def _compute_ensemble_stats(traj_ensemble, q_lo=0.1, q_hi=0.9):
    """
    Compute ensemble statistics (mean and quantile bands).
    
    Args:
        traj_ensemble: Trajectory ensemble, shape (S, T, d)
        q_lo: Lower quantile
        q_hi: Upper quantile
    
    Returns:
        mean: Ensemble mean, shape (T, d)
        band: Quantile band, shape (T, d, 2) where last dim is [lower, upper]
    """
    mean = traj_ensemble.mean(axis=0)                  # (T, d)
    lo = np.quantile(traj_ensemble, q_lo, axis=0)      # (T, d)
    hi = np.quantile(traj_ensemble, q_hi, axis=0)      # (T, d)
    band = np.stack([lo, hi], axis=-1)                 # (T, d, 2)
    return mean, band


def plot_unconditional_series_all_coords(pred_trajs_np, dt, mode_tag, outdir, 
                                         true_traj_np=None, system="lorenz63"):
    """
    Plot per-coordinate unconditional series: ensemble mean + central band.
    
    Args:
        pred_trajs_np: Predicted trajectories, shape (S, T, d)
        dt: Time step size
        mode_tag: Model mode tag
        outdir: Output directory
        true_traj_np: Optional reference trajectory, shape (T, d)
        system: System name ('dho' or 'lorenz63')
    """
    S, T, d = pred_trajs_np.shape
    mean, band = _compute_ensemble_stats(pred_trajs_np)
    t_axis = np.arange(T) * dt
    coord_labels = ["x", "y", "z"] if d == 3 else ["x", "v"]

    for k, lab in enumerate(coord_labels):
        plt.figure(figsize=(10, 4))
        # Uncertainty band
        plt.fill_between(t_axis, band[:,k,0], band[:,k,1], alpha=0.25, label="Ensemble band (10-90%)")
        # Ensemble mean
        plt.plot(t_axis, mean[:,k], label="Ensemble mean", linewidth=2)
        # Optional reference trajectory
        if true_traj_np is not None:
            plt.plot(t_axis, true_traj_np[:T,k], "--", label="Reference true", linewidth=2)
        plt.xlabel("Time"); plt.ylabel(lab)
        plt.title(f"Unconditional {lab}(t) (mean ± band) [{system} | {mode_tag}]")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"uncond_{lab}_{mode_tag}.png"), dpi=150)
        plt.close()


def plot_unconditional_phase_portrait(pred_trajs_np, mode_tag, outdir, 
                                      true_traj_np=None, system="lorenz63", n_show=5):
    """
    Plot unconditional phase portraits.
    - DHO: x–v in 2D for n_show sample trajectories
    - Lorenz: 3D phase portraits for n_show sample trajectories
    
    Args:
        pred_trajs_np: Predicted trajectories, shape (S, T, d)
        mode_tag: Model mode tag
        outdir: Output directory
        true_traj_np: Optional reference trajectory
        system: System name
        n_show: Number of sample trajectories to show
    """
    S, T, d = pred_trajs_np.shape
    idx = np.linspace(0, S-1, min(n_show, S)).astype(int)

    if d == 2:  # DHO
        plt.figure(figsize=(6, 6))
        for s in idx:
            plt.plot(pred_trajs_np[s,:,0], pred_trajs_np[s,:,1], alpha=0.8)
        if true_traj_np is not None:
            plt.plot(true_traj_np[:T,0], true_traj_np[:T,1], "k--", linewidth=2, label="Reference true")
        plt.xlabel("x"); plt.ylabel("v")
        plt.title(f"DHO Unconditional Phase Portrait [{mode_tag}]")
        if true_traj_np is not None: plt.legend()
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"uncond_dho_phase_{mode_tag}.png"), dpi=150)
        plt.close()
    elif d == 3:  # Lorenz
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for s in idx:
            ax.plot(pred_trajs_np[s,:,0], pred_trajs_np[s,:,1], pred_trajs_np[s,:,2], alpha=0.9)
        if true_traj_np is not None:
            ax.plot(true_traj_np[:T,0], true_traj_np[:T,1], true_traj_np[:T,2], 
                   "k--", linewidth=2, label="Reference true")
            ax.legend()
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"Lorenz-63 Unconditional Phase Portrait [{mode_tag}]")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"uncond_l63_phase_{mode_tag}.png"), dpi=150)
        plt.close()


def plot_acf_results(true_acf: np.ndarray, pred_acf: np.ndarray, 
                     max_lag: int, mode_tag: str, outdir: str):
    """
    Plot the true vs. predicted mean ACF.
    
    Args:
        true_acf: True mean ACF, shape (max_lag+1,)
        pred_acf: Predicted mean ACF, shape (max_lag+1,)
        max_lag: Maximum lag
        mode_tag: Model mode tag
        outdir: Output directory
    """
    lags = np.arange(max_lag + 1)
    mae = np.mean(np.abs(true_acf - pred_acf))
    
    plt.figure(figsize=(10, 6))
    plt.plot(lags, true_acf, 'b-', marker='o', label='True Mean ACF', linewidth=2)
    plt.plot(lags, pred_acf, 'r--', marker='x', label='Predicted Mean ACF', linewidth=2)
    plt.title(f"ACF Similarity for Unconditional Generation (MAE: {mae:.4f}) [{mode_tag}]")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"acf_similarity_{mode_tag}.png"), dpi=150)
    plt.close()

