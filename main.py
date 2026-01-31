"""
main.py - Main script for flow-matching-inspired training-free samplers for time series generation

This script trains and evaluates forecasting models on:
1. Damped Harmonic Oscillator (DHO) - 2D system
2. Lorenz-63 - 3D chaotic system

Tasks: forecast future states and generate unconditional trajectories
"""
import os
import time
import argparse
import numpy as np
import torch

from utils import set_device, seed_everything, DHOParams, Lorenz63Params
from data import (make_dho_dataset_np, make_lorenz_dataset_np, 
                  build_memory_bank_np, split_train_val)
from models import DenseCFMForecaster, TopKCFMForecaster
from train import grid_search_cfm
from forecast import multi_step_forecast_torch, generate_trajectories_torch
from metrics import mse_torch, crps_torch, evaluate_acf_similarity
from plotting import (plot_dho_conditional_full, plot_lorenz_conditional_coords,
                      plot_unconditional_series_all_coords, plot_unconditional_phase_portrait,
                      plot_acf_results)


def main():
    # ========================================================================
    # ARGUMENT PARSING
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Training-free samplers for time series generation."
    )
    parser.add_argument("--system", type=str, default="lorenz63", 
                       choices=["dho", "lorenz63"], 
                       help="Dynamical system to model.")
    parser.add_argument("--mode", type=str, default="dense", 
                       choices=["dense", "topk"], 
                       help="Mode (dense or top-k).")
    parser.add_argument("--integrator", type=str, default="euler", 
                       choices=["euler", "rk4"], 
                       help="ODE integrator.")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Computation device (e.g., 'cuda:0' or 'cpu').")
    
    # Dataset parameters
    parser.add_argument("--N_train", type=int, default=50, 
                       help="Number of training sequences.")
    parser.add_argument("--N_test_seqs", type=int, default=5,
                       help="Number of test sequences for evaluation.")
    parser.add_argument("--T_steps", type=int, default=1000, 
                       help="Data points per sequence (total length).")
    
    # Generation parameters
    parser.add_argument("--S", type=int, default=50, 
                       help="Number of Monte Carlo samples to be generated.")
    parser.add_argument("--ode_steps", type=int, default=10, 
                       help="ODE integration steps per generation.")
    parser.add_argument("--horizon", type=int, default=500, 
                       help="Number of steps to forecast.")
    
    # Model parameters
    parser.add_argument("--time_grid", type=int, default=100, 
                       help="Number of time points for bridge precomputation.")
    parser.add_argument("--topk", type=int, default=256, 
                       help="K for truncation of summation in the mixture model of the velocity field.")
    
    # Evaluation parameters
    parser.add_argument("--max_lag_acf", type=int, default=50, 
                       help="Maximum lag for ACF calculation.")
    parser.add_argument("--N_acf_true", type=int, default=50, 
                       help="Number of true trajectories for ACF ensemble.")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=123, 
                       help="Global RNG seed.")

    args = parser.parse_args()

    # ========================================================================
    # SETUP
    # ========================================================================
    seed_everything(args.seed)
    device = set_device(args.device)
    mode_tag = args.mode
    
    # Output directory setup
    outdir_base = (f"./outputs/{args.system}/{args.mode}/"
                   f"{args.integrator}/steps{args.ode_steps}")
    os.makedirs(outdir_base, exist_ok=True)
    print(f"Saving outputs to: {outdir_base}")

    
    # ========================================================================
    # DATA GENERATION
    # ========================================================================
    print("\n" + "="*70)
    print("DATA GENERATION")
    print("="*70)

    if args.system == "dho":
        data_params = DHOParams(steps=args.T_steps)
        make_dataset_func = make_dho_dataset_np
        print(f"System: Damped Harmonic Oscillator "
              f"(ω={data_params.omega}, ζ={data_params.zeta}, dt={data_params.dt})")
    else:
        data_params = Lorenz63Params(steps=args.T_steps)
        make_dataset_func = make_lorenz_dataset_np
        print(f"System: Lorenz-63 (σ={data_params.sigma}, ρ={data_params.rho}, "
              f"β={data_params.beta:.3f})")

    print(f"Generating {args.N_train} training sequences of length {args.T_steps}")

    # Generate dataset
    N_test_seqs = args.N_test_seqs
    N_total = args.N_train + N_test_seqs

    all_seqs = make_dataset_func(N_total, data_params)

    # Split data
    train_val_seqs = all_seqs[:-N_test_seqs]
    test_seqs = all_seqs[-N_test_seqs:]

    train_seqs, val_seqs = split_train_val(train_val_seqs, val_frac=0.25)

    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} validation, {N_test_seqs} test")

    # ========================================================================
    # MODEL SETUP
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL SETUP")
    print("="*70)

    if args.mode == "dense":
        ForecasterClass = DenseCFMForecaster
        forecaster_kwargs = {"time_grid": args.time_grid}
        print(f"Model: Dense Forecaster (full attention)")
    else:  # topk
        ForecasterClass = TopKCFMForecaster
        forecaster_kwargs = {"time_grid": args.time_grid, "k": args.topk}
        print(f"Model: TopK Forecaster (k={args.topk})")

    print("\nHyperparameter grid search for sigma_min and sigma...")
    sigma_min_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    sigma_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    df_results, best_hp, best_model = grid_search_cfm(
        ForecasterClass, forecaster_kwargs,
        train_seqs, val_seqs,
        sigma_min_grid, sigma_grid,
        integrator=args.integrator, steps=args.ode_steps, S=args.S,
        device=device
    )

    # Save grid search results
    df_results.to_csv(os.path.join(outdir_base, "grid_search_results.csv"), index=False)

    # ========================================================================
    # CONDITIONAL FORECASTING (Short-Term Prediction)
    # ========================================================================
    print("\n" + "="*70)
    print("CONDITIONAL FORECASTING")
    print("="*70)

    T_obs = args.T_steps - args.horizon
    print(f"Observed timesteps: {T_obs}")
    print(f"Forecast horizon per test sequence: {args.horizon}")
    print(f"Monte Carlo samples: {args.S}")
    print(f"Integration steps per forecast: {args.ode_steps}")
    print(f"Number of test sequences: {len(test_seqs)}")

    all_test_mse = []
    all_test_crps = []

    for test_idx, test_seq in enumerate(test_seqs):
        print(f"\n--- Test sequence {test_idx+1}/{len(test_seqs)} ---")

        observed_part = test_seq[:T_obs]
        future_true = test_seq[T_obs:T_obs + args.horizon]

        pred_mu, pred_samples = multi_step_forecast_torch(
            best_model, observed_part, args.horizon,
            steps=args.ode_steps, S=args.S,
            integrator=args.integrator, device=device
        )

        # Calculate metrics for this test sequence
        test_mse_list = []
        test_crps_list = []

        for i in range(args.horizon):
            y_true = torch.tensor(future_true[i], dtype=torch.float32, device=device)
            mu_pred = pred_mu[i]
            samples = pred_samples[i]

            test_mse_list.append(mse_torch(y_true, mu_pred))
            test_crps_list.append(crps_torch(samples, y_true))

        final_mse = np.mean(test_mse_list)
        final_crps = np.mean(test_crps_list)

        all_test_mse.append(final_mse)
        all_test_crps.append(final_crps)

        print(f"Test sequence {test_idx+1}: MSE = {final_mse:.6f}, CRPS = {final_crps:.6f}")

        # Plot conditional results for this test sequence
        outdir_cond = os.path.join(outdir_base, f"test_seq_{test_idx+1}")
        os.makedirs(outdir_cond, exist_ok=True)

        if args.system == "dho":
            plot_dho_conditional_full(
                observed_part, future_true, pred_mu,
                data_params, args.horizon, mode_tag, outdir_cond
            )
        else:
            plot_lorenz_conditional_coords(
                observed_part, future_true, pred_mu,
                data_params, args.horizon, mode_tag, outdir_cond
            )

    if len(all_test_mse) > 0:
        mean_mse = float(np.mean(all_test_mse))
        mean_crps = float(np.mean(all_test_crps))
        print("\n--- Aggregate Conditional Forecast Metrics over all test sequences ---")
        print(f"Average MSE  = {mean_mse:.6f}")
        print(f"Average CRPS = {mean_crps:.6f}")

    print(f"Conditional forecast plots saved under {outdir_base}")

    # ========================================================================
    # UNCONDITIONAL GENERATION (Long-Term Dynamics)
    # ========================================================================
    print("\n" + "="*70)
    print("UNCONDITIONAL GENERATION")
    print("="*70)

    # Generate true trajectories for reference ACF
    print(f"Generating {args.N_acf_true} reference trajectories...")
    if args.system == "lorenz63":
        init_range = (-15, 15)
        true_trajectories = make_lorenz_dataset_np(
            args.N_acf_true, data_params, init_box=init_range
        )
    else:  # DHO
        init_range = (0.5, 1.5)
        vel_range = (-0.5, 0.5)
        true_trajectories = make_dho_dataset_np(
            args.N_acf_true, data_params,
            x0_range=init_range, v0_range=vel_range
        )

    # Generate model trajectories
    print(f"Generating {args.S} model trajectory samples...")
    # use the initial state of the first test sequence as the anchor for unconditional generation

    start_gen = time.time()

    z0_unconditional = test_seqs[0][0]
    _, predicted_trajectories_np = generate_trajectories_torch(
        best_model, z0_unconditional, args.T_steps - 1, data_params.dt,
        S=args.S, steps=args.ode_steps, integrator=args.integrator, device=device
    )

    end_gen = time.time()
    generation_time = end_gen - start_gen

    print(f"Time to generate {args.S} new trajectories: {generation_time:.4f} seconds")
    print(f"Average time per trajectory: {generation_time / args.S:.4f} seconds")
    # -----------------------------------------------------

    # Get reference trajectory for plotting
    true_ref = true_trajectories[0] if len(true_trajectories) > 0 else None

    # Plot time series with uncertainty bands
    plot_unconditional_series_all_coords(
        predicted_trajectories_np, data_params.dt,
        mode_tag, outdir_base, true_traj_np=true_ref, system=args.system
    )

    # Plot phase portraits
    plot_unconditional_phase_portrait(
        predicted_trajectories_np, mode_tag, outdir_base,
        true_traj_np=true_ref, system=args.system, n_show=5
    )

    print(f"Unconditional generation plots saved to {outdir_base}")

    # ========================================================================
    # ACF EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("AUTOCORRELATION FUNCTION (ACF) EVALUATION")
    print("="*70)

    print(f"Computing ACF up to lag {args.max_lag_acf}...")
    acf_mae, mean_true_acf, mean_pred_acf = evaluate_acf_similarity(
        true_trajectories, predicted_trajectories_np,
        max_lag=args.max_lag_acf,
        n_components=predicted_trajectories_np.shape[2]
    )

    plot_acf_results(
        mean_true_acf, mean_pred_acf, 
        args.max_lag_acf, mode_tag, outdir_base
    )

    print(f"ACF MAE: {acf_mae:.6f}")
    print(f"ACF plot saved to {outdir_base}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best Hyperparameters:")
    print(f"  σ_min = {best_hp['sigma_min']:.4f}")
    print(f"  σ = {best_hp['sigma']:.4f}")
    print(f"\nUnconditional Generation:")
    print(f"  ACF MAE = {acf_mae:.6f}")
    print(f"\nAll outputs saved to: {outdir_base}")
    print(f"Generation Time:")
    print(f"  Total = {generation_time:.4f} seconds")
    print(f"  Per trajectory = {generation_time / args.S:.4f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()
