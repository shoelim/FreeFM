from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from utils import HyperParams
from data import build_memory_bank_np
from forecast import forecast_distribution_batched
from metrics import mse_torch, crps_torch


def grid_search_cfm(ForecasterClass,
                    forecaster_kwargs: dict,
                    train_seqs: List[np.ndarray],
                    val_seqs: List[np.ndarray],
                    sigma_min_grid: List[float],
                    sigma_grid: List[float],
                    integrator: str = "euler",
                    steps: int = 12,
                    S: int = 30,
                    max_pairs: int = 400,
                    device: torch.device = torch.device("cpu")
                    ) -> Tuple[pd.DataFrame, Dict, object]:
    """
    Grid search hyperparameters (σ_min, σ) for the empirical CFM velocity.

    For each combo, we:
      - build a CFM forecaster from train transitions,
      - evaluate 1-step forecasts on validation transitions,
      - compute MSE and CRPS on Val targets,
      - pick the best (lowest val CRPS).
    """
    # Build memory bank for train and val
    X1_train_np, X2_train_np = build_memory_bank_np(train_seqs)
    X1_val_np, X2_val_np = build_memory_bank_np(val_seqs)

    Mv = X1_val_np.shape[0]
    idx_val = np.arange(Mv)
    if Mv > max_pairs:
        idx_val = np.random.choice(Mv, size=max_pairs, replace=False)

    results = []
    best_row = None
    best_model = None

    for sigma_min in sigma_min_grid:
        for sigma in sigma_grid:
            hp = HyperParams(sigma_min=float(sigma_min), sigma=float(sigma))

            # Instantiate CFM forecaster using TRAIN memory only
            model = ForecasterClass(
                X1_train_np, X2_train_np,
                hp=hp,
                device=device,
                **forecaster_kwargs
            )

            preds = []
            crps_list = []

            for j in idx_val:
                x1_val = torch.tensor(X1_val_np[j], dtype=torch.float32, device=device)
                x2_val = torch.tensor(X2_val_np[j], dtype=torch.float32, device=device)

                mu, _, samples = forecast_distribution_batched(
                    model, x1_val,
                    S=S, steps=steps,
                    integrator=integrator, device=device
                )

                preds.append(mu.unsqueeze(0))
                crps_list.append(crps_torch(samples, x2_val))

            preds = torch.cat(preds, dim=0)  # (num_val, d)
            y = torch.tensor(X2_val_np[idx_val], dtype=torch.float32, device=device)

            val_mse = mse_torch(y, preds)
            val_crps = float(np.mean(crps_list))

            row = {
                "sigma_min": sigma_min,
                "sigma": sigma,
                "val_MSE": val_mse,
                "val_CRPS": val_crps,
            }
            results.append(row)

            if best_row is None or val_crps < best_row["val_CRPS"]:
                best_row = row
                best_model = model

    df = pd.DataFrame(results).sort_values(by="val_CRPS")
    return df, best_row, best_model
