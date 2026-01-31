import math
from typing import List, Tuple

import torch

from utils import HyperParams


class BaseCFMForecaster:
    """
    Base class implementing the empirical CFM velocity field for transitions:

        X^{(j)} = (X_1^{(j)}, X_2^{(j)}) ∈ R^d × R^d,  j = 1,...,M.

    Brownian-bridge conditional path:

        m_t^{(j)} = (1 - t) X_1^{(j)} + t X_2^{(j)},
        c_t^2     = σ_min^2 + σ^2 t(1 - t),
        G_t       = σ^2 (1 - 2t) / [2 c_t^2].

    Empirical CFM velocity:

        v(t,z) = G_t z + sum_j α_j(z,t) [ d m_t^{(j)} - G_t m_t^{(j)} ],

    where α_j(z,t) is the Gaussian responsibility with variance c_t^2.
    """

    def __init__(self,
                 X1_np,
                 X2_np,
                 hp: HyperParams,
                 time_grid: int = 100,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            X1_np: (M, d) array of X_1^{(j)} (start states).
            X2_np: (M, d) array of X_2^{(j)} (next states).
            hp:    HyperParams with sigma_min and sigma.
            time_grid: number of precomputed time points in [0,1].
            device:    computation device.
        """
        self.device = device
        self.hp = hp
        self.time_grid = time_grid

        # Store memory bank X_1^{(j)}, X_2^{(j)}
        X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
        X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
        assert X1.shape == X2.shape
        self.X1 = X1
        self.X2 = X2
        self.M, self.d = X1.shape

        # Uniform priors over transitions (not used explicitly but kept for clarity)
        self.pi = torch.ones(self.M, device=device) / self.M

        # Precompute everything on a fixed time grid
        self._precompute_time_grid()

    # -------------------------------------------------------------
    # Time grid and helpers
    # -------------------------------------------------------------
    def _precompute_time_grid(self):
        """
        Precompute for t ∈ {0,...,1}:

          - G_t (scalar),
          - c_t^2 (scalar),
          - logdet(c_t^2 I_d) = d * log(c_t^2),
          - m_t^{(j)} for all j (μ_t_list),
          - B_t^{(j)} = \dot m_t^{(j)} - G_t m_t^{(j)} for all j.
        """
        hp = self.hp
        sigma_min2 = float(hp.sigma_min ** 2)
        sigma2 = float(hp.sigma ** 2)

        ts = torch.linspace(0.0, 1.0, self.time_grid, device=self.device)
        self.ts = ts

        self.G_list = torch.empty(self.time_grid, device=self.device)
        self.c2_list = torch.empty(self.time_grid, device=self.device)
        self.logdet_list = torch.empty(self.time_grid, device=self.device)

        self.mu_list: List[torch.Tensor] = []  # m_t^{(j)}
        self.B_list: List[torch.Tensor] = []   # B_t^{(j)}

        X1 = self.X1
        X2 = self.X2

        for i, t_val in enumerate(ts.tolist()):
            t = float(t_val)

            # Brownian-bridge-like variance schedule
            c2_t = sigma_min2 + sigma2 * t * (1.0 - t)  # c_t^2
            # avoid degeneracy numerically
            c2_t = max(c2_t, 1e-12)

            # Global linear term G_t = σ^2 (1 - 2t) / (2 c_t^2)
            G_t = 0.0
            if sigma2 > 0.0:
                G_t = sigma2 * (1.0 - 2.0 * t) / (2.0 * c2_t)

            self.c2_list[i] = c2_t
            self.G_list[i] = G_t
            self.logdet_list[i] = self.d * math.log(c2_t)

            # Conditional path mean m_t^{(j)} and its time derivative
            a_t = 1.0 - t
            b_t = t
            m_t = a_t * X1 + b_t * X2                     # (M, d)
            dm_t = X2 - X1                                # (M, d)  (constant in t)

            # B_t^{(j)} = dm_t^{(j)} - G_t * m_t^{(j)}
            B_t = dm_t - G_t * m_t                        # (M, d)

            self.mu_list.append(m_t)
            self.B_list.append(B_t)

    def _idx_t(self, t: torch.Tensor) -> int:
        """
        Map a continuous t ∈ [0,1] to the nearest precomputed grid index.
        """
        idx = torch.clamp(torch.round(t * (self.time_grid - 1)),
                          0, self.time_grid - 1)
        return int(idx.item())

    # -------------------------------------------------------------
    # Drift interface (to be implemented in subclasses)
    # -------------------------------------------------------------
    def drift(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute v(t,z) for a batch of states z ∈ R^{B×d} at time t ∈ [0,1].
        """
        raise NotImplementedError


class DenseCFMForecaster(BaseCFMForecaster):
    """
    Empirical CFM velocity with full attention over all transitions:

        v(t,z) = G_t z + sum_j α_j(z,t) B_t^{(j)},

    where B_t^{(j)} = d m_t^{(j)} - G_t m_t^{(j)} and α_j is Gaussian attention.
    """

    def drift(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d) batch of states.
            t: scalar tensor in [0,1].

        Returns:
            dzdt: (B, d)
        """
        assert z.dim() == 2
        Bbatch, d = z.shape
        assert d == self.d

        i = self._idx_t(t)
        G_t = float(self.G_list[i].item())
        c2_t = float(self.c2_list[i].item())
        logdet = float(self.logdet_list[i].item())

        m_t = self.mu_list[i]   # (M, d)
        B_t = self.B_list[i]    # (M, d)

        # Gaussian responsibilities α_j(z,t) ∝ N(z; m_t^{(j)}, c_t^2 I)
        # Compute pairwise differences: (B, M, d)
        diff = z.unsqueeze(1) - m_t.unsqueeze(0)
        mahal = diff.pow(2).sum(dim=2) / c2_t  # (B, M)

        # log p_t(z | X^{(j)}) = const - 0.5 * mahal
        const = -0.5 * (self.d * math.log(2.0 * math.pi) + logdet)
        logw = const - 0.5 * mahal  # (B, M)
        w = torch.softmax(logw, dim=1)  # (B, M)

        # Nonlinear correction: sum_j α_j(z,t) B_t^{(j)}
        correction = w @ B_t  # (B, d)

        # Total velocity: v(t,z) = G_t z + correction
        return G_t * z + correction


class TopKCFMForecaster(BaseCFMForecaster):
    """
    Same empirical CFM velocity, but restricting attention to the k nearest
    transitions for each query state z at time t.
    """

    def __init__(self,
                 X1_np,
                 X2_np,
                 hp: HyperParams,
                 time_grid: int = 100,
                 k: int = 256,
                 device: torch.device = torch.device("cpu")):
        super().__init__(X1_np, X2_np, hp, time_grid=time_grid, device=device)
        self.k = min(k, self.M)

    def drift(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d)
            t: scalar in [0,1]
        """
        assert z.dim() == 2
        Bbatch, d = z.shape
        assert d == self.d

        i = self._idx_t(t)
        G_t = float(self.G_list[i].item())
        c2_t = float(self.c2_list[i].item())
        logdet = float(self.logdet_list[i].item())

        m_t = self.mu_list[i]   # (M, d)
        B_t = self.B_list[i]    # (M, d)

        # Compute distances to all m_t^{(j)} and pick top-k nearest
        # dist: (B, M)
        dist = torch.cdist(z, m_t)  # Euclidean
        topk_vals, topk_idx = dist.topk(self.k, dim=1, largest=False)  # (B, k)

        # Gather m_t and B_t for these indices
        m_k = m_t[topk_idx]   # (B, k, d)
        B_k = B_t[topk_idx]   # (B, k, d)

        # Gaussian responsibilities over the k neighbors
        diff_k = z.unsqueeze(1) - m_k        # (B, k, d)
        mahal_k = diff_k.pow(2).sum(dim=2) / c2_t   # (B, k)

        const = -0.5 * (self.d * math.log(2.0 * math.pi) + logdet)
        logw_k = const - 0.5 * mahal_k      # (B, k)
        w_k = torch.softmax(logw_k, dim=1)  # (B, k)

        # Weighted sum over k neighbors
        correction = torch.bmm(w_k.unsqueeze(1), B_k).squeeze(1)  # (B, d)

        return G_t * z + correction
