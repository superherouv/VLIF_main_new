"""
Spike Encoding Schemes for Image Input to SNNs.

Converting continuous pixel intensities [0,1] into spike trains is a critical
design choice that strongly affects both accuracy and energy consumption.

Encoding comparison:
┌───────────────┬──────────┬──────────┬──────────────────────────────────────┐
│ Scheme        │ T steps  │ Sparsity │ Notes                                │
├───────────────┼──────────┼──────────┼──────────────────────────────────────┤
│ Rate          │ High     │ Low      │ Classic; energy-inefficient          │
│ Temporal      │ 1        │ High     │ Latency coding; fast but noisy       │
│ Phase         │ log₂(N)  │ Medium   │ Binary representation; efficient     │
│ Direct/ANN    │ 1        │ N/A      │ No spike encoding; pseudo-SNN        │
│ Population    │ Medium   │ Medium   │ Gaussian basis; richer representation│
└───────────────┴──────────┴──────────┴──────────────────────────────────────┘

Key insight for low-level vision:
  - Rate coding needs many timesteps (T≥16) for accurate pixel reconstruction
  - Direct encoding + SNN layers is the standard in vision tasks (T=4-8)
  - Temporal coding struggles with smooth gradients needed for per-pixel output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RateEncoder(nn.Module):
    """
    Rate (Poisson) encoding: pixel intensity → spike rate.

    For each timestep t, spike if U(0,1) < pixel_value.
    Energy cost scales with T and mean pixel intensity.

    Args:
        T (int): Number of timesteps.
        gain (float): Scaling factor for firing rate.
    """

    def __init__(self, T: int = 8, gain: float = 1.0):
        super().__init__()
        self.T = T
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) normalized to [0, 1]

        Returns:
            spikes: (T, B, C, H, W) binary
        """
        x_scaled = (x * self.gain).clamp(0.0, 1.0)
        spikes = []
        for _ in range(self.T):
            noise = torch.rand_like(x_scaled)
            spikes.append((noise < x_scaled).float())
        return torch.stack(spikes, dim=0)  # (T, B, C, H, W)

    def extra_repr(self) -> str:
        return f"T={self.T}, gain={self.gain}"


class TemporalEncoder(nn.Module):
    """
    Time-to-first-spike (TTFS) / latency encoding.

    Brighter pixels fire earlier. Encoding:
        t_fire = T * (1 - x)   → earlier spike for higher intensity

    Very sparse (each neuron fires at most once), extremely energy efficient,
    but requires careful decoding and struggles with spatial precision.

    Args:
        T (int): Maximum timestep window.
    """

    def __init__(self, T: int = 8):
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) in [0,1]

        Returns:
            spikes: (T, B, C, H, W) with exactly one spike per pixel
        """
        # t_fire ∈ [0, T-1]: brighter → smaller t
        t_fire = (self.T * (1.0 - x.clamp(1e-6, 1.0))).long().clamp(0, self.T - 1)
        spikes = torch.zeros(self.T, *x.shape, device=x.device, dtype=x.dtype)
        for t in range(self.T):
            spikes[t] = (t_fire == t).float()
        return spikes

    def extra_repr(self) -> str:
        return f"T={self.T}"


class PhaseEncoder(nn.Module):
    """
    Phase / binary encoding.

    Decomposes pixel value into its binary representation:
        x ≈ Σ_{k=0}^{K-1} b_k · 2^(-k-1)   where b_k ∈ {0,1}

    Requires only K = ⌈log₂(levels)⌉ timesteps.
    For 8-bit pixels: K = 8 timesteps, but practically K=4 suffices.
    Good balance between temporal resolution and sparsity.

    Args:
        T (int): Number of bits (timesteps).
    """

    def __init__(self, T: int = 8):
        super().__init__()
        self.T = T
        # Powers of 2 for binary decomposition
        powers = torch.tensor([2**(-k - 1) for k in range(T)])
        self.register_buffer("powers", powers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) in [0, 1]

        Returns:
            spikes: (T, B, C, H, W) binary
        """
        # Quantize to 2^T levels
        levels = 2**self.T
        x_int = (x.clamp(0, 1) * (levels - 1)).long()

        spikes = []
        for k in range(self.T - 1, -1, -1):
            bit = (x_int >> k) & 1
            spikes.append(bit.float())
        # Stack: first spike = MSB
        return torch.stack(spikes, dim=0)  # (T, B, C, H, W)

    def extra_repr(self) -> str:
        return f"T={self.T}"


class DirectEncoder(nn.Module):
    """
    Direct encoding: pass pixel values directly as membrane input.

    No spike conversion at input; first layer receives analog values.
    This is the dominant approach in modern deep SNN papers for vision
    (VLIF, SpikingResNet, etc.) because it avoids information loss
    from stochastic/discrete encoding.

    The network then learns what to spike about internally.

    Args:
        T (int): Timesteps (same input repeated T times, or T learned projections).
        mode (str): 'repeat' | 'project'
            - 'repeat': same frame at each timestep (standard)
            - 'project': learned linear projection per timestep
        channels (int): Only needed for mode='project'.
    """

    def __init__(self, T: int = 4, mode: str = "repeat", channels: Optional[int] = None):
        super().__init__()
        self.T = T
        self.mode = mode
        if mode == "project":
            assert channels is not None, "channels required for project mode"
            # One conv per timestep (weight-sharing option: share all)
            self.projections = nn.ModuleList([
                nn.Conv2d(channels, channels, 1, bias=False)
                for _ in range(T)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (T, B, C, H, W)
        """
        if self.mode == "repeat":
            return x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)
        else:
            return torch.stack([proj(x) for proj in self.projections], dim=0)

    def extra_repr(self) -> str:
        return f"T={self.T}, mode={self.mode}"


class PopulationEncoder(nn.Module):
    """
    Population (Gaussian receptive field) encoding.

    Each pixel value is encoded by K neurons with Gaussian tuning curves
    centered at μ_k = k/(K-1), σ = 1/(2(K-1)).

    Response: r_k(x) = exp(-(x - μ_k)² / (2σ²))
    Spike if r_k(x) > uniform threshold.

    Provides richer distributed representation than scalar rate coding.
    Energy cost: K × rate_cost.

    Args:
        K (int): Number of neurons per pixel (population size).
        T (int): Timesteps.
    """

    def __init__(self, K: int = 10, T: int = 4):
        super().__init__()
        self.K = K
        self.T = T

        mu = torch.linspace(0.0, 1.0, K)  # (K,)
        sigma = 1.0 / (2.0 * (K - 1)) if K > 1 else 1.0
        self.register_buffer("mu", mu)
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) in [0,1]

        Returns:
            spikes: (T, B, K*C, H, W)
        """
        B, C, H, W = x.shape
        # Gaussian response for each center: (B, C, H, W, K)
        x_exp = x.unsqueeze(-1)  # (B, C, H, W, 1)
        mu_exp = self.mu.view(1, 1, 1, 1, self.K)
        responses = torch.exp(-((x_exp - mu_exp)**2) / (2 * self.sigma**2))
        # Flatten population dim into channels: (B, K*C, H, W)
        responses = responses.permute(0, 1, 4, 2, 3).reshape(B, C * self.K, H, W)

        # Rate encode the responses
        spikes = []
        for _ in range(self.T):
            noise = torch.rand_like(responses)
            spikes.append((noise < responses).float())
        return torch.stack(spikes, dim=0)

    def extra_repr(self) -> str:
        return f"K={self.K}, T={self.T}, sigma={self.sigma:.4f}"
