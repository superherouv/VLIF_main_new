"""
Evaluation metrics for image restoration.

Standard metrics:
  - PSNR: Peak Signal-to-Noise Ratio (dB), higher=better
  - SSIM: Structural Similarity Index [0,1], higher=better
  - LPIPS: Learned Perceptual Image Patch Similarity (optional, needs lpips pkg)
  - MAE: Mean Absolute Error, lower=better
  - FID: Fréchet Inception Distance (for perceptual quality, optional)

SNN-specific metrics:
  - Mean Firing Rate: average spike rate across all layers
  - SynOps: synaptic operations (proxy for energy)
  - SparsityIndex: 1 - mean_firing_rate (higher = more sparse = less energy)
  - EnergyEfficiency: PSNR / SynOps (quality per unit energy)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    reduction: str = "mean",
) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 20·log10(max_val / RMSE)

    Args:
        pred, target: (B, C, H, W) tensors in [0, max_val]
        reduction: 'mean' over batch | 'none' returns per-image

    Returns:
        PSNR in dB
    """
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])  # (B,)
    psnr_vals = 20.0 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse + 1e-8))
    if reduction == "mean":
        return psnr_vals.mean().item()
    return psnr_vals


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    reduction: str = "mean",
) -> float:
    """
    Structural Similarity Index (SSIM).

    Args:
        pred, target: (B, C, H, W) in [0,1]
        window_size: Gaussian kernel size
        reduction: 'mean' | 'none'

    Returns:
        SSIM value [0, 1]
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Gaussian kernel
    def gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.unsqueeze(1) * g.unsqueeze(0)
        return kernel

    kernel = gaussian_kernel(window_size).to(pred.device)
    C = pred.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu12

    ssim_map = (
        (2 * mu12 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )  # (B, C, H, W)

    ssim_per_image = ssim_map.mean(dim=[1, 2, 3])  # (B,)
    if reduction == "mean":
        return ssim_per_image.mean().item()
    return ssim_per_image


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return (pred - target).abs().mean().item()


class MetricCalculator:
    """
    Unified metric calculator for all restoration tasks.

    Tracks running averages across a dataset split.
    """

    def __init__(self, task_name: str = "unknown"):
        self.task_name = task_name
        self.reset()

    def reset(self):
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._mae_sum = 0.0
        self._count = 0
        self._firing_rate_sum = 0.0
        self._synops_sum = 0.0
        self._fr_count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        firing_rate: Optional[float] = None,
        synops: Optional[float] = None,
    ):
        """
        Update running averages with a batch.

        Args:
            pred: (B, C, H, W) predictions in [0,1]
            target: (B, C, H, W) ground truth in [0,1]
            firing_rate: optional mean SNN firing rate
            synops: optional SNN synaptic operations count
        """
        B = pred.shape[0]
        self._psnr_sum += psnr(pred, target) * B
        self._ssim_sum += ssim(pred, target) * B
        self._mae_sum += mae(pred, target) * B
        self._count += B

        if firing_rate is not None:
            self._firing_rate_sum += firing_rate
            self._fr_count += 1
        if synops is not None:
            self._synops_sum += synops

    def compute(self) -> Dict[str, float]:
        """Return averaged metrics dict."""
        if self._count == 0:
            return {}
        metrics = {
            "psnr": self._psnr_sum / self._count,
            "ssim": self._ssim_sum / self._count,
            "mae": self._mae_sum / self._count,
        }
        if self._fr_count > 0:
            metrics["mean_firing_rate"] = self._firing_rate_sum / self._fr_count
            metrics["sparsity_index"] = 1.0 - metrics["mean_firing_rate"]
            if self._synops_sum > 0:
                metrics["synops"] = self._synops_sum / self._fr_count
                # Energy efficiency: PSNR per unit SynOps (log scale)
                metrics["energy_efficiency"] = metrics["psnr"] / (
                    np.log1p(metrics["synops"])
                )
        return metrics

    def summary(self) -> str:
        """Human-readable metrics summary."""
        m = self.compute()
        lines = [f"=== {self.task_name.upper()} Metrics ==="]
        for k, v in m.items():
            lines.append(f"  {k:25s}: {v:.4f}")
        return "\n".join(lines)
