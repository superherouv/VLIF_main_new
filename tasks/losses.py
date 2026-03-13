"""
Loss functions for image restoration tasks.

Commonly used losses:
  - L1 / Charbonnier: robust pixel loss
  - L2 / MSE: standard but penalizes large errors more
  - SSIM Loss: structural similarity
  - Perceptual / VGG Loss: feature-level similarity
  - Frequency Loss: FFT-domain penalty for high-freq artifacts
  - Edge Loss: Sobel-gradient based

For SNN training specifically:
  - L1 is preferred over L2 (SNN outputs can have saturated regions)
  - Frequency loss helps recover sharp edges lost due to binary spike limitations
  - Charbonnier often best overall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CharbonnierLoss(nn.Module):
    """
    Charbonnier (generalized L1) loss: √(|x-y|² + ε²).

    More robust than L1 at exact zeros, smoother than L2 for large errors.
    Widely used in deraining (MPRNet), deblurring, super-resolution.
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps**2)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) based loss: 1 - SSIM(pred, target).

    Args:
        window_size (int): Gaussian window size.
        channel (int): Number of image channels.
        reduction (str): 'mean' | 'none'
    """

    def __init__(self, window_size: int = 11, channel: int = 3,
                 reduction: str = "mean"):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.reduction = reduction
        self.register_buffer("window", self._create_window(window_size, channel))

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        return g / g.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        C1 = 0.01**2
        C2 = 0.03**2
        window = self.window.to(img1.device)
        pad = self.window_size // 2
        groups = img1.shape[1]

        mu1 = F.conv2d(img1, window, padding=pad, groups=groups)
        mu2 = F.conv2d(img2, window, padding=pad, groups=groups)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=groups) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=groups) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=groups) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_val = self._ssim(pred, target)
        loss = 1.0 - ssim_val
        if self.reduction == "mean":
            return loss.mean()
        return loss


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss: L1 on FFT magnitude spectrum.

    Encourages recovery of high-frequency details (edges, textures)
    that are often suppressed by pixel-domain losses.
    Particularly useful for super-resolution and denoising.

    Ref: Cho et al., "Rethinking Coarse-to-Fine Approach in Single Image
    Deblurring", ICCV 2021.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute FFT of each channel
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        loss = F.l1_loss(pred_mag, target_mag, reduction=self.reduction)
        return loss


class EdgeLoss(nn.Module):
    """
    Edge-aware loss using Sobel gradients.

    Penalizes mismatches in gradient magnitude and direction,
    helping preserve sharp boundaries.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude, x: (B, C, H, W)."""
        B, C, H, W = x.shape
        x_gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        gx = F.conv2d(x_gray, self.kx, padding=1)
        gy = F.conv2d(x_gray, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._sobel(pred), self._sobel(target))


class CombinedRestorationLoss(nn.Module):
    """
    Combined loss for image restoration tasks.

    Default: Charbonnier + SSIM + optionally Frequency.

    Args:
        weights (dict): Weight for each loss component.
            Default: {'char': 1.0, 'ssim': 0.1, 'freq': 0.05}
        channels (int): Image channels for SSIM window.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        channels: int = 3,
        use_freq: bool = False,
        use_edge: bool = False,
    ):
        super().__init__()
        if weights is None:
            weights = {"char": 1.0, "ssim": 0.1}
            if use_freq:
                weights["freq"] = 0.05
            if use_edge:
                weights["edge"] = 0.05

        self.weights = weights
        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss(channel=channels)
        self.freq_loss = FrequencyLoss() if use_freq else None
        self.edge_loss = EdgeLoss() if use_edge else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ):
        """
        Returns:
            (total_loss, components_dict)
        """
        components = {}
        total = 0.0

        if "char" in self.weights:
            l = self.char_loss(pred, target)
            components["char"] = l.item()
            total = total + self.weights["char"] * l

        if "ssim" in self.weights and self.ssim_loss is not None:
            l = self.ssim_loss(pred, target)
            components["ssim"] = l.item()
            total = total + self.weights["ssim"] * l

        if "freq" in self.weights and self.freq_loss is not None:
            l = self.freq_loss(pred, target)
            components["freq"] = l.item()
            total = total + self.weights["freq"] * l

        if "edge" in self.weights and self.edge_loss is not None:
            l = self.edge_loss(pred, target)
            components["edge"] = l.item()
            total = total + self.weights["edge"] * l

        return total, components
