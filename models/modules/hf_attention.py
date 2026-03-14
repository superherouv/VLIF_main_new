"""
High-Frequency Enhancement Module (HFEM)  —  Intervention 1
=============================================================
Addresses the core limitation identified in VLIF and this study:
  *"SNN naturally biases toward low-frequency propagation, causing
   representation degradation on high-frequency-demanding tasks."*

Architecture overview
---------------------

    x_in  (B, C, H, W)
      │
      ├─── Low-pass branch ───────────────────────────────────────────┐
      │        (Gaussian smooth → standard SNN conv path)              │
      │                                                                │
      └─── High-pass branch ──────────────────────────────────────────┤
               (Laplacian / DoG → HF-aware conv → SE attention)        │
                    │                                                   │
                    └─── gate α (learnable, SigSTE) ──── + ────── x_out

Key design choices
------------------
* **HighFreqExtractor** uses a fixed Gaussian kernel (no learned parameters)
  to produce `HF = x − Gaussian(x)`.  This is deterministic and numerically
  stable — avoids introducing high-frequency artifacts during training.

* **SpikeSEAttention** applies squeeze-and-excitation channel attention on the
  HF path.  The attention weights are binarised during inference (straight-
  through in training) so the module remains spike-compatible.

* **Learnable gate** α (per-channel) controls how much HF enhancement to blend
  in.  Initialised to a small positive value (0.3) so the module starts as a
  minor perturbation and grows as it proves useful.

* **FrequencySplitBlock** implements the full frequency-split hybrid path:
  LF through SNN-style depthwise conv, HF through a separate (optionally
  ANN-full-precision) branch.  The merge is soft (sigmoid gate), enabling
  graceful fallback to the baseline when HF information is not useful.

Usage in boundary experiments
------------------------------

  # Intervention 1: drop HFEM into any SNN restorer
  from models.modules import HFEnhancementModule
  hfem = HFEnhancementModule(channels=64, T=4)
  x_enhanced = hfem(x_spike_feature)  # per-timestep

  # Intervention 3 (partial): frequency-split hybrid decoder block
  from models.modules import FrequencySplitBlock
  fsb = FrequencySplitBlock(channels=64, hybrid=True)
  x_out = fsb(x_feature)

Reference:
  VLIF (2024) — motivates HF enhancement for SNN.
  SENet (Hu et al., 2018) — channel attention baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class HighFreqExtractor(nn.Module):
    """
    Extracts the high-frequency component of a feature map.

    HF = x − LP(x)   where LP is a fixed Gaussian low-pass filter.

    The kernel is registered as a buffer (not a parameter) so it is
    not affected by the optimiser — HF extraction is always correct.
    """

    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.channels = channels

        # Build fixed Gaussian kernel
        k = kernel_size
        coords = torch.arange(k).float() - k // 2
        gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss_1d /= gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]   # (k, k)

        # Expand to (C, 1, k, k) for depthwise conv
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, k, k)
        self.register_buffer("gauss_kernel", kernel.clone())
        self.padding = k // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            hf: (B, C, H, W) high-frequency residual
        """
        lp = F.conv2d(
            x, self.gauss_kernel,
            padding=self.padding,
            groups=self.channels,
        )
        return x - lp

    def extra_repr(self) -> str:
        return f"channels={self.channels}"


class SpikeSEAttention(nn.Module):
    """
    Spike-compatible Squeeze-and-Excitation channel attention.

    Uses straight-through binarisation so the module remains compatible
    with SNN spike frameworks while still learning meaningful channel weights.

    The binarisation threshold is learnable via a softplus-parameterised scalar
    to give the module some flexibility in its operating point.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        init_threshold: float = 0.5,
    ):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        # Learnable binarisation threshold (reparameterised to stay in (0,1))
        self.raw_threshold = nn.Parameter(
            torch.tensor(float(init_threshold)).log()   # softplus-style init
        )

    @property
    def threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze
        s = x.mean(dim=[2, 3])         # (B, C)
        # Excitation
        attn = self.fc(s)               # (B, C) in [0, 1]
        # Spike-compatible binarisation (straight-through estimator in training)
        thr = self.threshold
        if self.training:
            attn_bin = attn + ((attn > thr).float() - attn).detach()
        else:
            attn_bin = (attn > thr).float()
        # Scale
        return x * attn_bin.unsqueeze(-1).unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# HFEM — High-Frequency Enhancement Module
# ──────────────────────────────────────────────────────────────────────────────

class HFEnhancementModule(nn.Module):
    """
    High-Frequency Enhancement Module for SNN feature maps.

    Injects a dedicated high-frequency processing path to compensate for
    the SNN's inherent low-frequency bias.

    Args:
        channels:       number of input/output channels
        T:              number of SNN timesteps (metadata; used in repr)
        hf_gate_init:   initial weight for HF blend-in gate (0.3 = modest)
        use_attention:  whether to apply SpikeSEAttention on the HF path
        kernel_size:    Gaussian kernel size for LP filter (odd number)
        sigma:          Gaussian sigma for LP filter
    """

    def __init__(
        self,
        channels: int,
        T: int = 4,
        hf_gate_init: float = 0.30,
        use_attention: bool = True,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.T = T

        # HF extraction (fixed, no parameters)
        self.hf_extractor = HighFreqExtractor(channels, kernel_size, sigma)

        # HF processing: point-wise mixing + batch norm
        self.hf_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Optional spike-compatible SE attention
        self.hf_attn: nn.Module = (
            SpikeSEAttention(channels, reduction=4)
            if use_attention
            else nn.Identity()
        )

        # Per-channel learnable gate (initialised to hf_gate_init)
        raw_init = torch.tensor(hf_gate_init).clamp(1e-4, 1 - 1e-4)
        # Inverse sigmoid so sigmoid(raw_gate) ≈ hf_gate_init
        raw = torch.log(raw_init / (1.0 - raw_init))
        self.raw_gate = nn.Parameter(
            raw.expand(1, channels, 1, 1).clone()
        )

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate)  # (1, C, 1, 1) in (0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — can be called per-timestep inside the SNN loop
        Returns:
            x_enhanced: (B, C, H, W)
        """
        hf = self.hf_extractor(x)      # (B, C, H, W) high-freq residual
        hf = self.hf_conv(hf)          # channel mixing
        hf = self.hf_attn(hf)          # spike-compatible attention
        return x + self.gate * hf      # gated blend-in

    def extra_repr(self) -> str:
        return f"channels={self.channels}, T={self.T}"


# ──────────────────────────────────────────────────────────────────────────────
# FrequencySplitBlock — full frequency-split hybrid path
# ──────────────────────────────────────────────────────────────────────────────

class FrequencySplitBlock(nn.Module):
    """
    Frequency-split processing block for hybrid SNN/ANN models.

    Splits input into LF and HF components, processes them through
    separate branches, then merges.  Implements Intervention 3 (hybrid decoder):
      - In SNN-only mode:    both branches use depthwise SNN-style convolutions.
      - In hybrid mode:      LF → SNN-style depthwise; HF → full ANN residual.

    This directly tests the hypothesis:
      "If replacing the HF path with an ANN recovers most of the PSNR gap,
       then SNN's bottleneck is in high-frequency reconstruction, not encoding."

    Args:
        channels:   feature channels
        T:          SNN timesteps (metadata)
        hybrid:     if True, use full-precision ANN conv on the HF path
    """

    def __init__(
        self,
        channels: int,
        T: int = 4,
        hybrid: bool = False,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.T = T
        self.hybrid = hybrid

        # Fixed HF extractor
        self.hf_extract = HighFreqExtractor(channels, kernel_size, sigma)

        # LF branch: depthwise conv (SNN-compatible, low parameter count)
        self.lf_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        # HF branch
        if hybrid:
            # Full-precision ANN conv (more expressive, for HF detail)
            self.hf_branch = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
        else:
            # SNN-style depthwise (same as LF branch, but separate weights)
            self.hf_branch = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )

        # Soft merge gate (sigmoid → [0, 1]; initialise to 0.5)
        self.merge_gate = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.merge_gate)  # scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            out: (B, C, H, W) frequency-merged output + residual connection
        """
        hf = self.hf_extract(x)   # high-freq component
        lf = x - hf               # low-freq complement

        lf_out = self.lf_branch(lf)
        hf_out = self.hf_branch(hf)

        alpha = self.alpha
        out = alpha * lf_out + (1.0 - alpha) * hf_out
        return out + x            # residual connection

    def extra_repr(self) -> str:
        return f"channels={self.channels}, T={self.T}, hybrid={self.hybrid}"
