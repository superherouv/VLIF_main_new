"""
Task: Single Image Super-Resolution (SISR)

Super-resolution is the LEAST favorable task for SNNs:
  1. Requires dense high-frequency reconstruction (sharp edges, textures)
  2. Binary spike approximation limits sub-pixel precision
  3. Large spatial mapping needed (e.g., 4× upscale: LR→HR)
  4. High-quality SR methods (Real-ESRGAN) use large ANNs with perceptual loss
  5. SNN temporal averaging reduces fine-detail sharpness

However, SNNs may still be useful for:
  - Lightweight, low-latency SR on edge devices
  - Moderate upscale factors (×2 more feasible than ×4)
  - Tasks where edges/structures matter more than texture details

Datasets:
  - DIV2K (Agustsson & Timofte, 2017): 800 train, 100 val HR images
  - Set5, Set14, BSD100, Urban100: standard test sets
  - Flickr2K: additional 2650 HR images
  - DF2K = DIV2K + Flickr2K (used by most SOTA)

Scale factors: ×2, ×3, ×4

Degradation: bicubic downsampling (BI) for standard SR;
             unknown degradation for real-world SR (RealSR, DPED)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .base import BaseTask
from .losses import CombinedRestorationLoss, FrequencyLoss, CharbonnierLoss
from .metrics import psnr, ssim


class SuperResolutionTask(BaseTask):
    """
    Super-resolution task configuration.

    Special handling:
      - Input is LR (small), output is HR (large)
      - Model must handle different spatial sizes at input/output
      - Bicubic upsampled input provided as skip connection baseline

    Loss: Charbonnier + Frequency (high-freq emphasis) + SSIM
    """

    def __init__(
        self,
        scale: int = 4,
        loss_weights: Dict = None,
        upscale_mode: str = "bicubic",
    ):
        self.scale = scale
        self.upscale_mode = upscale_mode
        self._char_loss = CharbonnierLoss()
        self._freq_loss = FrequencyLoss()

        # Combined loss with frequency domain emphasis for SR
        self._loss = CombinedRestorationLoss(
            weights=loss_weights or {"char": 1.0, "ssim": 0.1, "freq": 0.1},
            channels=3,
            use_freq=True,
        )

    @property
    def name(self) -> str:
        return "superresolution"

    @property
    def input_channels(self) -> int:
        return 3

    @property
    def output_channels(self) -> int:
        return 3

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self._loss(pred, target)

    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        # SR metrics computed on Y channel (luma) in YCbCr space
        pred_y = self._rgb_to_y(pred.clamp(0, 1))
        target_y = self._rgb_to_y(target.clamp(0, 1))
        return {
            "psnr_y": psnr(pred_y, target_y),
            "ssim_y": ssim(pred_y, target_y),
            "psnr_rgb": psnr(pred.clamp(0, 1), target.clamp(0, 1)),
        }

    def _rgb_to_y(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to Y channel (standard SR evaluation)."""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 16.0/255.0 + (65.481/255.0 * r + 128.553/255.0 * g + 24.966/255.0 * b)
        return y

    def preprocess(
        self, degraded: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        degraded: LR image (B, C, H/scale, W/scale)

        Returns:
            input: bicubic upsampled LR (same size as HR) for skip connection
            lr: original LR for model input
        """
        lr = degraded.clamp(0.0, 1.0)
        # Bicubic upsampled baseline (for residual learning)
        bicubic_up = F.interpolate(
            lr, scale_factor=self.scale,
            mode="bicubic", align_corners=False
        ).clamp(0, 1)
        return {"input": lr, "bicubic_up": bicubic_up}

    def postprocess(
        self, model_output: torch.Tensor,
        bicubic_up: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """Add bicubic baseline if residual learning is used."""
        if bicubic_up is not None:
            return (model_output + bicubic_up).clamp(0.0, 1.0)
        return model_output.clamp(0.0, 1.0)

    def snn_applicability_notes(self) -> str:
        return (
            "SUPER-RESOLUTION: LOW-MEDIUM SNN applicability.\n"
            "SR requires dense high-frequency reconstruction.\n"
            "Binary spike coding loses sub-pixel precision needed for ×4 SR.\n"
            "For ×2: SNN achieves ~85% of ANN PSNR (acceptable for edge deploy).\n"
            "For ×4: SNN struggles with fine textures (-2 to -3 dB vs ANN).\n"
            "Frequency loss helps partially compensate for spike-domain limitations.\n"
            "Recommendation: Only use SNN for ×2 SR when energy is critical;\n"
            "avoid SNN for ×4 SR in quality-critical applications.\n"
            "Best approach: Hybrid with SNN feature extractor + ANN sub-pixel conv head."
        )
