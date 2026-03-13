"""
Task: Image Denoising

Denoising is a medium-difficulty task for SNNs:
  - Gaussian noise is dense (affects every pixel) → reduces SNN energy advantage
  - BUT: noise patterns are stationary → SNN temporal memory useful
  - Real noise (SIDD, DND) is more structured → better for SNN

Datasets:
  - CBSD68 + Train400: Gaussian denoising (σ=15,25,50)
  - SIDD (Abdelhamed et al., 2018): real smartphone noise, 30,000 patches
  - DND (Plotz & Roth, 2017): real DSLR noise
  - PolyU (Liu et al., 2017): real noise with multiple captures

SNN notes:
  - Dense noise means high firing rate (all pixels affected)
  - Energy advantage diminishes vs deraining
  - SNN may still be competitive for structured/pattern noise
  - Temporal averaging over T steps provides implicit denoising (like BM3D block matching)
"""

import torch
from typing import Dict, Tuple
from .base import BaseTask
from .losses import CombinedRestorationLoss
from .metrics import psnr, ssim


class DenoisingTask(BaseTask):
    """
    Denoising task configuration.

    Loss: L1 + SSIM
    No edge loss (noise is not edge-preserving, edge loss can cause artifacts)
    """

    def __init__(
        self,
        noise_level: float = 25.0,  # sigma for blind denoising range
        blind: bool = True,
        loss_weights: Dict = None,
    ):
        self.noise_level = noise_level
        self.blind = blind
        self._loss = CombinedRestorationLoss(
            weights=loss_weights or {"char": 1.0, "ssim": 0.05},
            channels=3,
        )

    @property
    def name(self) -> str:
        return "denoising"

    @property
    def input_channels(self) -> int:
        return 3  # 4 if noise map concatenated (non-blind)

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
        return {
            "psnr": psnr(pred.clamp(0, 1), target.clamp(0, 1)),
            "ssim": ssim(pred.clamp(0, 1), target.clamp(0, 1)),
        }

    def preprocess(
        self, degraded: torch.Tensor, noise_sigma: float = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        For non-blind: concatenate noise level map as 4th channel.
        For blind: just normalize.
        """
        inp = degraded.clamp(0.0, 1.0)
        if not self.blind and noise_sigma is not None:
            # Noise level map: single channel filled with σ/255
            B, C, H, W = inp.shape
            noise_map = torch.full((B, 1, H, W), noise_sigma / 255.0,
                                   device=inp.device, dtype=inp.dtype)
            inp = torch.cat([inp, noise_map], dim=1)
        return {"input": inp}

    def postprocess(self, model_output: torch.Tensor, **kwargs) -> torch.Tensor:
        return model_output.clamp(0.0, 1.0)

    def snn_applicability_notes(self) -> str:
        return (
            "DENOISING: MEDIUM SNN applicability.\n"
            "Gaussian noise is spatially dense → reduces spike sparsity.\n"
            "At σ=25: expected 70-80% of ANN quality; 10-20% energy saving.\n"
            "At σ=50: SNN performance drops more due to dense high-magnitude input.\n"
            "Structured/real noise (SIDD) likely better for SNN than Gaussian.\n"
            "Temporal averaging over T steps provides implicit denoising effect.\n"
            "Recommendation: Use Hybrid (SNN-enc + ANN-dec) for best tradeoff."
        )
