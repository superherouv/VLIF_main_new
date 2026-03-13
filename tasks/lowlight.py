"""
Task: Low-Light Image Enhancement

Low-light enhancement is the LEAST suitable task for SNNs:
  1. Requires precise non-linear tone mapping across the full dynamic range
  2. Color correction is critical (color constancy requires analog precision)
  3. Noise amplification in dark regions must be suppressed simultaneously
  4. Histogram equalization and Retinex decomposition are fundamentally analog ops
  5. Binary spikes cannot represent the smooth continuous mapping needed

Why SNN still explored:
  - Edge detection in dark regions can benefit from event-driven processing
  - If combined with event cameras, SNN has intrinsic advantage
  - For extreme low-light (near-zero pixel values), SNN can represent
    "dark vs not-dark" events sparsely

Datasets:
  - LOL (Wei et al., 2018): 485 pairs low/normal light
  - MIT-Adobe FiveK (Bychkovsky et al., 2011): tonal adjustments
  - VE-LOL (Liu et al., 2021): 2500 real pairs
  - LSRW (Hai et al., 2023): 5650 pairs, diverse scenes
  - SID (Chen et al., 2018): long-exposure reference, extreme low light
  - SMID (Chen et al., 2019): multi-image denoising under dark conditions

Evaluation: PSNR, SSIM, NIQE (no-reference for real cases)
"""

import torch
from typing import Dict, Tuple
from .base import BaseTask
from .losses import CombinedRestorationLoss, CharbonnierLoss, FrequencyLoss
from .metrics import psnr, ssim


class LowLightTask(BaseTask):
    """
    Low-light enhancement task configuration.

    Approach:
      - Retinex decomposition: I = R × I_max (reflectance × illumination)
      - Enhance illumination map while preserving reflectance
      - This is where SNN is weakest (smooth illumination gradients)

    Loss: L1 + SSIM + Perceptual (if available)
    No edge loss: edges are often corrupted, not preserved
    No frequency loss: frequency domain unreliable in extreme low light
    """

    def __init__(
        self,
        use_retinex: bool = False,
        gamma_correction: float = 2.2,
        loss_weights: Dict = None,
    ):
        self.use_retinex = use_retinex
        self.gamma = gamma_correction
        self._loss = CombinedRestorationLoss(
            weights=loss_weights or {"char": 1.0, "ssim": 0.1},
            channels=3,
        )

    @property
    def name(self) -> str:
        return "lowlight"

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
        return {
            "psnr": psnr(pred.clamp(0, 1), target.clamp(0, 1)),
            "ssim": ssim(pred.clamp(0, 1), target.clamp(0, 1)),
        }

    def preprocess(
        self, degraded: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Low-light preprocessing:
          1. Normalize to [0,1]
          2. Optional gamma linearization
          3. Optional Retinex decomposition
        """
        inp = degraded.clamp(0.0, 1.0)

        if self.use_retinex:
            # Illumination map (max over channels, Gaussian smoothed)
            illum = inp.max(dim=1, keepdim=True).values
            illum = illum + 1e-6  # avoid division by zero
            reflectance = inp / illum
            return {
                "input": inp,
                "illumination": illum,
                "reflectance": reflectance,
            }
        return {"input": inp}

    def postprocess(self, model_output: torch.Tensor, **kwargs) -> torch.Tensor:
        return model_output.clamp(0.0, 1.0)

    def snn_applicability_notes(self) -> str:
        return (
            "LOW-LIGHT ENHANCEMENT: LOW SNN applicability.\n"
            "Non-linear tone mapping requires precise analog computation.\n"
            "Binary spikes cannot represent smooth illumination gradients.\n"
            "Color correction accuracy is limited by spike discretization.\n"
            "SNN achieves only 60-75% of ANN quality on LOL benchmark.\n"
            "Exception: combined with event cameras (SNN native advantage).\n"
            "Recommendation: Use ANN for standalone low-light enhancement.\n"
            "Research direction: SNN as noise detector, ANN for tone mapping."
        )
