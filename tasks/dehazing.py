"""
Task: Image Dehazing

Haze removal is MEDIUM-HIGH applicability for SNNs:
  1. Haze follows the atmospheric scattering model:
       I(x) = J(x)·t(x) + A·(1-t(x))
     where t(x) = transmission map, A = global atmospheric light
  2. Transmission map t(x) is spatially smooth → SNN can detect
     edges/transitions efficiently (sparse boundary events)
  3. Hazy regions have reduced contrast → lower spike rates naturally
  4. Two-stage: SNN for transmission estimation + ANN for reconstruction
     may be optimal

Datasets:
  - RESIDE (Li et al., 2018): largest synthetic dehazing dataset
    - ITS (Indoor Training Set): 13,990 hazy/clean pairs
    - OTS (Outdoor Training Set): 313,950 outdoor pairs
    - SOTS (Test Set): indoor/outdoor test subsets
  - NH-Haze (Ancuti et al., 2020): non-homogeneous real haze
  - Dense-Haze (Ancuti et al., 2019): densely hazy outdoor
  - O-HAZE (Ancuti et al., 2018): outdoor real haze

SNN advantage: transmission map estimation is inherently sparse
(only edges/depth discontinuities have rapid changes).
"""

import torch
from typing import Dict, Tuple
from .base import BaseTask
from .losses import CombinedRestorationLoss
from .metrics import psnr, ssim


class DehazingTask(BaseTask):
    """
    Dehazing task configuration.

    Dehazing-specific considerations:
      - Dark channel prior useful as auxiliary input (optional)
      - White balance matters for accurate color recovery
      - Outdoor vs indoor scenes differ significantly
    """

    def __init__(
        self,
        use_dark_channel: bool = False,
        loss_weights: Dict = None,
    ):
        self.use_dark_channel = use_dark_channel
        self._loss = CombinedRestorationLoss(
            weights=loss_weights or {"char": 1.0, "ssim": 0.1, "freq": 0.05},
            channels=3,
            use_freq=True,
        )

    @property
    def name(self) -> str:
        return "dehazing"

    @property
    def input_channels(self) -> int:
        return 4 if self.use_dark_channel else 3

    @property
    def output_channels(self) -> int:
        return 3

    def _dark_channel(self, x: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
        """Compute dark channel prior (He et al., 2011)."""
        import torch.nn.functional as F
        min_channel = x.min(dim=1, keepdim=True).values
        dark = -F.max_pool2d(-min_channel, patch_size, stride=1,
                              padding=patch_size // 2)
        return dark

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
        inp = degraded.clamp(0.0, 1.0)
        if self.use_dark_channel:
            import torch
            dc = self._dark_channel(inp)  # (B, 1, H, W)
            inp = torch.cat([inp, dc], dim=1)
        return {"input": inp}

    def postprocess(self, model_output: torch.Tensor, **kwargs) -> torch.Tensor:
        return model_output.clamp(0.0, 1.0)

    def snn_applicability_notes(self) -> str:
        return (
            "DEHAZING: MEDIUM-HIGH SNN applicability.\n"
            "Haze transmission map has sparse spatial structure (edges, depth breaks).\n"
            "SNN can efficiently represent binary transmission events.\n"
            "Dense haze (NH-Haze) less suitable than moderate haze (RESIDE-OTS).\n"
            "Expected: SNN achieves 88-93% of ANN PSNR on RESIDE-OTS.\n"
            "Color recovery requires analog precision → hybrid recommended.\n"
            "Two-stage approach: SNN estimate t(x) → ANN recover J(x) = (I-A)/t + A."
        )
