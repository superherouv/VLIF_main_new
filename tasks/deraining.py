"""
Task: Image Deraining

Rain removal is the task most amenable to SNNs because:
  1. Rain streaks are sparse, high-frequency, directional artifacts
  2. SNN spike coding naturally captures sparse events
  3. Temporal coding can encode rain direction/velocity
  4. Energy advantage: rain regions are sparse → low spike rate

Datasets:
  - Rain100H / Rain100L (Yang et al., 2017): synthetic heavy/light rain
  - Rain1400 (Fu et al., 2017): 14 rain streak patterns
  - SPA-Data (Wang et al., 2019): real-world rain
  - DID-MDN (Zhang et al., 2018): density-based
  - GT-RAIN (Ba et al., 2022): realistic rain with depth

Evaluation:
  - PSNR / SSIM on Rain100H and Rain100L (standard benchmarks)
  - Measured on Y channel (luminance) for fair comparison with literature

SNN notes from literature:
  - ESDNet (Song et al., 2024): shows SNN-based deraining achieves ~35dB PSNR
    on Rain100L with T=4 timesteps; ~40% energy saving vs ANN
  - Chen et al. (2025): Poisson encoding best for rain; temporal encoding
    better captures rain dynamics in video
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .base import BaseTask
from .losses import CombinedRestorationLoss
from .metrics import MetricCalculator, psnr, ssim


class DerainingTask(BaseTask):
    """
    Deraining task configuration.

    Loss: Charbonnier (primary) + SSIM (0.1) + Edge (0.05)
    Reasoning: Rain streaks affect edges strongly; edge loss helps.
    """

    def __init__(
        self,
        loss_weights: Dict = None,
        sigma_clip: float = 0.0,
    ):
        self._loss = CombinedRestorationLoss(
            weights=loss_weights or {"char": 1.0, "ssim": 0.1, "edge": 0.05},
            channels=3,
            use_edge=True,
        )
        self._metrics = MetricCalculator(task_name="deraining")

    @property
    def name(self) -> str:
        return "deraining"

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

    def preprocess(self, degraded: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Normalize to [0,1]."""
        return {"input": degraded.clamp(0.0, 1.0)}

    def postprocess(self, model_output: torch.Tensor, **kwargs) -> torch.Tensor:
        return model_output.clamp(0.0, 1.0)

    def snn_applicability_notes(self) -> str:
        return (
            "DERAINING: HIGH SNN applicability.\n"
            "Rain streaks are sparse, elongated, high-frequency artifacts.\n"
            "SNN spike coding maps naturally to sparse event detection.\n"
            "Expected: SNN achieves 90-95% of ANN PSNR with 30-50% energy savings.\n"
            "Best encoding: Direct (T=4) or Rate (T=8) for static images;\n"
            "Temporal encoding for video rain sequences.\n"
            "Key paper: ESDNet (Song et al., 2024) - first SNN deraining network."
        )
