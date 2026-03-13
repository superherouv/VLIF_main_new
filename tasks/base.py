"""Base class for all low-level vision tasks."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional


class BaseTask(ABC):
    """
    Abstract base class defining the interface for each restoration task.

    Responsibilities:
      - Provide loss function(s) for training
      - Provide evaluation metrics
      - Define input/output normalization conventions
      - Specify dataset paths and augmentation strategy
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task identifier."""
        ...

    @property
    @abstractmethod
    def input_channels(self) -> int:
        """Number of input channels."""
        ...

    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Number of output channels."""
        ...

    @abstractmethod
    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        Returns:
            (total_loss, dict of component losses for logging)
        """
        ...

    @abstractmethod
    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (PSNR, SSIM, etc.).

        Returns:
            dict of metric_name → value
        """
        ...

    @abstractmethod
    def preprocess(self, degraded: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Task-specific preprocessing (normalization, upsampling for SR, etc.).

        Returns:
            dict with at least 'input' key for model input
        """
        ...

    @abstractmethod
    def postprocess(self, model_output: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Post-process model output to final prediction (denormalization, clamping).
        """
        ...

    def snn_applicability_notes(self) -> str:
        """Return human-readable notes about SNN applicability for this task."""
        return "No notes available."
