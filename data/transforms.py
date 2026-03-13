"""
Data augmentation transforms for image restoration tasks.

IMPORTANT: Restoration augmentation must apply IDENTICAL transforms
to both input (degraded) and target (clean) pairs.
Standard augmentation: random crop, horizontal/vertical flip, rotation (90°).
Do NOT use color jitter (modifies clean target).

For SNN-specific augmentation:
  - No temporal augmentation needed (static images)
  - Contrast normalization can help with encoding
"""

import torch
import torchvision.transforms.functional as TF
import random
from typing import Tuple, Optional


def get_train_transforms(patch_size: int = 128):
    """Returns transform function for training (degraded, clean) pairs."""
    return TrainTransform(patch_size=patch_size)


def get_val_transforms():
    """Returns transform function for validation (no augmentation)."""
    return ValTransform()


class TrainTransform:
    """
    Augmentation for paired image restoration datasets.

    Applies:
      1. Random crop to patch_size × patch_size
      2. Random horizontal flip
      3. Random vertical flip
      4. Random 90° rotation (×4 variants)

    All operations applied identically to (degraded, clean) pair.
    """

    def __init__(self, patch_size: int = 128):
        self.patch_size = patch_size

    def __call__(
        self, degraded: torch.Tensor, clean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure both have same spatial size
        assert degraded.shape[-2:] == clean.shape[-2:], (
            f"Size mismatch: degraded={degraded.shape}, clean={clean.shape}"
        )

        # Random crop
        _, H, W = degraded.shape
        p = self.patch_size
        if H > p and W > p:
            top = random.randint(0, H - p)
            left = random.randint(0, W - p)
            degraded = degraded[:, top:top+p, left:left+p]
            clean = clean[:, top:top+p, left:left+p]

        # Random horizontal flip
        if random.random() > 0.5:
            degraded = TF.hflip(degraded)
            clean = TF.hflip(clean)

        # Random vertical flip
        if random.random() > 0.5:
            degraded = TF.vflip(degraded)
            clean = TF.vflip(clean)

        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            degraded = torch.rot90(degraded, k, dims=[-2, -1])
            clean = torch.rot90(clean, k, dims=[-2, -1])

        return degraded, clean


class ValTransform:
    """Validation transform: just tensor conversion (no augmentation)."""

    def __call__(
        self, degraded: torch.Tensor, clean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return degraded, clean


class SRTrainTransform:
    """
    Super-resolution train transform.
    crop is on HR; LR is derived by downsampling.

    Args:
        hr_patch_size (int): HR patch size.
        scale (int): Upscale factor.
        degradation (str): 'bicubic' | 'bilinear' (LR generation).
    """

    def __init__(self, hr_patch_size: int = 256, scale: int = 4,
                 degradation: str = "bicubic"):
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.scale = scale
        self.degradation = degradation

    def __call__(
        self, hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hr: (C, H, W) high-resolution image

        Returns:
            (lr, hr_patch): LR and corresponding HR patches
        """
        _, H, W = hr.shape
        p = self.hr_patch_size

        # Random crop on HR
        if H >= p and W >= p:
            top = random.randint(0, H - p)
            left = random.randint(0, W - p)
            hr_patch = hr[:, top:top+p, left:left+p]
        else:
            hr_patch = hr

        # Augmentation
        if random.random() > 0.5:
            hr_patch = TF.hflip(hr_patch)
        if random.random() > 0.5:
            hr_patch = TF.vflip(hr_patch)
        k = random.randint(0, 3)
        if k > 0:
            hr_patch = torch.rot90(hr_patch, k, dims=[-2, -1])

        # Generate LR by downsampling
        import torch.nn.functional as F
        lr = F.interpolate(
            hr_patch.unsqueeze(0),
            scale_factor=1.0 / self.scale,
            mode=self.degradation,
            align_corners=False if self.degradation == "bicubic" else None,
        ).squeeze(0)

        return lr, hr_patch
