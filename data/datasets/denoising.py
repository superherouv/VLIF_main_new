"""
Denoising Datasets.

Gaussian denoising: noise added on-the-fly during training.
Real noise: load pre-captured noisy/clean pairs.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import glob
import random
import numpy as np
from typing import Optional, Callable, Tuple, List, Union
from .base import PairedImageDataset


class NoisyDataset(Dataset):
    """
    Gaussian denoising dataset with on-the-fly noise synthesis.

    Supports:
      - Fixed noise level (σ = const): classic CBSD68 evaluation
      - Blind noise (σ ∈ [σ_min, σ_max]): DnCNN-blind training

    Args:
        root: Path to clean images.
        split: 'train' | 'val' | 'test'
        sigma: Fixed sigma OR (sigma_min, sigma_max) for blind
        transform: Paired transform
        patch_size: Crop size for training
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        sigma: Union[float, Tuple[float, float]] = 25.0,
        transform: Optional[Callable] = None,
        patch_size: int = 128,
        color: bool = True,
    ):
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.color = color

        if isinstance(sigma, (int, float)):
            self.sigma_fixed = float(sigma)
            self.sigma_range = None
        else:
            self.sigma_fixed = None
            self.sigma_range = sigma  # (min, max)

        # Load clean image paths
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.clean_paths: List[str] = []
        for ext in extensions:
            self.clean_paths.extend(
                glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
            )
        self.clean_paths = sorted(self.clean_paths)

        n = len(self.clean_paths)
        if split == "train":
            self.clean_paths = self.clean_paths[:int(0.9 * n)]
        elif split == "val":
            self.clean_paths = self.clean_paths[int(0.9 * n):]

    def _sample_sigma(self) -> float:
        if self.sigma_fixed is not None:
            return self.sigma_fixed
        return random.uniform(self.sigma_range[0], self.sigma_range[1])

    def _add_noise(self, clean: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add AWGN: noise ~ N(0, σ/255)."""
        noise = torch.randn_like(clean) * (sigma / 255.0)
        return (clean + noise).clamp(0, 1)

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Returns (noisy, clean, sigma)."""
        img = Image.open(self.clean_paths[idx])
        if self.color:
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        clean = TF.to_tensor(img)

        sigma = self._sample_sigma()
        noisy = self._add_noise(clean, sigma)

        if self.transform is not None:
            noisy, clean = self.transform(noisy, clean)

        return noisy, clean, sigma


class SIDDDataset(PairedImageDataset):
    """
    SIDD (Smartphone Image Denoising Dataset) loader.

    SIDD-Small contains 1280 noisy/GT pairs (320×320 patches).

    Structure:
        root/
            input_crops/    ← noisy patches
            gt_crops/       ← clean patches
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, split=split, transform=transform)

    def __repr__(self) -> str:
        return f"SIDDDataset(split={self.split}, n_pairs={len(self)})"
