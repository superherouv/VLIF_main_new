"""
Deraining Datasets.

Supports:
  - Rain100H / Rain100L (Yang et al., 2017): most cited benchmarks
  - Rain1400 (Fu et al., 2017): 14 rain types × 100 images
  - SPA-Data (Wang et al., 2019): real-world rain from video
  - DID-MDN (Zhang et al., 2018): density-classified rain

Dataset paths (configure in configs/):
    Rain100H:  {data_root}/Rain100H/{train,test}/{input,target}/
    Rain100L:  {data_root}/Rain100L/{train,test}/{input,target}/
    SPA-Data:  {data_root}/SPA-Data/{rain,gt}/
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import glob
from typing import Optional, Callable, List, Tuple
from .base import PairedImageDataset


class RainDataset(PairedImageDataset):
    """
    Paired rain removal dataset.

    Rain100H naming: rain-{N:04d}.png ↔ norain-{N:04d}.png
    Rain100L naming: same convention
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        dataset_name: str = "Rain100H",
        transform: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name
        data_root = os.path.join(root, dataset_name)
        if not os.path.isdir(data_root):
            data_root = root  # direct path provided

        super().__init__(data_root, split=split, transform=transform)

    def __repr__(self) -> str:
        return (f"RainDataset(dataset={self.dataset_name}, "
                f"split={self.split}, n_pairs={len(self)})")


class SyntheticRainDataset(Dataset):
    """
    Synthetic rain dataset with on-the-fly rain generation.
    For ablation studies where real rain datasets are not available.

    Generates synthetic rain streaks on clean images.
    """

    def __init__(
        self,
        clean_root: str,
        transform: Optional[Callable] = None,
        rain_density: str = "medium",
        split: str = "train",
    ):
        """
        Args:
            clean_root: Path to clean images.
            rain_density: 'light' | 'medium' | 'heavy'
            split: 'train' | 'test'
        """
        self.clean_root = clean_root
        self.transform = transform
        self.rain_params = {
            "light": {"n_streaks": 50, "length_range": (10, 30), "opacity": 0.3},
            "medium": {"n_streaks": 200, "length_range": (20, 60), "opacity": 0.5},
            "heavy": {"n_streaks": 500, "length_range": (30, 100), "opacity": 0.7},
        }[rain_density]

        extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.clean_paths = []
        for ext in extensions:
            self.clean_paths.extend(
                glob.glob(os.path.join(clean_root, f"**/*{ext}"), recursive=True)
            )
        self.clean_paths = sorted(self.clean_paths)

        # Train/test split
        n = len(self.clean_paths)
        if split == "train":
            self.clean_paths = self.clean_paths[:int(0.9 * n)]
        else:
            self.clean_paths = self.clean_paths[int(0.9 * n):]

    def _generate_rain(self, clean: torch.Tensor) -> torch.Tensor:
        """Generate synthetic rain on clean image."""
        import math, random
        _, H, W = clean.shape
        rain = clean.clone()

        n = self.rain_params["n_streaks"]
        min_len, max_len = self.rain_params["length_range"]
        opacity = self.rain_params["opacity"]

        for _ in range(n):
            # Random streak position and angle
            x0 = random.randint(0, W - 1)
            y0 = random.randint(0, H - 1)
            length = random.randint(min_len, max_len)
            angle = random.uniform(-30, 30)  # degrees from vertical

            dx = math.sin(math.radians(angle))
            dy = -math.cos(math.radians(angle))

            for t in range(length):
                x = int(x0 + t * dx)
                y = int(y0 + t * dy)
                if 0 <= x < W and 0 <= y < H:
                    rain[:, y, x] = rain[:, y, x] * (1 - opacity) + opacity

        return rain.clamp(0, 1)

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = TF.to_tensor(Image.open(self.clean_paths[idx]).convert("RGB"))
        rain = self._generate_rain(clean)

        if self.transform is not None:
            rain, clean = self.transform(rain, clean)

        return rain, clean
