"""Base dataset class for paired image restoration datasets."""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Callable, List, Tuple
import glob


class PairedImageDataset(Dataset):
    """
    Generic paired image dataset for (degraded, clean) pairs.

    Directory structure expected:
        root/
            input/   ← degraded images
            target/  ← clean reference images

    Or:
        root/
            train/
                input/
                target/
            test/
                input/
                target/

    Args:
        root (str): Path to dataset root.
        split (str): 'train' | 'val' | 'test'.
        transform (callable): Transform applied to (degraded, clean) pair.
        extensions (tuple): Valid image extensions.
        pair_by_name (bool): If True, match by filename; else by sorted order.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'),
        pair_by_name: bool = True,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.extensions = extensions

        self.degraded_paths, self.clean_paths = self._load_paths(pair_by_name)

        if len(self.degraded_paths) == 0:
            raise RuntimeError(
                f"No images found in {root} for split='{split}'.\n"
                f"Expected structure: {root}/{{input,target}}/ or "
                f"{root}/{split}/{{input,target}}/"
            )

    def _find_images(self, folder: str) -> List[str]:
        """Find all images in folder with valid extensions."""
        paths = []
        for ext in self.extensions:
            paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
            paths.extend(glob.glob(os.path.join(folder, f"**/*{ext.upper()}"), recursive=True))
        return sorted(set(paths))

    def _load_paths(self, pair_by_name: bool) -> Tuple[List[str], List[str]]:
        """Discover degraded/clean image pairs."""
        # Try split-specific subdirectory first
        split_root = os.path.join(self.root, self.split)
        if os.path.isdir(split_root):
            base = split_root
        else:
            base = self.root

        # Try various naming conventions
        input_candidates = ["input", "degraded", "rain", "hazy", "low", "noisy", "lr"]
        target_candidates = ["target", "clean", "gt", "sharp", "clear", "high", "hr"]

        input_dir = None
        target_dir = None
        for name in input_candidates:
            path = os.path.join(base, name)
            if os.path.isdir(path):
                input_dir = path
                break
        for name in target_candidates:
            path = os.path.join(base, name)
            if os.path.isdir(path):
                target_dir = path
                break

        if input_dir is None or target_dir is None:
            # Last resort: assume first two subdirs are (input, target)
            subdirs = sorted([
                d for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d))
            ])
            if len(subdirs) >= 2:
                input_dir = os.path.join(base, subdirs[0])
                target_dir = os.path.join(base, subdirs[1])

        if input_dir is None:
            return [], []

        degraded = self._find_images(input_dir)
        clean = self._find_images(target_dir)

        if pair_by_name:
            # Match by filename stem
            clean_map = {os.path.splitext(os.path.basename(p))[0]: p for p in clean}
            paired_deg, paired_clean = [], []
            for dp in degraded:
                stem = os.path.splitext(os.path.basename(dp))[0]
                # Try exact match, then prefix match
                if stem in clean_map:
                    paired_deg.append(dp)
                    paired_clean.append(clean_map[stem])
            if len(paired_deg) > 0:
                return paired_deg, paired_clean

        # Fall back to sorted order pairing
        min_len = min(len(degraded), len(clean))
        return degraded[:min_len], clean[:min_len]

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image as float tensor (C, H, W) in [0, 1]."""
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)  # (3, H, W), [0, 1]

    def __len__(self) -> int:
        return len(self.degraded_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        deg = self._load_image(self.degraded_paths[idx])
        clean = self._load_image(self.clean_paths[idx])

        if self.transform is not None:
            deg, clean = self.transform(deg, clean)

        return deg, clean

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"split={self.split}, n_pairs={len(self)})")
