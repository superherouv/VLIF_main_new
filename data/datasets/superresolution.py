"""Super-resolution dataset loader."""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import glob
import random
from typing import Optional, Callable, Tuple, List


class SRDataset(Dataset):
    """
    Super-resolution dataset. LR generated on-the-fly from HR.

    Supported: DIV2K, Flickr2K, Set5, Set14, BSD100, Urban100

    Args:
        root: HR image directory.
        scale: Upscale factor (2, 3, or 4).
        split: 'train' | 'val' | 'test'
        hr_patch_size: HR crop size during training.
        degradation: 'bicubic' | 'bilinear' (LR generation method)
    """

    def __init__(
        self,
        root: str,
        scale: int = 4,
        split: str = "train",
        hr_patch_size: int = 256,
        degradation: str = "bicubic",
        transform: Optional[Callable] = None,
    ):
        self.scale = scale
        self.split = split
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.degradation = degradation
        self.transform = transform

        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.hr_paths: List[str] = []
        for ext in extensions:
            self.hr_paths.extend(
                glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
            )
        self.hr_paths = sorted(self.hr_paths)

        n = len(self.hr_paths)
        if split == "train":
            self.hr_paths = self.hr_paths[:int(0.9 * n)]
        else:
            self.hr_paths = self.hr_paths[int(0.9 * n):]

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (lr, hr)."""
        hr = TF.to_tensor(Image.open(self.hr_paths[idx]).convert("RGB"))
        _, H, W = hr.shape

        if self.split == "train":
            # Random HR crop
            p = self.hr_patch_size
            if H >= p and W >= p:
                top = random.randint(0, H - p)
                left = random.randint(0, W - p)
                hr = hr[:, top:top+p, left:left+p]

            # Augment
            if random.random() > 0.5:
                hr = TF.hflip(hr)
            if random.random() > 0.5:
                hr = TF.vflip(hr)
            k = random.randint(0, 3)
            if k > 0:
                hr = torch.rot90(hr, k, dims=[-2, -1])

        # Generate LR
        mode = "bicubic" if self.degradation == "bicubic" else "bilinear"
        align = False if mode == "bicubic" else None
        lr = F.interpolate(
            hr.unsqueeze(0), scale_factor=1.0/self.scale,
            mode=mode,
            align_corners=align,
        ).squeeze(0).clamp(0, 1)

        return lr, hr
