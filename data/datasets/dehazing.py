"""Dehazing dataset loader (RESIDE-style)."""
from .base import PairedImageDataset
from typing import Optional, Callable


class HazeDataset(PairedImageDataset):
    """
    Dehazing dataset following RESIDE directory structure.

        root/
            hazy/   ← hazy input images (format: {id}_{beta}_{A}.png)
            gt/     ← clean ground truth images

    Args:
        root: RESIDE dataset root.
        split: 'train' (ITS/OTS) | 'test' (SOTS-indoor or SOTS-outdoor)
        subset: 'indoor' | 'outdoor' | None
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        subset: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        import os
        if subset is not None:
            root = os.path.join(root, subset)
        super().__init__(root, split=split, transform=transform)

    def __repr__(self) -> str:
        return f"HazeDataset(split={self.split}, n_pairs={len(self)})"
