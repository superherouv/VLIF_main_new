"""Low-light enhancement dataset loader (LOL-style)."""
from .base import PairedImageDataset
from typing import Optional, Callable


class LowLightDataset(PairedImageDataset):
    """
    Low-light enhancement dataset (LOL, VE-LOL, LSRW, etc.).

    Expected structure:
        root/
            low/    ← dark input images
            high/   ← normal-light reference images
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, split=split, transform=transform)

    def __repr__(self) -> str:
        return f"LowLightDataset(split={self.split}, n_pairs={len(self)})"
