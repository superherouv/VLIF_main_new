"""
Dataset factory: build DataLoaders from config.
"""

import torch
from torch.utils.data import DataLoader, random_split
from .transforms import get_train_transforms, get_val_transforms, SRTrainTransform
from .datasets.deraining import RainDataset, SyntheticRainDataset
from .datasets.denoising import NoisyDataset, SIDDDataset
from .datasets.superresolution import SRDataset
from .datasets.dehazing import HazeDataset
from .datasets.lowlight import LowLightDataset


def build_dataloader(
    task: str,
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    patch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    **task_kwargs,
) -> DataLoader:
    """
    Build a DataLoader for the given task.

    Args:
        task: One of 'deraining', 'denoising', 'superresolution', 'dehazing', 'lowlight'.
        data_root: Root path for the dataset.
        split: 'train' | 'val' | 'test'
        batch_size: Mini-batch size.
        patch_size: Crop size for training.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        **task_kwargs: Task-specific kwargs (e.g., sigma for denoising, scale for SR).

    Returns:
        DataLoader instance
    """
    is_train = (split == "train")
    transform = get_train_transforms(patch_size) if is_train else get_val_transforms()

    if task == "deraining":
        dataset_name = task_kwargs.get("dataset_name", "Rain100H")
        dataset = RainDataset(data_root, split=split,
                              dataset_name=dataset_name, transform=transform)

    elif task == "denoising":
        sigma = task_kwargs.get("sigma", (0, 50) if is_train else 25.0)
        dataset = NoisyDataset(data_root, split=split,
                               sigma=sigma, transform=transform,
                               patch_size=patch_size)

    elif task == "superresolution":
        scale = task_kwargs.get("scale", 4)
        hr_patch = task_kwargs.get("hr_patch_size", 256)
        dataset = SRDataset(data_root, scale=scale, split=split,
                            hr_patch_size=hr_patch)

    elif task == "dehazing":
        subset = task_kwargs.get("subset", None)
        dataset = HazeDataset(data_root, split=split, subset=subset, transform=transform)

    elif task == "lowlight":
        dataset = LowLightDataset(data_root, split=split, transform=transform)

    else:
        raise ValueError(f"Unknown task '{task}'")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=(num_workers > 0),
    )
    return loader
