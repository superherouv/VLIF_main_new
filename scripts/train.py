"""
Main training script for SNN Applicability Study.

Usage:
    # Train SNN for deraining
    python scripts/train.py --task deraining --model_type snn --T 4

    # Train ANN baseline
    python scripts/train.py --task deraining --model_type ann

    # Train hybrid
    python scripts/train.py --task denoising --model_type hybrid --T 4

    # Full sweep: all model types for one task
    python scripts/train.py --task deraining --sweep_models
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from tasks import build_task
from training.trainer import Trainer, build_optimizer, build_scheduler
from data.factory import build_dataloader


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ],
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single(args):
    """Train a single model configuration."""
    log_dir = os.path.join(
        args.log_dir, args.task, f"{args.model_type}_T{args.T}"
    )
    setup_logging(log_dir)
    set_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info(f"Training: task={args.task}, model={args.model_type}, T={args.T}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")

    # Build task
    task_kwargs = {}
    if args.task == "denoising":
        task_kwargs["sigma"] = (0, 50) if not hasattr(args, "sigma") else args.sigma
    elif args.task == "superresolution":
        task_kwargs["scale"] = getattr(args, "scale", 4)
    task = build_task(args.task, **task_kwargs)

    # Build model
    in_ch = task.input_channels
    out_ch = task.output_channels

    model_kwargs = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        base_channels=args.base_channels,
        n_levels=args.n_levels,
    )
    if args.model_type in ("snn", "hybrid"):
        model_kwargs.update(
            T=args.T,
            tau=args.tau,
            threshold=args.threshold,
            surrogate=args.surrogate,
        )

    model = build_model(args.model_type, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # Build data loaders
    loader_kwargs = dict(
        task=args.task,
        data_root=args.data_root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )
    if args.task == "superresolution":
        loader_kwargs["scale"] = getattr(args, "scale", 4)
    elif args.task == "denoising":
        loader_kwargs["sigma"] = (0, 50)

    train_loader = build_dataloader(split="train", **loader_kwargs)
    val_loader = build_dataloader(split="test", **loader_kwargs)

    # Build optimizer and scheduler
    opt_cfg = {"name": "adamw", "lr": args.lr, "weight_decay": args.weight_decay}
    optimizer = build_optimizer(model, opt_cfg)
    sched_cfg = {"name": "warmup_cosine", "warmup_epochs": 5, "min_lr": 1e-6}
    scheduler = build_scheduler(optimizer, sched_cfg, args.n_epochs)

    # Build trainer
    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        log_dir=log_dir,
        model_type=args.model_type,
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        save_every=args.save_every,
        val_every=args.val_every,
    )

    logger.info(f"Training complete. Best PSNR: {trainer._best_psnr:.2f} dB")
    return trainer._best_psnr


def main():
    parser = argparse.ArgumentParser(
        description="SNN Applicability Study - Training Script"
    )

    # Task and model
    parser.add_argument("--task", choices=[
        "deraining", "denoising", "superresolution", "dehazing", "lowlight"
    ], required=True)
    parser.add_argument("--model_type", choices=["snn", "ann", "hybrid"],
                        default="snn")
    parser.add_argument("--sweep_models", action="store_true",
                        help="Train all three model types sequentially")

    # Model architecture
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--T", type=int, default=4, help="SNN timesteps")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--surrogate", default="atan",
                        choices=["atan", "sigmoid", "triangle", "multi_gaussian"])

    # Training
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1)

    # Data
    parser.add_argument("--data_root", type=str, default="/data/restoration/")
    parser.add_argument("--num_workers", type=int, default=4)

    # System
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default="experiments/")

    # Task-specific
    parser.add_argument("--scale", type=int, default=4, help="SR scale factor")
    parser.add_argument("--sigma", type=float, default=25.0,
                        help="Noise sigma for fixed denoising")

    args = parser.parse_args()

    if args.sweep_models:
        results = {}
        for model_type in ["snn", "ann", "hybrid"]:
            args.model_type = model_type
            best_psnr = train_single(args)
            results[model_type] = best_psnr
        print("\n=== Sweep Results ===")
        for mt, psnr in results.items():
            print(f"  {mt:10s}: {psnr:.2f} dB")
    else:
        train_single(args)


if __name__ == "__main__":
    main()
