"""
Ablation study runner for SNN applicability research.

Runs the key ablation experiments:
  1. T sweep: PSNR vs timesteps (T=1,2,4,8,16)
  2. Surrogate gradient comparison
  3. Tau value comparison
  4. Model type comparison: SNN vs ANN vs Hybrid
  5. Encoding scheme comparison

This script generates the data needed for all figures in the paper.
"""

import os
import sys
import json
import itertools
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TASKS = ["deraining", "denoising", "superresolution", "dehazing", "lowlight"]
MODEL_TYPES = ["snn", "ann", "hybrid"]
T_VALUES = [1, 2, 4, 8, 16]
SURROGATE_TYPES = ["atan", "sigmoid", "triangle", "multi_gaussian"]
TAU_VALUES = [0.25, 0.5, 0.75]


def run_experiment(cmd: list, dry_run: bool = False) -> int:
    """Run a training command."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def ablation_model_types(
    task: str, data_root: str, base_args: list, dry_run: bool = False
):
    """Ablation 1: Compare SNN vs ANN vs Hybrid for each task."""
    print(f"\n[Ablation] Model types for task={task}")
    for model_type in MODEL_TYPES:
        cmd = base_args + [
            "--task", task,
            "--model_type", model_type,
            "--T", "4",
            "--data_root", data_root,
            "--log_dir", f"experiments/ablation/model_types/{task}/",
        ]
        run_experiment(cmd, dry_run)


def ablation_timesteps(
    task: str, data_root: str, base_args: list, dry_run: bool = False
):
    """Ablation 2: PSNR vs T for SNN."""
    print(f"\n[Ablation] Timesteps for task={task}")
    for T in T_VALUES:
        cmd = base_args + [
            "--task", task,
            "--model_type", "snn",
            "--T", str(T),
            "--data_root", data_root,
            "--log_dir", f"experiments/ablation/timesteps/{task}/T{T}/",
            "--n_epochs", "100",  # shorter for ablation
        ]
        run_experiment(cmd, dry_run)


def ablation_surrogate(
    task: str, data_root: str, base_args: list, dry_run: bool = False
):
    """Ablation 3: Surrogate gradient comparison."""
    print(f"\n[Ablation] Surrogate gradients for task={task}")
    for surr in SURROGATE_TYPES:
        cmd = base_args + [
            "--task", task,
            "--model_type", "snn",
            "--T", "4",
            "--surrogate", surr,
            "--data_root", data_root,
            "--log_dir", f"experiments/ablation/surrogate/{task}/{surr}/",
            "--n_epochs", "100",
        ]
        run_experiment(cmd, dry_run)


def ablation_tau(
    task: str, data_root: str, base_args: list, dry_run: bool = False
):
    """Ablation 4: Membrane time constant (tau) comparison."""
    print(f"\n[Ablation] Tau values for task={task}")
    for tau in TAU_VALUES:
        cmd = base_args + [
            "--task", task,
            "--model_type", "snn",
            "--T", "4",
            "--tau", str(tau),
            "--data_root", data_root,
            "--log_dir", f"experiments/ablation/tau/{task}/tau{tau}/",
            "--n_epochs", "100",
        ]
        run_experiment(cmd, dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="SNN Applicability Study - Ablation Runner"
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        help="Tasks to ablate")
    parser.add_argument("--ablations", nargs="+",
                        default=["model_types", "timesteps", "surrogate", "tau"],
                        help="Ablation types to run")
    parser.add_argument("--data_root", required=True,
                        help="Dataset root directory")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_script = os.path.join(os.path.dirname(__file__), "train.py")
    base_args = [
        "python", train_script,
        "--n_epochs", str(args.n_epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--seed", str(args.seed),
    ]

    total_runs = len(args.tasks) * len(args.ablations)
    print(f"\nSNN Applicability Ablation Study")
    print(f"Tasks: {args.tasks}")
    print(f"Ablations: {args.ablations}")
    print(f"Estimated runs: {total_runs}")

    for task in args.tasks:
        task_data_root = os.path.join(args.data_root, task)

        if "model_types" in args.ablations:
            ablation_model_types(task, task_data_root, base_args, args.dry_run)

        if "timesteps" in args.ablations:
            ablation_timesteps(task, task_data_root, base_args, args.dry_run)

        if "surrogate" in args.ablations:
            ablation_surrogate(task, task_data_root, base_args, args.dry_run)

        if "tau" in args.ablations:
            ablation_tau(task, task_data_root, base_args, args.dry_run)

    print("\n[Done] All ablation experiments queued/completed.")
    if args.dry_run:
        print("(DRY RUN - no experiments were actually run)")


if __name__ == "__main__":
    main()
