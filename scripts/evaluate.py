"""
Evaluation script with energy profiling.

Usage:
    # Evaluate SNN on deraining
    python scripts/evaluate.py \
        --task deraining \
        --model_type snn \
        --checkpoint experiments/deraining/snn_T4/ckpt_best.pth \
        --data_root /data/restoration/Rain100H/

    # Evaluate all checkpoints and generate applicability report
    python scripts/evaluate.py --generate_report \
        --experiments_dir experiments/
"""

import os
import sys
import argparse
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from tasks import build_task
from tasks.metrics import MetricCalculator
from data.factory import build_dataloader
from training.energy_monitor import EnergyMonitor
from analysis.applicability import ApplicabilityAnalyzer
from analysis.spike_analysis import SpikeAnalyzer
from analysis.visualizer import ApplicabilityVisualizer


def evaluate_single(args) -> dict:
    """Evaluate a single model checkpoint."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_type = ckpt.get("model_type", args.model_type)
    task_name = ckpt.get("task", args.task)

    # Build task and model
    task = build_task(task_name)
    model = build_model(
        model_type,
        in_channels=task.input_channels,
        out_channels=task.output_channels,
        base_channels=args.base_channels,
        n_levels=args.n_levels,
        T=args.T,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Build test loader
    test_loader = build_dataloader(
        task=task_name,
        data_root=args.data_root,
        split="test",
        batch_size=1,  # eval one at a time
        num_workers=2,
    )

    # Energy monitor
    monitor = EnergyMonitor(model, model_type=model_type, task=task_name)
    monitor.attach_hooks()

    # Spike analyzer
    if model_type in ("snn", "hybrid"):
        spike_analyzer = SpikeAnalyzer(model)
        spike_analyzer.attach_hooks(model)

    # Metrics calculator
    calc = MetricCalculator(task_name)

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                degraded, clean = batch
            else:
                degraded, clean = batch[0], batch[1]

            degraded = degraded.to(device)
            clean = clean.to(device)

            # Reset SNN states
            if model_type in ("snn", "hybrid") and hasattr(model, "reset_states"):
                model.reset_states()

            proc = task.preprocess(degraded)
            pred = model(proc["input"])
            if "bicubic_up" in proc:
                pred = task.postprocess(pred, bicubic_up=proc["bicubic_up"])
            pred = pred.clamp(0, 1)

            # Energy stats
            energy_stats = model.get_energy_stats() if hasattr(model, "get_energy_stats") else {}
            firing_rate = energy_stats.get("mean_firing_rate")

            calc.update(pred, clean.clamp(0, 1), firing_rate=firing_rate)

    monitor.remove_hooks()
    if model_type in ("snn", "hybrid"):
        spike_analyzer.remove_hooks()

    # Compute energy profile
    input_shape = (1, task.input_channels, 256, 256)
    profile = monitor.compute_profile(psnr=calc.compute().get("psnr", 0))
    monitor.print_report(profile)

    metrics = calc.compute()
    print(f"\n{calc.summary()}")

    return {
        "task": task_name,
        "model_type": model_type,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "energy": {
            "total_mac_ops": profile.total_mac_ops,
            "total_synops": profile.total_synops,
            "mean_firing_rate": profile.mean_firing_rate,
            "energy_ratio": profile.energy_ratio_vs_ann,
        },
        "T": args.T,
    }


def generate_applicability_report(results_dir: str, save_path: str):
    """
    Aggregate all evaluation results and generate applicability report.

    Args:
        results_dir: Directory containing eval_results.json files per experiment.
        save_path: Path to save the final report.
    """
    analyzer = ApplicabilityAnalyzer()
    visualizer = ApplicabilityVisualizer(save_dir=os.path.join(results_dir, "figures"))

    # Load all results
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname == "eval_results.json":
                with open(os.path.join(root, fname)) as f:
                    result = json.load(f)
                task = result["task"]
                model_type = result["model_type"]
                metrics = result.get("metrics", {})
                energy = result.get("energy", {})

                analyzer.add_result(
                    task=task,
                    model_type=model_type,
                    psnr=metrics.get("psnr", metrics.get("psnr_y", 0.0)),
                    ssim=metrics.get("ssim", metrics.get("ssim_y", 0.0)),
                    mean_firing_rate=energy.get("mean_firing_rate", 0.0),
                    energy_ratio=energy.get("energy_ratio", 1.0),
                    synops=energy.get("total_synops", 0.0),
                    T=result.get("T", 4),
                )

    # Compute and print report
    scored = analyzer.compute_applicability_scores()
    analyzer.print_report()
    analyzer.save_report(save_path)

    # Generate visualizations (requires matplotlib)
    try:
        visualizer.plot_applicability_radar(scored)
        visualizer.plot_applicability_heatmap(scored)
        print(f"\nVisualizations saved to: {visualizer.save_dir}")
    except ImportError:
        print("matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Visualization error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SNN Applicability Study - Evaluation Script"
    )
    parser.add_argument("--task", default="deraining")
    parser.add_argument("--model_type", default="snn",
                        choices=["snn", "ann", "hybrid"])
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint path")
    parser.add_argument("--data_root", default="/data/restoration/")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="experiments/eval/")

    # Report generation
    parser.add_argument("--generate_report", action="store_true",
                        help="Aggregate all results and generate applicability report")
    parser.add_argument("--experiments_dir", default="experiments/",
                        help="Directory containing all experiment results")

    args = parser.parse_args()

    if args.generate_report:
        report_path = os.path.join(args.experiments_dir, "applicability_report.json")
        generate_applicability_report(args.experiments_dir, report_path)
    elif args.checkpoint:
        os.makedirs(args.save_dir, exist_ok=True)
        result = evaluate_single(args)
        # Save
        out_path = os.path.join(args.save_dir, "eval_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
