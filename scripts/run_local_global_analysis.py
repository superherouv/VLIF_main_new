"""
Local–Global Frequency Chain Analysis Runner
============================================
验证核心论点：「局部 spike 激活偏好与全局传播偏好不一致」

支持三种运行模式：
  1. 单模型分析（需要 checkpoint + 数据）
  2. 跨模型对比（SNN vs ANN，同一任务）
  3. 跨任务汇总（4 任务 × N 模型 → 论文核心表格）

用法 (所有命令)
--------------

# ── A. 单任务单模型（最快验证，约 2 分钟）──────────────────────────────────
python scripts/run_local_global_analysis.py \\
    --task deraining \\
    --model_type snn \\
    --checkpoint experiments/deraining/snn/ckpt_best.pth \\
    --data_root /data/restoration/ \\
    --save_dir experiments/local_global/

# ── B. 单任务，SNN vs ANN 对比 ────────────────────────────────────────────
python scripts/run_local_global_analysis.py \\
    --task deraining \\
    --model_type snn ann \\
    --checkpoint experiments/deraining/snn/ckpt_best.pth \\
               experiments/deraining/ann/ckpt_best.pth \\
    --data_root /data/restoration/ \\
    --save_dir experiments/local_global/

# ── C. 4 任务全扫（需要 12 个 checkpoint，是论文最终跑法）─────────────────
python scripts/run_local_global_analysis.py \\
    --tasks deraining denoising dehazing super_resolution \\
    --model_types snn ann hybrid \\
    --checkpoint_dir experiments/ \\
    --data_root /data/restoration/ \\
    --save_dir experiments/local_global/ \\
    --cross_task_summary

# ── D. 无 checkpoint 模式（结构验证 / 快速调试，用随机初始化模型）──────────
python scripts/run_local_global_analysis.py \\
    --task deraining \\
    --model_type snn \\
    --no_checkpoint \\
    --save_dir experiments/local_global/

# ── E. 使用模拟数据（完全无数据依赖，用于 CI / 快速验证管道）────────────────
python scripts/run_local_global_analysis.py \\
    --task deraining \\
    --model_type snn \\
    --no_checkpoint --synthetic_data \\
    --save_dir experiments/local_global/
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_model_for_analysis(model_type: str, task_obj, args, device: str):
    """Load or build model, optionally from checkpoint."""
    import torch
    from models import build_model

    kwargs = dict(
        in_channels=task_obj.input_channels,
        out_channels=task_obj.output_channels,
        base_channels=args.base_channels,
        n_levels=args.n_levels,
    )
    if model_type in ("snn", "hybrid"):
        kwargs["T"] = args.T
    model = build_model(model_type=model_type, **kwargs)

    if not args.no_checkpoint and args.checkpoint:
        ckpts = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
        # Match checkpoint to model_type by index or by keyword in filename
        ckpt_path = None
        for c in ckpts:
            if model_type in os.path.basename(os.path.dirname(c)):
                ckpt_path = c
                break
        if ckpt_path is None and len(ckpts) == 1:
            ckpt_path = ckpts[0]

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  Warning: no checkpoint found for {model_type}, using random init")
    else:
        print(f"  Using random-initialized {model_type} model")

    model = model.to(device)
    model.eval()
    return model


def _build_synthetic_loader(task_obj, n_batches: int = 20, img_size: int = 128):
    """Generate synthetic (degraded, clean) pairs for pipeline testing."""
    import torch

    class SyntheticLoader:
        def __init__(self, n, C_in, C_out, H, W):
            self.n = n
            self.C_in = C_in
            self.C_out = C_out
            self.H, self.W = H, W

        def __iter__(self):
            for _ in range(self.n):
                deg = torch.rand(1, self.C_in, self.H, self.W)
                cln = torch.rand(1, self.C_out, self.H, self.W)
                yield deg, cln

        def __len__(self):
            return self.n

    return SyntheticLoader(n_batches,
                           task_obj.input_channels, task_obj.output_channels,
                           img_size, img_size)


def run_single(task_name: str, model_type: str, args, device: str) -> dict:
    """Run local–global analysis for one (task, model_type) pair."""
    import torch
    from tasks import build_task
    from analysis.frequency.local_global_freq_chain import LocalGlobalFreqAnalyzer

    print(f"\n{'─'*60}")
    print(f"  Task: {task_name}  |  Model: {model_type}")
    print(f"{'─'*60}")

    task = build_task(task_name)

    # Data
    if args.synthetic_data:
        loader = _build_synthetic_loader(task, n_batches=args.n_batches)
    else:
        from data.factory import build_dataloader
        loader = build_dataloader(
            task=task_name,
            data_root=args.data_root,
            split="test",
            batch_size=1,
            num_workers=2,
        )

    # Model
    model = _build_model_for_analysis(model_type, task, args, device)

    # Analyzer
    analyzer = LocalGlobalFreqAnalyzer(
        lf_cutoff=0.15,
        hf_cutoff=0.30,
        max_samples=args.max_samples,
    )
    analyzer.attach_hooks(model, model_type)

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.max_samples:
                break
            degraded, clean = (batch[0], batch[1]) if len(batch) >= 2 else batch[:2]
            degraded = degraded.to(device)
            clean = clean.to(device)

            if model_type in ("snn", "hybrid") and hasattr(model, "reset_states"):
                model.reset_states()

            proc = task.preprocess(degraded)
            pred = model(proc["input"])
            if "bicubic_up" in proc:
                pred = task.postprocess(pred, bicubic_up=proc["bicubic_up"])
            pred = pred.clamp(0, 1)

            analyzer.collect_input(degraded.cpu().numpy())
            analyzer.collect_pred_target(pred.cpu().numpy(), clean.cpu().clamp(0, 1).numpy())
            n_done += degraded.shape[0]

    report = analyzer.build_report(task=task_name, model_type=model_type)
    analyzer.print_report(report)

    return report.to_dict()


def run_cross_task(args, device: str) -> None:
    """Run 4×3 matrix and print cross-task summary."""
    from analysis.frequency.local_global_freq_chain import LocalGlobalFreqAnalyzer, LocalGlobalReport
    import json, os

    all_reports: dict[str, dict] = {}
    tasks = args.tasks
    model_types = args.model_types

    for task in tasks:
        for mtype in model_types:
            key = f"{task}/{mtype}"
            try:
                result = run_single(task, mtype, args, device)
                all_reports[key] = result
            except Exception as e:
                print(f"  ERROR {key}: {e}")
                all_reports[key] = {"task": task, "model_type": mtype, "error": str(e)}

    # Save aggregate JSON
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "local_global_cross_task.json")
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nFull results → {out_path}")

    # Cross-task summary table
    print(f"\n{'='*95}")
    print("Cross-Task Local–Global Frequency Discrepancy Summary")
    print(f"{'='*95}")
    print(f"  {'Task':<20} {'Model':>8} │ {'spike HF/LF':>11} {'feat HF/LF':>11} │"
          f" {'HF err':>7} │ {'Peak Disc':>10} │ Conclusion")
    print("  " + "─" * 90)
    for key, r in all_reports.items():
        if "error" in r:
            print(f"  {r['task']:<20} {r['model_type']:>8} │  ERROR: {r['error'][:50]}")
            continue
        conc = r.get("conclusion", "—")[:40]
        print(f"  {r['task']:<20} {r['model_type']:>8} │ "
              f"{r.get('mean_spike_hf_lf', 0):>11.3f} {r.get('mean_feat_hf_lf', 0):>11.3f} │"
              f" {r.get('output_hf_error_ratio', 0):>7.3f} │ "
              f"{r.get('peak_discrepancy', 0):>10.3f} │ {conc}")
    print(f"{'='*95}")
    print("  spike HF/LF > 1.2: local activation favors HF (VLIF claim)")
    print("  feat  HF/LF < 0.8: global propagation is LF-biased (MaxFormer claim)")
    print("  Discrepancy > 0.5: local–global mismatch significant")


def main():
    parser = argparse.ArgumentParser(
        description="Local–Global Frequency Chain Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Task / Model ───────────────────────────────────────────────────────────
    parser.add_argument("--task", default="deraining",
                        help="Single task (used when --tasks not given)")
    parser.add_argument("--tasks", nargs="+",
                        default=["deraining", "denoising", "dehazing", "super_resolution"],
                        help="Task list for cross-task mode")
    parser.add_argument("--model_type", default="snn",
                        help="Single model type (used when --model_types not given)")
    parser.add_argument("--model_types", nargs="+", default=["snn", "ann", "hybrid"],
                        help="Model types for cross-task mode")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="Checkpoint path(s). Multiple: matched by model_type directory name")
    parser.add_argument("--checkpoint_dir", default="experiments/",
                        help="Root dir; auto-discovers experiments/{task}/{model_type}/ckpt_best.pth")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Use random-initialized model (structure debug / quick test)")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_root", default="/data/restoration/")
    parser.add_argument("--synthetic_data", action="store_true",
                        help="Use random synthetic data (no real dataset needed)")
    parser.add_argument("--n_batches", type=int, default=30,
                        help="Number of batches when using synthetic data")

    # ── Model arch ────────────────────────────────────────────────────────────
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--T", type=int, default=4)

    # ── Analysis ──────────────────────────────────────────────────────────────
    parser.add_argument("--max_samples", type=int, default=64,
                        help="Max images to analyse per run")
    parser.add_argument("--cross_task_summary", action="store_true",
                        help="Run full 4×3 matrix and print cross-task table")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--save_dir", default="experiments/local_global/")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print(f"CUDA not available, using CPU")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Auto-discover checkpoints ──────────────────────────────────────────────
    if not args.no_checkpoint and args.checkpoint is None:
        # Try experiments/{task}/{model_type}/ckpt_best.pth
        discovered = []
        for mtype in args.model_types:
            task = args.task
            candidate = os.path.join(args.checkpoint_dir, task, mtype, "ckpt_best.pth")
            if os.path.exists(candidate):
                discovered.append(candidate)
        if discovered:
            args.checkpoint = discovered

    # ── Run ───────────────────────────────────────────────────────────────────
    # Determine which model_types to run:
    # --model_type snn             → [snn]  (explicit single)
    # --model_types snn ann        → [snn, ann]
    # (no flags)                   → [snn]  (default single)
    _explicit_multi = any("--model_types" in a for a in sys.argv)
    if args.cross_task_summary:
        run_cross_task(args, device)
    else:
        # Single task run (possibly multiple model_types for quick compare)
        model_types = args.model_types if _explicit_multi else [args.model_type]
        all_results = {}
        for mtype in model_types:
            result = run_single(args.task, mtype, args, device)
            key = f"{args.task}/{mtype}"
            all_results[key] = result

            out_path = os.path.join(args.save_dir,
                                    f"local_global_{args.task}_{mtype}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Saved → {out_path}")

        if len(all_results) > 1:
            print(f"\n{'='*80}")
            print(f"Same-task comparison: {args.task}")
            print(f"{'='*80}")
            print(f"  {'Model':>8} │ {'spike HF/LF':>11} {'feat HF/LF':>11} │"
                  f" {'HF err':>7} │ {'Peak Disc':>10}")
            print("  " + "─" * 55)
            for key, r in all_results.items():
                mtype = key.split("/")[1]
                print(f"  {mtype:>8} │ "
                      f"{r.get('mean_spike_hf_lf',0):>11.3f} {r.get('mean_feat_hf_lf',0):>11.3f} │"
                      f" {r.get('output_hf_error_ratio',0):>7.3f} │ "
                      f"{r.get('peak_discrepancy',0):>10.3f}")


if __name__ == "__main__":
    main()
