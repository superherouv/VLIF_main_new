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
from analysis.frequency.band_analysis import FrequencyBandAnalyzer
from analysis.frequency.error_spectrum import ErrorSpectrumAnalyzer
from analysis.frequency.wavelet import WaveletDecomposer
from analysis.representation.cka import CKAAnalyzer
from analysis.representation.manifold import ManifoldAnalyzer
from analysis.representation.membrane_phase import MembranePhaseSpace


def run_deep_analysis(
    preds: list,          # list of np.ndarray (C,H,W) [0,1]
    targets: list,        # list of np.ndarray (C,H,W) [0,1]
    bottleneck_feats: list,  # list of np.ndarray (C,H,W) flattened per sample
    membrane_records: dict,  # {layer_name: np.ndarray (T,B,C,H,W)} — may be empty
    model_type: str,
    task: str,
) -> dict:
    """
    Run 6-metric deep analysis on collected predictions.

    Metrics:
      1. Band-wise PSNR  (FrequencyBandAnalyzer)
      2. Error spectrum   (ErrorSpectrumAnalyzer)
      3. Wavelet subband  (WaveletDecomposer)      — requires SNN+ANN pair; skipped if only one model
      4. CKA              (CKAAnalyzer)            — within-model, bottleneck vs mean prediction
      5. Effective rank   (ManifoldAnalyzer)       — bottleneck features
      6. Membrane potential histogram (MembranePhaseSpace) — SNN/hybrid only
    """
    import numpy as np
    results = {}
    n = len(preds)
    if n == 0:
        return results

    # ── 1. Band-wise PSNR ────────────────────────────────────────────────────
    band_analyzer = FrequencyBandAnalyzer(n_bands=16)
    band_profiles = [band_analyzer.analyze_pair(p, t, model_type, task)
                     for p, t in zip(preds, targets)]
    # Average per-band PSNR across images
    n_bands = len(band_profiles[0].bands)
    band_psnr_avg = [
        float(sum(bp.bands[i].psnr for bp in band_profiles) / n)
        for i in range(n_bands)
    ]
    first = band_profiles[0]
    results["band_psnr"] = {
        "low_freq_psnr":  float(sum(b.low_freq_psnr  for b in band_profiles) / n),
        "mid_freq_psnr":  float(sum(b.mid_freq_psnr  for b in band_profiles) / n),
        "high_freq_psnr": float(sum(b.high_freq_psnr for b in band_profiles) / n),
        "hf_lf_ratio":    float(sum(b.hf_lf_ratio    for b in band_profiles) / n),
        "per_band_psnr":  band_psnr_avg,
        "freq_ranges":    [(b.freq_range[0], b.freq_range[1]) for b in first.bands],
    }

    # ── 2. Error spectrum ────────────────────────────────────────────────────
    err_analyzer = ErrorSpectrumAnalyzer()
    err_stats_list = [err_analyzer.analyze(p, t, model_type, task)
                      for p, t in zip(preds, targets)]
    results["error_spectrum"] = {
        "spectral_centroid": float(sum(s.spectral_centroid for s in err_stats_list) / n),
        "spectral_flatness": float(sum(s.spectral_flatness for s in err_stats_list) / n),
        "hf_error_ratio":    float(sum(s.hf_error_ratio    for s in err_stats_list) / n),
        "lf_error_ratio":    float(sum(s.lf_error_ratio    for s in err_stats_list) / n),
        "anisotropy_index":  float(sum(s.anisotropy_index  for s in err_stats_list) / n),
        "error_sparsity":    float(sum(s.error_sparsity    for s in err_stats_list) / n),
        "mean_error":        float(sum(s.mean_error        for s in err_stats_list) / n),
    }

    # ── 3. Wavelet subband error ─────────────────────────────────────────────
    # WaveletDecomposer.analyze() requires both SNN and ANN preds.
    # When a single model is evaluated we compare against a bicubic baseline (target itself
    # used as "ANN reference") so the ratio is always 1.  Cross-model comparison is done
    # in generate_applicability_report by loading two saved result JSONs.
    # We still run it so the subband structure is populated.
    wavelet = WaveletDecomposer(n_levels=3)
    wavelet_results = [wavelet.analyze(p, t, t) for p, t in zip(preds, targets)]
    results["wavelet_subband"] = {
        "ll_ratio": float(sum(r.ll_ratio for r in wavelet_results) / n),
        "lh_ratio": float(sum(r.lh_ratio for r in wavelet_results) / n),
        "hl_ratio": float(sum(r.hl_ratio for r in wavelet_results) / n),
        "hh_ratio": float(sum(r.hh_ratio for r in wavelet_results) / n),
        "worst_subband": wavelet_results[0].worst_subband,
        "best_subband":  wavelet_results[0].best_subband,
        "note": "SNN/ANN ratio vs target; cross-model comparison needs two result files",
    }

    # ── 4. CKA (within-model: bottleneck feat vs mean-pooled pred) ───────────
    if len(bottleneck_feats) >= 4:
        import numpy as np
        cka_analyzer = CKAAnalyzer()
        # X: bottleneck features (N, D_bottleneck)
        X = np.stack([f.flatten() for f in bottleneck_feats])
        # Y: prediction vectors (N, D_pred)
        Y = np.stack([p.flatten() for p in preds])
        # Subsample to 256 dims for tractability
        if X.shape[1] > 256:
            idx = np.random.choice(X.shape[1], 256, replace=False)
            X = X[:, idx]
        if Y.shape[1] > 256:
            idx = np.random.choice(Y.shape[1], 256, replace=False)
            Y = Y[:, idx]
        cka_result = cka_analyzer.compute_cka(X, Y, "bottleneck", "output")
        results["cka"] = {
            "bottleneck_vs_output": cka_result.cka_score,
            "hsic_ab": cka_result.hsic_ab,
            "n_samples": cka_result.n_samples,
        }
    else:
        results["cka"] = {"note": "too few samples for CKA (need >= 4)"}

    # ── 5. Effective rank (bottleneck features) ───────────────────────────────
    if len(bottleneck_feats) >= 4:
        import numpy as np
        manifold = ManifoldAnalyzer()
        X = np.stack([f.flatten() for f in bottleneck_feats])
        mstats = manifold.analyze_layer(X, labels=None,
                                         layer_name="bottleneck",
                                         model_type=model_type)
        results["effective_rank"] = {
            "bottleneck_effective_rank": mstats.effective_rank,
            "participation_ratio":       mstats.participation_ratio,
            "pca_dim_90":                mstats.pca_dim_90,
            "total_variance":            mstats.total_variance,
        }
    else:
        results["effective_rank"] = {"note": "too few samples (need >= 4)"}

    # ── 6. Membrane potential histogram ──────────────────────────────────────
    if membrane_records:
        mem_analyzer = MembranePhaseSpace(threshold=1.0)
        mem_stats = mem_analyzer.analyze(membrane_records)
        results["membrane_histogram"] = {}
        for layer_name, stats in mem_stats.items():
            entry = {
                "mean_v": stats.mean_v,
                "std_v":  stats.std_v,
                "threshold_crossing_rate": stats.threshold_crossing_rate,
                "spike_efficiency":        stats.spike_efficiency,
            }
            if stats.v_hist_counts is not None:
                entry["hist_counts"] = stats.v_hist_counts.tolist()
                entry["hist_edges"]  = stats.v_hist_edges.tolist()
            results["membrane_histogram"][layer_name] = entry
    else:
        if model_type in ("snn", "hybrid"):
            results["membrane_histogram"] = {
                "note": "no membrane records collected; attach LIF hooks before forward"
            }

    return results


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

    # ── Deep analysis collectors ─────────────────────────────────────────────
    MAX_ANALYSIS_SAMPLES = 100   # cap for tractability
    collected_preds   = []       # (C,H,W) numpy [0,1]
    collected_targets = []
    collected_bottleneck = []    # flattened bottleneck feature per sample

    # Hook bottleneck features (works for all model types)
    _bottleneck_buf = []
    def _bottleneck_hook(module, input, output):
        if len(collected_bottleneck) < MAX_ANALYSIS_SAMPLES:
            # output shape: (B, C, H, W) or (T,B,C,H,W) — take first/mean
            t = output.detach().cpu().float()
            if t.ndim == 5:
                t = t.mean(0)   # (B,C,H,W)
            collected_bottleneck.append(t[0].numpy())  # first sample in batch

    # Attach to the model's bottleneck submodule (naming differs by model type)
    _hook_handle = None
    for name, submod in model.named_modules():
        if "bottleneck" in name and hasattr(submod, "forward_step"):
            # SNNConvBlock — hook its second conv output via neuron2
            for sname, ssub in submod.named_modules():
                if "neuron2" in sname or (sname == "" and hasattr(submod, "neuron2")):
                    pass
            # Simpler: hook the SNNConvBlock itself
            _hook_handle = submod.register_forward_hook(
                lambda m, i, o: collected_bottleneck.append(
                    o.detach().cpu().float()[0].numpy()
                ) if len(collected_bottleneck) < MAX_ANALYSIS_SAMPLES else None
            )
            break
        elif "bottleneck" in name and isinstance(submod, torch.nn.Conv2d):
            _hook_handle = submod.register_forward_hook(
                lambda m, i, o: collected_bottleneck.append(
                    o.detach().cpu().float()[0].numpy()
                ) if len(collected_bottleneck) < MAX_ANALYSIS_SAMPLES else None
            )
            break

    # Collect membrane potentials for SNN/hybrid (hook LIF neurons)
    _membrane_bufs = {}   # {layer_name: list of (T,B,C,H,W)}
    _membrane_hooks = []
    if model_type in ("snn", "hybrid"):
        from models.snn.neurons import LIFNeuron
        for name, submod in model.named_modules():
            if isinstance(submod, LIFNeuron):
                capture_name = name  # closure capture
                def _make_mem_hook(lname):
                    def _hook(m, inp, out):
                        if m.membrane is not None and lname not in _membrane_bufs:
                            v = m.membrane.detach().cpu().float().numpy()
                            # Shape may be (B,C,H,W); wrap to (1,B,C,H,W) for T dim
                            _membrane_bufs[lname] = v[None] if v.ndim == 4 else v
                    return _hook
                h = submod.register_forward_hook(_make_mem_hook(capture_name))
                _membrane_hooks.append(h)

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

            # Collect for deep analysis
            if len(collected_preds) < MAX_ANALYSIS_SAMPLES:
                collected_preds.append(pred[0].detach().cpu().float().numpy())
                collected_targets.append(clean[0].detach().cpu().float().clamp(0, 1).numpy())

            # Energy stats
            energy_stats = model.get_energy_stats() if hasattr(model, "get_energy_stats") else {}
            firing_rate = energy_stats.get("mean_firing_rate")

            calc.update(pred, clean.clamp(0, 1), firing_rate=firing_rate)

    # Remove hooks
    if _hook_handle is not None:
        _hook_handle.remove()
    for h in _membrane_hooks:
        h.remove()

    monitor.remove_hooks()
    if model_type in ("snn", "hybrid"):
        spike_analyzer.remove_hooks()

    # Compute energy profile
    input_shape = (1, task.input_channels, 256, 256)
    profile = monitor.compute_profile(psnr=calc.compute().get("psnr", 0))
    monitor.print_report(profile)

    metrics = calc.compute()
    print(f"\n{calc.summary()}")

    # ── 6-metric deep analysis ───────────────────────────────────────────────
    print("\nRunning deep analysis (6 metrics)...")
    deep = run_deep_analysis(
        preds=collected_preds,
        targets=collected_targets,
        bottleneck_feats=collected_bottleneck,
        membrane_records=_membrane_bufs,
        model_type=model_type,
        task=task_name,
    )
    print(f"  band_psnr HF={deep.get('band_psnr', {}).get('high_freq_psnr', 'n/a'):.2f}  "
          f"error_centroid={deep.get('error_spectrum', {}).get('spectral_centroid', 'n/a'):.4f}  "
          f"eff_rank={deep.get('effective_rank', {}).get('bottleneck_effective_rank', 'n/a')}")

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
        "deep_analysis": deep,
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
