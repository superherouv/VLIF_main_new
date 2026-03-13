"""
Benchmark runner: evaluates a model checkpoint on standard test sets.

Computes PSNR, SSIM, latency, throughput, and energy estimates.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Optional
from tasks.metrics import MetricCalculator
from training.energy_monitor import EnergyMonitor


class BenchmarkRunner:
    """
    Comprehensive benchmark for SNN/ANN comparison.

    Measures:
      - Image quality (PSNR, SSIM per benchmark set)
      - Inference speed (ms/image, FPS)
      - Memory usage (peak GPU memory)
      - Energy estimate (SynOps, energy ratio)
    """

    def __init__(
        self,
        model: nn.Module,
        task,
        model_type: str = "ann",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.task = task
        self.model_type = model_type
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def run(self, test_loader, n_warmup: int = 5) -> Dict:
        """
        Run benchmark on test_loader.

        Args:
            test_loader: DataLoader for test set.
            n_warmup: Warmup iterations (excluded from timing).

        Returns:
            Dict with all benchmark results.
        """
        # Energy monitor
        monitor = EnergyMonitor(self.model, model_type=self.model_type,
                                task=self.task.name)
        monitor.attach_hooks()

        calc = MetricCalculator(self.task.name)
        latencies = []

        for i, batch in enumerate(test_loader):
            if len(batch) == 2:
                degraded, clean = batch
            else:
                degraded, clean = batch[0], batch[1]

            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            # SNN state reset
            if self.model_type in ("snn", "hybrid") and hasattr(self.model, "reset_states"):
                self.model.reset_states()

            proc = self.task.preprocess(degraded)

            # Timed inference
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = self.model(proc["input"])

            if self.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= n_warmup:
                latencies.append((t1 - t0) * 1000)  # ms

            if "bicubic_up" in proc:
                pred = self.task.postprocess(pred, bicubic_up=proc["bicubic_up"])
            pred = pred.clamp(0, 1)

            # Get firing rate if SNN
            firing_rate = None
            if hasattr(self.model, "get_energy_stats"):
                energy = self.model.get_energy_stats()
                firing_rate = energy.get("mean_firing_rate")

            calc.update(pred, clean.clamp(0, 1), firing_rate=firing_rate)

        monitor.remove_hooks()

        # Energy profile
        profile = monitor.compute_profile(
            psnr=calc.compute().get("psnr", calc.compute().get("psnr_y", 0))
        )

        # Memory
        if self.device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            peak_mem_mb = 0.0

        metrics = calc.compute()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            **metrics,
            "latency_ms": avg_latency,
            "fps": 1000.0 / avg_latency if avg_latency > 0 else 0,
            "peak_memory_mb": peak_mem_mb,
            "total_mac_ops": profile.total_mac_ops,
            "total_synops": profile.total_synops,
            "mean_firing_rate": profile.mean_firing_rate,
            "energy_ratio_vs_ann": profile.energy_ratio_vs_ann,
            "energy_efficiency": profile.energy_efficiency,
        }
