"""
SNN Applicability Boundary Analysis.

This is the central analysis module for the research question:
"When are SNNs suitable for low-level vision, and when are they not?"

Framework:
  For each task, we measure 6 key dimensions:
    1. Quality Gap: ΔPSNR = PSNR(ANN) - PSNR(SNN)  [lower = better for SNN]
    2. Energy Ratio: E(SNN) / E(ANN)               [lower = better for SNN]
    3. Spike Sparsity: 1 - mean_firing_rate         [higher = more sparse = more SNN-friendly]
    4. Temporal Benefit: PSNR(T=8) - PSNR(T=1)    [higher = more temporal gain]
    5. Hybrid Gain: PSNR(Hybrid) - PSNR(SNN)       [higher = ANN decoder needed]
    6. Task Signal Sparsity: estimated from data    [higher = more SNN-friendly]

These dimensions are combined into a single "SNN Applicability Score" [0-100].

Score interpretation:
  80-100: Highly suitable (deraining with heavy rain)
  60-80:  Suitable with caveats (dehazing, denoising σ≤25)
  40-60:  Marginal (denoising σ>25, super-res ×2)
  20-40:  Not recommended (super-res ×4, low-light)
  0-20:   Avoid SNN (extreme low-light, fine texture SR)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os


@dataclass
class TaskApplicabilityResult:
    """
    Complete applicability analysis result for one task.
    """
    task_name: str
    model_type: str  # snn, ann, hybrid

    # Metrics
    psnr: float = 0.0
    ssim: float = 0.0
    mean_firing_rate: float = 0.0
    sparsity_index: float = 0.0
    energy_ratio: float = 1.0
    synops: float = 0.0

    # Derived analysis
    quality_gap_vs_ann: float = 0.0     # ΔPSNR = PSNR(ANN) - PSNR(this)
    temporal_benefit: float = 0.0       # PSNR gain from T=1 to T=8
    hybrid_gain: float = 0.0            # PSNR(Hybrid) - PSNR(SNN)
    task_signal_sparsity: float = 0.0   # Estimated input signal sparsity

    # Final score
    applicability_score: float = 0.0
    applicability_level: str = ""       # HIGH / MEDIUM / LOW / VERY_LOW
    recommendation: str = ""

    # Extra metadata
    T: int = 4
    extra: Dict = field(default_factory=dict)


class ApplicabilityAnalyzer:
    """
    Computes and visualizes SNN applicability boundaries across tasks.

    Usage:
        analyzer = ApplicabilityAnalyzer()

        # Register results as experiments run
        analyzer.add_result("deraining", "snn",
                            psnr=35.2, ssim=0.94, mean_firing_rate=0.12, ...)
        analyzer.add_result("deraining", "ann", psnr=37.1, ssim=0.96, ...)
        ...

        # Compute applicability scores
        scores = analyzer.compute_applicability_scores()

        # Generate report
        analyzer.print_report()
        analyzer.save_report("analysis/results/report.json")
    """

    # Weight coefficients for applicability score
    SCORE_WEIGHTS = {
        "quality_gap_penalty": 0.35,     # ΔPSNR matters most
        "sparsity_bonus": 0.25,          # Spike sparsity
        "energy_efficiency": 0.20,       # Energy savings
        "temporal_benefit": 0.10,        # Temporal coding benefit
        "signal_sparsity": 0.10,         # Task-level signal sparsity
    }

    # Prior knowledge about signal sparsity per task
    # (estimated from domain knowledge and literature)
    TASK_SIGNAL_SPARSITY = {
        "deraining": 0.85,     # Rain streaks are sparse (5-15% of pixels)
        "denoising": 0.15,     # Gaussian noise is dense (all pixels)
        "superresolution": 0.30,  # LR→HR: some edges, mostly smooth
        "dehazing": 0.50,      # Transmission map has sparse boundaries
        "lowlight": 0.10,      # Uniform illumination changes, dense
    }

    def __init__(self):
        self._results: Dict[Tuple[str, str], TaskApplicabilityResult] = {}

    def add_result(
        self,
        task: str,
        model_type: str,
        **metrics,
    ):
        """
        Add experimental result for (task, model_type) pair.

        Args:
            task: Task name.
            model_type: 'snn' | 'ann' | 'hybrid'
            **metrics: psnr, ssim, mean_firing_rate, energy_ratio, synops, T, etc.
        """
        result = TaskApplicabilityResult(
            task_name=task,
            model_type=model_type,
            psnr=metrics.get("psnr", 0.0),
            ssim=metrics.get("ssim", 0.0),
            mean_firing_rate=metrics.get("mean_firing_rate", 0.0),
            sparsity_index=1.0 - metrics.get("mean_firing_rate", 0.0),
            energy_ratio=metrics.get("energy_ratio", 1.0),
            synops=metrics.get("synops", 0.0),
            T=metrics.get("T", 4),
            extra=metrics.get("extra", {}),
        )
        result.task_signal_sparsity = self.TASK_SIGNAL_SPARSITY.get(task, 0.5)
        self._results[(task, model_type)] = result

    def compute_applicability_scores(self) -> Dict[str, TaskApplicabilityResult]:
        """
        Compute applicability scores for all SNN results.

        For each task with SNN result, compute score relative to ANN baseline.

        Returns:
            Dict mapping task_name → updated SNN TaskApplicabilityResult
        """
        scored = {}

        for (task, model_type), result in self._results.items():
            if model_type != "snn":
                continue

            # Get ANN baseline
            ann_result = self._results.get((task, "ann"))
            hybrid_result = self._results.get((task, "hybrid"))

            # 1. Quality gap penalty (ΔPSNR)
            if ann_result is not None:
                result.quality_gap_vs_ann = ann_result.psnr - result.psnr
            else:
                result.quality_gap_vs_ann = 3.0  # assume 3dB gap if no ANN data

            # 2. Hybrid gain
            if hybrid_result is not None:
                result.hybrid_gain = hybrid_result.psnr - result.psnr
            else:
                result.hybrid_gain = result.quality_gap_vs_ann / 2.0

            # --- Compute score components [0, 100] ---

            # (a) Quality: penalize for PSNR gap
            # ΔPSNR=0 → score=100; ΔPSNR=5 → score=0
            quality_score = max(0, 100 * (1 - result.quality_gap_vs_ann / 5.0))

            # (b) Sparsity bonus: reward sparse firing
            # firing_rate=0.05 → score=100; firing_rate=0.8 → score=0
            sparsity_score = max(0, 100 * (1 - result.mean_firing_rate / 0.8))

            # (c) Energy efficiency: reward low energy ratio
            # ratio=0.1 → score=100; ratio=1.0 → score=0
            energy_score = max(0, 100 * (1 - result.energy_ratio))

            # (d) Task signal sparsity (prior knowledge)
            signal_score = result.task_signal_sparsity * 100

            # (e) Temporal benefit
            temporal_score = min(100, result.temporal_benefit * 20)  # 5dB gain → 100

            # Weighted sum
            w = self.SCORE_WEIGHTS
            result.applicability_score = (
                w["quality_gap_penalty"] * quality_score +
                w["sparsity_bonus"] * sparsity_score +
                w["energy_efficiency"] * energy_score +
                w["signal_sparsity"] * signal_score +
                w["temporal_benefit"] * temporal_score
            )

            # Applicability level
            s = result.applicability_score
            if s >= 75:
                result.applicability_level = "HIGH"
            elif s >= 55:
                result.applicability_level = "MEDIUM"
            elif s >= 35:
                result.applicability_level = "LOW"
            else:
                result.applicability_level = "VERY_LOW"

            # Recommendation
            result.recommendation = self._generate_recommendation(result)

            scored[task] = result

        return scored

    def _generate_recommendation(self, r: TaskApplicabilityResult) -> str:
        """Generate actionable recommendation based on scores."""
        if r.applicability_level == "HIGH":
            return (
                f"✓ SNN recommended for {r.task_name}. "
                f"PSNR gap={r.quality_gap_vs_ann:.2f}dB, "
                f"energy={r.energy_ratio:.2f}×ANN. "
                f"Direct encoding T={r.T} optimal."
            )
        elif r.applicability_level == "MEDIUM":
            return (
                f"△ SNN viable for {r.task_name} with tradeoffs. "
                f"Consider Hybrid (SNN-enc + ANN-dec) for "
                f"better quality-energy balance. "
                f"Gap={r.quality_gap_vs_ann:.2f}dB."
            )
        elif r.applicability_level == "LOW":
            return (
                f"✗ SNN marginal for {r.task_name}. "
                f"Only if energy is critical and quality tolerance > {r.quality_gap_vs_ann:.1f}dB. "
                f"Hybrid architecture strongly preferred."
            )
        else:
            return (
                f"✗✗ SNN not recommended for {r.task_name}. "
                f"Quality gap={r.quality_gap_vs_ann:.2f}dB exceeds acceptable threshold. "
                f"Use ANN. SNN only viable with event cameras."
            )

    def print_report(self):
        """Print comprehensive applicability report to stdout."""
        scored = self.compute_applicability_scores()

        print("\n" + "="*80)
        print("SNN APPLICABILITY BOUNDARY ANALYSIS FOR LOW-LEVEL VISION TASKS")
        print("="*80)
        print(f"{'Task':<18} {'SNN PSNR':>10} {'ANN PSNR':>10} {'ΔPSNR':>8} "
              f"{'FireRate':>10} {'Energy':>8} {'Score':>7} {'Level':<12}")
        print("-"*80)

        task_order = ["deraining", "denoising", "dehazing", "superresolution", "lowlight"]
        for task in task_order:
            if task not in scored:
                continue
            r = scored[task]
            ann = self._results.get((task, "ann"))
            ann_psnr = ann.psnr if ann else 0.0
            print(
                f"{task:<18} {r.psnr:>10.2f} {ann_psnr:>10.2f} "
                f"{r.quality_gap_vs_ann:>8.2f} "
                f"{r.mean_firing_rate:>10.4f} "
                f"{r.energy_ratio:>8.3f} "
                f"{r.applicability_score:>7.1f} "
                f"{r.applicability_level:<12}"
            )

        print("="*80)
        print("\nDETAILED RECOMMENDATIONS:")
        for task in task_order:
            if task in scored:
                print(f"\n[{task.upper()}]")
                print(f"  {scored[task].recommendation}")

        print("\n" + "="*80)
        print("APPLICABILITY MAP (Signal Sparsity vs Quality Gap):")
        self._print_ascii_scatter(scored)

    def _print_ascii_scatter(self, scored: Dict):
        """Print ASCII scatter plot: x=signal_sparsity, y=quality_gap."""
        print()
        print("  Quality")
        print("  Gap(dB)")
        print("  5.0 │")
        tasks = {
            "deraining": ("D", 0.85),
            "denoising": ("N", 0.15),
            "dehazing": ("H", 0.50),
            "superresolution": ("S", 0.30),
            "lowlight": ("L", 0.10),
        }
        # Simple 2D text grid
        W, H_grid = 50, 12
        grid = [[' '] * W for _ in range(H_grid)]
        for task, (sym, sparsity) in tasks.items():
            if task in scored:
                gap = scored[task].quality_gap_vs_ann
                x = int(sparsity * (W - 1))
                y = int((5.0 - gap) / 5.0 * (H_grid - 1))
                y = max(0, min(H_grid - 1, y))
                grid[y][x] = sym

        for i, row in enumerate(grid):
            gap_val = 5.0 - i * 5.0 / (H_grid - 1)
            print(f"  {gap_val:4.1f} │ {''.join(row)}")
        print("       └" + "─" * W)
        print("         0%                  Signal Sparsity                 100%")
        print("\n  D=Deraining  N=Denoising  H=Dehazing  S=SuperRes  L=LowLight")
        print("  Top-left = SNN unfavorable (high gap, low sparsity)")
        print("  Bottom-right = SNN favorable (low gap, high sparsity)")

    def save_report(self, path: str):
        """Save report to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scored = self.compute_applicability_scores()
        data = {}
        for task, result in scored.items():
            data[task] = {
                "psnr": result.psnr,
                "ssim": result.ssim,
                "quality_gap_vs_ann": result.quality_gap_vs_ann,
                "mean_firing_rate": result.mean_firing_rate,
                "sparsity_index": result.sparsity_index,
                "energy_ratio": result.energy_ratio,
                "applicability_score": result.applicability_score,
                "applicability_level": result.applicability_level,
                "recommendation": result.recommendation,
                "T": result.T,
            }
        # Add ANN baselines
        for (task, model_type), result in self._results.items():
            if model_type == "ann":
                data.setdefault(task, {})
                data[task]["ann_psnr"] = result.psnr
                data[task]["ann_ssim"] = result.ssim
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Report saved to {path}")


def estimate_signal_sparsity(images: "torch.Tensor") -> float:
    """
    Estimate task-specific signal sparsity from image statistics.

    For degradation images:
      - Compute gradient magnitude (edges/artifacts)
      - High-frequency content that differs from clean background
      - Used to verify our prior TASK_SIGNAL_SPARSITY values

    Args:
        images: (B, C, H, W) degraded input images

    Returns:
        Estimated sparsity index in [0, 1]
    """
    import torch
    import torch.nn.functional as F

    # Laplacian edge detection
    lap_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)

    gray = images.mean(dim=1, keepdim=True)
    edges = F.conv2d(gray, lap_kernel, padding=1).abs()

    # Sparsity: fraction of pixels with significant edge response
    threshold = edges.mean() + 2 * edges.std()
    sparsity = (edges > threshold).float().mean().item()
    return sparsity
