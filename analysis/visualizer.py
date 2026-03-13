"""
Visualization tools for SNN applicability analysis.

Generates:
  1. Spider/radar charts comparing SNN vs ANN across tasks
  2. Quality-Energy tradeoff curves (Pareto fronts)
  3. Firing rate heatmaps (spatial distribution)
  4. Temporal spike dynamics plots
  5. Applicability score bar charts
  6. PSNR vs T (timestep) curves
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple


class ApplicabilityVisualizer:
    """
    Generates publication-quality figures for the applicability study.

    Requires matplotlib (imported lazily to avoid hard dependency).
    """

    TASK_COLORS = {
        "deraining": "#2196F3",        # Blue
        "denoising": "#FF9800",        # Orange
        "superresolution": "#9C27B0",  # Purple
        "dehazing": "#4CAF50",         # Green
        "lowlight": "#F44336",         # Red
    }

    MODEL_STYLES = {
        "snn": {"linestyle": "-", "marker": "o"},
        "ann": {"linestyle": "--", "marker": "s"},
        "hybrid": {"linestyle": "-.", "marker": "^"},
    }

    def __init__(self, save_dir: str = "experiments/figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _get_plt(self):
        """Lazy matplotlib import."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            return plt, mpatches
        except ImportError:
            raise ImportError(
                "matplotlib required for visualization. "
                "Install with: pip install matplotlib"
            )

    def plot_applicability_radar(
        self,
        scored_results: Dict,
        save_name: str = "applicability_radar.png",
    ):
        """
        Radar/spider chart showing SNN performance dimensions per task.

        Dimensions:
          - Quality (normalized PSNR)
          - Energy Efficiency
          - Spike Sparsity
          - Signal Sparsity
          - Applicability Score
        """
        plt, mpatches = self._get_plt()

        tasks = list(scored_results.keys())
        if not tasks:
            return

        categories = ["Quality", "Energy\nEff.", "Spike\nSparsity",
                      "Signal\nSparsity", "Overall"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # close polygon

        fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 5),
                                  subplot_kw=dict(polar=True))
        if len(tasks) == 1:
            axes = [axes]

        for ax, task in zip(axes, tasks):
            r = scored_results[task]
            values = [
                max(0, 100 - r.quality_gap_vs_ann * 20),    # Quality
                max(0, (1 - r.energy_ratio) * 100),          # Energy efficiency
                r.sparsity_index * 100,                       # Spike sparsity
                r.task_signal_sparsity * 100,                 # Signal sparsity
                r.applicability_score,                        # Overall
            ]
            values += values[:1]

            ax.plot(angles, values, linewidth=2,
                    color=self.TASK_COLORS.get(task, "#888"))
            ax.fill(angles, values, alpha=0.25,
                    color=self.TASK_COLORS.get(task, "#888"))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=9)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7)
            ax.set_title(f"{task.capitalize()}\n"
                         f"Score: {r.applicability_score:.0f}/100\n"
                         f"[{r.applicability_level}]",
                         pad=15, size=10, weight="bold")

        plt.suptitle("SNN Applicability Radar Charts", y=1.02, fontsize=13, weight="bold")
        plt.tight_layout()
        path = os.path.join(self.save_dir, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    def plot_quality_energy_tradeoff(
        self,
        all_results: Dict,  # (task, model_type) → TaskApplicabilityResult-like dict
        save_name: str = "quality_energy_tradeoff.png",
    ):
        """
        Scatter plot: x=Energy Ratio, y=PSNR, marker=model_type, color=task.
        Pareto front highlights best quality/energy tradeoffs.
        """
        plt, mpatches = self._get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))

        for (task, model_type), result in all_results.items():
            if isinstance(result, dict):
                psnr = result.get("psnr", 0)
                energy = result.get("energy_ratio", 1.0)
            else:
                psnr = result.psnr
                energy = result.energy_ratio

            style = self.MODEL_STYLES.get(model_type, {})
            color = self.TASK_COLORS.get(task, "#888")

            ax.scatter(energy, psnr, c=color, s=150,
                       marker=style.get("marker", "o"),
                       label=f"{task}/{model_type}",
                       zorder=3)
            ax.annotate(f"{task[:3]}/{model_type[:3]}",
                        (energy, psnr), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        ax.set_xlabel("Energy Ratio (SNN/ANN)", fontsize=12)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_title("Quality vs Energy Tradeoff Across Tasks and Model Types", fontsize=13)
        ax.grid(True, alpha=0.3)

        # Add reference lines
        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="ANN energy baseline")
        ax.axvline(x=0.2, color="green", linestyle="--", alpha=0.5, label="20% energy target")

        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.save_dir, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    def plot_psnr_vs_timesteps(
        self,
        timestep_results: Dict[str, Dict[int, float]],
        save_name: str = "psnr_vs_timesteps.png",
    ):
        """
        Line plot: x=T (timesteps), y=PSNR, one line per task.

        Shows how temporal resolution affects quality for each task.
        Tasks where quality saturates at T=2-4 are more SNN-friendly.

        Args:
            timestep_results: {task: {T: psnr_value}}
        """
        plt, _ = self._get_plt()
        fig, ax = plt.subplots(figsize=(8, 5))

        for task, t_results in timestep_results.items():
            Ts = sorted(t_results.keys())
            psnrs = [t_results[t] for t in Ts]
            ax.plot(Ts, psnrs,
                    color=self.TASK_COLORS.get(task, "#888"),
                    marker="o", linewidth=2, label=task.capitalize())

        ax.set_xlabel("Number of Timesteps (T)", fontsize=12)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_title("SNN PSNR vs Timesteps: Quality-Latency Tradeoff", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 16])

        plt.tight_layout()
        path = os.path.join(self.save_dir, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    def plot_applicability_heatmap(
        self,
        scored_results: Dict,
        save_name: str = "applicability_heatmap.png",
    ):
        """
        Heatmap: rows=tasks, columns=dimensions, color=score.
        """
        plt, _ = self._get_plt()

        tasks = ["deraining", "denoising", "dehazing", "superresolution", "lowlight"]
        dims = ["Quality", "Sparsity", "Energy", "Signal", "Overall"]

        matrix = np.zeros((len(tasks), len(dims)))
        for i, task in enumerate(tasks):
            if task not in scored_results:
                continue
            r = scored_results[task]
            matrix[i, 0] = max(0, 100 - r.quality_gap_vs_ann * 20)
            matrix[i, 1] = r.sparsity_index * 100
            matrix[i, 2] = max(0, (1 - r.energy_ratio) * 100)
            matrix[i, 3] = r.task_signal_sparsity * 100
            matrix[i, 4] = r.applicability_score

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims, fontsize=11)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([t.capitalize() for t in tasks], fontsize=11)

        # Annotate cells
        for i in range(len(tasks)):
            for j in range(len(dims)):
                val = matrix[i, j]
                text_color = "white" if val < 40 or val > 80 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=11, color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Score [0-100]")
        ax.set_title("SNN Applicability Heatmap (Red=Low, Green=High)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.save_dir, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
