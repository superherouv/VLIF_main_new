"""
Demo: Simulate applicability analysis results (without running actual training).

Populates the ApplicabilityAnalyzer with representative results from the
literature and our framework, then generates the full analysis report.

This can be run immediately to see the analysis framework in action.

Literature sources for simulated values:
  - ESDNet (Song et al., 2024): SNN deraining PSNR ~35dB on Rain100L
  - Chen et al. (2025): SNN deraining with various encodings
  - NAFNet (Chen et al., 2022): ANN baseline for all tasks
  - DnCNN (Zhang et al., 2017): ANN denoising baseline
  - RCAN (Zhang et al., 2018): ANN SR baseline
  - FFA-Net (Qin et al., 2020): ANN dehazing baseline
  - SNR-Aware (Wu et al., 2022): ANN low-light baseline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.applicability import ApplicabilityAnalyzer
from analysis.visualizer import ApplicabilityVisualizer


def main():
    analyzer = ApplicabilityAnalyzer()

    # ─────────────────────────────────────────────────────────────
    # Simulated experimental results (representative of literature)
    # ─────────────────────────────────────────────────────────────

    # DERAINING (Rain100H benchmark)
    analyzer.add_result("deraining", "ann",   psnr=30.41, ssim=0.90,
                        mean_firing_rate=1.0, energy_ratio=1.0, T=1)
    analyzer.add_result("deraining", "snn",   psnr=28.15, ssim=0.87,
                        mean_firing_rate=0.12, energy_ratio=0.24, T=4)
    analyzer.add_result("deraining", "hybrid", psnr=29.50, ssim=0.89,
                        mean_firing_rate=0.18, energy_ratio=0.45, T=4)

    # DENOISING (CBSD68, σ=25)
    analyzer.add_result("denoising", "ann",   psnr=31.73, ssim=0.89,
                        mean_firing_rate=1.0, energy_ratio=1.0, T=1)
    analyzer.add_result("denoising", "snn",   psnr=30.21, ssim=0.86,
                        mean_firing_rate=0.42, energy_ratio=0.60, T=4)
    analyzer.add_result("denoising", "hybrid", psnr=31.10, ssim=0.88,
                        mean_firing_rate=0.35, energy_ratio=0.52, T=4)

    # SUPER-RESOLUTION (DIV2K, ×4)
    analyzer.add_result("superresolution", "ann",   psnr=32.64, ssim=0.90,
                        mean_firing_rate=1.0, energy_ratio=1.0, T=1)
    analyzer.add_result("superresolution", "snn",   psnr=29.85, ssim=0.84,
                        mean_firing_rate=0.35, energy_ratio=0.55, T=4)
    analyzer.add_result("superresolution", "hybrid", psnr=31.50, ssim=0.87,
                        mean_firing_rate=0.28, energy_ratio=0.48, T=4)

    # DEHAZING (SOTS-Indoor)
    analyzer.add_result("dehazing", "ann",   psnr=36.92, ssim=0.99,
                        mean_firing_rate=1.0, energy_ratio=1.0, T=1)
    analyzer.add_result("dehazing", "snn",   psnr=34.85, ssim=0.97,
                        mean_firing_rate=0.18, energy_ratio=0.32, T=4)
    analyzer.add_result("dehazing", "hybrid", psnr=35.90, ssim=0.98,
                        mean_firing_rate=0.22, energy_ratio=0.40, T=4)

    # LOW-LIGHT (LOL-v1)
    analyzer.add_result("lowlight", "ann",   psnr=23.65, ssim=0.87,
                        mean_firing_rate=1.0, energy_ratio=1.0, T=1)
    analyzer.add_result("lowlight", "snn",   psnr=19.30, ssim=0.78,
                        mean_firing_rate=0.22, energy_ratio=0.38, T=4)
    analyzer.add_result("lowlight", "hybrid", psnr=21.80, ssim=0.83,
                        mean_firing_rate=0.25, energy_ratio=0.42, T=4)

    # ─────────────────────────────────────────────────────────────
    # Compute and print analysis
    # ─────────────────────────────────────────────────────────────
    scored = analyzer.compute_applicability_scores()
    analyzer.print_report()

    # Save report
    os.makedirs("experiments/analysis", exist_ok=True)
    analyzer.save_report("experiments/analysis/applicability_report.json")

    # Generate visualizations (matplotlib optional)
    try:
        viz = ApplicabilityVisualizer(save_dir="experiments/figures")
        viz.plot_applicability_heatmap(scored)
        viz.plot_applicability_radar(scored)

        # PSNR vs T simulation
        psnr_vs_T = {
            "deraining": {1: 25.5, 2: 27.2, 4: 28.15, 8: 28.4, 16: 28.5},
            "denoising": {1: 28.0, 2: 29.5, 4: 30.21, 8: 30.35, 16: 30.40},
            "superresolution": {1: 27.0, 2: 28.5, 4: 29.85, 8: 30.1, 16: 30.15},
            "dehazing": {1: 31.0, 2: 33.2, 4: 34.85, 8: 35.0, 16: 35.1},
            "lowlight": {1: 17.5, 2: 18.5, 4: 19.30, 8: 19.6, 16: 19.7},
        }
        viz.plot_psnr_vs_timesteps(psnr_vs_T)

        print(f"\nAll figures saved to: {viz.save_dir}")
    except ImportError:
        print("\nmatplotlib not installed - skipping figure generation")
        print("To generate figures: pip install matplotlib")


if __name__ == "__main__":
    main()
