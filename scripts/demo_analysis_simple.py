"""
Demo analysis - pure Python, no torch/numpy required.
Tests the core applicability scoring logic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standalone implementation (no torch needed for analysis itself)
from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass
class TaskApplicabilityResult:
    task_name: str
    model_type: str
    psnr: float = 0.0
    ssim: float = 0.0
    mean_firing_rate: float = 0.0
    sparsity_index: float = 0.0
    energy_ratio: float = 1.0
    quality_gap_vs_ann: float = 0.0
    hybrid_gain: float = 0.0
    task_signal_sparsity: float = 0.0
    applicability_score: float = 0.0
    applicability_level: str = ""
    recommendation: str = ""
    T: int = 4


TASK_SIGNAL_SPARSITY = {
    "deraining": 0.85,
    "denoising": 0.15,
    "superresolution": 0.30,
    "dehazing": 0.50,
    "lowlight": 0.10,
}

SCORE_WEIGHTS = {
    "quality_gap_penalty": 0.35,
    "sparsity_bonus": 0.25,
    "energy_efficiency": 0.20,
    "temporal_benefit": 0.10,
    "signal_sparsity": 0.10,
}

results = {}

def add_result(task, model_type, **kw):
    r = TaskApplicabilityResult(
        task_name=task, model_type=model_type,
        psnr=kw.get("psnr", 0), ssim=kw.get("ssim", 0),
        mean_firing_rate=kw.get("mean_firing_rate", 0),
        sparsity_index=1 - kw.get("mean_firing_rate", 0),
        energy_ratio=kw.get("energy_ratio", 1.0),
        T=kw.get("T", 4),
    )
    r.task_signal_sparsity = TASK_SIGNAL_SPARSITY.get(task, 0.5)
    results[(task, model_type)] = r

# ── Simulated results from literature ────────────────────────────────────
add_result("deraining",    "ann",    psnr=30.41, ssim=0.90, mean_firing_rate=1.0,  energy_ratio=1.0,  T=1)
add_result("deraining",    "snn",    psnr=28.15, ssim=0.87, mean_firing_rate=0.12, energy_ratio=0.24, T=4)
add_result("deraining",    "hybrid", psnr=29.50, ssim=0.89, mean_firing_rate=0.18, energy_ratio=0.45, T=4)
add_result("denoising",    "ann",    psnr=31.73, ssim=0.89, mean_firing_rate=1.0,  energy_ratio=1.0,  T=1)
add_result("denoising",    "snn",    psnr=30.21, ssim=0.86, mean_firing_rate=0.42, energy_ratio=0.60, T=4)
add_result("denoising",    "hybrid", psnr=31.10, ssim=0.88, mean_firing_rate=0.35, energy_ratio=0.52, T=4)
add_result("superresolution","ann",  psnr=32.64, ssim=0.90, mean_firing_rate=1.0,  energy_ratio=1.0,  T=1)
add_result("superresolution","snn",  psnr=29.85, ssim=0.84, mean_firing_rate=0.35, energy_ratio=0.55, T=4)
add_result("superresolution","hybrid",psnr=31.50, ssim=0.87, mean_firing_rate=0.28, energy_ratio=0.48, T=4)
add_result("dehazing",     "ann",    psnr=36.92, ssim=0.99, mean_firing_rate=1.0,  energy_ratio=1.0,  T=1)
add_result("dehazing",     "snn",    psnr=34.85, ssim=0.97, mean_firing_rate=0.18, energy_ratio=0.32, T=4)
add_result("dehazing",     "hybrid", psnr=35.90, ssim=0.98, mean_firing_rate=0.22, energy_ratio=0.40, T=4)
add_result("lowlight",     "ann",    psnr=23.65, ssim=0.87, mean_firing_rate=1.0,  energy_ratio=1.0,  T=1)
add_result("lowlight",     "snn",    psnr=19.30, ssim=0.78, mean_firing_rate=0.22, energy_ratio=0.38, T=4)
add_result("lowlight",     "hybrid", psnr=21.80, ssim=0.83, mean_firing_rate=0.25, energy_ratio=0.42, T=4)

# ── Compute applicability scores ─────────────────────────────────────────
scored = {}
for (task, model_type), r in results.items():
    if model_type != "snn":
        continue
    ann = results.get((task, "ann"))
    r.quality_gap_vs_ann = ann.psnr - r.psnr if ann else 3.0
    hybrid = results.get((task, "hybrid"))
    r.hybrid_gain = hybrid.psnr - r.psnr if hybrid else r.quality_gap_vs_ann / 2

    quality_score  = max(0, 100 * (1 - r.quality_gap_vs_ann / 5.0))
    sparsity_score = max(0, 100 * (1 - r.mean_firing_rate / 0.8))
    energy_score   = max(0, 100 * (1 - r.energy_ratio))
    signal_score   = r.task_signal_sparsity * 100
    temporal_score = 0.0  # placeholder

    r.applicability_score = (
        SCORE_WEIGHTS["quality_gap_penalty"] * quality_score +
        SCORE_WEIGHTS["sparsity_bonus"]       * sparsity_score +
        SCORE_WEIGHTS["energy_efficiency"]    * energy_score +
        SCORE_WEIGHTS["signal_sparsity"]      * signal_score +
        SCORE_WEIGHTS["temporal_benefit"]     * temporal_score
    )

    s = r.applicability_score
    if s >= 75:   r.applicability_level = "HIGH"
    elif s >= 55: r.applicability_level = "MEDIUM"
    elif s >= 35: r.applicability_level = "LOW"
    else:         r.applicability_level = "VERY_LOW"

    scored[task] = r

# ── Print report ─────────────────────────────────────────────────────────
task_order = ["deraining", "denoising", "dehazing", "superresolution", "lowlight"]

print("\n" + "="*80)
print("SNN APPLICABILITY BOUNDARY ANALYSIS FOR LOW-LEVEL VISION TASKS")
print("="*80)
print(f"{'Task':<18} {'SNN PSNR':>10} {'ANN PSNR':>10} {'ΔPSNR':>8} "
      f"{'FireRate':>10} {'Energy':>8} {'Score':>7} {'Level':<12}")
print("-"*80)

for task in task_order:
    if task not in scored:
        continue
    r = scored[task]
    ann = results.get((task, "ann"))
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
print("\nAPPLICABILITY MAP (text scatter: bottom-right = SNN favorable)")
W, H_grid = 50, 12
grid = [[' '] * W for _ in range(H_grid)]
SYMBOLS = {"deraining":"D","denoising":"N","dehazing":"H","superresolution":"S","lowlight":"L"}
SPARSITY = {"deraining":0.85,"denoising":0.15,"dehazing":0.50,"superresolution":0.30,"lowlight":0.10}
for task in task_order:
    if task not in scored:
        continue
    r = scored[task]
    x = int(SPARSITY[task] * (W-1))
    y = int((5.0 - r.quality_gap_vs_ann) / 5.0 * (H_grid-1))
    y = max(0, min(H_grid-1, y))
    grid[y][x] = SYMBOLS[task]

for i, row in enumerate(grid):
    gap_val = 5.0 - i * 5.0 / (H_grid-1)
    print(f"  ΔPSNR={gap_val:4.1f}│ {''.join(row)}")
print("          └" + "─"*W)
print("           0%             Signal Sparsity              100%")
print("\nD=Deraining  N=Denoising  H=Dehazing  S=SuperRes  L=LowLight")
print("Top-left = SNN poor (high gap, low sparsity) | Bottom-right = SNN favorable")

print("\n" + "="*80)
print("HYBRID vs SNN PSNR GAINS:")
for task in task_order:
    if task in scored:
        r = scored[task]
        print(f"  {task:<18}: Hybrid gains +{r.hybrid_gain:.2f} dB over SNN")

# Save JSON report
os.makedirs("experiments/analysis", exist_ok=True)
report = {}
for task, r in scored.items():
    report[task] = {
        "psnr_snn": r.psnr,
        "psnr_ann": results.get((task,"ann"), r).psnr,
        "delta_psnr": r.quality_gap_vs_ann,
        "mean_firing_rate": r.mean_firing_rate,
        "energy_ratio": r.energy_ratio,
        "score": round(r.applicability_score, 1),
        "level": r.applicability_level,
        "hybrid_gain": round(r.hybrid_gain, 2),
    }
with open("experiments/analysis/applicability_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n✓ Report saved: experiments/analysis/applicability_report.json")
print("="*80)
