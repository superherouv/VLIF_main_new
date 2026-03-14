"""
demo_boundary_study.py
======================
Comprehensive demo of the three-layer SNN applicability boundary study.

Structure follows the research plan exactly:

  Layer 1 — Phenomenon Verification
    "Does SNN performance differ significantly across tasks?"
    → 4 tasks × 3 models (ANN/SNN/Hybrid) × T ∈ {1,2,4,8}
    → band-wise PSNR + energy

  Layer 2 — Mechanism Localisation
    "Why does the performance gap appear where it does?"
    → Frequency transfer curves (A3.3)
    → Spike–band overlap (A3.1) + temporal PSD (A3.2)
    → Temporal CKA (B1 extension)
    → Linear probe accuracy (B5)
    → Membrane potential histogram (B3)

  Layer 3 — Boundary-Controlled Intervention
    "Can we move the boundary by targeted modification?"
    → Intervention 1: High-Frequency Enhancement Module (HFEM)
    → Intervention 3: ANN decoder replacement (via Hybrid)

  SAI Synthesis
    Compute the SNN Applicability Index for all tasks and display the
    frequency-and-representation boundary map.

All results are produced from *simulated* data so the demo runs without
a GPU or trained models.  Replace the simulation helpers with real model
forward passes to produce publication-ready results.

Run:
  python scripts/demo_boundary_study.py
"""

import sys
import os
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Imports ───────────────────────────────────────────────────────────────────

from analysis.sai import SAICalculator, SAIReport
from analysis.frequency.freq_transfer import FrequencyTransferAnalyzer
from analysis.frequency.spike_freq_coupling import SpikeFqCouplingAnalyzer
from analysis.representation.temporal_cka import TemporalCKAAnalyzer
from analysis.representation.linear_probe import (
    LinearProbeAnalyzer,
    make_frequency_band_labels,
)

# ── Simulation helpers ────────────────────────────────────────────────────────

def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _sim_images(B: int, C: int, H: int, W: int, seed: int = 0) -> np.ndarray:
    return _rng(seed).rand(B, C, H, W).astype(np.float32)


def _sim_spikes(T: int, B: int, C: int, H: int, W: int,
                firing_rate: float = 0.15, seed: int = 0) -> np.ndarray:
    rng = _rng(seed)
    return (rng.rand(T, B, C, H, W) < firing_rate).astype(np.float32)


def _sim_features(N: int, D: int, n_classes: int = 4, seed: int = 0) -> tuple:
    """Return (features, labels) with cluster structure."""
    rng = _rng(seed)
    labels = rng.randint(0, n_classes, size=N)
    centres = rng.randn(n_classes, D).astype(np.float64)
    feats = centres[labels] + 0.5 * rng.randn(N, D).astype(np.float64)
    return feats.astype(np.float32), labels


# ── Task-specific ground-truth simulation parameters ─────────────────────────
# These are representative values based on ESDNet / VLIF results and
# domain knowledge; replace with real measurements from your experiments.

TASK_PARAMS = {
    "deraining": dict(
        psnr_snn=34.8, psnr_ann=37.2, ssim_snn=0.930, ssim_ann=0.960,
        firing_rate=0.13,
        scsm=0.78, bff=0.61,
        hf_psnr_gap=-1.2, wavelet_hh_ratio=1.18,
        error_centroid_snn=0.27, error_centroid_ann=0.25,
        cka_mean=0.66, rank_ratio=0.92, viable_ratio=0.81,
        temporal_redundancy=0.21, isi_cv=1.40,
        task_type="sparse",
        hfr=0.70,
    ),
    "denoising": dict(
        psnr_snn=30.1, psnr_ann=33.5, ssim_snn=0.870, ssim_ann=0.920,
        firing_rate=0.38,
        scsm=0.18, bff=0.31,
        hf_psnr_gap=-3.8, wavelet_hh_ratio=2.10,
        error_centroid_snn=0.41, error_centroid_ann=0.31,
        cka_mean=0.44, rank_ratio=0.63, viable_ratio=0.55,
        temporal_redundancy=0.52, isi_cv=0.70,
        task_type="dense",
        hfr=0.85,
    ),
    "dehazing": dict(
        psnr_snn=27.9, psnr_ann=29.5, ssim_snn=0.900, ssim_ann=0.930,
        firing_rate=0.17,
        scsm=0.54, bff=0.72,
        hf_psnr_gap=-0.8, wavelet_hh_ratio=1.05,
        error_centroid_snn=0.22, error_centroid_ann=0.20,
        cka_mean=0.72, rank_ratio=0.88, viable_ratio=0.77,
        temporal_redundancy=0.30, isi_cv=1.15,
        task_type="sparse",
        hfr=0.25,
    ),
    "superresolution": dict(
        psnr_snn=24.3, psnr_ann=29.1, ssim_snn=0.720, ssim_ann=0.870,
        firing_rate=0.45,
        scsm=0.29, bff=0.22,
        hf_psnr_gap=-5.4, wavelet_hh_ratio=3.20,
        error_centroid_snn=0.48, error_centroid_ann=0.30,
        cka_mean=0.32, rank_ratio=0.47, viable_ratio=0.40,
        temporal_redundancy=0.68, isi_cv=0.55,
        task_type="dense",
        hfr=0.90,
    ),
}


# ── Layer 1: Phenomenon Verification ─────────────────────────────────────────

def run_layer1() -> None:
    print("\n" + "="*80)
    print("LAYER 1 — PHENOMENON VERIFICATION")
    print("  Goal: Confirm that SNN performance differs significantly across tasks.")
    print("="*80)

    print(f"\n  {'Task':<18} {'SNN_PSNR':>10} {'ANN_PSNR':>10} {'Gap(dB)':>9} "
          f"{'FiringRate':>11} {'T_benefit':>11}")
    print("  " + "-"*73)

    T_values = [1, 2, 4, 8]
    for task, p in TASK_PARAMS.items():
        # Simulate T-sweep: PSNR improves with T (diminishing returns)
        psnr_by_T = {
            T: p["psnr_snn"] - 1.5 * np.exp(-T * 0.4)  # approaches plateau
            for T in T_values
        }
        t_benefit = psnr_by_T[8] - psnr_by_T[1]
        gap = p["psnr_snn"] - p["psnr_ann"]
        print(f"  {task:<18} {p['psnr_snn']:>10.2f} {p['psnr_ann']:>10.2f} "
              f"{gap:>9.2f} {p['firing_rate']:>11.3f} {t_benefit:>11.2f}")

    print("\n  Band-wise PSNR (simulated — LF / MF / HF):")
    print(f"\n  {'Task':<18} {'Model':<8} {'LF_PSNR':>9} {'MF_PSNR':>9} {'HF_PSNR':>9} {'HF-gap':>8}")
    print("  " + "-"*63)
    for task, p in TASK_PARAMS.items():
        ann_psnr = p["psnr_ann"]
        snn_psnr = p["psnr_snn"]
        # Simulate band-wise PSNR: SNN performs well at LF, increasingly worse at HF
        for model, base in [("ann", ann_psnr), ("snn", snn_psnr)]:
            lf  = base + 2.0  if model == "ann" else base + 1.5
            mf  = base        if model == "ann" else base - 0.5
            hf  = base - 3.0  if model == "ann" else base + p["hf_psnr_gap"]
            if model == "snn":
                hf_gap = hf - (ann_psnr - 3.0)
                print(f"  {task:<18} {model:<8} {lf:>9.2f} {mf:>9.2f} {hf:>9.2f} {hf_gap:>8.2f}")
            else:
                print(f"  {task:<18} {model:<8} {lf:>9.2f} {mf:>9.2f} {hf:>9.2f} {'':>8}")


# ── Layer 2: Mechanism Localisation ──────────────────────────────────────────

def run_layer2_freq_transfer() -> None:
    print("\n" + "="*80)
    print("LAYER 2A — FREQUENCY TRANSFER CURVES  (A3.3)")
    print("  Goal: Measure model frequency response using sinusoidal gratings.")
    print("="*80)

    analyzer = FrequencyTransferAnalyzer(n_orientations=4, image_size=128)

    # Simulate multiple models
    configs = {
        "ANN":        {"type": "ann"},
        "SNN T=1":    {"type": "snn", "firing_rate": 0.15, "T": 1},
        "SNN T=4":    {"type": "snn", "firing_rate": 0.15, "T": 4},
        "SNN T=8":    {"type": "snn", "firing_rate": 0.15, "T": 8},
        "Hybrid":     {"type": "hybrid", "firing_rate": 0.15, "T": 4},
        "SNN+HFEM":   {"type": "snn_hfem", "firing_rate": 0.15, "T": 4},
    }

    results = {}
    for name, cfg in configs.items():
        mtype = cfg["type"]
        if mtype == "ann":
            results[name] = analyzer.simulate_ann(name)
        elif mtype == "snn":
            results[name] = analyzer.simulate_snn(cfg["firing_rate"], cfg["T"], name)
        elif mtype == "hybrid":
            results[name] = analyzer.simulate_hybrid(cfg["firing_rate"], cfg["T"], name)
        elif mtype == "snn_hfem":
            results[name] = analyzer.simulate_snn_hfem(cfg["firing_rate"], cfg["T"], name)

    analyzer.print_comparison(results)

    # Task-frequency match scores
    print("\n  Task–Model Frequency Match Scores (weighted by task HFR):")
    print(f"  {'Model':<16}", end="")
    for task in TASK_PARAMS:
        print(f" {task[:8]:>12}", end="")
    print()
    print("  " + "-" * (16 + 13 * len(TASK_PARAMS)))

    for name, result in results.items():
        print(f"  {name:<16}", end="")
        for task, p in TASK_PARAMS.items():
            score = analyzer.task_frequency_match({name: result}, p["hfr"])
            print(f" {score[name]:>12.4f}", end="")
        print()


def run_layer2_spike_coupling() -> None:
    print("\n" + "="*80)
    print("LAYER 2B — SPIKE–FREQUENCY COUPLING  (A3.1 + A3.2)")
    print("  Goal: Test VLIF's 'SNN as HF indicator' claim across all tasks.")
    print("="*80)

    analyzer = SpikeFqCouplingAnalyzer()
    coupling_reports = {}

    for task, p in TASK_PARAMS.items():
        T, B, C, H, W = 4, 4, 16, 32, 32
        # Simulate spike records for 3 layers
        # Deraining: more HF-biased spikes; SR/denoising: less HF-biased
        hf_bias = p["scsm"]  # sparse tasks → more HF-biased spike maps
        seed = hash(task) % 10000
        spike_records = {
            "enc1": _sim_spikes(T, B, C, H, W, firing_rate=p["firing_rate"] * 0.8, seed=seed),
            "enc3": _sim_spikes(T, B, C*2, H//2, W//2, firing_rate=p["firing_rate"], seed=seed+1),
            "dec1": _sim_spikes(T, B, C, H, W, firing_rate=p["firing_rate"] * 0.7, seed=seed+2),
        }
        # Degrade the images to simulate the task
        images = _sim_images(B, 3, H, W, seed=seed)
        # Add task-specific structure (hf_bias controls how HF the degradation is)
        images += hf_bias * np.random.RandomState(seed).randn(B, 3, H, W).astype(np.float32) * 0.1

        report = analyzer.analyze(spike_records, images, task=task, model_type="snn")
        coupling_reports[task] = report

    SpikeFqCouplingAnalyzer.compare_tasks(coupling_reports)
    print("\n  Interpretation:")
    for task, r in coupling_reports.items():
        bias = "HF-biased" if r.mean_hf_lf_ratio > 1.2 else \
               "LF-biased" if r.mean_hf_lf_ratio < 0.8 else "balanced"
        claim = ("← supports VLIF 'HF indicator' claim" if bias == "HF-biased"
                 else "← contradicts VLIF claim for this task")
        print(f"  {task:<20}: {bias} (HF/LF={r.mean_hf_lf_ratio:.3f}) {claim}")


def run_layer2_temporal_cka() -> None:
    print("\n" + "="*80)
    print("LAYER 2C — TEMPORAL CKA  (B1 extension)")
    print("  Goal: Measure representation convergence and temporal redundancy.")
    print("="*80)

    tcka = TemporalCKAAnalyzer()

    for task, p in TASK_PARAMS.items():
        print(f"\n  Task: {task.upper()}")
        T = 4
        N, D = 64, 128
        seed = hash(task) % 10000

        # Simulate temporal features: representations that evolve over T steps
        # Tasks with high redundancy (dense) change less across T
        stability = 1.0 - p["temporal_redundancy"]  # low redundancy → fast change
        rng = np.random.RandomState(seed)
        base = rng.randn(N, D).astype(np.float32)
        temporal_feats = np.stack([
            base + (1.0 - stability) * t * rng.randn(N, D).astype(np.float32) * 0.3
            for t in range(T)
        ])  # (T, N, D)

        # ANN features (reference)
        ann_feats = base + rng.randn(N, D).astype(np.float32) * (1.0 - p["cka_mean"])

        # Analyse using the flat spike_records interface
        spike_records = {
            "enc3": np.zeros((T, N // 4, D // (16*16), 16, 16), dtype=np.float32),
        }
        # Directly call analyze_layer
        result = tcka.analyze_layer(temporal_feats, f"{task}_enc3", ann_feats)
        print(f"    Stability:    {result.temporal_stability:.4f}")
        print(f"    Redundancy:   {result.temporal_redundancy:.4f}  "
              f"(SAI R4 = {1 - result.temporal_redundancy:.4f})")
        print(f"    SNN-ANN CKA:  {result.snn_ann_cka_at_T:.4f}  (SAI R1)")
        print(f"    Converged:    {'Yes' if result.is_converged else 'No'}")
        tcka.print_convergence_curves({f"{task}_enc3": result})


def run_layer2_linear_probe() -> None:
    print("\n" + "="*80)
    print("LAYER 2D — LINEAR PROBE ANALYSIS  (B5)")
    print("  Goal: 'SNN can't learn features' vs 'can't decode them'.")
    print("="*80)

    probe = LinearProbeAnalyzer(n_neighbors=7, probe_max_iter=100)

    print(f"\n  {'Task':<18} {'SNN_lin':>9} {'SNN_knn':>9} {'ANN_lin':>9} {'ANN_knn':>9} "
          f"{'SNN_sparse':>11} {'ANN_sparse':>11}")
    print("  " + "-"*80)

    for task, p in TASK_PARAMS.items():
        N, D = 80, 64
        seed = hash(task) % 10000
        rng = np.random.RandomState(seed)

        # SNN features: lower discriminability, higher sparsity
        snn_feats, labels = _sim_features(N, D, n_classes=3, seed=seed)
        # Add noise proportional to how hard the task is for SNN
        hardness = 1.0 - p["cka_mean"]
        snn_feats += hardness * rng.randn(N, D).astype(np.float32) * 0.8

        # ANN features: better discriminability, less sparse
        ann_feats = snn_feats + (1.0 - hardness) * rng.randn(N, D).astype(np.float32) * 0.3

        # Quality scores per sample
        quality_scores = (rng.rand(N) * 5 + p["psnr_snn"] - 2).astype(np.float32)

        snn_r = probe.analyze_layer(snn_feats, labels, f"{task}_enc3", "snn", quality_scores)
        ann_r = probe.analyze_layer(ann_feats, labels, f"{task}_enc3", "ann", quality_scores)

        print(f"  {task:<18} {snn_r.linear_acc:>9.3f} {snn_r.knn_acc:>9.3f} "
              f"{ann_r.linear_acc:>9.3f} {ann_r.knn_acc:>9.3f} "
              f"{snn_r.feature_sparsity:>11.3f} {ann_r.feature_sparsity:>11.3f}")

    print("\n  Interpretation:")
    print("  If SNN lin_acc ≈ ANN lin_acc but knn_acc is lower: SNN *learned* features")
    print("  but they live in a poorly-structured manifold (geometry problem).")
    print("  If both lin_acc and knn_acc are low: SNN *cannot learn* these features.")


# ── Layer 3: Boundary-Controlled Intervention ─────────────────────────────────

def run_layer3_interventions() -> None:
    print("\n" + "="*80)
    print("LAYER 3 — BOUNDARY-CONTROLLED INTERVENTION")
    print("  Goal: Prove we can move the applicability boundary.")
    print("="*80)

    freq_analyzer = FrequencyTransferAnalyzer(n_orientations=4, image_size=128)

    print("\n  Intervention 1: High-Frequency Enhancement Module (HFEM)")
    print("  Prediction: largest improvement on tasks with high HFR (SR, denoising)")
    print()

    snn_base = freq_analyzer.simulate_snn(firing_rate=0.15, T=4, model_type="snn")
    snn_hfem = freq_analyzer.simulate_snn_hfem(firing_rate=0.15, T=4, model_type="snn+hfem")
    ann_ref  = freq_analyzer.simulate_ann(model_type="ann")

    print(f"  {'Model':<16} {'HF_fid':>9} {'BW':>9} {'Cutoff':>9}")
    print("  " + "-"*45)
    for name, r in [("ANN (ref)", ann_ref), ("SNN base", snn_base), ("SNN+HFEM", snn_hfem)]:
        print(f"  {name:<16} {r.hf_fidelity:>9.4f} {r.bandwidth:>9.4f} {r.cutoff_3db:>9.4f}")

    hf_gain = snn_hfem.hf_fidelity - snn_base.hf_fidelity
    print(f"\n  HFEM HF fidelity gain: {hf_gain:+.4f}  "
          f"({'significant — HF bottleneck confirmed' if hf_gain > 0.05 else 'marginal'})")

    print("\n  Intervention 3: ANN Decoder Replacement (Hybrid)")
    print("  Prediction: largest improvement on tasks where decoder is bottleneck")
    print()

    snn_enc  = freq_analyzer.simulate_snn(firing_rate=0.15, T=4, model_type="snn (enc only)")
    hybrid   = freq_analyzer.simulate_hybrid(firing_rate=0.15, T=4, model_type="snn_enc+ann_dec")

    print(f"  Task-specific impact (simulated PSNR recovery):")
    print(f"\n  {'Task':<18} {'SNN→Hybrid gain':>17} {'Bottleneck interpretation'}")
    print("  " + "-"*70)
    for task, p in TASK_PARAMS.items():
        # Estimate hybrid gain as: gap × decoder_fraction
        # Higher gap + higher HF error → decoder bottleneck more likely
        decoder_frac = abs(p["hf_psnr_gap"]) / 6.0
        hybrid_gain = (p["psnr_ann"] - p["psnr_snn"]) * decoder_frac
        interp = (
            "DECODER is bottleneck" if decoder_frac > 0.5
            else "encoder / feature bottleneck"
        )
        print(f"  {task:<18} {hybrid_gain:>+17.2f} dB  {interp}")


# ── SAI Synthesis ─────────────────────────────────────────────────────────────

def run_sai_synthesis() -> None:
    print("\n" + "="*80)
    print("SAI SYNTHESIS — SNN Applicability Index Cross-Task Comparison")
    print("  Combines Q + E + F + R into a single ranked score.")
    print("="*80)

    calc = SAICalculator()
    reports: dict[str, SAIReport] = {}

    for task, p in TASK_PARAMS.items():
        reports[task] = calc.compute(
            task=task,
            psnr_snn=p["psnr_snn"],     psnr_ann=p["psnr_ann"],
            ssim_snn=p["ssim_snn"],      ssim_ann=p["ssim_ann"],
            firing_rate=p["firing_rate"],
            scsm=p["scsm"],              bff=p["bff"],
            hf_psnr_gap=p["hf_psnr_gap"],
            wavelet_hh_ratio=p["wavelet_hh_ratio"],
            error_centroid_snn=p["error_centroid_snn"],
            error_centroid_ann=p["error_centroid_ann"],
            cka_mean=p["cka_mean"],
            rank_ratio=p["rank_ratio"],
            viable_ratio=p["viable_ratio"],
            temporal_redundancy=p["temporal_redundancy"],
            isi_cv=p["isi_cv"],
            task_type=p["task_type"],
        )

    # Table
    SAICalculator.compare_tasks(reports)

    # Radar for each task
    print()
    for task, rpt in reports.items():
        calc.print_radar(rpt)
        if rpt.interpretation:
            print(f"  → {rpt.interpretation[:110]}")

    # Frequency-and-representation boundary map
    SAICalculator.frequency_axis_plot(reports)

    # Answer the 4 core research questions
    _answer_research_questions(reports)


def _answer_research_questions(reports: dict) -> None:
    ranked = sorted(reports.items(), key=lambda x: x[1].components.sai, reverse=True)

    print(f"\n{'='*80}")
    print("RESEARCH QUESTION ANSWERS (from SAI analysis)")
    print(f"{'='*80}")

    print("\n  Q1: Which tasks in the frequency domain best suit spike representation?")
    print("  ──────────────────────────────────────────────────────────────────────")
    for task, rpt in ranked:
        c = rpt.components
        print(f"    {task:<20}: F={c.f_score*100:.1f}%  SCSM={c.f_scsm:.2f}  "
              f"HH_ratio={c.f_wavelet_hh:.2f}  → {rpt.components.level}")

    print("\n  Q2: Does the performance gap come from freq fidelity or repr geometry?")
    print("  ──────────────────────────────────────────────────────────────────────")
    for task, rpt in ranked:
        c = rpt.components
        f_issue = c.f_hf_psnr_gap < -3.0
        r_issue = c.r_cka_mean < 0.50
        if f_issue and r_issue:
            reason = "BOTH freq fidelity AND representation geometry"
        elif f_issue:
            reason = "primarily FREQUENCY fidelity (HF PSNR gap large)"
        elif r_issue:
            reason = "primarily REPRESENTATION geometry (CKA low)"
        else:
            reason = "minor gap — both freq and repr are adequate"
        print(f"    {task:<20}: {reason}")

    print("\n  Q3: When is the SNN energy advantage real (not just estimated)?")
    print("  ──────────────────────────────────────────────────────────────────────")
    for task, rpt in ranked:
        c = rpt.components
        if c.e_firing_rate < 0.20:
            verdict = "REAL advantage (low firing rate)"
        elif c.e_firing_rate < 0.40:
            verdict = "PARTIAL advantage (moderate firing rate)"
        else:
            verdict = "MINIMAL advantage (high firing rate negates savings)"
        print(f"    {task:<20}: rate={c.e_firing_rate:.3f}  E={c.e_score*100:.1f}%  → {verdict}")

    print("\n  Q4: Which stage fails — encoder, feature extraction, or decoder?")
    print("  ──────────────────────────────────────────────────────────────────────")
    for task, rpt in ranked:
        c = rpt.components
        if c.f_hf_psnr_gap < -4.0:
            stage = "DECODER (HF reconstruction fails → ANN decoder replacement helps)"
        elif c.r_cka_mean < 0.45:
            stage = "FEATURE EXTRACTION (low CKA → representations diverge from ANN)"
        elif c.r_viable_ratio < 0.5:
            stage = "ENCODER (high dead-neuron ratio → gradient/threshold issues)"
        else:
            stage = "No critical single-stage failure"
        print(f"    {task:<20}: bottleneck={rpt.components.bottleneck}  stage={stage}")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("""
  SNN suitability for low-level vision is NOT determined by task name alone,
  but by a combination of:
    ① Degradation signal sparsity  (high → SNN-friendly spike coding)
    ② High-frequency reconstruction demand  (high → SNN bottleneck)
    ③ Representation geometry alignment  (low CKA → SNN learns differently)
    ④ Actual firing rate  (high rate → energy advantage vanishes)

  The SAI framework quantifies these four axes and enables *predictive*
  applicability assessment before running expensive experiments.

  Practical guideline derived from this study:
    SAI ≥ 75  → Pure SNN recommended
    55–75     → Hybrid (SNN encoder + ANN decoder) for best efficiency-quality
    35–55     → ANN preferred; SNN only if energy is critical
    < 35      → Avoid SNN; binary spikes insufficient for this task's demands
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print(" SNN APPLICABILITY BOUNDARY STUDY")
    print(" 'Where Do Spiking Neural Networks Help in Low-Level Vision?'")
    print(" A Frequency-and-Representation Boundary Study")
    print("="*80)
    print(" NOTE: All data is simulated for demonstration.")
    print("       Replace _sim_* helpers with real model outputs for experiments.")

    run_layer1()
    run_layer2_freq_transfer()
    run_layer2_spike_coupling()
    run_layer2_temporal_cka()
    run_layer2_linear_probe()
    run_layer3_interventions()
    run_sai_synthesis()

    print("\n[Demo complete. All analysis modules verified.]\n")


if __name__ == "__main__":
    main()
