"""
SNN Applicability Index (SAI)
==============================
Formal multi-dimensional scoring framework for predicting SNN suitability
on low-level vision tasks.

Formula:
    SAI = w_Q·Q + w_E·E + w_F·F + w_R·R  (scaled to [0, 100])

Components:
  Q  Quality Gap         — How much PSNR/SSIM SNN loses vs ANN baseline
  E  Energy Benefit      — Actual firing_rate × (e_spike/e_mac) energy ratio
  F  Frequency Adaptation— How well task freq structure matches spike coding
  R  Representation Fit  — CKA + effective rank + viable neurons + redundancy

Sub-components of F:
  F1: SCSM   — degradation signal sparsity (Gini coefficient)
  F2: BFF    — binary frequency fidelity (can binary spikes preserve the signal?)
  F3: HFR_gap— SNN HF PSNR gap vs ANN (inverted — gap → penalty)
  F4: WavHH  — wavelet HH subband ratio (inverted)
  F5: ErrCen — error spectrum centroid shift (higher shift → penalty)

Sub-components of R:
  R1: CKA    — SNN-ANN layer representation alignment
  R2: Rank   — effective rank ratio (SNN/ANN)
  R3: Viable — viable neuron ratio
  R4: Redund — temporal redundancy (inverted — high redundancy is wasteful)
  R5: ISI    — ISI coding match for task type

This formulation answers the 4 core research questions:
  Q1: Which tasks suit spike representation? → F score ranking
  Q2: Where does quality gap originate?      → F vs R breakdown
  Q3: When is energy advantage real?         → E score
  Q4: Which stage fails (enc/feat/dec)?      → Hybrid gain analysis

Reference:
  "Where Do Spiking Neural Networks Help in Low-Level Vision?
   A Frequency-and-Representation Boundary Study"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Default weights — empirically set so each component contributes at comparable scale
DEFAULT_WEIGHTS: Dict[str, float] = {
    "quality":         0.35,   # Q: task performance is most directly measurable
    "energy":          0.15,   # E: energy advantage (often overestimated in papers)
    "frequency":       0.30,   # F: frequency-domain fit (core novel contribution)
    "representation":  0.20,   # R: representation alignment
}


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SAIComponents:
    """Detailed breakdown of all SAI sub-components."""

    # ── Q: Quality Gap ────────────────────────────────────────────────
    q_psnr_gap: float = 0.0      # SNN PSNR − ANN PSNR (negative = SNN worse)
    q_ssim_gap: float = 0.0      # SNN SSIM − ANN SSIM
    q_score: float = 0.0         # [0, 1] normalized quality component

    # ── E: Energy Benefit ─────────────────────────────────────────────
    e_firing_rate: float = 0.0   # mean firing rate [0, 1]
    e_synops_ratio: float = 0.0  # SNN SynOps / ANN MACs (estimated)
    e_energy_ratio: float = 0.0  # estimated E(SNN) / E(ANN)
    e_score: float = 0.0         # [0, 1] energy benefit component

    # ── F: Frequency Adaptation ───────────────────────────────────────
    f_scsm: float = 0.0          # degradation signal sparsity (Gini)
    f_bff: float = 0.0           # binary frequency fidelity
    f_hf_psnr_gap: float = 0.0   # SNN−ANN PSNR gap in HF band (negative = SNN worse)
    f_wavelet_hh: float = 0.0    # wavelet HH error ratio (SNN/ANN; >1 = SNN worse)
    f_error_centroid_shift: float = 0.0  # SNN centroid − ANN centroid (positive = SNN shifts to HF)
    f_score: float = 0.0         # [0, 1] frequency adaptation component

    # ── R: Representation Fit ─────────────────────────────────────────
    r_cka_mean: float = 0.0          # mean CKA SNN-ANN across layers
    r_rank_ratio: float = 0.0        # SNN effective rank / ANN effective rank
    r_viable_ratio: float = 0.0      # fraction of viable (non-dead/non-saturated) neurons
    r_temporal_redundancy: float = 0.0  # temporal redundancy in spike sequence
    r_isi_cv: float = 0.0            # coefficient of variation of ISI
    r_score: float = 0.0             # [0, 1] representation fit component

    # ── Final SAI ─────────────────────────────────────────────────────
    sai: float = 0.0
    level: str = "UNKNOWN"         # HIGH / MEDIUM / LOW / VERY_LOW
    bottleneck: str = "UNKNOWN"    # which component drags the score down


@dataclass
class SAIReport:
    """Complete SAI analysis report for a single task."""
    task: str
    components: SAIComponents
    weights: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)
    # Fine-grained F sub-scores (for visualization / ablation)
    f_subcomponents: Dict[str, float] = field(default_factory=dict)
    # Fine-grained R sub-scores
    r_subcomponents: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _level(score: float) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 55:
        return "MEDIUM"
    if score >= 35:
        return "LOW"
    return "VERY_LOW"


# ──────────────────────────────────────────────────────────────────────────────
# SAI Calculator
# ──────────────────────────────────────────────────────────────────────────────

class SAICalculator:
    """
    Computes the SNN Applicability Index from experimental measurements.

    All inputs are simple floats extracted from experiment results —
    no PyTorch / model objects needed at scoring time.

    Typical usage
    -------------
    calc = SAICalculator()
    report = calc.compute(
        task="deraining",
        psnr_snn=35.2, psnr_ann=37.1,
        ssim_snn=0.93, ssim_ann=0.96,
        firing_rate=0.12,
        scsm=0.78, bff=0.61,
        hf_psnr_gap=-1.3,
        wavelet_hh_ratio=1.15,
        error_centroid_snn=0.28, error_centroid_ann=0.25,
        cka_mean=0.64, rank_ratio=0.91,
        viable_ratio=0.82, temporal_redundancy=0.22,
        isi_cv=1.35, task_type="sparse",
    )
    calc.print_radar(report)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    # ── Q Component ───────────────────────────────────────────────────

    def _q_score(
        self,
        psnr_snn: float, psnr_ann: float,
        ssim_snn: float, ssim_ann: float,
    ) -> Tuple[float, float, float]:
        """
        Map PSNR/SSIM gap to quality score.
          gap  =  0 dB  →  Q = 1.0  (SNN matches ANN)
          gap  = -3 dB  →  Q ≈ 0.50
          gap  = -6 dB  →  Q = 0.00
        """
        psnr_gap = psnr_snn - psnr_ann   # typically ≤ 0
        ssim_gap = ssim_snn - ssim_ann

        q_psnr = float(np.clip((psnr_gap + 6.0) / 6.0, 0.0, 1.0))
        q_ssim = float(np.clip((ssim_gap + 0.10) / 0.10, 0.0, 1.0)) if ssim_ann > 0 else q_psnr

        q = 0.70 * q_psnr + 0.30 * q_ssim
        return float(q), float(psnr_gap), float(ssim_gap)

    # ── E Component ───────────────────────────────────────────────────

    def _e_score(
        self,
        firing_rate: float,
        synops_ratio: Optional[float] = None,
        e_spike_pj: float = 0.9,   # pJ per synaptic op on neuromorphic chip
        e_mac_pj: float = 4.6,     # pJ per MAC on GPU/ASIC
    ) -> Tuple[float, float]:
        """
        Energy benefit score.

        E(SNN) / E(ANN) ≈ synops_ratio × (e_spike / e_mac)
        score = 1 - energy_ratio  (lower ratio = more benefit)

        Penalty applied when firing_rate > 0.5 (high rate negates efficiency).
        """
        sr = synops_ratio if synops_ratio is not None else firing_rate
        energy_ratio = float(np.clip(sr * (e_spike_pj / e_mac_pj), 0.0, 1.0))

        e = float(np.clip(1.0 - energy_ratio, 0.0, 1.0))
        if firing_rate > 0.5:
            e *= 0.60   # high firing rate discount

        return e, energy_ratio

    # ── F Component ───────────────────────────────────────────────────

    def _f_score(
        self,
        scsm: float,                    # [0,1] degradation signal Gini sparsity
        bff: float,                     # [0,1] binary frequency fidelity
        hf_psnr_gap: float,             # SNN−ANN HF PSNR gap (≤ 0 typically)
        wavelet_hh_ratio: float,        # SNN/ANN HH error ratio (≥ 1 = SNN worse)
        error_centroid_snn: float,      # SNN error spectral centroid [0,1]
        error_centroid_ann: float,      # ANN error spectral centroid [0,1]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Frequency adaptation score — 5 sub-components:

        F1  SCSM   — Sparse signal → few discrete spike events needed → SNN-friendly
        F2  BFF    — High binary fidelity → binary spikes can represent signal well
        F3  HF gap — Small HF PSNR gap → SNN not losing accuracy in critical freq band
        F4  WavHH  — HH ratio ≈ 1 → SNN preserves high-freq diagonal detail
        F5  CenSft — SNN error centroid not shifted to higher freq than ANN
        """
        f1 = float(np.clip(scsm, 0.0, 1.0))

        f2 = float(np.clip(bff, 0.0, 1.0))

        # f3: 0 gap → 1.0; -6 dB → 0.0
        f3 = float(np.clip((hf_psnr_gap + 6.0) / 6.0, 0.0, 1.0))

        # f4: ratio=1.0 → 1.0; ratio=3.0 → 0.0
        f4 = float(np.clip(1.0 - (wavelet_hh_ratio - 1.0) / 2.0, 0.0, 1.0))

        # f5: zero centroid shift → 1.0; +0.3 shift (SNN error more HF) → 0.0
        centroid_shift = error_centroid_snn - error_centroid_ann
        f5 = float(np.clip(1.0 - centroid_shift / 0.30, 0.0, 1.0))

        # Weights reflect research importance
        f = 0.30 * f1 + 0.15 * f2 + 0.25 * f3 + 0.20 * f4 + 0.10 * f5

        sub = {
            "scsm_sparsity": round(f1, 4),
            "binary_fidelity": round(f2, 4),
            "hf_psnr_gap": round(f3, 4),
            "wavelet_hh": round(f4, 4),
            "centroid_shift": round(f5, 4),
        }
        return float(f), sub

    # ── R Component ───────────────────────────────────────────────────

    def _r_score(
        self,
        cka_mean: float,            # [0,1] mean CKA SNN-ANN
        rank_ratio: float,          # [0,∞] SNN eff rank / ANN eff rank
        viable_ratio: float,        # [0,1] fraction of alive neurons
        temporal_redundancy: float, # [0,1] temporal redundancy (lower = better)
        isi_cv: float,              # coefficient of variation of ISI
        task_type: str = "sparse",  # "sparse" | "dense"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Representation fit score — 5 sub-components:

        R1  CKA     — SNN-ANN alignment: high CKA = learned similar features
        R2  Rank    — Effective rank ratio: low rank = information bottleneck
        R3  Viable  — Active neurons: dead neurons waste capacity
        R4  Redund  — Temporal redundancy: high redundancy = wasted timesteps
        R5  ISI     — Coding match: sparse tasks need irregular (high-CV) coding;
                      dense tasks benefit from regular (low-CV) rate coding
        """
        r1 = float(np.clip(cka_mean, 0.0, 1.0))

        # rank_ratio > 1 is fine (SNN uses more dimensions) → clip to 1
        r2 = float(np.clip(rank_ratio, 0.0, 1.0))

        r3 = float(np.clip(viable_ratio, 0.0, 1.0))

        r4 = float(np.clip(1.0 - temporal_redundancy, 0.0, 1.0))

        # ISI match depends on task type
        if task_type == "sparse":
            # Sparse degradation (rain streaks) → burst/irregular coding preferred → high CV good
            r5 = float(np.clip(isi_cv / 2.0, 0.0, 1.0))
        else:
            # Dense degradation (noise, SR) → rate coding preferred → low CV good
            r5 = float(np.clip(1.0 - (isi_cv - 0.5) / 1.5, 0.0, 1.0))

        r = 0.30 * r1 + 0.20 * r2 + 0.25 * r3 + 0.15 * r4 + 0.10 * r5

        sub = {
            "cka_alignment": round(r1, 4),
            "rank_ratio": round(r2, 4),
            "viable_neurons": round(r3, 4),
            "temporal_efficiency": round(r4, 4),
            "isi_coding_match": round(r5, 4),
        }
        return float(r), sub

    # ── Bottleneck identification ──────────────────────────────────────

    def _bottleneck(self, comp: SAIComponents) -> str:
        scores = {
            "quality":         comp.q_score,
            "energy":          comp.e_score,
            "frequency":       comp.f_score,
            "representation":  comp.r_score,
        }
        return min(scores, key=scores.get)

    # ── Interpretation & recommendations ──────────────────────────────

    def _interpret(self, comp: SAIComponents, task: str) -> Tuple[str, List[str]]:
        msgs, recs = [], []

        if comp.sai >= 75:
            msgs.append(f"Task '{task}' is highly suitable for SNN processing.")
        elif comp.sai >= 55:
            msgs.append(f"Task '{task}' is moderately suitable — Hybrid model recommended.")
        elif comp.sai >= 35:
            msgs.append(f"Task '{task}' is poorly suited for pure SNN — prefer ANN baseline.")
        else:
            msgs.append(f"Task '{task}' is NOT suitable for SNN — use ANN.")

        if comp.q_psnr_gap < -3.0:
            msgs.append(f"Large quality gap ({comp.q_psnr_gap:.1f} dB) vs ANN.")
        if comp.f_hf_psnr_gap < -3.0:
            msgs.append("SNN severely underperforms in the high-frequency band.")
            recs.append("Apply HF-Enhancement Module (HFEM) or replace decoder with ANN.")
        if comp.f_scsm < 0.3:
            msgs.append("Dense degradation signal poorly matched to sparse spike coding.")
            recs.append("Consider ANN or Hybrid (dense tasks need continuous representation).")
        if comp.r_cka_mean < 0.45:
            msgs.append("Low SNN-ANN representation similarity — SNN learns very different features.")
            recs.append("Apply ANN→SNN representation distillation to align features.")
        if comp.r_viable_ratio < 0.40:
            msgs.append(f"High dead neuron ratio ({100*(1-comp.r_viable_ratio):.0f}%).")
            recs.append("Tune LIF threshold, use PLIF or ALIF for better neuron utilization.")
        if comp.e_firing_rate > 0.5:
            msgs.append(f"High firing rate ({100*comp.e_firing_rate:.0f}%) negates energy advantage.")
        if comp.r_temporal_redundancy > 0.6:
            msgs.append("High temporal redundancy — timesteps T can be reduced safely.")
            recs.append("Reduce T; apply temporal sparsity regularization.")

        return ". ".join(msgs), recs

    # ── Main API ──────────────────────────────────────────────────────

    def compute(
        self,
        task: str,
        # Q inputs
        psnr_snn: float = 0.0, psnr_ann: float = 0.0,
        ssim_snn: float = 0.0, ssim_ann: float = 0.0,
        # E inputs
        firing_rate: float = 0.20, synops_ratio: Optional[float] = None,
        # F inputs
        scsm: float = 0.50, bff: float = 0.50,
        hf_psnr_gap: float = 0.0,
        wavelet_hh_ratio: float = 1.0,
        error_centroid_snn: float = 0.30,
        error_centroid_ann: float = 0.30,
        # R inputs
        cka_mean: float = 0.70, rank_ratio: float = 0.80,
        viable_ratio: float = 0.50, temporal_redundancy: float = 0.50,
        isi_cv: float = 1.0,
        task_type: str = "sparse",
    ) -> SAIReport:
        """Compute SAI for one task. All inputs are plain Python floats."""
        comp = SAIComponents()

        comp.q_score, comp.q_psnr_gap, comp.q_ssim_gap = self._q_score(
            psnr_snn, psnr_ann, ssim_snn, ssim_ann
        )

        comp.e_firing_rate = firing_rate
        comp.e_score, comp.e_energy_ratio = self._e_score(firing_rate, synops_ratio)
        comp.e_synops_ratio = synops_ratio or firing_rate

        comp.f_scsm = scsm
        comp.f_bff = bff
        comp.f_hf_psnr_gap = hf_psnr_gap
        comp.f_wavelet_hh = wavelet_hh_ratio
        comp.f_error_centroid_shift = error_centroid_snn - error_centroid_ann
        comp.f_score, f_sub = self._f_score(
            scsm, bff, hf_psnr_gap, wavelet_hh_ratio,
            error_centroid_snn, error_centroid_ann,
        )

        comp.r_cka_mean = cka_mean
        comp.r_rank_ratio = rank_ratio
        comp.r_viable_ratio = viable_ratio
        comp.r_temporal_redundancy = temporal_redundancy
        comp.r_isi_cv = isi_cv
        comp.r_score, r_sub = self._r_score(
            cka_mean, rank_ratio, viable_ratio,
            temporal_redundancy, isi_cv, task_type,
        )

        w = self.weights
        comp.sai = (
            w["quality"]        * comp.q_score * 100
            + w["energy"]       * comp.e_score * 100
            + w["frequency"]    * comp.f_score * 100
            + w["representation"] * comp.r_score * 100
        )
        comp.level = _level(comp.sai)
        comp.bottleneck = self._bottleneck(comp)

        interp, recs = self._interpret(comp, task)
        return SAIReport(
            task=task,
            components=comp,
            weights=dict(self.weights),
            interpretation=interp,
            recommendations=recs,
            f_subcomponents=f_sub,
            r_subcomponents=r_sub,
        )

    # ── Display helpers ───────────────────────────────────────────────

    def print_radar(self, report: SAIReport) -> None:
        """ASCII radar bar for one task."""
        c = report.components

        def bar(v: float, w: int = 20) -> str:
            n = max(0, min(w, int(round(v * w))))
            return "█" * n + "░" * (w - n)

        print(f"\n  ┌── SAI Radar: {report.task.upper()} ──────────────────────────── SAI={c.sai:.1f} [{c.level}]")
        print(f"  │  Q [{bar(c.q_score)}] {c.q_score*100:5.1f}%  quality gap ({c.q_psnr_gap:+.2f} dB)")
        print(f"  │  E [{bar(c.e_score)}] {c.e_score*100:5.1f}%  energy benefit (rate={c.e_firing_rate:.3f})")
        print(f"  │  F [{bar(c.f_score)}] {c.f_score*100:5.1f}%  freq adaptation")
        for k, v in report.f_subcomponents.items():
            print(f"  │      {k:<22} {bar(v, 12)} {v:.3f}")
        print(f"  │  R [{bar(c.r_score)}] {c.r_score*100:5.1f}%  representation fit")
        for k, v in report.r_subcomponents.items():
            print(f"  │      {k:<22} {bar(v, 12)} {v:.3f}")
        print(f"  │")
        print(f"  │  Bottleneck: {c.bottleneck.upper()}")
        print(f"  └─────────────────────────────────────────────────────────────")
        if report.recommendations:
            print(f"  Recommendations:")
            for rec in report.recommendations:
                print(f"    → {rec}")

    @staticmethod
    def compare_tasks(
        reports: Dict[str, "SAIReport"],
        task_order: Optional[List[str]] = None,
    ) -> None:
        """Print cross-task SAI comparison table."""
        if task_order is None:
            task_order = ["deraining", "dehazing", "denoising", "superresolution", "lowlight"]
        ordered = [t for t in task_order if t in reports]
        ordered += [t for t in reports if t not in task_order]

        print(f"\n{'='*95}")
        print("SNN APPLICABILITY INDEX (SAI) — Cross-Task Comparison")
        print(f"{'='*95}")
        print(f"  {'Task':<18} {'SAI':>6} {'Level':<10} "
              f"{'Q':>6} {'E':>6} {'F':>6} {'R':>6}   "
              f"{'Bottleneck':<16} {'HF-gap':>8} {'CKA':>6}")
        print("  " + "-" * 91)
        for task in ordered:
            r = reports[task]
            c = r.components
            print(f"  {task:<18} {c.sai:>6.1f} {c.level:<10} "
                  f"{c.q_score*100:>6.1f} {c.e_score*100:>6.1f} "
                  f"{c.f_score*100:>6.1f} {c.r_score*100:>6.1f}   "
                  f"{c.bottleneck:<16} {c.f_hf_psnr_gap:>8.2f} {c.r_cka_mean:>6.3f}")
        print(f"{'='*95}")
        print("  Weights: Q=35%  E=15%  F=30%  R=20%")
        print("  Level: HIGH≥75 | MEDIUM≥55 | LOW≥35 | VERY_LOW<35")

    @staticmethod
    def frequency_axis_plot(reports: Dict[str, "SAIReport"]) -> None:
        """
        ASCII scatter: x = SCSM (task degradation sparsity),
                       y = SAI score.
        This is the 'frequency-and-representation boundary' visualization.
        """
        W, H = 55, 15
        grid = [[" "] * W for _ in range(H)]
        labels = []
        sym_map = {
            "deraining": "D", "denoising": "N", "dehazing": "H",
            "superresolution": "S", "lowlight": "L",
        }
        for task, rpt in reports.items():
            sym = sym_map.get(task, task[0].upper())
            x = int(np.clip(rpt.components.f_scsm, 0, 1) * (W - 1))
            y = int((1.0 - np.clip(rpt.components.sai / 100, 0, 1)) * (H - 1))
            grid[y][x] = sym
            labels.append(f"  {sym}={task:<18} SAI={rpt.components.sai:.1f} [{rpt.components.level}]")

        print(f"\n  SAI Score")
        print(f"  100 ┤")
        for i, row in enumerate(grid):
            score_val = 100 - i * 100 / (H - 1)
            print(f"  {score_val:3.0f} │ {''.join(row)}")
        print(f"      └" + "─" * W)
        print(f"       0%            Task Signal Sparsity (SCSM)            100%")
        print()
        for lbl in labels:
            print(lbl)
        print(f"  Top-right = SNN favorable (high SAI, sparse degradation)")
        print(f"  Bottom-left = SNN unfavorable (low SAI, dense degradation)")
