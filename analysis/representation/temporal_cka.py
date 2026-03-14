"""
Temporal CKA Analysis  (B1 extension)
======================================
Extends standard CKA to the *time dimension* of SNN representations,
enabling three new research questions:

  1. Temporal convergence
     Does the SNN representation become more ANN-like as t → T?
     → "More timesteps improve quality because representation converges to ANN."

  2. Temporal redundancy
     How similar are representations across different time steps?
     → High CKA(t_i, t_j) for i ≠ j means timesteps are wasteful.
     → Directly informs optimal T choice vs. energy trade-off.

  3. Cross-task temporal dynamics
     Do different tasks require different temporal accumulation patterns?
     → Deraining (sparse events) may show fast convergence.
     → Super-resolution (dense texture) may show slow / non-convergent dynamics.

Metric definitions
------------------
temporal_cka_matrix[i, j] = CKA(feat_{t=i},  feat_{t=j})
convergence_curve[t]       = CKA(feat_{t},    feat_{t=T-1})
temporal_stability         = mean CKA between adjacent time steps
temporal_redundancy        = mean CKA between non-adjacent pairs (gap ≥ 2)
snn_ann_cka_at_T           = CKA(feat_{t=T-1},  ann_feat)

Reference:
  Kornblith et al., "Similarity of Neural Network Representations Revisited",
  ICML 2019.
  Zheng et al., "Going Deeper With Directly-Trained Larger Spiking Neural
  Networks", AAAI 2021 — first to observe temporal representation drift.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalCKAResult:
    """Results of temporal CKA analysis for one layer."""
    layer_name: str
    n_timesteps: int

    # (T × T) matrix; element [i,j] = CKA(feat_t=i, feat_t=j)
    temporal_cka_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1))
    )

    # CKA(feat_t, feat_{T-1}): how close each step is to the final step
    convergence_curve: List[float] = field(default_factory=list)

    # mean CKA between adjacent steps (t, t+1)
    temporal_stability: float = 0.0

    # mean CKA between non-adjacent pairs (gap ≥ 2)
    temporal_redundancy: float = 0.0

    # Has the representation stopped changing? (True if last delta < threshold)
    is_converged: bool = False

    # CKA between SNN at t=T-1 and ANN (if ANN features provided)
    snn_ann_cka_at_T: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class TemporalCKAAnalyzer:
    """
    Analyses how SNN representations evolve across time steps.

    Usage
    -----
    analyzer = TemporalCKAAnalyzer()

    # temporal_features: {layer: np.ndarray (T, N, D)}  — SNN activations
    # ann_features:      {layer: np.ndarray (N, D)}      — ANN activations (optional)
    results = analyzer.analyze_all_layers(temporal_features, ann_features)
    analyzer.print_summary(results)
    """

    def __init__(self, convergence_delta: float = 0.05):
        """
        Args:
            convergence_delta: if CKA change between last two steps < this,
                               the representation is considered converged.
        """
        self.convergence_delta = convergence_delta

    # ── CKA primitives ────────────────────────────────────────────────

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Double-center a kernel matrix: H K H, where H = I − 1/n·11ᵀ."""
        n = K.shape[0]
        H = np.eye(n) - 1.0 / n
        return H @ K @ H

    @staticmethod
    def _linear_kernel(X: np.ndarray) -> np.ndarray:
        return X @ X.T

    def _hsic(self, Kx: np.ndarray, Ky: np.ndarray) -> float:
        n = Kx.shape[0]
        if n < 4:
            return 0.0
        Kx_c = self._center_kernel(Kx)
        Ky_c = self._center_kernel(Ky)
        return float(np.trace(Kx_c @ Ky_c) / (n - 1) ** 2)

    def _cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Unbiased linear CKA between two feature matrices (N, D_x), (N, D_y)."""
        N = min(X.shape[0], Y.shape[0])
        X, Y = X[:N].astype(np.float64), Y[:N].astype(np.float64)
        Kx = self._linear_kernel(X)
        Ky = self._linear_kernel(Y)
        hsic_xy = self._hsic(Kx, Ky)
        hsic_xx = self._hsic(Kx, Kx)
        hsic_yy = self._hsic(Ky, Ky)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-8
        return float(np.clip(hsic_xy / denom, 0.0, 1.0))

    # ── Per-layer analysis ─────────────────────────────────────────────

    def analyze_layer(
        self,
        temporal_features: np.ndarray,          # (T, N, D) SNN features
        layer_name: str = "layer",
        ann_features: Optional[np.ndarray] = None,  # (N, D) ANN reference
        max_samples: int = 512,                 # cap for speed
    ) -> TemporalCKAResult:
        """
        Analyse temporal evolution of SNN representations for one layer.

        Args:
            temporal_features: (T, N, D) — SNN feature activations per timestep
            layer_name: identifier string
            ann_features: (N, D) — ANN features for convergence-target comparison
            max_samples: maximum N to use (subsampled for speed)
        """
        T, N, D = temporal_features.shape
        result = TemporalCKAResult(layer_name=layer_name, n_timesteps=T)

        if T < 2:
            result.temporal_cka_matrix = np.eye(T)
            result.convergence_curve = [1.0]
            return result

        # Sub-sample samples for speed
        if N > max_samples:
            idx = np.random.choice(N, max_samples, replace=False)
            temporal_features = temporal_features[:, idx, :]
            if ann_features is not None:
                ann_features = ann_features[idx, :]
            N = max_samples

        # Reduce dimension if D is very large (random projection)
        D_max = 256
        if D > D_max:
            proj = np.random.randn(D, D_max).astype(np.float64) / np.sqrt(D_max)
            temporal_features = temporal_features.reshape(T * N, D) @ proj
            temporal_features = temporal_features.reshape(T, N, D_max)
            if ann_features is not None:
                ann_features = (ann_features.astype(np.float64) @ proj)

        # ── Temporal CKA matrix (T × T) ─────────────────────────────
        cka_mat = np.eye(T)
        for i in range(T):
            for j in range(i + 1, T):
                c = self._cka(temporal_features[i], temporal_features[j])
                cka_mat[i, j] = c
                cka_mat[j, i] = c
        result.temporal_cka_matrix = cka_mat

        # ── Convergence curve: CKA(feat_t, feat_{T-1}) ──────────────
        final = temporal_features[-1]
        conv = [self._cka(temporal_features[t], final) for t in range(T)]
        result.convergence_curve = conv

        # ── Temporal stability: mean CKA between adjacent steps ──────
        adj_ckas = [cka_mat[t, t + 1] for t in range(T - 1)]
        result.temporal_stability = float(np.mean(adj_ckas))

        # ── Temporal redundancy: mean CKA of non-adjacent pairs ──────
        non_adj = [
            cka_mat[i, j]
            for i in range(T)
            for j in range(i + 2, T)
        ]
        result.temporal_redundancy = float(np.mean(non_adj)) if non_adj else 0.0

        # ── Convergence detection ─────────────────────────────────────
        if len(conv) >= 3:
            delta = abs(conv[-1] - conv[-2])
            result.is_converged = bool(delta < self.convergence_delta)

        # ── SNN–ANN CKA at final timestep ────────────────────────────
        if ann_features is not None:
            result.snn_ann_cka_at_T = self._cka(final, ann_features)

        return result

    # ── Multi-layer analysis ───────────────────────────────────────────

    def analyze_all_layers(
        self,
        spike_records: Dict[str, np.ndarray],       # {layer: (T, B, C, H, W)}
        ann_features: Optional[Dict[str, np.ndarray]] = None,  # {layer: (N, D)}
        flatten_spatial: bool = True,
    ) -> Dict[str, TemporalCKAResult]:
        """
        Analyse temporal CKA for all recorded SNN layers.

        Args:
            spike_records: raw spike tensors from the SNN forward pass
            ann_features: optional dict of ANN feature matrices for comparison
            flatten_spatial: if True, reshape (T, B, C, H, W) → (T, B, C·H·W)
        """
        results: Dict[str, TemporalCKAResult] = {}

        for layer, spikes in spike_records.items():
            T, B, C, H, W = spikes.shape
            if flatten_spatial:
                feats = spikes.reshape(T, B, C * H * W)
            else:
                feats = spikes.reshape(T, B, C)

            ann_ref = None
            if ann_features and layer in ann_features:
                ann_ref = ann_features[layer]

            results[layer] = self.analyze_layer(feats, layer, ann_ref)

        return results

    # ── Aggregate metrics ─────────────────────────────────────────────

    def redundancy_score(self, results: Dict[str, TemporalCKAResult]) -> float:
        """Mean temporal redundancy across all layers (used in SAI.R4)."""
        vals = [r.temporal_redundancy for r in results.values()]
        return float(np.mean(vals)) if vals else 0.0

    def stability_score(self, results: Dict[str, TemporalCKAResult]) -> float:
        """Mean temporal stability across all layers."""
        vals = [r.temporal_stability for r in results.values()]
        return float(np.mean(vals)) if vals else 0.0

    def snn_ann_alignment(self, results: Dict[str, TemporalCKAResult]) -> float:
        """Mean SNN–ANN CKA at final timestep across all layers (used in SAI.R1)."""
        vals = [r.snn_ann_cka_at_T for r in results.values() if r.snn_ann_cka_at_T > 0]
        return float(np.mean(vals)) if vals else 0.0

    # ── Display ────────────────────────────────────────────────────────

    def print_summary(self, results: Dict[str, TemporalCKAResult]) -> None:
        print(f"\n{'='*80}")
        print("Temporal CKA Analysis — SNN Representation Evolution Across Time Steps")
        print(f"{'='*80}")
        print(f"  {'Layer':<22} {'T':>4} {'Stability':>10} "
              f"{'Redundancy':>11} {'Converged':>10} {'SNN-ANN':>9}")
        print("  " + "-" * 70)
        for layer, r in results.items():
            print(f"  {layer[:21]:<22} {r.n_timesteps:>4} "
                  f"{r.temporal_stability:>10.4f} "
                  f"{r.temporal_redundancy:>11.4f} "
                  f"{'Yes' if r.is_converged else 'No':>10} "
                  f"{r.snn_ann_cka_at_T:>9.4f}")
        print(f"{'='*80}")
        print("  Stability:   mean CKA between adjacent timesteps (high = slow drift)")
        print("  Redundancy:  mean CKA between non-adjacent pairs (high = wasteful T)")
        print("  SNN-ANN:     CKA vs ANN features at final timestep")
        # Aggregate
        agg_redund = self.redundancy_score(results)
        agg_align  = self.snn_ann_alignment(results)
        print(f"\n  → Mean redundancy: {agg_redund:.4f}  "
              f"(SAI R4 input: temporal_efficiency = {1-agg_redund:.4f})")
        print(f"  → Mean SNN-ANN alignment: {agg_align:.4f}  (SAI R1 input)")

    def print_convergence_curves(self, results: Dict[str, TemporalCKAResult]) -> None:
        """ASCII plot of convergence curves per layer."""
        print(f"\n  Convergence Curves (CKA to final timestep)")
        for layer, r in results.items():
            if not r.convergence_curve:
                continue
            print(f"  {layer[:18]:<18}: ", end="")
            for v in r.convergence_curve:
                n = int(v * 10)
                print(f"{'█'*n}{'░'*(10-n)}[{v:.2f}] ", end="")
            conv_str = "✓converged" if r.is_converged else "✗not-conv."
            print(f"← {conv_str}")
