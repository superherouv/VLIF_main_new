"""
Linear Probe Analysis  (B5 — Information Proxy Metrics)
=========================================================
Answers the question: **"Does SNN fail because it can't *learn* the features,
or because it can't *decode* them from the learned representation?"**

Probes
------
1. **Linear probe**  — train a linear classifier on *frozen* features.
   Target labels (auto-computed from input images, no human annotation needed):
     - degradation type (0=rain, 1=noise, 2=haze, …)  → task discriminability
     - frequency band label (LF/MF/HF of degradation energy) → freq sensitivity
     - edge presence (Sobel response > threshold)      → structural sensitivity

2. **kNN probe** — k-nearest-neighbour classification (geometry-based).
   Measures *local* feature structure without optimising a head.

3. **Feature entropy** — mean activation entropy per dimension.
   High entropy = information-rich; low entropy = collapsed features.

4. **Feature sparsity** — Gini coefficient of |activation| values.
   High sparsity + high linear accuracy → efficient sparse coding.

5. **Quality correlation** — correlation between feature magnitude and
   per-sample reconstruction PSNR.
   Strong correlation → features encode quality-relevant information.

All methods are pure NumPy; no external ML library required.

Reference:
  Alain & Bengio, "Understanding Intermediate Layers Using Linear Classifier
  Probes", ICLR 2017 Workshop.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LinearProbeResult:
    """Per-layer probe analysis result."""
    layer_name: str
    model_type: str    # "snn" | "ann" | "hybrid"
    n_samples: int = 0

    # Probe accuracies
    linear_acc: float = 0.0   # linear probe accuracy [0, 1]
    knn_acc: float = 0.0      # kNN probe accuracy [0, 1]

    # Feature statistics
    feature_entropy: float = 0.0    # mean activation entropy [nats]
    feature_sparsity: float = 0.0   # Gini coefficient [0, 1]
    pca_var_3: float = 0.0          # variance explained by top-3 PCA dims [0, 1]

    # Quality correlation: |feature| vs per-sample PSNR
    quality_correlation: float = 0.0  # Pearson r [-1, 1]

    # Shortcut detection: does linear probe on degradation *type* exceed kNN?
    # (big gap → linear head learns shortcut, not geometry)
    linear_knn_gap: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class LinearProbeAnalyzer:
    """
    Evaluates feature representational quality via linear and kNN probes.

    Usage
    -----
    analyzer = LinearProbeAnalyzer()

    # features: (N, D) float array
    # labels:   (N,)   int array   (can be any of the auto-label types above)
    result = analyzer.analyze_layer(features, labels, "enc3", "snn")
    print(result)

    # Or compare SNN vs ANN across all layers:
    comparison = analyzer.compare_snn_ann(snn_feats, ann_feats, labels)
    analyzer.print_comparison(comparison)
    """

    def __init__(
        self,
        n_neighbors: int = 7,
        probe_lr: float = 0.02,
        probe_max_iter: int = 150,
        pca_n_components: int = 3,
    ):
        self.k = n_neighbors
        self.lr = probe_lr
        self.max_iter = probe_max_iter
        self.pca_n = pca_n_components

    # ── Preprocessing ──────────────────────────────────────────────────

    def _preprocess(self, X: np.ndarray, max_dim: int = 128) -> np.ndarray:
        """Normalise and (optionally) PCA-reduce features."""
        X = X.astype(np.float64)
        # Z-score per feature
        mu = X.mean(0)
        sd = X.std(0) + 1e-8
        X = (X - mu) / sd
        # Random-projection if still large (preserves distances)
        if X.shape[1] > max_dim:
            rng = np.random.RandomState(0)
            proj = rng.randn(X.shape[1], max_dim) / np.sqrt(max_dim)
            X = X @ proj
        return X

    # ── Linear probe (multinomial logistic regression) ─────────────────

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        e = np.exp(Z - Z.max(1, keepdims=True))
        return e / (e.sum(1, keepdims=True) + 1e-8)

    def _train_linear_probe(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Train multinomial logistic regression via mini-batch gradient descent.
        Returns weight matrix W (D, n_classes).
        """
        N, D = X.shape
        n_cls = int(y.max()) + 1
        W = np.zeros((D, n_cls), dtype=np.float64)
        b = np.zeros(n_cls, dtype=np.float64)

        for _ in range(self.max_iter):
            logits = X @ W + b                          # (N, C)
            probs = self._softmax(logits)               # (N, C)
            Y_oh = np.zeros_like(probs)
            Y_oh[np.arange(N), y.astype(int)] = 1.0
            dL = (probs - Y_oh) / N                    # (N, C)
            W -= self.lr * (X.T @ dL) + 1e-4 * W      # L2 reg
            b -= self.lr * dL.sum(0)

        return W, b

    def _linear_probe_accuracy(
        self, X: np.ndarray, y: np.ndarray
    ) -> float:
        """5-fold cross-validated linear probe accuracy."""
        N = X.shape[0]
        n_cls = int(y.max()) + 1
        if n_cls < 2 or N < n_cls * 3:
            return 1.0 / n_cls  # chance level

        n_folds = min(5, N // n_cls)
        if n_folds < 2:
            # No CV possible: train on all, evaluate on all (optimistic)
            W, b = self._train_linear_probe(X, y)
            preds = (X @ W + b).argmax(1)
            return float((preds == y).mean())

        fold_size = N // n_folds
        accs = []
        for f in range(n_folds):
            val_idx = np.arange(f * fold_size, (f + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(N), val_idx)
            if len(train_idx) < n_cls:
                continue
            W, b = self._train_linear_probe(X[train_idx], y[train_idx])
            preds = (X[val_idx] @ W + b).argmax(1)
            accs.append(float((preds == y[val_idx]).mean()))
        return float(np.mean(accs)) if accs else 1.0 / n_cls

    # ── kNN probe ──────────────────────────────────────────────────────

    def _knn_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Leave-one-out kNN accuracy."""
        N = X.shape[0]
        if N < self.k + 1:
            return 0.0

        # Pairwise L2² distance matrix
        sq = (X ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        D2 += np.eye(N) * 1e9   # mask self

        correct = 0
        for i in range(N):
            nn_idx = np.argsort(D2[i])[:self.k]
            nn_labels = y[nn_idx].astype(int)
            pred = np.bincount(nn_labels, minlength=int(y.max()) + 1).argmax()
            correct += int(pred == int(y[i]))
        return float(correct / N)

    # ── Feature statistics ─────────────────────────────────────────────

    def _entropy(self, X: np.ndarray, n_bins: int = 20) -> float:
        """Mean marginal entropy over feature dimensions."""
        X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)
        D = X.shape[1]
        entropies = []
        for d in range(min(D, 64)):   # sample up to 64 dims
            hist, _ = np.histogram(X_norm[:, d], bins=n_bins, density=True)
            hist = hist / (hist.sum() + 1e-10)
            h = -np.sum(hist * np.log(hist + 1e-10))
            entropies.append(float(h))
        return float(np.mean(entropies)) if entropies else 0.0

    @staticmethod
    def _gini(X: np.ndarray) -> float:
        """Gini coefficient of |X| values (0 = equal; 1 = fully sparse)."""
        vals = np.sort(np.abs(X).flatten())
        n = len(vals)
        if vals.sum() < 1e-10:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * idx - n - 1).dot(vals) / (n * vals.sum()))

    def _pca_variance(self, X: np.ndarray) -> float:
        """Fraction of variance explained by top-n PCA components."""
        X_c = X - X.mean(0)
        try:
            _, S, _ = np.linalg.svd(X_c, full_matrices=False)
            var_ratio = (S ** 2) / ((S ** 2).sum() + 1e-10)
            return float(var_ratio[:self.pca_n].sum())
        except np.linalg.LinAlgError:
            return 0.0

    # ── Main API ───────────────────────────────────────────────────────

    def analyze_layer(
        self,
        features: np.ndarray,                         # (N, D)
        labels: Optional[np.ndarray] = None,           # (N,) int
        layer_name: str = "layer",
        model_type: str = "snn",
        quality_scores: Optional[np.ndarray] = None,  # (N,) per-sample PSNR
    ) -> LinearProbeResult:
        """
        Analyse representational quality for one feature layer.

        Args:
            features:       (N, D) feature matrix
            labels:         (N,) integer class labels (optional)
            layer_name:     identifier
            model_type:     "snn" | "ann" | "hybrid"
            quality_scores: (N,) per-sample PSNR (for quality correlation)
        """
        N, D = features.shape
        result = LinearProbeResult(
            layer_name=layer_name, model_type=model_type, n_samples=N
        )

        X = self._preprocess(features)

        # ── Probe accuracies ─────────────────────────────────────────
        if labels is not None and len(np.unique(labels)) >= 2:
            result.linear_acc = self._linear_probe_accuracy(X, labels)
            result.knn_acc    = self._knn_accuracy(X, labels)
        else:
            n_cls = int(labels.max()) + 1 if labels is not None and len(labels) > 0 else 2
            result.linear_acc = 1.0 / n_cls
            result.knn_acc    = 1.0 / n_cls

        result.linear_knn_gap = result.linear_acc - result.knn_acc

        # ── Feature statistics ────────────────────────────────────────
        result.feature_entropy  = self._entropy(features)
        result.feature_sparsity = self._gini(features)
        result.pca_var_3        = self._pca_variance(X)

        # ── Quality correlation ───────────────────────────────────────
        if quality_scores is not None and len(quality_scores) == N:
            mag = np.linalg.norm(features, axis=1)
            if mag.std() > 1e-8 and quality_scores.std() > 1e-8:
                r = np.corrcoef(mag, quality_scores)[0, 1]
                result.quality_correlation = float(r)

        return result

    def compare_snn_ann(
        self,
        snn_feats: Dict[str, np.ndarray],          # {layer: (N, D)}
        ann_feats: Dict[str, np.ndarray],           # {layer: (N, D)}
        labels: Optional[np.ndarray] = None,
        quality_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[LinearProbeResult, LinearProbeResult]]:
        """
        Compare linear probe results between SNN and ANN layers.

        Returns:
            {layer: (snn_result, ann_result)}
        """
        comparison = {}
        for layer in snn_feats:
            if layer not in ann_feats:
                continue
            snn_r = self.analyze_layer(snn_feats[layer], labels, layer, "snn", quality_scores)
            ann_r = self.analyze_layer(ann_feats[layer], labels, layer, "ann", quality_scores)
            comparison[layer] = (snn_r, ann_r)
        return comparison

    # ── Display ────────────────────────────────────────────────────────

    @staticmethod
    def print_comparison(
        comparison: Dict[str, Tuple[LinearProbeResult, LinearProbeResult]],
    ) -> None:
        print(f"\n{'='*90}")
        print("Linear Probe Analysis: SNN vs ANN Feature Quality")
        print(f"{'='*90}")
        print(f"  {'Layer':<20}  {'── SNN ──':^40}  {'── ANN ──':^40}")
        hdr2 = (f"  {'':20}  {'Lin':>6} {'kNN':>6} {'Entropy':>8} {'Sparse':>8} {'Corr':>6}"
                f"  {'Lin':>6} {'kNN':>6} {'Entropy':>8} {'Sparse':>8} {'Corr':>6}")
        print(hdr2)
        print("  " + "-" * 86)
        for layer, (snn, ann) in comparison.items():
            print(f"  {layer[:19]:<20}  "
                  f"{snn.linear_acc:>6.3f} {snn.knn_acc:>6.3f} "
                  f"{snn.feature_entropy:>8.3f} {snn.feature_sparsity:>8.3f} "
                  f"{snn.quality_correlation:>6.3f}  "
                  f"{ann.linear_acc:>6.3f} {ann.knn_acc:>6.3f} "
                  f"{ann.feature_entropy:>8.3f} {ann.feature_sparsity:>8.3f} "
                  f"{ann.quality_correlation:>6.3f}")
        print(f"{'='*90}")
        print("  Lin/kNN: classification accuracy for degradation-type labels")
        print("  Entropy: feature information richness (higher = more informative)")
        print("  Sparse:  Gini sparsity (higher = sparser, more spike-like coding)")
        print("  Corr:    Pearson r between feature magnitude and per-sample PSNR")


# ──────────────────────────────────────────────────────────────────────────────
# Auto-label generators (no human annotation required)
# ──────────────────────────────────────────────────────────────────────────────

def make_degradation_type_labels(n_per_task: int, tasks: List[str]) -> np.ndarray:
    """
    Generate integer labels for degradation type classification probe.
    task i → label i.  Assumes n_per_task samples per task, stacked in order.
    """
    return np.repeat(np.arange(len(tasks)), n_per_task)


def make_frequency_band_labels(
    images: np.ndarray,    # (N, C, H, W) degraded images
    lf_thresh: float = 0.15,
    hf_thresh: float = 0.30,
) -> np.ndarray:
    """
    Assign each image to LF (0), MF (1), or HF (2) based on the dominant
    frequency band of its FFT energy.
    """
    labels = []
    for img in images:
        gray = img.mean(0) if img.ndim == 3 else img
        fft = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.abs(fft)

        H, W = gray.shape
        fu = np.fft.fftshift(np.fft.fftfreq(W))
        fv = np.fft.fftshift(np.fft.fftfreq(H))
        FU, FV = np.meshgrid(fu, fv)
        r = np.sqrt(FU ** 2 + FV ** 2)

        lf_e = float(mag[r < lf_thresh].sum())
        mf_e = float(mag[(r >= lf_thresh) & (r <= hf_thresh)].sum())
        hf_e = float(mag[r > hf_thresh].sum())

        label = int(np.argmax([lf_e, mf_e, hf_e]))
        labels.append(label)

    return np.array(labels, dtype=int)


def make_edge_labels(
    images: np.ndarray,    # (N, C, H, W)
    threshold_percentile: float = 70.0,
) -> np.ndarray:
    """
    Binary edge-presence label: 1 if image has strong edges (Sobel), 0 otherwise.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T

    labels = []
    for img in images:
        gray = img.mean(0) if img.ndim == 3 else img
        # Manual 2D convolution via stride tricks (small kernel)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(gray, 1, mode='reflect')
        patches = sliding_window_view(pad, (3, 3))  # (H, W, 3, 3)
        gx = (patches * sobel_x).sum((-1, -2))
        gy = (patches * sobel_y).sum((-1, -2))
        strength = np.sqrt(gx ** 2 + gy ** 2).mean()
        labels.append(float(strength))

    threshold = np.percentile(labels, threshold_percentile)
    return (np.array(labels) > threshold).astype(int)


# Expose List in this module's namespace for the type hint above
from typing import List
