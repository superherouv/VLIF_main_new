"""
Spike–Frequency Coupling Analysis  (A3.1 + A3.2)
==================================================
This module tests the core claim from VLIF:
  "Traditional LIF in deraining acts more like a high-frequency indicator."

It does so by quantifying the *spatial overlap* between spike activation maps
and Fourier frequency bands, and by analysing the *temporal power spectral
density* (PSD) of spike sequences.

Two analyses
------------
A3.1  Spike–Band Overlap
    For each SNN layer, compute the overlap between spike activation map and
    frequency-band masks (LF / MF / HF) derived from the input/target features.
    Tells us: do spikes preferentially fire where the *signal* is low-freq or high-freq?

A3.2  Spike Temporal PSD
    For each layer, compute the power spectral density of the spike sequence
    across time (treating each neuron's spike train as a 1D discrete signal).
    Tells us: does the neural population encode the task signal at
    low temporal frequencies (rate coding) or high temporal frequencies (burst)?

Scientific insight
------------------
If VLIF's claim is task-specific (only deraining), we should see:
  deraining:       spike–HF overlap HIGH  (spikes activate on rain streaks)
  super-res / SR:  spike–HF overlap LOW   (spikes don't fire on texture detail)
  dehazing:        spike–LF overlap HIGH  (spikes fire on scene-level haze)

Combining with temporal PSD:
  deraining:  high temporal bandwidth needed (streak timing matters)
  denoising:  moderate bandwidth (noise is spatially diffuse)
  dehazing:   low temporal bandwidth (smooth global change)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeBandOverlapResult:
    """
    Per-layer spike–frequency-band overlap statistics.

    Overlap is computed as the inner product between:
      - normalised spike activation map  (H × W mean over T and C)
      - binary/soft frequency-band mask  (H × W, derived from FFT of input)
    divided by the total spike energy, giving a value in [0, 1].
    """
    layer_name: str

    # Mean overlap with each frequency band
    lf_overlap: float = 0.0   # [0,1] — low-frequency band overlap
    mf_overlap: float = 0.0   # [0,1] — mid-frequency band overlap
    hf_overlap: float = 0.0   # [0,1] — high-frequency band overlap

    # Dominant band (which band captures most spike energy)
    dominant_band: str = "mf"  # "lf" | "mf" | "hf"

    # Ratio HF / LF (> 1 means spikes prefer high-freq regions)
    hf_lf_ratio: float = 1.0


@dataclass
class SpikeTemporalPSDResult:
    """
    Per-layer temporal power spectral density of spike sequences.

    Treats each neuron's binary spike train (T,) as a discrete 1D signal
    and computes the ensemble-average PSD.
    """
    layer_name: str
    T: int = 4

    # PSD at each temporal frequency bin (length T//2+1)
    psd: List[float] = field(default_factory=list)

    # Temporal bandwidth: first frequency bin where cumulative power > 95%
    temporal_bandwidth: float = 0.0

    # 1/f exponent: negative slope of log–log PSD (higher = more 1/f, slower dynamics)
    one_f_exponent: float = 0.0

    # Burstiness: fraction of power above the mean temporal frequency
    burstiness: float = 0.0

    # Coding type inferred from PSD shape
    coding_type: str = "mixed"  # "rate" | "burst" | "temporal" | "mixed"


@dataclass
class SpikeCouplingReport:
    """Combined report for one task / model."""
    task: str
    model_type: str = "snn"
    band_overlaps: Dict[str, SpikeBandOverlapResult] = field(default_factory=dict)
    temporal_psds: Dict[str, SpikeTemporalPSDResult] = field(default_factory=dict)

    # Aggregate signatures
    mean_hf_lf_ratio: float = 1.0   # mean over layers
    mean_temporal_bw: float = 0.0   # mean temporal bandwidth
    task_coding_signature: str = "" # combined description


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class SpikeFqCouplingAnalyzer:
    """
    Analyses the coupling between spike activation patterns and
    spatial / temporal frequency content.

    Usage
    -----
    analyzer = SpikeFqCouplingAnalyzer()

    # spike_records: {layer_name: np.ndarray shape (T, B, C, H, W)}
    # input_images:  np.ndarray shape (B, C, H, W)  [degraded input]
    report = analyzer.analyze(spike_records, input_images, task="deraining")
    analyzer.print_report(report)
    """

    def __init__(
        self,
        lf_cutoff: float = 0.15,   # c/px below = low-freq
        hf_cutoff: float = 0.30,   # c/px above = high-freq
    ):
        self.lf_cutoff = lf_cutoff
        self.hf_cutoff = hf_cutoff

    # ── Frequency-band masks from spatial image ────────────────────────

    def _freq_masks(
        self, image: np.ndarray,   # (H, W) grayscale
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spatial LF / MF / HF binary masks from image FFT.

        Returns
        -------
        lf_mask, mf_mask, hf_mask : each (H, W) float32 in [0, 1]
        """
        H, W = image.shape
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Frequency coordinate grid (cycles/pixel)
        fu = np.fft.fftshift(np.fft.fftfreq(W))
        fv = np.fft.fftshift(np.fft.fftfreq(H))
        FU, FV = np.meshgrid(fu, fv)
        freq_radius = np.sqrt(FU ** 2 + FV ** 2)

        lf_mask = (freq_radius < self.lf_cutoff).astype(np.float32)
        hf_mask = (freq_radius > self.hf_cutoff).astype(np.float32)
        mf_mask = 1.0 - lf_mask - hf_mask
        mf_mask = np.clip(mf_mask, 0.0, 1.0)

        # Soft masks weighted by magnitude (emphasise energy-rich frequency regions)
        lf_soft = lf_mask * magnitude
        mf_soft = mf_mask * magnitude
        hf_soft = hf_mask * magnitude

        # Convert to spatial domain (inverse FFT of masked magnitude)
        def _to_spatial(soft_mask: np.ndarray) -> np.ndarray:
            masked = soft_mask * np.exp(1j * np.angle(fft_shift))
            spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(masked)))
            mn, mx = spatial.min(), spatial.max()
            return (spatial - mn) / (mx - mn + 1e-8)

        return _to_spatial(lf_soft), _to_spatial(mf_soft), _to_spatial(hf_soft)

    # ── A3.1: Spike–Band Overlap ───────────────────────────────────────

    def _band_overlap(
        self,
        spikes: np.ndarray,     # (T, B, C, H, W) binary spikes
        images: np.ndarray,     # (B, C, H, W) reference images for mask
        layer_name: str,
    ) -> SpikeBandOverlapResult:
        """
        Compute overlap between per-layer spike map and freq-band masks.

        The spike activation map is the temporal mean of absolute spike values,
        averaged over channels:  spike_map[b] = mean_{t,c} |S[t,b,c,:,:]|
        """
        T, B, C, H, W = spikes.shape
        result = SpikeBandOverlapResult(layer_name=layer_name)

        # Aggregate over time and channel → (B, H, W)
        spike_map = np.abs(spikes).mean(axis=(0, 2))  # (B, H, W)

        lf_overlaps, mf_overlaps, hf_overlaps = [], [], []
        for b in range(min(B, 8)):  # process up to 8 samples
            gray = images[b].mean(0) if images[b].ndim == 3 else images[b]

            # Resize masks to match feature map if sizes differ
            if gray.shape != (H, W):
                h_s, w_s = gray.shape
                gray_r = _resize_nearest(gray, H, W)
            else:
                gray_r = gray

            lf_m, mf_m, hf_m = self._freq_masks(gray_r)

            smap = spike_map[b]  # (H, W)
            total = smap.sum() + 1e-8

            lf_overlaps.append(float((smap * lf_m).sum() / total))
            mf_overlaps.append(float((smap * mf_m).sum() / total))
            hf_overlaps.append(float((smap * hf_m).sum() / total))

        result.lf_overlap = float(np.mean(lf_overlaps))
        result.mf_overlap = float(np.mean(mf_overlaps))
        result.hf_overlap = float(np.mean(hf_overlaps))

        overlaps = {"lf": result.lf_overlap, "mf": result.mf_overlap, "hf": result.hf_overlap}
        result.dominant_band = max(overlaps, key=overlaps.get)
        result.hf_lf_ratio = result.hf_overlap / (result.lf_overlap + 1e-8)

        return result

    # ── A3.2: Spike Temporal PSD ───────────────────────────────────────

    def _temporal_psd(
        self,
        spikes: np.ndarray,   # (T, B, C, H, W)
        layer_name: str,
    ) -> SpikeTemporalPSDResult:
        """
        Compute ensemble-average temporal PSD of spike trains.

        For each neuron (b, c, h, w), the spike train is the T-length binary
        vector.  We compute the PSD and average over all neurons.
        """
        T, B, C, H, W = spikes.shape
        result = SpikeTemporalPSDResult(layer_name=layer_name, T=T)

        if T < 2:
            result.psd = [1.0]
            return result

        # Reshape to (N_neurons, T)
        trains = spikes.transpose(1, 2, 3, 4, 0).reshape(-1, T)  # (N, T)

        # Sub-sample for speed (max 2000 neurons)
        if trains.shape[0] > 2000:
            idx = np.random.choice(trains.shape[0], 2000, replace=False)
            trains = trains[idx]

        # DFT along T axis; use real FFT
        n_freq = T // 2 + 1
        psd_sum = np.zeros(n_freq)
        for train in trains:
            fft_mag = np.abs(np.fft.rfft(train.astype(np.float32))) ** 2
            psd_sum += fft_mag
        psd = psd_sum / (trains.shape[0] + 1e-8)

        result.psd = psd.tolist()

        # Temporal bandwidth: first freq bin where cumulative power ≥ 95%
        cum_psd = np.cumsum(psd)
        cum_psd /= cum_psd[-1] + 1e-8
        bw_idx = int(np.searchsorted(cum_psd, 0.95))
        result.temporal_bandwidth = float(bw_idx / (n_freq - 1 + 1e-8))

        # 1/f exponent via log–log regression (skip DC bin at index 0)
        if n_freq > 3:
            freqs_log = np.log(np.arange(1, n_freq) + 1e-8)
            psd_log = np.log(psd[1:] + 1e-8)
            slope = float(np.polyfit(freqs_log, psd_log, 1)[0])
            result.one_f_exponent = -slope   # positive = steeper 1/f

        # Burstiness: power above the median frequency bin
        mid_idx = n_freq // 2
        result.burstiness = float(psd[mid_idx:].sum() / (psd.sum() + 1e-8))

        # Coding type inference
        if result.burstiness > 0.40:
            result.coding_type = "burst"
        elif result.temporal_bandwidth < 0.20:
            result.coding_type = "rate"
        elif result.one_f_exponent > 1.5:
            result.coding_type = "temporal"
        else:
            result.coding_type = "mixed"

        return result

    # ── Main API ───────────────────────────────────────────────────────

    def analyze(
        self,
        spike_records: Dict[str, np.ndarray],   # {layer: (T,B,C,H,W)}
        input_images: Optional[np.ndarray] = None,  # (B,C,H,W)
        task: str = "unknown",
        model_type: str = "snn",
    ) -> SpikeCouplingReport:
        """
        Run full A3.1 + A3.2 analysis for all recorded layers.

        Args:
            spike_records: spike tensors from each layer
            input_images:  original degraded images (for frequency mask computation)
            task:          task label (metadata)
            model_type:    model label (metadata)
        """
        report = SpikeCouplingReport(task=task, model_type=model_type)

        # Use zero images if none provided (masks will be uninformative but won't crash)
        if input_images is None:
            T, B, C, H, W = list(spike_records.values())[0].shape
            input_images = np.zeros((B, 3, H, W), dtype=np.float32)

        for layer_name, spikes in spike_records.items():
            # A3.1: band overlap
            try:
                report.band_overlaps[layer_name] = self._band_overlap(
                    spikes, input_images, layer_name
                )
            except Exception as e:
                report.band_overlaps[layer_name] = SpikeBandOverlapResult(layer_name=layer_name)

            # A3.2: temporal PSD
            try:
                report.temporal_psds[layer_name] = self._temporal_psd(
                    spikes, layer_name
                )
            except Exception as e:
                report.temporal_psds[layer_name] = SpikeTemporalPSDResult(
                    layer_name=layer_name
                )

        # Aggregate signatures
        hf_lf_ratios = [v.hf_lf_ratio for v in report.band_overlaps.values()]
        bws = [v.temporal_bandwidth for v in report.temporal_psds.values()
               if v.temporal_bandwidth > 0]
        report.mean_hf_lf_ratio = float(np.mean(hf_lf_ratios)) if hf_lf_ratios else 1.0
        report.mean_temporal_bw = float(np.mean(bws)) if bws else 0.0

        coding_counts: Dict[str, int] = {}
        for v in report.temporal_psds.values():
            coding_counts[v.coding_type] = coding_counts.get(v.coding_type, 0) + 1
        dominant_coding = max(coding_counts, key=coding_counts.get) if coding_counts else "mixed"

        bias = "HF-biased" if report.mean_hf_lf_ratio > 1.2 else \
               "LF-biased" if report.mean_hf_lf_ratio < 0.8 else "balanced"
        report.task_coding_signature = f"{bias}, {dominant_coding}-coding"

        return report

    # ── Display ────────────────────────────────────────────────────────

    def print_report(self, report: SpikeCouplingReport) -> None:
        """Print spike–frequency coupling report."""
        print(f"\n{'='*75}")
        print(f"Spike–Frequency Coupling Report: {report.task.upper()} ({report.model_type})")
        print(f"{'='*75}")
        print(f"  Signature: {report.task_coding_signature}")
        print(f"  Mean HF/LF ratio: {report.mean_hf_lf_ratio:.3f}  "
              f"(>1 = spikes prefer HF regions)")
        print(f"  Mean temporal BW: {report.mean_temporal_bw:.3f}")
        print()
        print(f"  A3.1 Spike–Band Overlap:")
        print(f"  {'Layer':<22} {'LF':>7} {'MF':>7} {'HF':>7} {'HF/LF':>8} {'Dominant':<8}")
        print("  " + "-" * 60)
        for name, r in report.band_overlaps.items():
            print(f"  {name[:21]:<22} {r.lf_overlap:>7.3f} {r.mf_overlap:>7.3f} "
                  f"{r.hf_overlap:>7.3f} {r.hf_lf_ratio:>8.3f} {r.dominant_band:<8}")

        print(f"\n  A3.2 Spike Temporal PSD:")
        print(f"  {'Layer':<22} {'T':>4} {'Bandwidth':>10} {'1/f exp':>9} "
              f"{'Burstiness':>11} {'Coding':<10}")
        print("  " + "-" * 70)
        for name, r in report.temporal_psds.items():
            print(f"  {name[:21]:<22} {r.T:>4} {r.temporal_bandwidth:>10.4f} "
                  f"{r.one_f_exponent:>9.3f} {r.burstiness:>11.4f} {r.coding_type:<10}")
        print(f"{'='*75}")

    @staticmethod
    def compare_tasks(
        reports: Dict[str, SpikeCouplingReport],
    ) -> None:
        """Print cross-task spike–frequency coupling comparison."""
        print(f"\n{'='*80}")
        print("Cross-Task Spike–Frequency Coupling Summary")
        print(f"{'='*80}")
        print(f"  {'Task':<18} {'HF/LF':>7} {'TempBW':>8} {'Signature':<30}")
        print("  " + "-" * 65)
        for task, r in reports.items():
            print(f"  {task:<18} {r.mean_hf_lf_ratio:>7.3f} "
                  f"{r.mean_temporal_bw:>8.4f} {r.task_coding_signature:<30}")
        print(f"{'='*80}")
        print("  HF/LF > 1.2: spikes prefer high-frequency spatial regions (VLIF claim)")
        print("  HF/LF < 0.8: spikes prefer low-frequency regions")
        print("  TempBW:      temporal bandwidth of spike trains (0=DC, 1=Nyquist)")


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _resize_nearest(img: np.ndarray, H_out: int, W_out: int) -> np.ndarray:
    """Nearest-neighbour resize (H,W) → (H_out, W_out). Pure NumPy."""
    H_in, W_in = img.shape[:2]
    row_idx = (np.arange(H_out) * H_in / H_out).astype(int)
    col_idx = (np.arange(W_out) * W_in / W_out).astype(int)
    return img[row_idx[:, None], col_idx[None, :]]
