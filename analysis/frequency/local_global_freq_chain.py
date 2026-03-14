"""
Local–Global Frequency Chain Analysis
=======================================
核心论点验证工具：
  「局部 spike 激活偏好与全局传播偏好可能并不一致」

证据链结构
----------
同一模型、同一任务下，逐层追踪：

  Layer       Depth   spike_hf_lf   feat_hf_ratio   解读
  stem        0       2.34          0.45            spike 激活在高频区，特征高频能量也高
  enc_L0      1       1.87          0.38            spike 仍偏高频，高频保留略降
  enc_L1      2       1.12          0.24            spike 趋于平衡，高频已明显衰减
  bottleneck  3       0.89          0.15            spike 偏低频，高频几乎丢失
  dec_L0      4       0.71          0.21            低频偏向，解码器少量恢复
  dec_L1      5       0.65          0.28            低频偏向，跳跃连接带回部分高频

  output_error         ——           hf_ratio=0.67   误差集中在高频 → 高频未被恢复

跨任务对照（deraining vs denoising vs dehazing vs super-res）：
  spike_hf_lf 和 hf_error_ratio 的「不匹配程度」强弱差异，才是核心结论。

用法
----
  analyzer = LocalGlobalFreqAnalyzer()
  analyzer.attach_hooks(snn_model, model_type="snn")
  for batch in loader:
      snn_model.reset_states()
      pred = snn_model(batch_input)
      analyzer.collect_input(batch_input.cpu().numpy())
      analyzer.collect_pred_target(pred.cpu().numpy(), batch_clean.cpu().numpy())
  report = analyzer.build_report(task="deraining", model_type="snn")
  analyzer.print_report(report)
  d = report.to_dict()    # → JSON 序列化
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .spike_freq_coupling import SpikeFqCouplingAnalyzer


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerFreqStats:
    """Per-layer frequency statistics — the single row in the evidence chain."""
    layer_name: str
    depth: int                  # 0=stem, 1..N=encoder, N+1=bottleneck, N+2..=decoder

    # ─ 局部激活偏好（spike-band overlap） ─────────────────────────────
    spike_lf_overlap: float = 0.0   # [0,1] spike 与低频区重叠
    spike_mf_overlap: float = 0.0
    spike_hf_overlap: float = 0.0
    spike_hf_lf_ratio: float = 1.0  # >1 → spike 偏高频
    spike_dominant_band: str = "mf"

    # ─ 全局传播偏好（feature HF energy ratio） ─────────────────────────
    feat_hf_ratio: float = 0.0      # HF energy / total energy in feature map FFT
    feat_lf_ratio: float = 0.0      # LF energy / total energy
    feat_hf_lf_ratio: float = 1.0   # >1 → 特征保留了更多高频信息

    # ─ 不一致程度 ────────────────────────────────────────────────────
    discrepancy: float = 0.0        # |spike_hf_lf - feat_hf_lf| / mean
    # >0.5: 局部激活偏高频但全局传播已偏低频 → 不一致明显

    # availability flags
    has_spike: bool = False
    has_feat: bool = False


@dataclass
class LocalGlobalReport:
    """Complete local–global chain report for one (task, model_type) pair."""
    task: str
    model_type: str
    n_samples: int = 0

    # Layer-wise chain (ordered by depth)
    layers: List[LayerFreqStats] = field(default_factory=list)

    # Output error spectrum
    output_hf_error_ratio: float = 0.0    # fraction of error energy in HF
    output_lf_error_ratio: float = 0.0
    output_spectral_centroid: float = 0.0

    # Aggregate signatures
    mean_spike_hf_lf: float = 1.0      # averaged over encoder layers
    mean_feat_hf_lf: float = 1.0       # averaged over encoder layers
    peak_discrepancy: float = 0.0      # max discrepancy layer
    peak_discrepancy_layer: str = ""

    # Cross-task key conclusion
    conclusion: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model_type": self.model_type,
            "n_samples": self.n_samples,
            "layers": [
                {
                    "layer_name": l.layer_name, "depth": l.depth,
                    "spike_hf_lf_ratio": l.spike_hf_lf_ratio,
                    "spike_dominant_band": l.spike_dominant_band,
                    "feat_hf_ratio": l.feat_hf_ratio,
                    "feat_hf_lf_ratio": l.feat_hf_lf_ratio,
                    "discrepancy": l.discrepancy,
                }
                for l in self.layers
            ],
            "output_hf_error_ratio": self.output_hf_error_ratio,
            "output_spectral_centroid": self.output_spectral_centroid,
            "mean_spike_hf_lf": self.mean_spike_hf_lf,
            "mean_feat_hf_lf": self.mean_feat_hf_lf,
            "peak_discrepancy": self.peak_discrepancy,
            "peak_discrepancy_layer": self.peak_discrepancy_layer,
            "conclusion": self.conclusion,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class LocalGlobalFreqAnalyzer:
    """
    逐层追踪「局部 spike 激活偏好」vs「全局特征传播偏好」的不一致性。

    工作原理
    --------
    1. 用 register_forward_hook 挂到 LIFNeuron（获取 spike，0/1 张量）
       和 nn.BatchNorm2d / nn.Conv2d（获取连续特征图）。
    2. 每次 forward 后汇总 T 个时间步的 spike 到 (T,B,C,H,W) 缓冲区。
    3. 对每层：
       - spike_band_overlap → 局部激活偏好
       - feat_hf_ratio      → 该层特征中高频能量占比（传播偏好代理）
    4. 同时收集输出预测和 GT，计算误差频谱（验证高频确实没被恢复）。
    """

    def __init__(
        self,
        lf_cutoff: float = 0.15,
        hf_cutoff: float = 0.30,
        max_samples: int = 64,       # 最多取多少图做分析（内存约束）
    ):
        self.lf_cutoff = lf_cutoff
        self.hf_cutoff = hf_cutoff
        self.max_samples = max_samples
        self._coupling = SpikeFqCouplingAnalyzer(lf_cutoff, hf_cutoff)

        # Buffers — 以 layer name 为 key
        self._spike_bufs: Dict[str, List[np.ndarray]] = {}  # (B,C,H,W) per timestep
        self._feat_bufs: Dict[str, List[np.ndarray]] = {}   # (B,C,H,W)
        self._hooks = []

        self._input_imgs: List[np.ndarray] = []    # (C,H,W) each
        self._pred_imgs: List[np.ndarray] = []
        self._tgt_imgs: List[np.ndarray] = []

        # Track current timestep per layer (to group T calls into one sample)
        self._layer_call_count: Dict[str, int] = {}

    # ── Hook attachment ────────────────────────────────────────────────────────

    def attach_hooks(self, model, model_type: str = "snn") -> None:
        """Attach forward hooks to LIF neurons and conv/bn layers."""
        import torch.nn as nn

        self._detach_all()
        self._spike_bufs.clear()
        self._feat_bufs.clear()
        self._layer_call_count.clear()

        if model_type in ("snn", "hybrid"):
            self._attach_snn_hooks(model)
        else:
            self._attach_ann_hooks(model)

    def _attach_snn_hooks(self, model) -> None:
        import torch.nn as nn
        try:
            from models.snn.neurons import LIFNeuron
        except ImportError:
            return

        for name, module in model.named_modules():
            depth = self._name_to_depth(name, model)
            if depth < 0:
                continue

            if isinstance(module, LIFNeuron):
                # Spike hook
                buf_name = name
                self._spike_bufs[buf_name] = []
                self._layer_call_count[buf_name] = 0

                def _make_spike_hook(bname):
                    def _hook(m, inp, out):
                        arr = out.detach().cpu().float().numpy()  # (B,C,H,W)
                        self._spike_bufs[bname].append(arr)
                    return _hook
                h = module.register_forward_hook(_make_spike_hook(buf_name))
                self._hooks.append(h)

            elif isinstance(module, nn.BatchNorm2d):
                # Continuous feature hook (pre-spike, after BN)
                buf_name = name
                self._feat_bufs[buf_name] = []

                def _make_feat_hook(bname):
                    def _hook(m, inp, out):
                        arr = out.detach().cpu().float().numpy()  # (B,C,H,W)
                        if len(self._feat_bufs[bname]) < self.max_samples:
                            self._feat_bufs[bname].append(arr[0])  # first sample
                    return _hook
                h = module.register_forward_hook(_make_feat_hook(buf_name))
                self._hooks.append(h)

    def _attach_ann_hooks(self, model) -> None:
        import torch.nn as nn
        for name, module in model.named_modules():
            depth = self._name_to_depth(name, model)
            if depth < 0:
                continue
            if isinstance(module, nn.BatchNorm2d):
                buf_name = name
                self._feat_bufs[buf_name] = []

                def _make_feat_hook(bname):
                    def _hook(m, inp, out):
                        arr = out.detach().cpu().float().numpy()
                        if len(self._feat_bufs[bname]) < self.max_samples:
                            self._feat_bufs[bname].append(arr[0])
                    return _hook
                h = module.register_forward_hook(_make_feat_hook(buf_name))
                self._hooks.append(h)

    def _detach_all(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset_buffers(self) -> None:
        """Call between runs (different tasks)."""
        self._detach_all()
        self._spike_bufs.clear()
        self._feat_bufs.clear()
        self._input_imgs.clear()
        self._pred_imgs.clear()
        self._tgt_imgs.clear()

    # ── Sample collection ──────────────────────────────────────────────────────

    def collect_input(self, x_np: np.ndarray) -> None:
        """x_np: (B,C,H,W) degraded input [0,1]."""
        if len(self._input_imgs) >= self.max_samples:
            return
        for b in range(min(x_np.shape[0], self.max_samples - len(self._input_imgs))):
            self._input_imgs.append(x_np[b])

    def collect_pred_target(self, pred_np: np.ndarray, tgt_np: np.ndarray) -> None:
        """pred_np, tgt_np: (B,C,H,W) [0,1]."""
        if len(self._pred_imgs) >= self.max_samples:
            return
        n = min(pred_np.shape[0], self.max_samples - len(self._pred_imgs))
        for b in range(n):
            self._pred_imgs.append(np.clip(pred_np[b], 0, 1))
            self._tgt_imgs.append(np.clip(tgt_np[b], 0, 1))

    # ── Frequency computations ─────────────────────────────────────────────────

    def _feat_hf_lf_ratio(self, feat_list: List[np.ndarray]) -> Tuple[float, float, float]:
        """
        Given list of (C,H,W) feature maps, compute mean HF/LF energy ratios.
        Returns: (hf_ratio, lf_ratio, hf_lf_ratio)
        """
        hf_ratios, lf_ratios = [], []
        for feat in feat_list[:32]:  # max 32 for speed
            C, H, W = feat.shape
            hf_e = lf_e = total_e = 0.0
            for c in range(C):
                fft = np.fft.fft2(feat[c])
                psd = np.abs(np.fft.fftshift(fft)) ** 2
                fy = np.fft.fftshift(np.fft.fftfreq(H))
                fx = np.fft.fftshift(np.fft.fftfreq(W))
                FX, FY = np.meshgrid(fx, fy)
                R = np.sqrt(FX ** 2 + FY ** 2)
                total_e += psd.sum()
                hf_e += psd[R > self.hf_cutoff].sum()
                lf_e += psd[R < self.lf_cutoff].sum()
            total_e = max(total_e, 1e-10)
            hf_ratios.append(hf_e / total_e)
            lf_ratios.append(lf_e / total_e)

        if not hf_ratios:
            return 0.0, 0.0, 1.0
        hfr = float(np.mean(hf_ratios))
        lfr = float(np.mean(lf_ratios))
        return hfr, lfr, hfr / (lfr + 1e-8)

    def _error_spectrum(self, preds, tgts) -> Tuple[float, float, float]:
        """
        Compute error spectrum stats across all collected pred/target pairs.
        Returns: (hf_error_ratio, lf_error_ratio, spectral_centroid)
        """
        from .error_spectrum import ErrorSpectrumAnalyzer
        ea = ErrorSpectrumAnalyzer()
        hf_list, lf_list, cent_list = [], [], []
        for p, t in zip(preds[:32], tgts[:32]):
            s = ea.analyze(p, t)
            hf_list.append(s.hf_error_ratio)
            lf_list.append(s.lf_error_ratio)
            cent_list.append(s.spectral_centroid)
        if not hf_list:
            return 0.0, 0.0, 0.0
        return (float(np.mean(hf_list)), float(np.mean(lf_list)),
                float(np.mean(cent_list)))

    # ── Depth assignment ───────────────────────────────────────────────────────

    @staticmethod
    def _name_to_depth(name: str, model) -> int:
        """
        Map module name to integer depth. Returns -1 if not a tracked layer.
        Convention:
          stem*             → 0
          enc_blocks.{i}.*  → i + 1
          bottleneck.*      → 100  (special)
          dec_blocks.{i}.*  → 200 + i
        """
        if "stem" in name:
            return 0
        if name.startswith("enc_blocks.") or "enc_blocks." in name:
            parts = name.split(".")
            try:
                idx = parts.index("enc_blocks")
                return int(parts[idx + 1]) + 1
            except (ValueError, IndexError):
                return -1
        if "bottleneck" in name:
            return 100
        if name.startswith("dec_blocks.") or "dec_blocks." in name:
            parts = name.split(".")
            try:
                idx = parts.index("dec_blocks")
                return 200 + int(parts[idx + 1])
            except (ValueError, IndexError):
                return -1
        return -1

    @staticmethod
    def _depth_to_label(depth: int) -> str:
        if depth == 0:
            return "stem"
        elif 1 <= depth <= 99:
            return f"enc_L{depth - 1}"
        elif depth == 100:
            return "bottleneck"
        elif depth >= 200:
            return f"dec_L{depth - 200}"
        return f"d{depth}"

    # ── Report building ────────────────────────────────────────────────────────

    def build_report(self, task: str, model_type: str) -> LocalGlobalReport:
        """Build the full local–global chain report from collected data."""
        report = LocalGlobalReport(
            task=task,
            model_type=model_type,
            n_samples=len(self._pred_imgs),
        )

        # ── Group spike bufs by depth ──────────────────────────────────────
        depth_to_spikes: Dict[int, List[np.ndarray]] = {}   # depth → list(B,C,H,W)
        depth_to_feats: Dict[int, List[np.ndarray]] = {}

        for layer_name, spike_list in self._spike_bufs.items():
            if not spike_list:
                continue
            depth = self._name_to_depth(layer_name, None)
            if depth < 0:
                continue
            depth_to_spikes.setdefault(depth, []).extend(spike_list[:self.max_samples])

        for layer_name, feat_list in self._feat_bufs.items():
            if not feat_list:
                continue
            depth = self._name_to_depth(layer_name, None)
            if depth < 0:
                continue
            depth_to_feats.setdefault(depth, []).extend(feat_list[:self.max_samples])

        # Collect all depths
        all_depths = sorted(set(list(depth_to_spikes.keys()) + list(depth_to_feats.keys())))
        input_imgs_np = (np.stack(self._input_imgs) if self._input_imgs
                         else np.zeros((1, 3, 64, 64), dtype=np.float32))

        for depth in all_depths:
            lname = self._depth_to_label(depth)
            stats = LayerFreqStats(layer_name=lname, depth=depth)

            # ── Spike-band overlap (local activation preference) ────────────
            if depth in depth_to_spikes:
                arr = np.stack(depth_to_spikes[depth], axis=0)   # (N, B, C, H, W) roughly
                # We need (T, B, C, H, W) — treat N as T (one per timestep collected)
                # Reshape: each element is (B,C,H,W), stack → (N,B,C,H,W), take as (T,1,C,H,W)
                T = min(arr.shape[0], 8)
                spikes_tbchw = arr[:T, :1, :, :, :]  # (T,1,C,H,W)
                # Build a small input_images for mask (1,C,H,W)
                ref_img = input_imgs_np[:1]
                try:
                    overlap = self._coupling._band_overlap(spikes_tbchw, ref_img, lname)
                    stats.spike_lf_overlap = overlap.lf_overlap
                    stats.spike_mf_overlap = overlap.mf_overlap
                    stats.spike_hf_overlap = overlap.hf_overlap
                    stats.spike_hf_lf_ratio = overlap.hf_lf_ratio
                    stats.spike_dominant_band = overlap.dominant_band
                    stats.has_spike = True
                except Exception:
                    pass

            # ── Feature HF/LF ratio (global propagation preference) ─────────
            if depth in depth_to_feats:
                try:
                    hfr, lfr, hfl_ratio = self._feat_hf_lf_ratio(depth_to_feats[depth])
                    stats.feat_hf_ratio = hfr
                    stats.feat_lf_ratio = lfr
                    stats.feat_hf_lf_ratio = hfl_ratio
                    stats.has_feat = True
                except Exception:
                    pass

            # ── Discrepancy ─────────────────────────────────────────────────
            if stats.has_spike and stats.has_feat:
                mean_ratio = (stats.spike_hf_lf_ratio + stats.feat_hf_lf_ratio) / 2
                stats.discrepancy = abs(stats.spike_hf_lf_ratio - stats.feat_hf_lf_ratio) / (mean_ratio + 1e-8)

            report.layers.append(stats)

        # Sort by depth
        report.layers.sort(key=lambda l: l.depth)

        # ── Output error spectrum ────────────────────────────────────────────
        if self._pred_imgs and self._tgt_imgs:
            hf_err, lf_err, centroid = self._error_spectrum(self._pred_imgs, self._tgt_imgs)
            report.output_hf_error_ratio = hf_err
            report.output_lf_error_ratio = lf_err
            report.output_spectral_centroid = centroid

        # ── Aggregate signatures ─────────────────────────────────────────────
        enc_layers = [l for l in report.layers if 1 <= l.depth <= 99]
        if enc_layers:
            report.mean_spike_hf_lf = float(np.mean([l.spike_hf_lf_ratio for l in enc_layers
                                                       if l.has_spike] or [1.0]))
            report.mean_feat_hf_lf = float(np.mean([l.feat_hf_lf_ratio for l in enc_layers
                                                      if l.has_feat] or [1.0]))

        if report.layers:
            peak_layer = max(report.layers, key=lambda l: l.discrepancy)
            report.peak_discrepancy = peak_layer.discrepancy
            report.peak_discrepancy_layer = peak_layer.layer_name

        # ── Conclusion string ────────────────────────────────────────────────
        spike_bias = ("HF" if report.mean_spike_hf_lf > 1.2
                      else "LF" if report.mean_spike_hf_lf < 0.8 else "balanced")
        feat_bias = ("HF" if report.mean_feat_hf_lf > 1.2
                     else "LF" if report.mean_feat_hf_lf < 0.8 else "balanced")
        err_dominant = ("HF" if report.output_hf_error_ratio > 0.5 else "LF")
        if spike_bias != feat_bias:
            report.conclusion = (
                f"LOCAL-GLOBAL DISCREPANCY: spike activation {spike_bias}-biased "
                f"but feature propagation {feat_bias}-biased; "
                f"output error {err_dominant}-dominant"
            )
        else:
            report.conclusion = (
                f"CONSISTENT: both spike ({spike_bias}) and feature ({feat_bias}) "
                f"align; output error {err_dominant}-dominant"
            )

        return report

    # ── Display ───────────────────────────────────────────────────────────────

    def print_report(self, report: LocalGlobalReport) -> None:
        W = 90
        print(f"\n{'='*W}")
        print(f"Local–Global Frequency Chain: {report.task.upper()} ({report.model_type})"
              f"  [{report.n_samples} samples]")
        print(f"{'='*W}")
        print(f"\n  Conclusion: {report.conclusion}")
        print(f"\n  ┌──────────────────────────────────────────────────────────────────────────┐")
        print(f"  │ {'Layer':<14} {'Depth':>5} │ {'spike HF/LF':>11} {'spike dom':>9} "
              f"│ {'feat HF/LF':>10} {'feat HFr':>8} │ {'Discrep':>8} │")
        print(f"  ├──────────────────────────────────────────────────────────────────────────┤")
        for l in report.layers:
            sp_str = f"{l.spike_hf_lf_ratio:>11.3f} {l.spike_dominant_band:>9}" if l.has_spike else f"{'—':>11} {'—':>9}"
            ft_str = f"{l.feat_hf_lf_ratio:>10.3f} {l.feat_hf_ratio:>8.3f}" if l.has_feat else f"{'—':>10} {'—':>8}"
            disc_str = f"{l.discrepancy:>8.3f}" if (l.has_spike and l.has_feat) else f"{'—':>8}"
            print(f"  │ {l.layer_name:<14} {l.depth:>5} │ {sp_str} │ {ft_str} │ {disc_str} │")
        print(f"  ├──────────────────────────────────────────────────────────────────────────┤")
        print(f"  │ {'output error':<14} {'—':>5} │ {'—':>11} {'—':>9} │ "
              f"{'—':>10} {'—':>8} │ {'HFr':>5}={report.output_hf_error_ratio:.3f} │")
        print(f"  └──────────────────────────────────────────────────────────────────────────┘")
        print(f"\n  Aggregates (encoder layers only):")
        print(f"    mean spike HF/LF  = {report.mean_spike_hf_lf:.3f}  (>1.2 → spike偏高频)")
        print(f"    mean feat  HF/LF  = {report.mean_feat_hf_lf:.3f}  (>1.2 → 特征保留高频)")
        print(f"    peak discrepancy  = {report.peak_discrepancy:.3f}  @ {report.peak_discrepancy_layer}")
        print(f"    output HF error   = {report.output_hf_error_ratio:.3f}  (>0.5 → 误差集中在高频)")
        print(f"{'='*W}")

    @staticmethod
    def cross_task_summary(reports: Dict[str, "LocalGlobalReport"]) -> None:
        """Print cross-task comparison table — the key paper table."""
        W = 95
        print(f"\n{'='*W}")
        print("Cross-Task Local–Global Frequency Discrepancy Summary")
        print("(验证：不一致性在不同任务下的强弱差异是普遍规律还是偶然)")
        print(f"{'='*W}")
        print(f"  {'Task':<18} {'Model':>8} │ {'spike HF/LF':>11} {'feat HF/LF':>11} "
              f"│ {'out HF err':>10} │ {'Peak Disc':>10} │ Conclusion")
        print("  " + "-" * 90)
        for task_model, r in reports.items():
            print(f"  {r.task:<18} {r.model_type:>8} │ "
                  f"{r.mean_spike_hf_lf:>11.3f} {r.mean_feat_hf_lf:>11.3f} │ "
                  f"{r.output_hf_error_ratio:>10.3f} │ "
                  f"{r.peak_discrepancy:>10.3f} │ {r.conclusion[:35]}")
        print(f"{'='*W}")
        print("  spike HF/LF > 1.2: local activation favors HF (VLIF claim)")
        print("  feat  HF/LF < 0.8: global propagation is LF-biased (MaxFormer claim)")
        print("  Discrepancy > 0.5: strong local–global mismatch → frequency is NOT the only bottleneck")
