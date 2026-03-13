"""
脉冲模式功率谱分析 (Spike Pattern Spectrum Analysis).

核心问题：SNN 内部的脉冲时空模式有什么频率特征？
  - 时间维度：脉冲序列的功率谱（类似神经科学中的 LFP 分析）
  - 空间维度：每个时间步的脉冲图的空间频谱
  - 联合时频：短时傅里叶变换 (STFT) of spike trains

三大发现预期：
  1. 去雨任务：脉冲功率谱应有高频峰（对应雨线的空间频率）
  2. 去噪任务：脉冲功率谱应相对平坦（噪声是宽频的）
  3. 低光任务：脉冲功率谱应集中在低频（光照是低频的）

这直接揭示了"为什么去雨适合 SNN"：
  SNN 的脉冲模式与任务的自然频率结构匹配得更好。

额外分析：
  - 1/f 噪声特性：神经元自发放电是否符合 1/f 功率谱？
  - 同步性 (synchrony)：不同通道的脉冲是否在特定频率同步？
  - 频率选择性：不同层是否对不同频段有选择性响应？
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SpikeSpectrumResult:
    """脉冲序列功率谱分析结果。"""
    layer_name: str
    # 时间域：对 T 个时间步的脉冲率序列做 FFT
    temporal_freqs: np.ndarray       # 频率轴 (T/2,)
    temporal_psd: np.ndarray         # 功率谱密度 (T/2,)
    temporal_dominant_freq: float    # 主频率
    temporal_1f_exponent: float      # 1/f^α 中的指数 α（α≈1 表示粉噪声）

    # 空间域：脉冲图的平均空间 PSD
    spatial_radial_psd: np.ndarray   # 径向平均 PSD (n_bins,)
    spatial_freqs: np.ndarray        # 空间频率轴 (n_bins,)
    spatial_peak_freq: float         # 空间主频率（对应退化模式的频率）

    # 联合统计
    mean_rate: float
    rate_variance: float             # 时间方差（高 = 更多脉冲波动）
    burstiness_index: float          # 突发性指数: (σ-μ)/(σ+μ), 范围[-1,1]


class SpikeSpectrumAnalyzer:
    """
    分析 SNN 各层脉冲模式的频率特性。

    用法：
        # 收集各层脉冲（格式：{layer_name: (T, B, C, H, W) tensor}）
        analyzer = SpikeSpectrumAnalyzer()
        results = analyzer.analyze(spike_records)
        analyzer.compare_tasks(results_per_task)
    """

    def __init__(self, n_radial_bins: int = 32):
        self.n_radial_bins = n_radial_bins

    def analyze_layer(
        self,
        spikes: np.ndarray,  # (T, B, C, H, W) 二值脉冲
        layer_name: str = "layer",
    ) -> SpikeSpectrumResult:
        """
        分析单层脉冲的频率特性。

        Args:
            spikes: (T, B, C, H, W) float32, 值为 0 或 1
            layer_name: 层名标签

        Returns:
            SpikeSpectrumResult
        """
        T, B, C, H, W = spikes.shape

        # ─── 1. 时间域分析 ───────────────────────────────────────────────
        # 时间序列：每步的平均发火率
        rate_series = spikes.mean(axis=(1, 2, 3, 4))  # (T,)
        mean_rate = float(rate_series.mean())
        rate_std = float(rate_series.std())

        # 功率谱（沿时间轴 FFT）
        if T > 2:
            fft_t = np.fft.rfft(rate_series - rate_series.mean())
            psd_t = (np.abs(fft_t)**2) / T
            freqs_t = np.fft.rfftfreq(T)
        else:
            psd_t = np.array([0.0])
            freqs_t = np.array([0.0])

        # 主频
        if len(psd_t) > 1:
            dom_idx = int(np.argmax(psd_t[1:])) + 1
            temporal_dominant_freq = float(freqs_t[dom_idx])
        else:
            temporal_dominant_freq = 0.0

        # 估计 1/f^α 指数（对 log-log PSD 线性拟合斜率）
        temporal_1f_exp = self._fit_1f_exponent(freqs_t, psd_t)

        # 突发性指数 (Shinomoto & Tsubo 2001)
        if (rate_std + mean_rate) > 0:
            burstiness = (rate_std - mean_rate) / (rate_std + mean_rate)
        else:
            burstiness = 0.0

        # ─── 2. 空间域分析 ───────────────────────────────────────────────
        # 对 T 个时间步的脉冲图做平均，再对每个通道做 FFT
        mean_spike_map = spikes.mean(axis=(0, 1))  # (C, H, W)
        spatial_psd_radial = self._radial_psd(mean_spike_map)
        spatial_freqs = np.linspace(0, 0.5, self.n_radial_bins)

        # 空间主频（退化模式对应的典型空间频率）
        spatial_peak_freq = float(
            spatial_freqs[np.argmax(spatial_psd_radial[1:])+1]
            if len(spatial_psd_radial) > 1 else 0.0
        )

        return SpikeSpectrumResult(
            layer_name=layer_name,
            temporal_freqs=freqs_t,
            temporal_psd=psd_t,
            temporal_dominant_freq=temporal_dominant_freq,
            temporal_1f_exponent=temporal_1f_exp,
            spatial_radial_psd=spatial_psd_radial,
            spatial_freqs=spatial_freqs,
            spatial_peak_freq=spatial_peak_freq,
            mean_rate=mean_rate,
            rate_variance=float(rate_std**2),
            burstiness_index=float(burstiness),
        )

    def _radial_psd(self, maps: np.ndarray) -> np.ndarray:
        """
        计算 (C, H, W) 脉冲图的径向平均功率谱。

        Returns:
            radial_psd: (n_radial_bins,) 归一化 PSD
        """
        C, H, W = maps.shape
        psd_accum = np.zeros(self.n_radial_bins)
        count = np.zeros(self.n_radial_bins)

        fy = np.fft.fftfreq(H)
        fx = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)  # 径向频率 [0, ~0.7]
        r_max = R.max() + 1e-8

        bin_edges = np.linspace(0, r_max, self.n_radial_bins + 1)

        for c in range(C):
            fft2 = np.fft.fft2(maps[c])
            psd2 = np.abs(fft2)**2

            for b in range(self.n_radial_bins):
                mask = (R >= bin_edges[b]) & (R < bin_edges[b+1])
                if mask.sum() > 0:
                    psd_accum[b] += psd2[mask].mean()
                    count[b] += 1

        radial_psd = psd_accum / (count + 1e-8)
        # 归一化
        total = radial_psd.sum() + 1e-12
        return radial_psd / total

    def _fit_1f_exponent(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> float:
        """
        拟合 1/f^α 指数：对 log(f) vs log(PSD) 做线性回归。

        返回 α。α≈0：白噪声；α≈1：粉噪声；α≈2：布朗噪声。
        对 SNN 而言，预期 α ∈ [0.5, 1.5]。
        """
        valid = (freqs > 0) & (psd > 0)
        if valid.sum() < 3:
            return 0.0
        log_f = np.log(freqs[valid])
        log_p = np.log(psd[valid])
        # 线性拟合
        coeffs = np.polyfit(log_f, log_p, 1)
        return float(-coeffs[0])  # 负斜率即为 α

    def analyze(
        self,
        spike_records: Dict[str, np.ndarray],  # {layer_name: (T,B,C,H,W)}
    ) -> Dict[str, SpikeSpectrumResult]:
        """分析所有层的脉冲频谱。"""
        return {
            name: self.analyze_layer(spikes, name)
            for name, spikes in spike_records.items()
        }

    def task_frequency_signature(
        self,
        results: Dict[str, SpikeSpectrumResult],
    ) -> Dict[str, float]:
        """
        提取任务的脉冲频率"指纹"：
          - mean_spatial_peak: 平均空间主频（高 = 任务关注高频）
          - mean_temporal_dominant: 平均时间主频
          - mean_burstiness: 平均突发性（高 = 脉冲更集中/阵发）
          - mean_1f_exponent: 平均 1/f 指数

        这些指纹可以解释为什么去雨（高空间主频）比低光（低频）更适合 SNN。
        """
        if not results:
            return {}
        return {
            "mean_spatial_peak_freq":   float(np.mean([r.spatial_peak_freq for r in results.values()])),
            "mean_temporal_dom_freq":   float(np.mean([r.temporal_dominant_freq for r in results.values()])),
            "mean_burstiness":          float(np.mean([r.burstiness_index for r in results.values()])),
            "mean_1f_exponent":         float(np.mean([r.temporal_1f_exponent for r in results.values()])),
            "mean_rate":                float(np.mean([r.mean_rate for r in results.values()])),
            "rate_variance":            float(np.mean([r.rate_variance for r in results.values()])),
        }

    def print_summary(self, results: Dict[str, SpikeSpectrumResult]):
        print(f"\n{'='*65}")
        print("Spike Spectrum Summary")
        print(f"{'Layer':<40} {'Rate':>7} {'Burst':>7} {'α(1/f)':>8} {'f_sp':>8}")
        print("-"*65)
        for name, r in sorted(results.items()):
            print(f"{name:<40} {r.mean_rate:>7.4f} "
                  f"{r.burstiness_index:>7.3f} "
                  f"{r.temporal_1f_exponent:>8.3f} "
                  f"{r.spatial_peak_freq:>8.4f}")
        print(f"{'='*65}")
