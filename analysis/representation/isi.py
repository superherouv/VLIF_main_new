"""
脉冲间隔统计分析 (Inter-Spike Interval Analysis).

ISI = 连续两次脉冲之间的时间间隔（以时间步为单位）。

在神经科学中，ISI 分布揭示了神经编码策略：
  - 泊松过程（指数分布）：随机发放，rate coding
  - 周期性发放（正态分布）：精确时间编码
  - 突发性发放（双模分布）：burst coding
  - 没有发放（ISI = ∞）：沉默神经元

在 SNN 图像恢复中的意义：
  1. 任务-ISI 关联：
     - 去雨（稀疏事件）→ 长 ISI（神经元很少发放）
     - 去噪（密集信号）→ 短 ISI（高频发放）
     → ISI 分布形状预测 SNN 的适用性

  2. 层-ISI 关系：
     - 编码器早期层：ISI 反映输入退化的稀疏性
     - 解码器晚期层：ISI 应趋近于期望的输出模式
     → 层间 ISI 的"转变"揭示信息流动

  3. 时间编码效率：
     - 如果 ISI 的 CV（变异系数）高 → 信息在时间中分布不均
     - CV = 1：泊松过程（rate coding）
     - CV < 1：规则发放（temporal coding）
     - CV > 1：突发性发放（burst coding）

  4. 任务-特异的 ISI 模式：
     - 去雨时，雨线位置的神经元 ISI 应短（多次激活）
     - 背景区域的神经元 ISI 应长（少激活）
     → ISI 空间图 ↔ 退化区域图的相关性

公式：
  ISI(n) = t_{n+1} - t_n（连续两次发放的时间差）
  CV_ISI = std(ISI) / mean(ISI)
  Lv（局部变异系数）= (3/(T-1)) Σ (ISI_{i+1} - ISI_i)² / (ISI_{i+1} + ISI_i)²
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ISIStats:
    """单层/单神经元的 ISI 统计。"""
    layer_name: str
    n_neurons_analyzed: int

    # 基本统计
    mean_isi: float = 0.0          # 平均 ISI（时间步）
    std_isi: float = 0.0
    cv_isi: float = 0.0            # 变异系数 = std/mean（1=泊松，<1=规则，>1=突发）
    median_isi: float = 0.0
    isi_min: float = 0.0
    isi_max: float = 0.0

    # 神经科学指标
    lv: float = 0.0                # 局部变异系数（比 CV 更鲁棒）
    firing_rate: float = 0.0       # 平均发放率
    burst_fraction: float = 0.0    # 短 ISI（≤2步）的比例

    # 编码类型推断
    coding_type: str = ""          # "rate" / "temporal" / "burst" / "silent"

    # 直方图（用于绘图）
    isi_histogram: Optional[np.ndarray] = None  # (n_bins,)
    isi_bin_edges: Optional[np.ndarray] = None  # (n_bins+1,)

    # 空间图：每个位置的平均 ISI（用于定位退化区域）
    spatial_isi_map: Optional[np.ndarray] = None  # (H, W)


class ISIAnalyzer:
    """
    脉冲间隔分析器。

    输入：(T, B, C, H, W) 二值脉冲序列
    输出：ISI 统计 + 编码类型判断 + 空间 ISI 图

    用法：
        analyzer = ISIAnalyzer()
        spikes = ...  # (T, B, C, H, W) binary
        stats = analyzer.analyze_layer(spikes, "encoder_0")
    """

    def __init__(self, n_histogram_bins: int = 20):
        self.n_hist_bins = n_histogram_bins

    def _compute_isi_for_neuron(
        self, spike_train: np.ndarray  # (T,) binary
    ) -> np.ndarray:
        """
        计算单个神经元的 ISI 序列。

        Returns:
            isi: (n_spikes-1,) 脉冲间隔序列，若脉冲数 < 2 则返回空数组
        """
        spike_times = np.where(spike_train > 0.5)[0]
        if len(spike_times) < 2:
            return np.array([])
        return np.diff(spike_times).astype(float)

    def _local_variation(self, isi: np.ndarray) -> float:
        """
        局部变异系数 Lv（Shinomoto et al. 2009）。
        比 CV 更对 spike count 变化鲁棒。
        """
        if len(isi) < 2:
            return 0.0
        # Lv = (3/(n-1)) Σ ((ISI_i+1 - ISI_i)/(ISI_i+1 + ISI_i))²
        a = isi[:-1]
        b = isi[1:]
        denom = a + b
        valid = denom > 0
        if valid.sum() == 0:
            return 0.0
        lv = 3.0 / (len(isi) - 1) * np.sum(((b[valid] - a[valid]) / denom[valid])**2)
        return float(lv)

    def _infer_coding_type(
        self, cv: float, burst_fraction: float, firing_rate: float
    ) -> str:
        """
        根据 ISI 统计推断编码类型。

        Rules:
          - firing_rate ≈ 0：silent neuron
          - CV ≈ 1 且 burst_fraction 低：rate coding (Poisson-like)
          - CV < 0.5：temporal/clock coding（规则发放）
          - CV > 1.5 且 burst_fraction 高：burst coding
          - CV > 1 但 burst_fraction 低：irregular (common in SNN)
        """
        if firing_rate < 0.02:
            return "silent"
        if burst_fraction > 0.3:
            return "burst"
        if cv < 0.5:
            return "temporal"
        if 0.7 < cv < 1.3:
            return "rate"
        return "irregular"

    def analyze_layer(
        self,
        spikes: np.ndarray,    # (T, B, C, H, W)
        layer_name: str = "layer",
        compute_spatial: bool = True,
    ) -> ISIStats:
        """
        分析单层的 ISI 统计。

        Args:
            spikes: (T, B, C, H, W) 二值脉冲
            layer_name: 层名
            compute_spatial: 是否计算空间 ISI 图（慢，可选）

        Returns:
            ISIStats
        """
        T, B, C, H, W = spikes.shape
        # 展平: (T, N)
        v = spikes.reshape(T, -1)
        N = v.shape[1]

        # 收集所有 ISI
        all_isi = []
        spatial_mean_isi = np.zeros(N)
        n_analyzed = 0

        for n in range(N):
            isi = self._compute_isi_for_neuron(v[:, n])
            if len(isi) > 0:
                all_isi.extend(isi.tolist())
                spatial_mean_isi[n] = float(isi.mean())
                n_analyzed += 1
            else:
                spatial_mean_isi[n] = float(T)  # 未发放 → ISI = 最大值

        # 全局 ISI 统计
        if all_isi:
            isi_arr = np.array(all_isi)
            mean_isi  = float(isi_arr.mean())
            std_isi   = float(isi_arr.std())
            cv_isi    = std_isi / (mean_isi + 1e-8)
            median_isi = float(np.median(isi_arr))
            isi_min   = float(isi_arr.min())
            isi_max   = float(isi_arr.max())
            lv        = self._local_variation(isi_arr)
            # 短 ISI (≤2步) 比例 → burst 指标
            burst_frac = float((isi_arr <= 2).mean())
            # 直方图
            hist, edges = np.histogram(
                isi_arr, bins=self.n_hist_bins,
                range=(1, min(float(T), float(isi_arr.max()) + 1))
            )
        else:
            mean_isi = float(T)
            std_isi = 0.0
            cv_isi = 0.0
            median_isi = float(T)
            isi_min = isi_max = float(T)
            lv = burst_frac = 0.0
            hist = np.zeros(self.n_hist_bins, dtype=int)
            edges = np.linspace(1, T, self.n_hist_bins + 1)

        firing_rate = float((v > 0.5).mean())
        coding_type = self._infer_coding_type(cv_isi, burst_frac, firing_rate)

        # 空间 ISI 图
        if compute_spatial:
            spatial_map = spatial_mean_isi.reshape(B, C, H, W).mean(0).mean(0)  # (H, W)
        else:
            spatial_map = None

        return ISIStats(
            layer_name=layer_name,
            n_neurons_analyzed=n_analyzed,
            mean_isi=mean_isi, std_isi=std_isi,
            cv_isi=cv_isi, median_isi=median_isi,
            isi_min=isi_min, isi_max=isi_max,
            lv=lv, firing_rate=firing_rate,
            burst_fraction=burst_frac,
            coding_type=coding_type,
            isi_histogram=hist,
            isi_bin_edges=edges,
            spatial_isi_map=spatial_map,
        )

    def analyze_all_layers(
        self,
        spike_records: Dict[str, np.ndarray],
    ) -> Dict[str, ISIStats]:
        """分析所有层。"""
        return {
            name: self.analyze_layer(spikes, name)
            for name, spikes in spike_records.items()
        }

    def compare_tasks(
        self,
        task_stats: Dict[str, Dict[str, ISIStats]],
    ) -> Dict[str, Dict[str, float]]:
        """
        对比不同任务的 ISI 特征（聚合所有层）。

        Args:
            task_stats: {task: {layer: ISIStats}}

        Returns:
            {task: {metric: mean_value}}
        """
        summary = {}
        for task, layer_stats in task_stats.items():
            all_cv = [s.cv_isi for s in layer_stats.values()]
            all_lv = [s.lv for s in layer_stats.values()]
            all_burst = [s.burst_fraction for s in layer_stats.values()]
            all_rate = [s.firing_rate for s in layer_stats.values()]
            coding_types = [s.coding_type for s in layer_stats.values()]
            dominant_coding = max(set(coding_types), key=coding_types.count)
            summary[task] = {
                "mean_cv": float(np.mean(all_cv)),
                "mean_lv": float(np.mean(all_lv)),
                "mean_burst_fraction": float(np.mean(all_burst)),
                "mean_firing_rate": float(np.mean(all_rate)),
                "dominant_coding_type": dominant_coding,
            }
        return summary

    def print_summary(self, stats: Dict[str, ISIStats]):
        print(f"\n{'='*80}")
        print("Inter-Spike Interval (ISI) Analysis")
        print(f"{'Layer':<35} {'Rate':>7} {'CV_ISI':>8} {'Lv':>7} "
              f"{'Burst%':>8} {'Coding':<12}")
        print("-"*80)
        for name, s in sorted(stats.items()):
            print(f"{name:<35} {s.firing_rate:>7.4f} "
                  f"{s.cv_isi:>8.3f} "
                  f"{s.lv:>7.3f} "
                  f"{s.burst_fraction:>8.3f} "
                  f"{s.coding_type:<12}")
        print(f"{'='*80}")
        print("  CV≈1: rate-coded | CV<0.5: temporal | CV>1.5+Burst>0.3: burst")
