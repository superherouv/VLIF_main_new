"""
神经元活性分析 (Neuron Viability Analysis).

"死神经元"是 SNN 训练中的核心问题：
  若代理梯度在某神经元的膜电位始终远离阈值时近似为 0，
  该神经元的梯度永远为 0 → 不再学习 → "死亡"。

与 ANN ReLU 的"死神经元"类似，但 SNN 中更严重：
  - ANN：神经元输出 < 0 → ReLU 输出 0 → 梯度消失
  - SNN：膜电位永远 < Vth 或永远 > Vth → 脉冲率=0 或 =1 → 代理梯度≈0

分析内容：
  1. 死神经元检测：
     - 沉默神经元（silent）：整个数据集上从不发放（rate=0）
     - 超活跃神经元（saturated）：几乎每步都发放（rate>0.9）
     - 两者都无法有效传递梯度

  2. 任务-死神经元相关性：
     - 哪些任务有更多死神经元？
     - 死神经元的空间分布（是否集中在某些层）？

  3. 退化区域-活性关系：
     - 退化区域（如雨线）上的神经元是否更活跃？
     - 背景区域上的神经元是否沉默？
     - 这是 SNN 高效处理稀疏退化的机制证据

  4. 训练过程中神经元死亡动态：
     - 随 epoch 增加，死神经元比例的变化曲线
     - 不同超参数（tau、surrogate）对死亡率的影响

  5. 层间活性梯度：
     - 从输入层到输出层，神经元活性的变化模式
     - SNN 是否有"活性坍缩"现象（深层神经元大量死亡）？
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NeuronViabilityStats:
    """单层神经元活性统计。"""
    layer_name: str
    total_neurons: int

    # 活性分类
    n_silent: int = 0              # 沉默神经元（rate < silent_threshold）
    n_saturated: int = 0           # 超活跃神经元（rate > saturated_threshold）
    n_normal: int = 0              # 正常神经元
    n_irregular: int = 0           # 异常活跃但未饱和

    # 比例
    silent_fraction: float = 0.0
    saturated_fraction: float = 0.0
    normal_fraction: float = 0.0
    effective_fraction: float = 0.0   # 真正有效传递梯度的神经元比例

    # 分布
    rate_histogram: Optional[np.ndarray] = None
    rate_mean: float = 0.0
    rate_std: float = 0.0
    rate_gini: float = 0.0           # 发放率分布的不均匀性

    # 空间分布
    silent_spatial_map: Optional[np.ndarray] = None    # (H,W) 沉默神经元密度
    saturated_spatial_map: Optional[np.ndarray] = None

    # 退化区域相关性
    activation_degradation_corr: float = 0.0  # 活跃区域 vs 退化区域的相关性


class NeuronViability:
    """
    神经元活性分析器。

    检测 SNN 中的沉默/饱和神经元，
    分析神经元活性与图像退化区域的空间对应关系。

    Args:
        silent_threshold: 低于此发放率认为是沉默神经元（推荐 0.01-0.05）
        saturated_threshold: 高于此发放率认为是饱和神经元（推荐 0.85-0.95）
        n_histogram_bins: 发放率直方图的 bins 数
    """

    def __init__(
        self,
        silent_threshold: float = 0.02,
        saturated_threshold: float = 0.9,
        n_histogram_bins: int = 50,
    ):
        self.silent_thr = silent_threshold
        self.saturated_thr = saturated_threshold
        self.n_bins = n_histogram_bins

    def analyze_layer(
        self,
        spikes: np.ndarray,           # (T, B, C, H, W) 二值脉冲
        layer_name: str = "layer",
        degradation_mask: Optional[np.ndarray] = None,  # (H, W) 退化区域掩码
    ) -> NeuronViabilityStats:
        """
        分析单层的神经元活性。

        Args:
            spikes: (T, B, C, H, W) 二值脉冲序列
            layer_name: 层名
            degradation_mask: (H, W) 退化区域二值掩码（1=退化，0=背景）

        Returns:
            NeuronViabilityStats
        """
        T, B, C, H, W = spikes.shape
        # 每个空间位置的发放率：对 T 和 B 求平均
        rate_map = spikes.mean(axis=(0, 1))  # (C, H, W)

        # 展平为 (C*H*W,)
        rates = rate_map.reshape(-1)
        N = len(rates)

        # 分类
        silent_mask    = rates < self.silent_thr
        saturated_mask = rates > self.saturated_thr
        normal_mask    = (~silent_mask) & (~saturated_mask)
        # "正常"但可能梯度仍然很小（代理梯度在远离阈值时衰减）
        # 有效范围：ATan 代理梯度在 v ≈ Vth 时最大，v 偏离 2σ 后≈0
        # 近似有效范围：rate ∈ [0.1, 0.7]
        effective_mask = (rates >= 0.1) & (rates <= 0.7)

        n_silent    = int(silent_mask.sum())
        n_saturated = int(saturated_mask.sum())
        n_normal    = int(normal_mask.sum())
        n_effective = int(effective_mask.sum())

        # 发放率 Gini 系数（测量神经元间活性的不均匀性）
        rates_sorted = np.sort(rates)
        idx = np.arange(1, N + 1)
        if rates_sorted.sum() > 0:
            gini = float((2*idx - N - 1).dot(rates_sorted) / (N * rates_sorted.sum()))
        else:
            gini = 0.0

        # 直方图
        hist, _ = np.histogram(rates, bins=self.n_bins, range=(0.0, 1.0))

        # 空间分布（对 C 求平均）
        silent_spatial    = (rate_map < self.silent_thr).astype(float).mean(0)     # (H, W)
        saturated_spatial = (rate_map > self.saturated_thr).astype(float).mean(0)  # (H, W)

        # 退化区域与活性的相关性
        act_deg_corr = 0.0
        if degradation_mask is not None and degradation_mask.shape == (H, W):
            # 活跃区域：rate > 0.1 的均值图
            activity_map = (rate_map > 0.1).astype(float).mean(0)  # (H, W)
            deg_flat = degradation_mask.flatten()
            act_flat = activity_map.flatten()
            if deg_flat.std() > 0 and act_flat.std() > 0:
                corr = np.corrcoef(deg_flat, act_flat)[0, 1]
                act_deg_corr = float(0.0 if np.isnan(corr) else corr)

        return NeuronViabilityStats(
            layer_name=layer_name,
            total_neurons=N,
            n_silent=n_silent, n_saturated=n_saturated,
            n_normal=n_normal,
            n_irregular=n_normal - n_effective,
            silent_fraction=n_silent / N,
            saturated_fraction=n_saturated / N,
            normal_fraction=n_normal / N,
            effective_fraction=n_effective / N,
            rate_histogram=hist,
            rate_mean=float(rates.mean()),
            rate_std=float(rates.std()),
            rate_gini=gini,
            silent_spatial_map=silent_spatial,
            saturated_spatial_map=saturated_spatial,
            activation_degradation_corr=act_deg_corr,
        )

    def analyze_all(
        self,
        spike_records: Dict[str, np.ndarray],
        degradation_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, NeuronViabilityStats]:
        """分析所有层。"""
        return {
            name: self.analyze_layer(spikes, name, degradation_mask)
            for name, spikes in spike_records.items()
        }

    def global_health_score(
        self, all_stats: Dict[str, NeuronViabilityStats]
    ) -> Dict[str, float]:
        """
        计算模型级别的神经元健康度指标。

        Returns:
            {
              'effective_neuron_ratio': 有效神经元平均比例,
              'dead_neuron_ratio': 死神经元（沉默+饱和）比例,
              'activity_gini': 活性不均匀性,
              'health_score': 综合健康分 [0,1]
            }
        """
        effective = [s.effective_fraction for s in all_stats.values()]
        silent    = [s.silent_fraction for s in all_stats.values()]
        saturated = [s.saturated_fraction for s in all_stats.values()]
        gini      = [s.rate_gini for s in all_stats.values()]

        mean_eff  = float(np.mean(effective))
        mean_dead = float(np.mean(silent) + np.mean(saturated))
        mean_gini = float(np.mean(gini))

        # 健康分：有效比例高、死亡比例低、活性不太不均匀
        health = (
            0.5 * mean_eff +
            0.3 * (1 - mean_dead) +
            0.2 * (1 - mean_gini)
        )

        return {
            "effective_neuron_ratio": mean_eff,
            "dead_neuron_ratio": mean_dead,
            "activity_gini": mean_gini,
            "health_score": float(health),
        }

    def print_viability_report(self, all_stats: Dict[str, NeuronViabilityStats]):
        print(f"\n{'='*80}")
        print("Neuron Viability Analysis")
        print(f"{'Layer':<35} {'N':>8} {'Silent%':>9} {'Satur%':>9} "
              f"{'Effect%':>9} {'Gini':>7} {'ActDegCorr':>12}")
        print("-"*80)
        for name, s in sorted(all_stats.items()):
            print(f"{name:<35} {s.total_neurons:>8} "
                  f"{100*s.silent_fraction:>8.1f}% "
                  f"{100*s.saturated_fraction:>8.1f}% "
                  f"{100*s.effective_fraction:>8.1f}% "
                  f"{s.rate_gini:>7.3f} "
                  f"{s.activation_degradation_corr:>12.4f}")

        health = self.global_health_score(all_stats)
        print("-"*80)
        print(f"  Global Health Score: {health['health_score']:.4f}")
        print(f"  Effective neurons:   {100*health['effective_neuron_ratio']:.1f}%")
        print(f"  Dead neurons:        {100*health['dead_neuron_ratio']:.1f}%")
        print(f"  Activity Gini:       {health['activity_gini']:.4f}")
        print(f"{'='*80}")
        print("  High ActDegCorr = neuron activation aligns with degradation regions")
        print("  (ideal: encoder fires at rain/noise/haze → efficient sparse coding)")
