"""
膜电位相空间分析 (Membrane Potential Phase Space Analysis).

这是 SNN 研究中**最独特**的分析方法，在图像恢复领域几乎未被探索。

核心思路：
  不仅记录脉冲（0/1），还记录膜电位的连续轨迹 v(t)。
  通过相空间重构，分析 SNN 的"计算轨迹"。

相空间分析内容：
  1. 相平面 (v, dv/dt)：
     膜电位 vs 其变化率，类似于弹簧-质量系统的相轨迹
     → 有无稳定不动点？极限环？混沌？
     → 不同退化类型是否产生不同的相轨迹？

  2. 吸引子分析：
     反复出现的膜电位模式 → SNN 是否有"记忆"退化模式的吸引子？
     → 使用 Takens 嵌入定理重构吸引子

  3. 阈值穿越动力学：
     膜电位接近阈值（v → Vth）时的统计特性
     → 阈值穿越速度 dv/dt|_{v→Vth} 是否与退化强度相关？

  4. 非线性度量：
     - Lyapunov 指数（局部稳定性）
     - 相空间体积（维数）
     → SNN vs ANN：SNN 是低维计算？还是高维混沌？

  5. 恢复功能映射：
     膜电位的均值场 E[v(t)] 是否是输入退化的线性/非线性函数？
     → 这揭示了 SNN 的实际"计算功能"

实现：
  纯 NumPy 实现，不依赖 PyDSTools 等专业库
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PhaseSpaceStats:
    """单层膜电位的相空间统计。"""
    layer_name: str
    T: int              # 时间步数
    n_neurons: int      # 神经元数量（C×H×W 展平）

    # 相平面统计
    mean_v: float = 0.0             # 平均膜电位
    std_v: float = 0.0              # 膜电位标准差
    mean_dv: float = 0.0            # 平均 dv/dt
    threshold_crossing_rate: float = 0.0  # 单步内接近阈值（v > 0.8 Vth）的比例
    subthreshold_variance: float = 0.0    # 阈下区域的膜电位方差

    # 轨迹特征
    autocorr_tau1: float = 0.0      # 膜电位时间自相关（lag=1）
    mean_trajectory_length: float = 0.0  # 相空间中轨迹长度（运动量）
    phase_volume: float = 0.0       # 相空间体积估计

    # 功能分析
    v_input_correlation: float = 0.0     # E[v(t)] 与输入的相关性
    spike_efficiency: float = 0.0        # 脉冲/信息量 比（低 = 高效）

    # 轨迹数据（可选，用于绘图）
    trajectory_sample: Optional[np.ndarray] = None  # (T, sample_neurons)


class MembranePhaseSpace:
    """
    膜电位相空间分析器。

    收集 SNN 各层在 T 个时间步的膜电位轨迹，
    进行相空间重构和动力学分析。

    用法：
        # 在 SNN 前向传播时记录膜电位
        # membrane_records: {layer_name: (T, B, C, H, W) float}
        analyzer = MembranePhaseSpace(threshold=1.0)
        stats = analyzer.analyze(membrane_records)
        analyzer.find_attractors(membrane_records['bottleneck'])
    """

    def __init__(self, threshold: float = 1.0, n_sample_neurons: int = 100):
        self.threshold = threshold
        self.n_sample = n_sample_neurons

    def analyze_layer(
        self,
        membrane: np.ndarray,   # (T, B, C, H, W)
        layer_name: str = "layer",
        input_signal: Optional[np.ndarray] = None,  # (B, C, H, W)
    ) -> PhaseSpaceStats:
        """
        分析单层的膜电位动力学。

        Args:
            membrane: (T, B, C, H, W) 连续膜电位值
            layer_name: 层名
            input_signal: 该层的输入信号（用于计算 MI）

        Returns:
            PhaseSpaceStats
        """
        T, B, C, H, W = membrane.shape
        # 展平空间维度：(T, N) 其中 N = B*C*H*W
        v = membrane.reshape(T, -1).astype(float)
        N = v.shape[1]

        # ── 基本统计 ─────────────────────────────────────────────────
        mean_v = float(v.mean())
        std_v  = float(v.std())

        # dv/dt（有限差分）
        if T > 1:
            dv = np.diff(v, axis=0)  # (T-1, N)
            mean_dv = float(np.abs(dv).mean())
        else:
            dv = np.zeros((0, N))
            mean_dv = 0.0

        # 阈值穿越率：v > 0.8 * Vth 的比例
        tcr = float((v > 0.8 * self.threshold).mean())

        # 阈下方差（v < Vth 的区域）
        sub_mask = v < self.threshold
        if sub_mask.sum() > 0:
            sub_var = float(v[sub_mask].var())
        else:
            sub_var = 0.0

        # ── 时间自相关 ───────────────────────────────────────────────
        if T > 2:
            # 对采样神经元计算 lag-1 自相关
            sample_n = min(self.n_sample, N)
            idx = np.random.choice(N, sample_n, replace=False)
            v_sample = v[:, idx]  # (T, sample_n)
            # 自相关：corr(v[t], v[t-1])
            if T > 1:
                v_centered = v_sample - v_sample.mean(0)
                v1 = v_centered[:-1]  # (T-1, sample_n)
                v2 = v_centered[1:]   # (T-1, sample_n)
                cov = (v1 * v2).mean()
                var = v_centered.var()
                ac = float(cov / (var + 1e-8))
            else:
                ac = 0.0
        else:
            ac = 0.0
            sample_n = min(self.n_sample, N)
            idx = np.random.choice(N, sample_n, replace=False)
            v_sample = v[:, idx]

        # ── 轨迹长度（相空间中运动量）────────────────────────────────
        if T > 1:
            traj_len = float(np.linalg.norm(dv, axis=1).mean())
        else:
            traj_len = 0.0

        # ── 相空间体积估计（协方差矩阵的行列式）──────────────────────
        if T > 2 and sample_n >= 2:
            # 用时间维度的协方差
            cov_matrix = np.cov(v_sample.T)   # (sample_n, sample_n)
            # 近似体积：log|Σ| 的下界
            try:
                eigvals = np.linalg.eigvalsh(cov_matrix)
                phase_vol = float(np.log(np.abs(eigvals) + 1e-8).sum())
            except:
                phase_vol = 0.0
        else:
            phase_vol = 0.0

        # ── 输入-膜电位相关性 ─────────────────────────────────────────
        v_input_corr = 0.0
        if input_signal is not None:
            inp_flat = input_signal.reshape(-1)[:N]
            v_mean_over_T = v.mean(0)  # (N,)
            min_len = min(len(inp_flat), len(v_mean_over_T))
            if min_len > 1:
                corr_mat = np.corrcoef(inp_flat[:min_len], v_mean_over_T[:min_len])
                v_input_corr = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0

        # ── 脉冲效率 ──────────────────────────────────────────────────
        spike_rate = float((v >= self.threshold).mean())
        spike_eff = spike_rate / (std_v + 1e-8)  # 每单位方差的脉冲数

        return PhaseSpaceStats(
            layer_name=layer_name,
            T=T, n_neurons=N,
            mean_v=mean_v, std_v=std_v,
            mean_dv=mean_dv,
            threshold_crossing_rate=tcr,
            subthreshold_variance=sub_var,
            autocorr_tau1=ac,
            mean_trajectory_length=traj_len,
            phase_volume=phase_vol,
            v_input_correlation=v_input_corr,
            spike_efficiency=float(spike_eff),
            trajectory_sample=v_sample if T <= 16 else None,
        )

    def analyze(
        self,
        membrane_records: Dict[str, np.ndarray],
    ) -> Dict[str, PhaseSpaceStats]:
        """分析所有层的膜电位相空间。"""
        return {
            name: self.analyze_layer(membrane, name)
            for name, membrane in membrane_records.items()
        }

    def find_attractors(
        self,
        membrane: np.ndarray,  # (T, B, C, H, W)
        n_clusters: int = 5,
    ) -> Dict[str, np.ndarray]:
        """
        使用 k-means 聚类找到膜电位轨迹的吸引子（代表性模式）。

        Returns:
            {'centroids': (k, T), 'labels': (B*C*H*W,), 'inertia': float}
        """
        T = membrane.shape[0]
        v = membrane.reshape(T, -1).T  # (N, T)

        # 简单 k-means（纯 NumPy 实现）
        N = v.shape[0]
        if N < n_clusters:
            return {}

        # 随机初始化中心
        np.random.seed(42)
        centroids = v[np.random.choice(N, n_clusters, replace=False)].copy()
        labels = np.zeros(N, dtype=int)

        for _ in range(50):  # 最多 50 次迭代
            # 分配
            dists = np.array([
                np.sum((v - c)**2, axis=1) for c in centroids
            ]).T  # (N, k)
            new_labels = dists.argmin(axis=1)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # 更新中心
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    centroids[k] = v[mask].mean(0)

        inertia = sum(
            np.sum((v[labels == k] - centroids[k])**2)
            for k in range(n_clusters)
            if (labels == k).sum() > 0
        )

        return {
            "centroids": centroids,   # (k, T) 吸引子轨迹
            "labels": labels,         # (N,) 聚类标签
            "inertia": float(inertia),
        }

    def print_summary(self, stats: Dict[str, PhaseSpaceStats]):
        print(f"\n{'='*80}")
        print("Membrane Potential Phase Space Analysis")
        print(f"{'Layer':<35} {'E[v]':>7} {'σ[v]':>7} {'TCR':>7} "
              f"{'AC(1)':>7} {'TrjLen':>8} {'SpikeEff':>10}")
        print("-"*80)
        for name, s in sorted(stats.items()):
            print(f"{name:<35} {s.mean_v:>7.4f} {s.std_v:>7.4f} "
                  f"{s.threshold_crossing_rate:>7.4f} "
                  f"{s.autocorr_tau1:>7.4f} "
                  f"{s.mean_trajectory_length:>8.4f} "
                  f"{s.spike_efficiency:>10.4f}")
        print(f"{'='*80}")
        print("  TCR: threshold crossing rate | AC(1): lag-1 autocorrelation")
        print("  TrjLen: phase-space trajectory length | SpikeEff: spikes per unit variance")
