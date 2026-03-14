"""
表征探究主线 (Representation Analysis Thread).

研究问题：
  Q1. SNN 和 ANN 学到的特征表征有多相似/不同？（CKA）
  Q2. SNN 各层保留了多少关于退化类型的互信息？（MI）
  Q3. 膜电位的动态轨迹揭示了什么计算机制？（Phase Space）
  Q4. 脉冲的时间间隔分布说明了什么？（ISI Statistics）
  Q5. 有多少神经元是"死的"或"过度激活的"？（Neuron Viability）
  Q6. SNN 特征是否形成了有意义的流形结构？（Manifold Analysis）

工具：
  CKAAnalyzer         - Centered Kernel Alignment 层间相似性
  MutualInfoAnalyzer  - 特征 vs 退化类型的互信息
  MembranePhaseSpace  - 膜电位相空间轨迹
  ISIAnalyzer         - 脉冲间隔统计（时间编码质量）
  NeuronViability     - 死神经元 / 饱和神经元检测
  ManifoldAnalyzer    - 特征流形分析（PCA/t-SNE on spike features）
  TemporalCKAAnalyzer - 时间维 CKA（B1 extension: convergence + redundancy）
  LinearProbeAnalyzer - 线性探针信息量分析（B5: 学到了还是解码不出来？）
"""

from .cka import CKAAnalyzer
from .mutual_info import MutualInfoAnalyzer
from .membrane_phase import MembranePhaseSpace
from .isi import ISIAnalyzer
from .neuron_viability import NeuronViability
from .manifold import ManifoldAnalyzer
from .temporal_cka import TemporalCKAAnalyzer, TemporalCKAResult
from .linear_probe import (
    LinearProbeAnalyzer,
    LinearProbeResult,
    make_degradation_type_labels,
    make_frequency_band_labels,
    make_edge_labels,
)

__all__ = [
    "CKAAnalyzer",
    "MutualInfoAnalyzer",
    "MembranePhaseSpace",
    "ISIAnalyzer",
    "NeuronViability",
    "ManifoldAnalyzer",
    # New B-thread modules
    "TemporalCKAAnalyzer",
    "TemporalCKAResult",
    "LinearProbeAnalyzer",
    "LinearProbeResult",
    "make_degradation_type_labels",
    "make_frequency_band_labels",
    "make_edge_labels",
]
