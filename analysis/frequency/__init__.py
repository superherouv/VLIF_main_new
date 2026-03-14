"""
Frequency Domain Analysis Thread.

Research questions addressed:
  Q1. 在哪个频率段 SNN 的重建质量最先崩溃？
  Q2. 脉冲模式本身的频谱结构是什么？是否存在频率偏好？
  Q3. 任务的频率复杂度（高频能量占比）与 SNN 适用性的相关性？
  Q4. SNN/ANN 的误差频谱形状是否不同？（SNN 误差更集中于高频？）
  Q5. 不同时间步 T 如何影响频率保真度？
  Q6. 代理梯度类型是否影响频率响应？

Tools:
  FrequencyBandAnalyzer      - 分频段 PSNR/MSE
  SpikeSpectrumAnalyzer      - 脉冲模式的功率谱分析
  WaveletDecomposer          - 小波多尺度分解
  FrequencyComplexity        - 任务频率复杂度指标
  ErrorSpectrumAnalyzer      - 重建误差的频谱分布
  FrequencyTransferAnalyzer  - 频率传递函数曲线 (A3.3: sinusoidal grating test)
  SpikeFqCouplingAnalyzer    - Spike–band overlap + temporal PSD (A3.1 + A3.2)
"""

from .band_analysis import FrequencyBandAnalyzer
from .spike_spectrum import SpikeSpectrumAnalyzer
from .wavelet import WaveletDecomposer
from .task_complexity import FrequencyComplexity
from .error_spectrum import ErrorSpectrumAnalyzer
from .freq_transfer import FrequencyTransferAnalyzer, FreqTransferResult
from .spike_freq_coupling import SpikeFqCouplingAnalyzer, SpikeCouplingReport

__all__ = [
    "FrequencyBandAnalyzer",
    "SpikeSpectrumAnalyzer",
    "WaveletDecomposer",
    "FrequencyComplexity",
    "ErrorSpectrumAnalyzer",
    # A3 new modules
    "FrequencyTransferAnalyzer",
    "FreqTransferResult",
    "SpikeFqCouplingAnalyzer",
    "SpikeCouplingReport",
]
