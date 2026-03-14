"""
SNN 适用边界研究 - 分析工具包

两条主线：
  Thread 1 - 频域主线: analysis.frequency
  Thread 2 - 表征主线: analysis.representation
  整合流水线: analysis.unified_pipeline
"""

from .applicability import ApplicabilityAnalyzer, TaskApplicabilityResult, estimate_signal_sparsity
from .visualizer import ApplicabilityVisualizer
from .spike_analysis import SpikeAnalyzer
from .unified_pipeline import UnifiedAnalysisPipeline, FullAnalysisReport
from .sai import SAICalculator, SAIReport, SAIComponents
from . import frequency
from . import representation

__all__ = [
    "ApplicabilityAnalyzer", "TaskApplicabilityResult", "estimate_signal_sparsity",
    "ApplicabilityVisualizer",
    "SpikeAnalyzer",
    "UnifiedAnalysisPipeline", "FullAnalysisReport",
    # Formal applicability index
    "SAICalculator", "SAIReport", "SAIComponents",
    "frequency",
    "representation",
]
