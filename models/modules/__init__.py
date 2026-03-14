"""
Pluggable modules for boundary-intervention experiments.

  HFEnhancementModule   — High-frequency enhancement for SNN (Intervention 1)
  FrequencySplitBlock   — Frequency-split processing (SNN-LF / ANN-HF paths)
"""

from .hf_attention import HFEnhancementModule, FrequencySplitBlock, HighFreqExtractor

__all__ = [
    "HFEnhancementModule",
    "FrequencySplitBlock",
    "HighFreqExtractor",
]
