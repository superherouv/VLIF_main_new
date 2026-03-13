"""
Model Registry for SNN Applicability Study.

Available models:
  - 'snn_restorer': Spiking U-Net for image restoration
  - 'ann_restorer': ANN NAFNet baseline
  - 'hybrid_restorer': SNN encoder + ANN decoder (ablation)
"""

from .snn_restorer import SNNRestorer
from .ann.restorer import ANNRestorer
from .hybrid_restorer import HybridRestorer

MODEL_REGISTRY = {
    "snn": SNNRestorer,
    "ann": ANNRestorer,
    "hybrid": HybridRestorer,
}


def build_model(model_type: str, **kwargs):
    """
    Build a model by type string.

    Args:
        model_type: 'snn' | 'ann' | 'hybrid'
        **kwargs: passed to model constructor

    Returns:
        nn.Module instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = ["SNNRestorer", "ANNRestorer", "HybridRestorer", "build_model", "MODEL_REGISTRY"]
