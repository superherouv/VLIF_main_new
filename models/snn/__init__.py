"""
SNN Core Modules for Low-Level Vision Tasks.

This package implements Spiking Neural Network building blocks:
- LIF (Leaky Integrate-and-Fire) neuron variants
- Surrogate gradient functions for BPTT training
- Spike encoding schemes (rate, temporal, phase)
- SNN-specific normalization layers
- Synaptic plasticity mechanisms
"""

from .neurons import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    ParametricLIFNeuron,
    IzhikevichNeuron,
)
from .surrogate import (
    SurrogateSigmoid,
    SurrogateATan,
    SurrogateTriangle,
    SurrogateMultiGaussian,
)
from .encoding import (
    RateEncoder,
    TemporalEncoder,
    PhaseEncoder,
    DirectEncoder,
    PopulationEncoder,
)
from .layers import (
    SNNConv2d,
    SNNLinear,
    SNNBatchNorm,
    SNNResidualBlock,
    SNNAttention,
    ThresholdBalancing,
)
from .membrane import MembraneStateManager

__all__ = [
    # Neurons
    "LIFNeuron", "AdaptiveLIFNeuron", "ParametricLIFNeuron", "IzhikevichNeuron",
    # Surrogate gradients
    "SurrogateSigmoid", "SurrogateATan", "SurrogateTriangle", "SurrogateMultiGaussian",
    # Encoders
    "RateEncoder", "TemporalEncoder", "PhaseEncoder", "DirectEncoder", "PopulationEncoder",
    # Layers
    "SNNConv2d", "SNNLinear", "SNNBatchNorm", "SNNResidualBlock", "SNNAttention",
    "ThresholdBalancing",
    # Memory management
    "MembraneStateManager",
]
