"""
Surrogate Gradient Functions for SNN Training via BPTT.

The core challenge of training SNNs: the Heaviside step function H(x)
has zero gradient almost everywhere. We replace it with smooth surrogates
in the backward pass only (straight-through estimator variant).

Reference:
    Neftci et al., "Surrogate Gradient Learning in Spiking Neural Networks", 2019.
    Zenke & Ganguli, "SuperSpike: Supervised Learning in Multilayer SNNs", 2018.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class SurrogateSigmoid(Function):
    """
    Sigmoid surrogate: forward=Heaviside, backward=sigmoid'(kx).

    Smooth approximation: σ'(kx) = k·σ(kx)·(1−σ(kx))
    Good default choice; k controls sharpness.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float = 1.0,
                scale: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.scale = scale
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        scale = ctx.scale
        x = membrane - threshold
        sig = torch.sigmoid(scale * x)
        surrogate_grad = scale * sig * (1.0 - sig)
        return grad_output * surrogate_grad, None, None


class SurrogateATan(Function):
    """
    Arctangent surrogate: forward=Heaviside, backward=α/(1+(παx)²).

    Widely used in SpikingJelly and STBP-based work.
    Provides heavier tails than sigmoid, better gradient flow.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float = 1.0,
                alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.alpha = alpha
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        x = membrane - threshold
        surrogate_grad = alpha / 2.0 / (1.0 + (torch.pi / 2.0 * alpha * x) ** 2)
        return grad_output * surrogate_grad, None, None


class SurrogateTriangle(Function):
    """
    Triangular / piecewise-linear surrogate.

    Computationally cheapest; good for hardware mapping.
    gradient = max(0, 1 - |x-threshold|/window)
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float = 1.0,
                window: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.window = window
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        window = ctx.window
        x = membrane - threshold
        surrogate_grad = torch.clamp(1.0 - torch.abs(x) / window, min=0.0)
        return grad_output * surrogate_grad, None, None


class SurrogateMultiGaussian(Function):
    """
    Multi-Gaussian surrogate (Yin et al., 2021).

    Combines a positive and two negative Gaussians to form
    a derivative-like shape: better at capturing sharp transitions.

    f'(x) ≈ A·G(x,σ) − B·(G(x-μ,σ) + G(x+μ,σ))
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        x = membrane - threshold
        # Parameters from Yin et al.
        h = 0.15
        s = 6.0
        mu = 0.5
        A = 0.5
        B = 0.25
        g1 = torch.exp(-x**2 * s) / (2 * torch.pi / s)**0.5
        g2 = torch.exp(-(x - mu)**2 * s) / (2 * torch.pi / s)**0.5
        g3 = torch.exp(-(x + mu)**2 * s) / (2 * torch.pi / s)**0.5
        surrogate_grad = h * (A * g1 - B * (g2 + g3))
        return grad_output * surrogate_grad, None


# Convenience wrappers as nn.Module (for use inside Sequential / ModuleDict)
class SurrogateFunction(nn.Module):
    """Base class for surrogate functions as nn.Module wrappers."""

    def __init__(self, name: str = "atan", **kwargs):
        super().__init__()
        self.name = name
        self.kwargs = kwargs
        _registry = {
            "sigmoid": SurrogateSigmoid,
            "atan": SurrogateATan,
            "triangle": SurrogateTriangle,
            "multi_gaussian": SurrogateMultiGaussian,
        }
        if name not in _registry:
            raise ValueError(f"Unknown surrogate '{name}'. Choose from {list(_registry)}")
        self._fn = _registry[name]

    def forward(self, membrane: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        return self._fn.apply(membrane, threshold, **self.kwargs)

    def extra_repr(self) -> str:
        return f"name={self.name}, kwargs={self.kwargs}"
