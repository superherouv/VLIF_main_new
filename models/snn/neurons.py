"""
Neuron Models for Spiking Neural Networks.

Implements various biologically-plausible and engineering-oriented neuron models:
  - LIFNeuron: standard Leaky Integrate-and-Fire (most common in deep SNN literature)
  - AdaptiveLIFNeuron: LIF + threshold adaptation (reduces burst firing)
  - ParametricLIFNeuron: learnable τ, Vth per-channel (PLIF, Fang et al. 2021)
  - IzhikevichNeuron: rich dynamics, two-variable system (rarely used in vision)

All neurons support:
  - Soft/hard reset after spike
  - Detach-reset trick to improve gradient flow
  - State dictionary for multi-step unfolding

Membrane dynamics (discrete, Euler):
    v[t] = τ * v[t-1] * (1 - spike[t-1]) + I[t]   (hard reset)
    v[t] = τ * v[t-1] - Vth * spike[t-1] + I[t]    (soft reset)
"""

import torch
import torch.nn as nn
from .surrogate import SurrogateATan, SurrogateFunction
from typing import Optional, Tuple


class LIFNeuron(nn.Module):
    """
    Standard Leaky Integrate-and-Fire neuron.

    Args:
        tau (float): Membrane time constant decay factor (0 < τ < 1).
            Larger τ → longer memory; typical range [0.2, 0.9].
        threshold (float): Firing threshold V_th.
        reset_mode (str): 'hard' zeros membrane; 'soft' subtracts V_th.
        detach_reset (bool): Detach spike from computation graph on reset
            (improves gradient flow, Zenke & Neftci 2021).
        surrogate (str): Which surrogate gradient to use.
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 1.0,
        reset_mode: str = "soft",
        detach_reset: bool = True,
        surrogate: str = "atan",
        surrogate_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_mode = reset_mode
        self.detach_reset = detach_reset
        surrogate_kwargs = surrogate_kwargs or {}
        self.surrogate_fn = SurrogateFunction(surrogate, **surrogate_kwargs)

        # Membrane state (initialized lazily to match input shape)
        self.register_buffer("membrane", torch.zeros(1), persistent=False)
        self._state_initialized = False

    def _init_state(self, x: torch.Tensor):
        self.membrane = torch.zeros_like(x)
        self._state_initialized = True

    def reset_state(self, batch_size: Optional[int] = None):
        """Reset membrane potential to zero (call between sequences)."""
        if self._state_initialized:
            self.membrane = torch.zeros_like(self.membrane)
            if batch_size is not None and self.membrane.shape[0] != batch_size:
                self.membrane = torch.zeros(
                    batch_size, *self.membrane.shape[1:],
                    device=self.membrane.device, dtype=self.membrane.dtype
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single time-step forward pass.

        Args:
            x: synaptic input, shape (B, C, H, W) or (B, C)

        Returns:
            spike: binary spike tensor, same shape as x
        """
        if not self._state_initialized or self.membrane.shape != x.shape:
            self._init_state(x)

        # Integrate: v[t] = τ·v[t-1] + I[t]
        self.membrane = self.tau * self.membrane.detach() + x

        # Fire
        spike = self.surrogate_fn(self.membrane, self.threshold)

        # Reset
        if self.detach_reset:
            spike_detached = spike.detach()
        else:
            spike_detached = spike

        if self.reset_mode == "hard":
            self.membrane = self.membrane * (1.0 - spike_detached)
        else:  # soft reset
            self.membrane = self.membrane - self.threshold * spike_detached

        return spike

    def extra_repr(self) -> str:
        return (f"tau={self.tau}, threshold={self.threshold}, "
                f"reset={self.reset_mode}, surrogate={self.surrogate_fn.name}")


class AdaptiveLIFNeuron(nn.Module):
    """
    LIF neuron with adaptive threshold (ALIF).

    After each spike, threshold increases by Δ, then decays with τ_adapt.
    This implements spike-frequency adaptation, preventing excessive firing
    and improving temporal coding fidelity.

    v[t] = τ·v[t-1] + I[t]
    Vth[t] = Vth_base + a[t-1]
    a[t] = τ_adapt · a[t-1] + b · spike[t]
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold_base: float = 1.0,
        tau_adapt: float = 0.8,
        b_adapt: float = 1.6,
        reset_mode: str = "soft",
        surrogate: str = "atan",
    ):
        super().__init__()
        self.tau = tau
        self.threshold_base = threshold_base
        self.tau_adapt = tau_adapt
        self.b_adapt = b_adapt
        self.reset_mode = reset_mode
        self.surrogate_fn = SurrogateFunction(surrogate)

        self.register_buffer("membrane", torch.zeros(1), persistent=False)
        self.register_buffer("threshold_adapt", torch.zeros(1), persistent=False)
        self._state_initialized = False

    def _init_state(self, x: torch.Tensor):
        self.membrane = torch.zeros_like(x)
        self.threshold_adapt = torch.zeros_like(x)
        self._state_initialized = True

    def reset_state(self):
        if self._state_initialized:
            self.membrane.zero_()
            self.threshold_adapt.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._state_initialized or self.membrane.shape != x.shape:
            self._init_state(x)

        # Dynamic threshold
        threshold = self.threshold_base + self.threshold_adapt

        self.membrane = self.tau * self.membrane.detach() + x
        spike = self.surrogate_fn(self.membrane, threshold=1.0)  # membrane vs threshold

        # Update adaptive threshold
        self.threshold_adapt = (self.tau_adapt * self.threshold_adapt.detach()
                                + self.b_adapt * spike.detach())

        # Reset
        if self.reset_mode == "hard":
            self.membrane = self.membrane * (1.0 - spike.detach())
        else:
            self.membrane = self.membrane - threshold * spike.detach()

        return spike


class ParametricLIFNeuron(nn.Module):
    """
    Parametric LIF (PLIF) with learnable per-channel τ.

    Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance
    Learning of Spiking Neural Networks", ICCV 2021.

    τ is learned via sigmoid reparameterization: τ = sigmoid(w),
    ensuring τ ∈ (0,1) without constraints.
    """

    def __init__(
        self,
        channels: int,
        threshold: float = 1.0,
        init_tau: float = 0.5,
        reset_mode: str = "soft",
        surrogate: str = "atan",
    ):
        super().__init__()
        self.threshold = threshold
        self.reset_mode = reset_mode
        self.surrogate_fn = SurrogateFunction(surrogate)

        # Learnable tau per channel: τ = sigmoid(w)
        init_w = torch.log(torch.tensor(init_tau / (1.0 - init_tau)))
        self.w_tau = nn.Parameter(init_w.expand(channels).clone())

        self.register_buffer("membrane", torch.zeros(1), persistent=False)
        self._state_initialized = False

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self.w_tau)

    def _init_state(self, x: torch.Tensor):
        self.membrane = torch.zeros_like(x)
        self._state_initialized = True

    def reset_state(self):
        if self._state_initialized:
            self.membrane.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._state_initialized or self.membrane.shape != x.shape:
            self._init_state(x)

        # Broadcast tau over spatial dims: (C,) -> (1,C,1,1) or (1,C)
        tau = self.tau
        if x.dim() == 4:
            tau = tau.view(1, -1, 1, 1)
        else:
            tau = tau.view(1, -1)

        self.membrane = tau * self.membrane.detach() + x
        spike = self.surrogate_fn(self.membrane, self.threshold)

        if self.reset_mode == "hard":
            self.membrane = self.membrane * (1.0 - spike.detach())
        else:
            self.membrane = self.membrane - self.threshold * spike.detach()

        return spike


class IzhikevichNeuron(nn.Module):
    """
    Izhikevich two-variable neuron model (simplified for deep learning).

    dv/dt = 0.04v² + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30mV: v ← c, u ← u + d

    Discretized with small Δt. Rarely competitive with LIF for image tasks
    but included for completeness and ablation studies.
    """

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 1.0,
        surrogate: str = "atan",
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.surrogate_fn = SurrogateFunction(surrogate)

        self.register_buffer("v", torch.zeros(1), persistent=False)
        self.register_buffer("u", torch.zeros(1), persistent=False)
        self._state_initialized = False

    def _init_state(self, x: torch.Tensor):
        self.v = torch.full_like(x, self.c)
        self.u = self.b * self.v
        self._state_initialized = True

    def reset_state(self):
        if self._state_initialized:
            self.v.fill_(self.c)
            self.u = self.b * self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._state_initialized or self.v.shape != x.shape:
            self._init_state(x)

        v, u = self.v.detach(), self.u.detach()
        dv = (0.04 * v**2 + 5.0 * v + 140.0 - u + x) * self.dt
        du = (self.a * (self.b * v - u)) * self.dt
        self.v = v + dv
        self.u = u + du

        # Spike at 30 mV
        spike = self.surrogate_fn(self.v, threshold=30.0)

        # Reset
        self.v = self.v * (1 - spike.detach()) + self.c * spike.detach()
        self.u = self.u + self.d * spike.detach()

        return spike
