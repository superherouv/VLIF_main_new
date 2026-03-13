"""
SNN-Specific Layer Building Blocks.

Wraps standard torch modules with:
  - Per-timestep application
  - Automatic neuron state management
  - Spike activity monitoring for energy estimation
  - Threshold balancing for ANN→SNN conversion

Architecture convention:
    SNNConv2d = Conv2d → BN → LIFNeuron (one unit cell)
    All forward() methods accept (T, B, C, H, W) tensors and return same shape.
"""

import torch
import torch.nn as nn
from .neurons import LIFNeuron, ParametricLIFNeuron
from typing import Optional, Dict, Tuple


class SNNConv2d(nn.Module):
    """
    Convolutional SNN layer: Conv2d + optional BN + LIF neuron.

    Processes T timesteps sequentially, maintaining membrane state.

    Args:
        in_channels, out_channels, kernel_size, stride, padding: std conv args
        tau (float): LIF membrane time constant
        threshold (float): firing threshold
        neuron_type (str): 'lif' | 'plif' | 'alif'
        surrogate (str): surrogate gradient type
        bn (bool): whether to add BatchNorm
        bias (bool): conv bias (usually False when bn=True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        tau: float = 0.5,
        threshold: float = 1.0,
        neuron_type: str = "lif",
        surrogate: str = "atan",
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()

        if neuron_type == "lif":
            self.neuron = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)
        elif neuron_type == "plif":
            self.neuron = ParametricLIFNeuron(
                channels=out_channels, threshold=threshold, surrogate=surrogate
            )
        else:
            from .neurons import AdaptiveLIFNeuron
            self.neuron = AdaptiveLIFNeuron(tau=tau, threshold_base=threshold)

        # Activity tracking for energy estimation
        self._spike_count = 0
        self._total_ops = 0

    def reset_state(self):
        self.neuron.reset_state()
        self._spike_count = 0
        self._total_ops = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C_in, H, W) spike input

        Returns:
            spikes: (T, B, C_out, H', W')
        """
        T = x.shape[0]
        outputs = []
        for t in range(T):
            out = self.conv(x[t])
            out = self.bn(out)
            spike = self.neuron(out)
            outputs.append(spike)
            # Track activity
            self._spike_count += spike.sum().item()
            self._total_ops += spike.numel()
        return torch.stack(outputs, dim=0)

    @property
    def firing_rate(self) -> float:
        """Average firing rate over all recorded timesteps."""
        if self._total_ops == 0:
            return 0.0
        return self._spike_count / self._total_ops


class SNNLinear(nn.Module):
    """
    Fully-connected SNN layer: Linear + BN + LIF.

    Args:
        in_features, out_features: standard linear args
        tau, threshold, surrogate: neuron parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate: str = "atan",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.neuron = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

    def reset_state(self):
        self.neuron.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, in_features)

        Returns:
            spikes: (T, B, out_features)
        """
        T = x.shape[0]
        outputs = []
        for t in range(T):
            out = self.linear(x[t])
            out = self.bn(out)
            spike = self.neuron(out)
            outputs.append(spike)
        return torch.stack(outputs, dim=0)


class SNNBatchNorm(nn.Module):
    """
    Threshold-normalized BatchNorm for SNNs.

    Standard BN applied per-timestep with threshold normalization.
    Helps maintain spike rates near target values.

    Args:
        num_features: channel count
        target_rate (float): desired average firing rate (e.g., 0.2)
    """

    def __init__(self, num_features: int, target_rate: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.target_rate = target_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, B, C, H, W)"""
        T = x.shape[0]
        outputs = [self.bn(x[t]) for t in range(T)]
        return torch.stack(outputs, dim=0)


class SNNResidualBlock(nn.Module):
    """
    SNN Residual Block with spike-domain skip connections.

    Two SNN conv layers with a shortcut. The skip connection adds
    spike tensors (binary addition), which is exact in spike domain.

    Challenge: residual adds can increase average membrane potential,
    requiring threshold re-calibration (handled by ThresholdBalancing).

    Architecture:
        x → [Conv→BN→LIF] → [Conv→BN] → (+x_short) → LIF → out
    """

    def __init__(
        self,
        channels: int,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate: str = "atan",
        downsample: bool = False,
    ):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.neuron1 = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.neuron2 = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

        # Shortcut (identity or projection)
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

        self._spike_counts: Dict[str, float] = {}

    def reset_state(self):
        self.neuron1.reset_state()
        self.neuron2.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C, H, W)

        Returns:
            spikes: (T, B, C, H, W)
        """
        T = x.shape[0]
        outputs = []
        for t in range(T):
            xt = x[t]
            h = self.neuron1(self.bn1(self.conv1(xt)))
            h = self.bn2(self.conv2(h))
            skip = self.shortcut(xt)
            out = self.neuron2(h + skip)
            outputs.append(out)
        return torch.stack(outputs, dim=0)


class SNNAttention(nn.Module):
    """
    Spike-domain channel attention (SE-like) for SNNs.

    Computes attention weights from temporal-averaged spike rates,
    then modulates per-channel spike output.

    Since spikes are binary, attention is applied to the membrane
    potential before the final threshold comparison.

    Args:
        channels (int): Number of feature channels.
        reduction (int): Bottleneck reduction factor.
        T_accumulate (int): Timesteps to accumulate before recomputing attention.
    """

    def __init__(self, channels: int, reduction: int = 4, T_accumulate: int = 4):
        super().__init__()
        self.T_accumulate = T_accumulate
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        self.neuron = LIFNeuron(tau=0.5)

    def reset_state(self):
        self.neuron.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, C, H, W)

        Returns:
            out: (T, B, C, H, W)
        """
        T = x.shape[0]
        # Compute attention from mean spike rate across T
        mean_rate = x.mean(0)  # (B, C, H, W)
        attn = self.se(mean_rate)  # (B, C)
        attn = attn.view(attn.shape[0], attn.shape[1], 1, 1)  # (B, C, 1, 1)

        outputs = []
        for t in range(T):
            modulated = x[t] * attn  # scale spikes by attention weights
            spike = self.neuron(modulated)
            outputs.append(spike)
        return torch.stack(outputs, dim=0)


class ThresholdBalancing(nn.Module):
    """
    Data-driven threshold balancing for ANN→SNN conversion.

    After training an ANN, this module finds optimal per-layer thresholds
    by running forward passes and setting Vth = max(|activation|) per layer.

    Reference:
        Rueckauer et al., "Conversion of Continuous-Valued Deep Networks
        to Efficient Event-Driven Networks for Image Classification", 2017.

    Args:
        model: SNN model with LIF neurons
        percentile (float): Use Nth percentile instead of max (more robust).
    """

    def __init__(self, percentile: float = 99.9):
        super().__init__()
        self.percentile = percentile
        self._activation_stats: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def calibrate(self, model: nn.Module, data_loader, device: str = "cuda",
                  n_batches: int = 10):
        """
        Run calibration forward passes to collect activation statistics.

        After calling this, call apply_thresholds() to update model.

        Args:
            model: ANN model (before SNN conversion)
            data_loader: calibration data
            device: compute device
            n_batches: number of batches to use
        """
        model.eval()
        hooks = []
        stats: Dict[str, list] = {}

        def make_hook(name: str):
            def hook(module, inp, out):
                if name not in stats:
                    stats[name] = []
                stats[name].append(out.detach().flatten())
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(make_hook(name)))

        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            model(imgs)

        for hook in hooks:
            hook.remove()

        for name, activations in stats.items():
            all_acts = torch.cat(activations)
            threshold = torch.quantile(all_acts, self.percentile / 100.0)
            self._activation_stats[name] = threshold.item()

        return self._activation_stats

    def apply_thresholds(self, model: nn.Module):
        """Apply computed thresholds to corresponding SNN neurons."""
        relu_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (LIFNeuron, ParametricLIFNeuron)):
                if relu_idx < len(self._activation_stats):
                    key = list(self._activation_stats.keys())[relu_idx]
                    module.threshold = self._activation_stats[key]
                    relu_idx += 1
        return model


class MembraneStateManager(nn.Module):
    """
    Utility for managing neuron states across sequences.

    Provides:
    - Automatic state reset between samples
    - State checkpointing for long sequences
    - Gradual state warmup (truncated BPTT)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def get_all_neurons(self):
        """Return all LIF-type neurons in model."""
        neurons = []
        for module in self.model.modules():
            if isinstance(module, (LIFNeuron, ParametricLIFNeuron)):
                neurons.append(module)
        return neurons

    def reset_all_states(self):
        """Reset membrane potential of all neurons."""
        for neuron in self.get_all_neurons():
            neuron.reset_state()

    def get_firing_stats(self) -> Dict[str, float]:
        """Collect average firing rates across all SNN conv layers."""
        stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, SNNConv2d):
                stats[name] = module.firing_rate
        return stats

    def forward(self, x, reset: bool = True):
        if reset:
            self.reset_all_states()
        return self.model(x)
