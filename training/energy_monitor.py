"""
Energy Consumption Monitor for SNN vs ANN comparison.

Energy model (Horowitz 2014, neuromorphic scaling):
  - ANN MAC (GPU/CPU): ~4.6 pJ (45nm process)
  - SNN spike add (neuromorphic): ~0.9 pJ
  - SNN DRAM access: ~640 pJ per word (dominates if off-chip)

For comparing energy efficiency across tasks:
  E(SNN)  ≈ firing_rate × n_MACs × E_spike
  E(ANN)  ≈ n_MACs × E_MAC
  Ratio   = firing_rate × (E_spike / E_MAC)
           ≈ firing_rate × 0.2   [at 45nm]

This means: if firing_rate < 0.2, SNN is more energy efficient.
In practice, neuromorphic hardware (Loihi, SpiNNaker) achieves
much higher efficiency than this simple model suggests.

SynOps (Synaptic Operations):
  SynOps = Σ_l (spike_rate_l × MAC_ops_l)
  This is the standard SNN energy proxy in literature.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LayerEnergyStats:
    """Energy stats for a single layer."""
    name: str
    mac_ops: int              # Total MAC operations (ANN)
    spike_rate: float         # Average firing rate [0,1]
    synops: float             # Synaptic operations = mac_ops × spike_rate
    input_shape: tuple = ()
    output_shape: tuple = ()


@dataclass
class ModelEnergyProfile:
    """Complete energy profile for a model forward pass."""
    model_type: str
    task: str
    total_mac_ops: int = 0
    total_synops: float = 0.0
    mean_firing_rate: float = 0.0
    energy_ratio_vs_ann: float = 1.0  # SNN/ANN energy ratio
    layers: List[LayerEnergyStats] = field(default_factory=list)

    # Per-task PSNR for efficiency metric
    psnr: float = 0.0
    energy_efficiency: float = 0.0  # PSNR / (SynOps normalized)


class EnergyMonitor:
    """
    Monitors and computes energy consumption for SNN and ANN models.

    Usage:
        monitor = EnergyMonitor(model, model_type='snn')
        monitor.attach_hooks()
        with torch.no_grad():
            output = model(input)
        profile = monitor.compute_profile()
        monitor.remove_hooks()
    """

    # Energy constants (pJ, 45nm)
    E_MAC = 4.6       # ANN multiply-accumulate
    E_SPIKE = 0.9     # SNN spike event (AC operation)
    E_DRAM = 200.0    # DRAM access per 32-bit word (simplified)

    def __init__(self, model: nn.Module, model_type: str = "ann",
                 task: str = "unknown"):
        self.model = model
        self.model_type = model_type
        self.task = task
        self._hooks = []
        self._layer_stats: Dict[str, dict] = {}

    def attach_hooks(self):
        """Attach forward hooks to all Conv2d and Linear layers."""
        self._layer_stats.clear()

        def make_hook(name: str, module_type: str):
            def hook(module, inp, out):
                in_tensor = inp[0] if isinstance(inp, tuple) else inp

                # Compute MAC count
                if isinstance(module, nn.Conv2d):
                    # MACs = out_h × out_w × out_c × (in_c/groups) × kh × kw
                    B, C_out, H_out, W_out = out.shape
                    C_in = in_tensor.shape[1]
                    kh, kw = module.kernel_size
                    macs = B * C_out * H_out * W_out * (C_in // module.groups) * kh * kw
                elif isinstance(module, nn.Linear):
                    B = in_tensor.shape[0]
                    macs = B * module.out_features * module.in_features
                else:
                    macs = 0

                # Compute spike rate (if input is spike tensor)
                # For SNN: input is binary spike → firing rate = mean
                # For ANN: input is continuous → firing rate = 1.0 (all active)
                if self.model_type in ("snn", "hybrid"):
                    # Check if input looks like spikes (mostly binary)
                    if in_tensor.dtype == torch.float32:
                        unique_vals = in_tensor.unique()
                        is_spike = (len(unique_vals) <= 3 and
                                    in_tensor.min() >= 0 and
                                    in_tensor.max() <= 1.0)
                        if is_spike:
                            spike_rate = in_tensor.mean().item()
                        else:
                            spike_rate = 1.0  # analog layer
                    else:
                        spike_rate = 1.0
                else:
                    spike_rate = 1.0

                synops = macs * spike_rate

                self._layer_stats[name] = {
                    "mac_ops": int(macs),
                    "spike_rate": spike_rate,
                    "synops": synops,
                    "in_shape": tuple(in_tensor.shape),
                    "out_shape": tuple(out.shape),
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(make_hook(name, type(module).__name__))
                self._hooks.append(h)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute_profile(self, psnr: float = 0.0) -> ModelEnergyProfile:
        """Compute energy profile from collected stats."""
        profile = ModelEnergyProfile(
            model_type=self.model_type,
            task=self.task,
            psnr=psnr,
        )

        for layer_name, stats in self._layer_stats.items():
            layer_stat = LayerEnergyStats(
                name=layer_name,
                mac_ops=stats["mac_ops"],
                spike_rate=stats["spike_rate"],
                synops=stats["synops"],
                input_shape=stats.get("in_shape", ()),
                output_shape=stats.get("out_shape", ()),
            )
            profile.layers.append(layer_stat)
            profile.total_mac_ops += stats["mac_ops"]
            profile.total_synops += stats["synops"]

        if len(profile.layers) > 0:
            profile.mean_firing_rate = profile.total_synops / max(profile.total_mac_ops, 1)

        # Energy ratio vs ANN equivalent
        ann_energy = profile.total_mac_ops * self.E_MAC
        snn_energy = profile.total_synops * self.E_SPIKE
        if ann_energy > 0:
            profile.energy_ratio_vs_ann = snn_energy / ann_energy
        else:
            profile.energy_ratio_vs_ann = 1.0

        # Energy efficiency metric
        if profile.total_synops > 0:
            normalized_synops = profile.total_synops / 1e9  # GigaSynOps
            profile.energy_efficiency = psnr / (normalized_synops + 1e-8)

        return profile

    def print_report(self, profile: ModelEnergyProfile):
        """Print human-readable energy report."""
        print(f"\n{'='*60}")
        print(f"Energy Profile: {profile.model_type.upper()} | Task: {profile.task}")
        print(f"{'='*60}")
        print(f"  Total MACs:        {profile.total_mac_ops:>15,}")
        print(f"  Total SynOps:      {profile.total_synops:>15.1f}")
        print(f"  Mean Firing Rate:  {profile.mean_firing_rate:>15.4f}")
        print(f"  Energy vs ANN:     {profile.energy_ratio_vs_ann:>15.4f}×")
        print(f"  PSNR:              {profile.psnr:>15.2f} dB")
        print(f"  Energy Efficiency: {profile.energy_efficiency:>15.2f} dB/GSynOps")
        print(f"\nTop-10 layers by MACs:")
        sorted_layers = sorted(profile.layers, key=lambda x: x.mac_ops, reverse=True)
        for layer in sorted_layers[:10]:
            print(f"  {layer.name:40s} | MACs={layer.mac_ops:>12,} | "
                  f"rate={layer.spike_rate:.3f}")
        print(f"{'='*60}\n")


def compute_macs(model: nn.Module, input_shape: tuple,
                 device: str = "cpu") -> int:
    """
    Quick MAC computation using a single forward pass.

    Args:
        model: The neural network.
        input_shape: (B, C, H, W) input shape.
        device: compute device.

    Returns:
        Total MAC count.
    """
    monitor = EnergyMonitor(model, model_type="ann")
    monitor.attach_hooks()

    dummy = torch.zeros(*input_shape).to(device)
    with torch.no_grad():
        model.eval()
        model(dummy)

    total = sum(s["mac_ops"] for s in monitor._layer_stats.values())
    monitor.remove_hooks()
    return total
