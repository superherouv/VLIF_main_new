"""
Spike Pattern Analysis for Low-Level Vision Tasks.

Analyzes:
  1. Spatial distribution of spikes (where does the SNN fire?)
  2. Temporal distribution across timesteps
  3. Per-layer firing rates (encoder vs decoder vs bottleneck)
  4. Correlation between spike patterns and degradation regions

This analysis reveals WHY SNNs succeed/fail on different tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SpikeMap:
    """Spatial spike density map for a layer."""
    layer_name: str
    shape: Tuple[int, ...]
    mean_rate: float
    spatial_map: Optional[np.ndarray] = None  # (H, W) averaged firing rate
    temporal_profile: Optional[np.ndarray] = None  # (T,) rate per timestep


class SpikeAnalyzer:
    """
    Records and analyzes spike patterns during forward passes.

    Usage:
        analyzer = SpikeAnalyzer(model)
        analyzer.attach_hooks(model)
        with torch.no_grad():
            output = model(input_spike_sequence)  # (T, B, C, H, W)
        maps = analyzer.get_spike_maps()
        analyzer.print_spatial_analysis()
        analyzer.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks = []
        self._spike_records: Dict[str, List[torch.Tensor]] = {}

    def attach_hooks(self, model: Optional[nn.Module] = None):
        """
        Attach forward hooks to all LIF neurons.
        Captures output spikes per timestep.
        """
        from models.snn.neurons import LIFNeuron, ParametricLIFNeuron
        model = model or self.model
        self._spike_records.clear()

        def make_hook(name: str):
            def hook(module, inp, out):
                if name not in self._spike_records:
                    self._spike_records[name] = []
                # out is (B, C, H, W) at a single timestep
                self._spike_records[name].append(out.detach().cpu())
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (LIFNeuron, ParametricLIFNeuron)):
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_spike_maps(self) -> Dict[str, SpikeMap]:
        """
        Compute spatial and temporal spike maps from recorded data.

        Returns:
            Dict mapping layer_name → SpikeMap
        """
        maps = {}
        for name, spike_list in self._spike_records.items():
            if len(spike_list) == 0:
                continue

            # Stack timesteps: (T, B, C, H, W) or (T, B, C)
            stacked = torch.stack(spike_list, dim=0)  # (T, B, ...)
            T = stacked.shape[0]

            mean_rate = stacked.mean().item()

            # Temporal profile: mean rate per timestep
            temporal = stacked.view(T, -1).mean(dim=1).numpy()

            # Spatial map: mean over T and B, then over C → (H, W) or scalar
            if stacked.dim() >= 5:
                # (T, B, C, H, W)
                spatial = stacked.mean(dim=[0, 1, 2]).numpy()  # (H, W)
            elif stacked.dim() == 4:
                # (T, B, C) = FC layer
                spatial = None
            else:
                spatial = None

            maps[name] = SpikeMap(
                layer_name=name,
                shape=tuple(stacked.shape),
                mean_rate=mean_rate,
                spatial_map=spatial,
                temporal_profile=temporal,
            )
        return maps

    def analyze_task_correlation(
        self,
        degraded: torch.Tensor,
        clean: torch.Tensor,
        maps: Dict[str, SpikeMap],
    ) -> Dict[str, float]:
        """
        Compute correlation between spike density and degradation regions.

        High correlation means SNN has learned to focus spikes on degraded areas,
        which is the desired behavior for efficient processing.

        Args:
            degraded: (B, C, H, W) degraded input
            clean: (B, C, H, W) clean reference
            maps: SpikeMap dict from get_spike_maps()

        Returns:
            Dict of layer_name → correlation coefficient
        """
        # Degradation mask: where degraded differs from clean
        diff = (degraded - clean).abs().mean(dim=1).cpu().numpy()  # (B, H, W)
        # Binarize: top 20% of diff values are degradation regions
        threshold = np.percentile(diff, 80)
        degradation_mask = (diff > threshold).astype(float)

        correlations = {}
        for name, smap in maps.items():
            if smap.spatial_map is None:
                continue
            # Resize spatial map to match input if needed
            spike_map = smap.spatial_map  # (H, W) or (H', W')
            if spike_map.shape != degradation_mask.shape[1:]:
                from PIL import Image
                spike_pil = Image.fromarray(
                    (spike_map * 255).astype(np.uint8)
                ).resize(
                    (degradation_mask.shape[2], degradation_mask.shape[1]),
                    Image.BILINEAR
                )
                spike_map = np.array(spike_pil) / 255.0

            # Pearson correlation
            for b in range(degradation_mask.shape[0]):
                sm = spike_map.flatten()
                dm = degradation_mask[b].flatten()
                if sm.std() > 0 and dm.std() > 0:
                    corr = float(np.corrcoef(sm, dm)[0, 1])
                    correlations[name] = correlations.get(name, [])
                    correlations[name].append(corr)

        return {k: float(np.mean(v)) for k, v in correlations.items()}

    def print_summary(self, maps: Dict[str, SpikeMap]):
        """Print spike pattern summary."""
        print("\n" + "="*60)
        print("SPIKE PATTERN ANALYSIS")
        print("="*60)
        print(f"{'Layer':<45} {'Rate':>8} {'Shape'}")
        print("-"*60)
        for name, smap in sorted(maps.items()):
            shape_str = str(smap.shape)
            print(f"{name:<45} {smap.mean_rate:>8.4f} {shape_str}")

        all_rates = [m.mean_rate for m in maps.values()]
        if all_rates:
            print("-"*60)
            print(f"{'MEAN':.<45} {np.mean(all_rates):>8.4f}")
            print(f"{'STD':.<45} {np.std(all_rates):>8.4f}")
            print(f"{'MIN':.<45} {np.min(all_rates):>8.4f}")
            print(f"{'MAX':.<45} {np.max(all_rates):>8.4f}")
        print("="*60)
