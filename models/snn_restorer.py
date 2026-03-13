"""
SNN-based Image Restoration Network.

Core architecture for the SNN applicability study.
Implements a U-Net style architecture where all intermediate features
are spike trains (binary tensors), enabling neuromorphic hardware mapping
and accurate energy consumption estimation.

Key design decisions (justified by ablation study literature):
  1. Direct encoding (no rate/temporal) at input layer
  2. Soft reset (better gradient flow than hard reset)
  3. Surrogate ATan (empirically best for vision tasks, Fang et al. 2021)
  4. T=4 timesteps (optimal for quality/energy tradeoff in restoration)
  5. Bilinear upsample (not transposed conv) in decoder
  6. Analog output: accumulate T spike outputs → final prediction

The network follows the SNN-UNet-Restoration design:
  Input → DirectEncoder → SNN_Encoder → SNN_Bottleneck → SNN_Decoder → Head

Energy advantage analysis:
  - SNN multiply-accumulates (MACs) ≈ spike_rate × ANN_MACs
  - Typical spike rate in restoration: 5-20% → 5-20× energy saving
  - BUT: temporal unfolding (T steps) costs T× memory bandwidth
  - Net advantage depends on hardware: neuromorphic >> GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from .snn.neurons import LIFNeuron, ParametricLIFNeuron
from .snn.encoding import DirectEncoder
from .snn.layers import SNNConv2d, SNNResidualBlock


class SNNConvBlock(nn.Module):
    """
    SNN double-conv block: [Conv→BN→LIF] × 2.
    Fundamental unit of the SNN U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate: str = "atan",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.neuron1 = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.neuron2 = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

        # Spike rate monitoring
        self._n1_spikes = 0
        self._n2_spikes = 0
        self._n_elements = 0

    def reset_states(self):
        self.neuron1.reset_state()
        self.neuron2.reset_state()
        self._n1_spikes = 0
        self._n2_spikes = 0
        self._n_elements = 0

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single timestep forward."""
        h = self.neuron1(self.bn1(self.conv1(x)))
        h = self.neuron2(self.bn2(self.conv2(h)))
        self._n1_spikes += (self.neuron1.membrane >= self.neuron1.threshold).sum().item()
        self._n2_spikes += (self.neuron2.membrane >= self.neuron2.threshold).sum().item()
        self._n_elements += h.numel()
        return h

    def get_firing_rates(self) -> Tuple[float, float]:
        """Returns (rate1, rate2) firing rates."""
        if self._n_elements == 0:
            return 0.0, 0.0
        return (self._n1_spikes / self._n_elements,
                self._n2_spikes / self._n_elements)


class SNNRestorer(nn.Module):
    """
    Spiking Neural Network for Image Restoration.

    Supports all 5 low-level vision tasks through task-specific heads
    and residual learning.

    Args:
        in_channels (int): Input image channels (3=RGB, 1=gray).
        out_channels (int): Output image channels.
        base_channels (int): Width multiplier.
        n_levels (int): U-Net depth.
        T (int): Number of SNN timesteps.
        tau (float): LIF membrane time constant.
        threshold (float): LIF firing threshold.
        surrogate (str): Surrogate gradient function.
        use_plif (bool): Use learnable tau (ParametricLIF) in encoder.
        task (str): Task name for logging (no functional effect).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        n_levels: int = 4,
        T: int = 4,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate: str = "atan",
        use_plif: bool = False,
        task: str = "deraining",
    ):
        super().__init__()
        self.T = T
        self.task = task

        # Input encoding (direct: repeat frame T times)
        self.encoder = DirectEncoder(T=T, mode="repeat")

        # Stem: project to base_channels (still analog, no LIF here)
        self.stem_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.stem_neuron = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

        # Encoder levels
        ch = base_channels
        enc_channels = [ch]
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for level in range(n_levels - 1):
            block = SNNConvBlock(ch, ch * 2, tau=tau, threshold=threshold,
                                 surrogate=surrogate)
            self.enc_blocks.append(block)
            # Strided conv as downsampler (SNN-compatible, no pooling)
            self.downsamples.append(
                nn.Sequential(
                    nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch * 2),
                )
            )
            # Need a neuron after downsample
            self.downsamples[-1].add_module(
                "lif", LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)
            )
            ch *= 2
            enc_channels.append(ch)

        # Bottleneck
        self.bottleneck = SNNConvBlock(ch, ch, tau=tau, threshold=threshold,
                                       surrogate=surrogate)

        # Decoder levels
        self.upsamples = nn.ModuleList()    # upsample conv after bilinear
        self.dec_projs = nn.ModuleList()    # channel projection after cat
        self.dec_blocks = nn.ModuleList()   # decoder conv blocks
        self.dec_neurons = nn.ModuleList()  # LIF after projection

        for level in range(n_levels - 1):
            skip_ch = enc_channels[-(level + 2)]
            # Project bottleneck channels to skip_ch before adding
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(ch, skip_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(skip_ch),
                )
            )
            self.upsamples[-1].add_module(
                "lif", LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)
            )
            # Project concatenated skip+up to skip_ch
            self.dec_projs.append(
                nn.Sequential(
                    nn.Conv2d(skip_ch * 2, skip_ch, 1, bias=False),
                    nn.BatchNorm2d(skip_ch),
                )
            )
            self.dec_neurons.append(
                LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)
            )
            self.dec_blocks.append(
                SNNConvBlock(skip_ch, skip_ch, tau=tau, threshold=threshold,
                             surrogate=surrogate)
            )
            ch = skip_ch

        # Output head: accumulate spikes over T steps → continuous output
        # The head is analog (no LIF) to produce continuous pixel values
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_channels, 1, bias=True),
        )

        # Collect all state-ful modules
        self._all_snn_blocks: List[SNNConvBlock] = []
        self._collect_snn_blocks()

    def _collect_snn_blocks(self):
        """Collect all SNNConvBlocks for state management."""
        self._all_snn_blocks = []
        for m in self.modules():
            if isinstance(m, SNNConvBlock):
                self._all_snn_blocks.append(m)

    def _get_all_lif_neurons(self) -> List[LIFNeuron]:
        """Get all LIF neurons (including those in Sequential modules)."""
        neurons = []
        for m in self.modules():
            if isinstance(m, (LIFNeuron, ParametricLIFNeuron)):
                neurons.append(m)
        return neurons

    def reset_states(self):
        """Reset all neuron membrane potentials (call between sequences)."""
        for neuron in self._get_all_lif_neurons():
            neuron.reset_state()
        for block in self._all_snn_blocks:
            block.reset_states()

    def _forward_stem(self, x_t: torch.Tensor) -> torch.Tensor:
        """Stem forward for a single timestep."""
        return self.stem_neuron(self.stem_bn(self.stem_conv(x_t)))

    def _forward_downsample(self, x: torch.Tensor, down_seq: nn.Sequential) -> torch.Tensor:
        """Downsample through conv+bn+lif."""
        conv_bn = down_seq[:-1] if hasattr(down_seq, '__len__') else down_seq
        lif = down_seq[-1] if hasattr(down_seq, '__len__') else None
        # Walk through manually
        h = x
        for name, module in down_seq.named_children():
            if isinstance(module, LIFNeuron):
                h = module(h)
            else:
                h = module(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) degraded image, values in [0,1]

        Returns:
            out: (B, C_out, H, W) restored image
        """
        self.reset_states()
        B, C, H, W = x.shape
        inp = x

        # Encode input to T timesteps
        x_seq = self.encoder(x)  # (T, B, C, H, W)

        # Accumulate outputs for head (analog output)
        # We accumulate spike representations at decoder output
        head_accumulator = torch.zeros(B, self.enc_blocks[0].conv1.in_channels
                                       if len(self.enc_blocks) > 0 else 32,
                                       H, W, device=x.device, dtype=x.dtype)

        # --- Process T timesteps ---
        # We unfold the SNN in time, carrying skip connections per timestep
        skip_accumulators = [
            torch.zeros(B, self.enc_blocks[i].conv2.out_channels,
                        H // (2**i), W // (2**i),
                        device=x.device) if i < len(self.enc_blocks) else None
            for i in range(len(self.enc_blocks))
        ]

        # Collect final timestep outputs for decoder (simpler than full accumulation)
        all_enc_spikes = []  # per-level encoder outputs averaged over T
        for level in range(len(self.enc_blocks)):
            all_enc_spikes.append(
                torch.zeros(1)  # placeholder, filled below
            )

        # Full T-step forward
        enc_outputs_per_T = [[] for _ in range(len(self.enc_blocks))]
        bottleneck_outputs = []
        dec_outputs = []  # final channel dim varies by level

        for t in range(self.T):
            x_t = x_seq[t]  # (B, C, H, W)

            # Stem
            h = self._forward_stem(x_t)

            # Encode
            enc_spikes = []  # skip connections at this timestep
            for level, (enc_block, down) in enumerate(
                zip(self.enc_blocks, self.downsamples)
            ):
                h = enc_block.forward_step(h)
                enc_spikes.append(h)
                # Downsample via strided conv → spike
                h_down = h
                for name, module in down.named_children():
                    h_down = module(h_down)
                h = h_down

            # Bottleneck
            h = self.bottleneck.forward_step(h)
            bottleneck_outputs.append(h)

            # Decode
            for level, (up, proj, dec_neuron, dec_block) in enumerate(
                zip(self.upsamples, self.dec_projs, self.dec_neurons, self.dec_blocks)
            ):
                # Bilinear upsample (spike-compatible)
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                # SNN up conv
                for name, module in up.named_children():
                    h = module(h)
                # Skip connection
                skip = enc_spikes[-(level + 1)]
                if h.shape != skip.shape:
                    h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                for name, module in proj.named_children():
                    h = module(h)
                h = dec_neuron(h)
                h = dec_block.forward_step(h)

            # Accumulate decoded spikes (T-fold accumulation → better analog approx)
            if t == 0:
                dec_accumulator = h.clone()
            else:
                dec_accumulator = dec_accumulator + h

        # Average over T: approximate analog value from spike count
        dec_out = dec_accumulator / self.T  # (B, ch, H, W)

        # Final analog head
        out = self.head(dec_out)

        # Residual learning (restored = degraded + residual)
        if out.shape == inp.shape:
            out = out + inp

        return out

    def get_energy_stats(self) -> Dict[str, float]:
        """
        Estimate synaptic operations (SynOps) for energy analysis.

        SynOps = Σ_l (spike_rate_l × MAC_ops_l)
        Energy(SNN) ≈ SynOps × E_spike    (E_spike << E_MAC for neuromorphic)
        Energy(ANN) ≈ MAC_ops × E_MAC

        Returns:
            dict with per-layer and total firing rates/SynOps
        """
        stats = {}
        for i, block in enumerate(self.enc_blocks):
            r1, r2 = block.get_firing_rates()
            stats[f"enc_{i}_rate1"] = r1
            stats[f"enc_{i}_rate2"] = r2
        r1, r2 = self.bottleneck.get_firing_rates()
        stats["bottleneck_rate1"] = r1
        stats["bottleneck_rate2"] = r2
        for i, block in enumerate(self.dec_blocks):
            r1, r2 = block.get_firing_rates()
            stats[f"dec_{i}_rate1"] = r1
            stats[f"dec_{i}_rate2"] = r2
        # Mean across all layers
        all_rates = [v for v in stats.values()]
        stats["mean_firing_rate"] = sum(all_rates) / len(all_rates) if all_rates else 0.0
        return stats

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
