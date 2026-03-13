"""
Hybrid SNN-ANN Restorer (Ablation Study Model).

Architecture: SNN encoder for feature extraction → ANN decoder for reconstruction.

Hypothesis: SNNs may be better at sparse event detection (noise, rain streaks)
while ANNs are better at precise pixel reconstruction. This hybrid tests that.

This is a key ablation for the applicability study:
  - Full ANN: best quality, high energy
  - Full SNN: lower quality, lowest energy (on neuromorphic)
  - Hybrid SNN-enc + ANN-dec: ??? (hypothesis: quality closer to ANN, energy < ANN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from .snn.neurons import LIFNeuron
from .snn.encoding import DirectEncoder
from .snn_restorer import SNNConvBlock
from .ann.restorer import NAFBlock
from .shared.norm import LayerNorm2d


class HybridRestorer(nn.Module):
    """
    Hybrid model: SNN encoder extracts sparse features,
    ANN decoder reconstructs the clean image.

    The interface between SNN and ANN: accumulated spike outputs
    (T-averaged) are passed as analog inputs to the ANN decoder.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        base_channels (int): Feature width.
        n_levels (int): U-Net depth.
        T (int): SNN timesteps.
        tau (float): LIF time constant.
        threshold (float): LIF firing threshold.
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
    ):
        super().__init__()
        self.T = T
        self.n_levels = n_levels

        # ---- SNN Encoder ----
        self.encoder_input = DirectEncoder(T=T, mode="repeat")
        self.stem_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.stem_neuron = LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate)

        ch = base_channels
        enc_channels = [ch]
        self.snn_enc_blocks = nn.ModuleList()
        self.snn_downsamples = nn.ModuleList()

        for _ in range(n_levels - 1):
            self.snn_enc_blocks.append(
                SNNConvBlock(ch, ch * 2, tau=tau, threshold=threshold, surrogate=surrogate)
            )
            self.snn_downsamples.append(nn.Sequential(
                nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch * 2),
                LIFNeuron(tau=tau, threshold=threshold, surrogate=surrogate),
            ))
            ch *= 2
            enc_channels.append(ch)

        self.snn_bottleneck = SNNConvBlock(ch, ch, tau=tau, threshold=threshold,
                                           surrogate=surrogate)

        # ---- ANN Decoder ----
        self.ann_upsamples = nn.ModuleList()
        self.ann_dec_projs = nn.ModuleList()
        self.ann_dec_blocks = nn.ModuleList()

        for level in range(n_levels - 1):
            skip_ch = enc_channels[-(level + 2)]
            self.ann_upsamples.append(nn.Sequential(
                nn.Conv2d(ch, skip_ch, 3, padding=1, bias=False),
                LayerNorm2d(skip_ch),
                nn.GELU(),
            ))
            self.ann_dec_projs.append(nn.Conv2d(skip_ch * 2, skip_ch, 1, bias=False))
            self.ann_dec_blocks.append(
                nn.Sequential(NAFBlock(skip_ch), NAFBlock(skip_ch))
            )
            ch = skip_ch

        self.head = nn.Conv2d(ch, out_channels, 3, padding=1, bias=True)

    def _get_all_lif(self) -> List[LIFNeuron]:
        return [m for m in self.modules() if isinstance(m, LIFNeuron)]

    def reset_states(self):
        for n in self._get_all_lif():
            n.reset_state()
        for b in self.snn_enc_blocks:
            b.reset_states()
        self.snn_bottleneck.reset_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)

        Returns:
            out: (B, C_out, H, W)
        """
        self.reset_states()
        inp = x
        x_seq = self.encoder_input(x)  # (T, B, C, H, W)

        # Accumulate SNN encoder outputs across T
        enc_accumulators = [None] * len(self.snn_enc_blocks)
        bottleneck_accum = None

        for t in range(self.T):
            xt = x_seq[t]

            # Stem
            h = self.stem_neuron(self.stem_bn(self.stem_conv(xt)))

            enc_spikes_t = []
            for level, (enc_block, down) in enumerate(
                zip(self.snn_enc_blocks, self.snn_downsamples)
            ):
                h = enc_block.forward_step(h)
                enc_spikes_t.append(h)
                for name, module in down.named_children():
                    h = module(h)

            h = self.snn_bottleneck.forward_step(h)

            # Accumulate
            if bottleneck_accum is None:
                bottleneck_accum = h.clone()
            else:
                bottleneck_accum = bottleneck_accum + h

            for level, spk in enumerate(enc_spikes_t):
                if enc_accumulators[level] is None:
                    enc_accumulators[level] = spk.clone()
                else:
                    enc_accumulators[level] = enc_accumulators[level] + spk

        # Average (spike count → analog value)
        bottleneck_feat = bottleneck_accum / self.T
        enc_feats = [acc / self.T for acc in enc_accumulators]

        # ANN Decoder
        h = bottleneck_feat
        for level, (up, proj, dec_block) in enumerate(
            zip(self.ann_upsamples, self.ann_dec_projs, self.ann_dec_blocks)
        ):
            h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False)
            h = up(h)
            skip = enc_feats[-(level + 1)]
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = proj(torch.cat([h, skip], dim=1))
            h = dec_block(h)

        out = self.head(h)
        if out.shape == inp.shape:
            out = out + inp
        return out

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
