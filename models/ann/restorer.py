"""
ANN Baseline Restorer for Low-Level Vision Tasks.

Architecture: U-Net with residual blocks + CBAM attention.
Matches parameter count to the SNN counterpart for fair comparison.

Design follows NAFNet (Chen et al., 2022) and MPRNet (Zamir et al., 2021)
patterns: encoder-decoder with skip connections and a residual output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..shared.attention import CBAM
from ..shared.norm import LayerNorm2d
from typing import Optional


class NAFBlock(nn.Module):
    """
    Simplified NAFNet block (Chen et al., 2022).
    Nonlinear Activation Free: uses SimpleGate instead of GELU.
    Excellent baseline for all restoration tasks.
    """

    def __init__(self, channels: int, ffn_expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        # Depthwise separable conv for efficiency
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.pw_conv1 = nn.Conv2d(channels, channels * ffn_expand * 2, 1, bias=True)
        self.pw_conv2 = nn.Conv2d(channels * ffn_expand, channels, 1, bias=True)

        # SimpleGate: split channels in half, multiply
        # FFN
        self.ffn1 = nn.Conv2d(channels, channels * ffn_expand * 2, 1, bias=True)
        self.ffn2 = nn.Conv2d(channels * ffn_expand, channels, 1, bias=True)

        # Channel attention (simplified)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def _simple_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Split in half along channel dim and multiply."""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention branch
        h = self.norm1(x)
        h = self.dw_conv(h)
        h = self._simple_gate(self.pw_conv1(h))
        h = h * self.ca(h)
        h = self.pw_conv2(h)
        x = x + h * self.beta

        # FFN branch
        h = self.norm2(x)
        h = self._simple_gate(self.ffn1(h))
        h = self.ffn2(h)
        x = x + h * self.gamma

        return self.drop(x)


class ANNRestorer(nn.Module):
    """
    ANN restoration network for any low-level vision task.

    Architecture:
        Input → Stem → [Encoder × n_levels] → Bottleneck → [Decoder × n_levels] → Head → Output

    Output = Input + residual (residual learning for all restoration tasks).

    Args:
        in_channels (int): Input channels (3 for RGB, 1 for grayscale).
        out_channels (int): Output channels.
        base_channels (int): Base feature width.
        n_levels (int): Encoder/decoder depth.
        n_blocks (int): NAFBlocks per level.
        ffn_expand (int): FFN expansion ratio.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        n_levels: int = 4,
        n_blocks: int = 2,
        ffn_expand: int = 2,
    ):
        super().__init__()
        self.n_levels = n_levels

        ch = base_channels
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1, bias=True)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        enc_channels = [ch]
        for _ in range(n_levels - 1):
            self.enc_blocks.append(
                nn.Sequential(*[NAFBlock(ch, ffn_expand) for _ in range(n_blocks)])
            )
            self.downsamples.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2
            enc_channels.append(ch)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[NAFBlock(ch, ffn_expand) for _ in range(n_blocks * 2)]
        )

        # Decoder
        self.upsamples = nn.ModuleList()
        self.dec_projs = nn.ModuleList()  # project after concat
        self.dec_blocks = nn.ModuleList()
        for i in range(n_levels - 1):
            skip_ch = enc_channels[-(i + 2)]
            self.upsamples.append(nn.ConvTranspose2d(ch, skip_ch, 2, stride=2))
            self.dec_projs.append(nn.Conv2d(skip_ch * 2, skip_ch, 1))
            self.dec_blocks.append(
                nn.Sequential(*[NAFBlock(skip_ch, ffn_expand) for _ in range(n_blocks)])
            )
            ch = skip_ch

        self.head = nn.Conv2d(ch, out_channels, 3, padding=1, bias=True)

        # Track parameter count
        self._param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) degraded image

        Returns:
            out: (B, C_out, H, W) restored image
        """
        inp = x
        x = self.stem(x)

        # Encode
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        for up, proj, dec, skip in zip(
            self.upsamples, self.dec_projs, self.dec_blocks, reversed(skips)
        ):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = proj(torch.cat([x, skip], dim=1))
            x = dec(x)

        out = self.head(x)

        # Residual output (works for deraining, denoising, dehazing, lowlight)
        # For super-resolution, input is upsampled first externally
        if out.shape == inp.shape:
            out = out + inp

        return out

    def get_param_count(self) -> int:
        return self._param_count
