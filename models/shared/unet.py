"""
UNet-style encoder/decoder for both SNN and ANN restoration networks.

Design principle for SNN:
  - Avoid transposed convolutions (introduce spatial artifacts with binary spikes)
  - Use bilinear upsample + conv instead
  - Skip connections use element-wise addition (supported in spike domain)
  - No MaxPool (use strided conv instead to preserve spike gradient flow)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """Standard double-conv block for ANN U-Net."""

    def __init__(self, in_ch: int, out_ch: int, act: str = "relu"):
        super().__init__()
        act_layer = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer,
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    """
    Encoder half of U-Net.

    Returns features at each scale for skip connections.
    Downsampling via strided convolutions (SNN-compatible).

    Args:
        in_channels (int): Input channels (3 for RGB).
        base_channels (int): First scale channel count.
        n_levels (int): Number of downsampling levels.
        act (str): Activation ('relu' for ANN, 'snn' handled externally).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_levels: int = 4,
        act: str = "relu",
    ):
        super().__init__()
        self.n_levels = n_levels
        ch = base_channels

        self.stem = ConvBlock(in_channels, ch, act)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for _ in range(n_levels - 1):
            self.downsamples.append(
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1, bias=False)
            )
            ch_out = ch * 2
            self.encoders.append(ConvBlock(ch_out, ch_out, act))
            ch = ch_out

        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            skips: list of feature maps [level0, level1, ..., bottleneck]
                   level0 = highest resolution
        """
        x = self.stem(x)
        skips = [x]
        for down, enc in zip(self.downsamples, self.encoders):
            x = down(x)
            x = enc(x)
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    """
    Decoder half of U-Net with skip connections.

    Upsampling via bilinear interpolation + conv (SNN-friendly).

    Args:
        base_channels (int): Must match encoder's base_channels.
        n_levels (int): Must match encoder's n_levels.
        out_channels (int): Final output channels.
        act (str): Activation function.
    """

    def __init__(
        self,
        base_channels: int = 64,
        n_levels: int = 4,
        out_channels: int = 3,
        act: str = "relu",
    ):
        super().__init__()
        self.n_levels = n_levels
        # Channel sizes at each level (from encoder, bottom to top)
        ch_list = [base_channels * (2 ** i) for i in range(n_levels)]  # [64, 128, 256, 512]
        ch_list = ch_list[::-1]  # [512, 256, 128, 64] (decoder order)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(n_levels - 1):
            ch_in = ch_list[i]
            ch_skip = ch_list[i + 1]
            ch_out = ch_list[i + 1]
            self.upsamples.append(
                nn.Conv2d(ch_in, ch_skip, 3, padding=1, bias=False)
            )
            self.decoders.append(ConvBlock(ch_skip * 2, ch_out, act))

        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            skips: list from UNetEncoder, [high_res, ..., bottleneck]

        Returns:
            out: (B, out_channels, H, W)
        """
        # Process from bottleneck upward
        x = skips[-1]
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = up(x)
            skip = skips[-(i + 2)]
            # Handle size mismatch (odd input sizes)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)
