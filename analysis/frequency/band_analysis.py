"""
频段分解 PSNR 分析 (Frequency-Band PSNR Analysis).

核心思路：
  将图像的 FFT 频谱按径向频率划分为 N 个环形频段，
  分别计算 SNN/ANN 在每个频段的重建误差。

  结论预期：
    - SNN 在低频段（DC ~ f_low）与 ANN 差距小（甚至持平）
    - SNN 在高频段（f_mid ~ Nyquist）差距显著扩大
    - 频段 PSNR 差距的"拐点"频率 f* 是 SNN 适用性的关键指标

公式：
  对于频段 B_k = {(u,v) : r_{k-1} ≤ √(u²+v²) < r_k}
  MSE_k(pred, target) = ||F(pred)_{B_k} - F(target)_{B_k}||² / |B_k|
  PSNR_k = 20·log10(1 / √MSE_k)

应用：
  - 绘制 PSNR-frequency 曲线，对比 SNN/ANN/Hybrid
  - 找到 SNN 性能下降的关键频率阈值
  - 量化"低频保真度"vs"高频重建能力"的 SNN/ANN 差异
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class BandPSNRResult:
    """每个频段的质量指标。"""
    band_idx: int
    freq_range: Tuple[float, float]   # 归一化频率 [0, 1]（0=DC, 1=Nyquist）
    mse: float
    psnr: float
    energy_ratio: float               # 该频段能量占总能量的比例
    n_coeffs: int                     # 频段内系数数量


@dataclass
class FrequencyBandProfile:
    """完整的频域分析结果。"""
    model_type: str        # snn / ann / hybrid
    task: str
    n_bands: int
    bands: List[BandPSNRResult] = field(default_factory=list)

    # 派生指标
    low_freq_psnr: float = 0.0       # 低频段（前1/3）平均 PSNR
    mid_freq_psnr: float = 0.0       # 中频段 PSNR
    high_freq_psnr: float = 0.0      # 高频段（后1/3）PSNR
    critical_freq: float = 0.0       # PSNR 下降超过 3dB 的频率阈值
    hf_lf_ratio: float = 0.0         # 高频/低频 PSNR 比（<1 说明高频重建差）


class FrequencyBandAnalyzer:
    """
    频段 PSNR 分析器。

    将 FFT 频谱按径向频率等分为 n_bands 个环，
    计算每环内的重建误差，生成 PSNR-frequency 曲线。

    Args:
        n_bands (int): 频段数量（推荐 16-32）。
        log_scale (bool): 是否按对数间隔划分频段（更关注低频）。
    """

    def __init__(self, n_bands: int = 16, log_scale: bool = False):
        self.n_bands = n_bands
        self.log_scale = log_scale

    def _build_radial_masks(
        self, H: int, W: int
    ) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
        """
        构建径向频率掩码列表。

        Returns:
            List of (mask, (freq_lo, freq_hi)) tuples.
            mask: boolean array (H, W//2+1) for rfft2 output.
        """
        # 归一化频率坐标 [0, 1]
        # rfft2 output shape: (H, W//2+1)
        fy = np.fft.fftfreq(H)           # (H,) — full for rows
        fx = np.fft.rfftfreq(W)          # (W//2+1,) — rfft for cols
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2) * np.sqrt(2)  # 归一化，max≈1

        if self.log_scale:
            edges = np.logspace(-2, 0, self.n_bands + 1)
        else:
            edges = np.linspace(0, 1.0, self.n_bands + 1)

        masks = []
        for i in range(self.n_bands):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                mask = R <= hi
            else:
                mask = (R > lo) & (R <= hi)
            masks.append((mask, (float(lo), float(hi))))
        return masks

    def analyze_pair(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        model_type: str = "snn",
        task: str = "unknown",
    ) -> FrequencyBandProfile:
        """
        对一对 (预测, 目标) 图像进行频段分析。

        Args:
            pred:   (H, W) 或 (C, H, W) 归一化图像 [0, 1]
            target: 同上
            model_type: 模型类型标签
            task:  任务名称标签

        Returns:
            FrequencyBandProfile
        """
        # 灰度化（在 Y 通道上分析，与 PSNR 评估一致）
        if pred.ndim == 3:
            pred_y   = 0.299 * pred[0] + 0.587 * pred[1] + 0.114 * pred[2]
            target_y = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
        else:
            pred_y, target_y = pred, target

        H, W = pred_y.shape

        # FFT（实数 FFT）
        P = np.fft.rfft2(pred_y)
        T = np.fft.rfft2(target_y)

        total_energy = np.abs(T)**2
        total_energy_sum = total_energy.sum() + 1e-12

        masks = self._build_radial_masks(H, W)
        profile = FrequencyBandProfile(
            model_type=model_type, task=task, n_bands=self.n_bands
        )

        psnr_vals = []
        for i, (mask, (lo, hi)) in enumerate(masks):
            n_coeffs = int(mask.sum())
            if n_coeffs == 0:
                continue
            diff = P[mask] - T[mask]
            mse = (np.abs(diff)**2).mean()
            psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
            energy_ratio = float(total_energy[mask].sum() / total_energy_sum)

            band = BandPSNRResult(
                band_idx=i,
                freq_range=(lo, hi),
                mse=float(mse),
                psnr=float(psnr),
                energy_ratio=energy_ratio,
                n_coeffs=n_coeffs,
            )
            profile.bands.append(band)
            psnr_vals.append(psnr)

        # 派生指标（按三等分频段）
        n = len(profile.bands)
        if n >= 3:
            lf = [b.psnr for b in profile.bands[:n//3]]
            mf = [b.psnr for b in profile.bands[n//3:2*n//3]]
            hf = [b.psnr for b in profile.bands[2*n//3:]]
            profile.low_freq_psnr  = float(np.mean(lf))
            profile.mid_freq_psnr  = float(np.mean(mf))
            profile.high_freq_psnr = float(np.mean(hf))
            profile.hf_lf_ratio    = profile.high_freq_psnr / (profile.low_freq_psnr + 1e-8)

        # Critical frequency: first band where PSNR < max_psnr - 3dB
        if psnr_vals:
            max_psnr = max(psnr_vals)
            for band, pv in zip(profile.bands, psnr_vals):
                if pv < max_psnr - 3.0:
                    profile.critical_freq = float(np.mean(band.freq_range))
                    break

        return profile

    def compare_models(
        self,
        pred_dict: Dict[str, np.ndarray],  # {model_type: pred_image}
        target: np.ndarray,
        task: str = "unknown",
    ) -> Dict[str, FrequencyBandProfile]:
        """
        同时分析多个模型的频段 PSNR，便于对比。

        Args:
            pred_dict: {'snn': arr, 'ann': arr, 'hybrid': arr}
            target: ground truth image

        Returns:
            {model_type: FrequencyBandProfile}
        """
        return {
            model_type: self.analyze_pair(pred, target, model_type, task)
            for model_type, pred in pred_dict.items()
        }

    def compute_snn_advantage_band(
        self,
        snn_profile: FrequencyBandProfile,
        ann_profile: FrequencyBandProfile,
    ) -> Dict[str, float]:
        """
        计算 SNN 相对于 ANN 在各频段的 PSNR 差值。

        正值 = SNN 更好；负值 = ANN 更好（几乎总是负值，但大小有差异）

        Returns:
            {
              'band_{i}': delta_psnr,
              'lf_advantage': mean delta in low-freq,
              'hf_advantage': mean delta in high-freq,
              'critical_gap_freq': 频率（SNN 差距超过 ANN 2倍 的第一个频段）
            }
        """
        result = {}
        deltas = []
        for sb, ab in zip(snn_profile.bands, ann_profile.bands):
            d = sb.psnr - ab.psnr
            result[f"band_{sb.band_idx}"] = d
            deltas.append(d)

        n = len(deltas)
        if n >= 3:
            result["lf_advantage"] = float(np.mean(deltas[:n//3]))
            result["mf_advantage"] = float(np.mean(deltas[n//3:2*n//3]))
            result["hf_advantage"] = float(np.mean(deltas[2*n//3:]))

        # 找到 SNN 差距 > 2× 初始差距的频段
        if deltas:
            baseline_gap = abs(deltas[0]) + 1e-8
            for i, d in enumerate(deltas):
                if abs(d) > 2 * baseline_gap and d < 0:
                    result["critical_gap_band"] = i
                    result["critical_gap_freq"] = float(
                        np.mean(snn_profile.bands[i].freq_range)
                    )
                    break

        return result

    def print_band_table(
        self, profiles: Dict[str, FrequencyBandProfile]
    ):
        """打印各模型的频段 PSNR 对比表。"""
        models = list(profiles.keys())
        print(f"\n{'='*70}")
        print(f"Frequency-Band PSNR Analysis (task={list(profiles.values())[0].task})")
        print(f"{'='*70}")
        header = f"{'Band':>5} {'FreqRange':>18} "
        for m in models:
            header += f"{m:>12}"
        print(header)
        print("-"*70)

        n_bands = min(len(p.bands) for p in profiles.values())
        for i in range(n_bands):
            bands = {m: profiles[m].bands[i] for m in models}
            freq_str = f"[{list(bands.values())[0].freq_range[0]:.3f},{list(bands.values())[0].freq_range[1]:.3f}]"
            row = f"{i:>5} {freq_str:>18} "
            for m in models:
                row += f"{bands[m].psnr:>12.2f}"
            print(row)

        print("-"*70)
        for m in models:
            p = profiles[m]
            print(f"{m:>10}: LF={p.low_freq_psnr:.2f} | MF={p.mid_freq_psnr:.2f} | "
                  f"HF={p.high_freq_psnr:.2f} | f*={p.critical_freq:.3f}")
        print(f"{'='*70}")
