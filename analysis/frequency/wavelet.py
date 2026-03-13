"""
小波多尺度分解分析 (Wavelet Multi-Scale Decomposition).

为什么用小波而不仅用 FFT？
  1. FFT 是全局频率分析，无法定位"哪里"有高频误差
  2. 小波提供时频局部化：同时知道"哪个位置"在"哪个尺度"出错
  3. 图像退化（雨线、噪声、雾）在小波子带中有特定的稀疏结构

分析内容：
  - 在各 Haar/DB 小波子带（LL/LH/HL/HH × 多级）上比较 SNN vs ANN 误差
  - 子带 SNR：SNN 在哪个子带损失最多
  - 稀疏性：SNN/ANN 特征在小波域的稀疏性差异
  - 退化定位：退化区域在小波子带中的空间分布

实现：纯 NumPy 实现 2D Haar 小波（无需 PyWavelets 依赖）
      支持 N 级分解，输出各子带的均方误差和稀疏度

核心发现预期：
  - 去雨：SNN 的 HH 子带（高高频）误差更小 → SNN 更好地定位了雨线
  - 去噪：SNN 的所有 H 子带误差均比 ANN 大 → SNN 高频重建弱
  - 低光：SNN 的 LL 子带（低低频）误差都大 → 连平滑区域都重建不好
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class WaveletSubbandResult:
    """单个子带的分析结果。"""
    name: str          # 如 "2_LH" = 第2级 LH 子带
    level: int
    subband_type: str  # LL / LH / HL / HH
    shape: Tuple[int, int]

    # 误差指标
    mse_snn: float = 0.0
    mse_ann: float = 0.0
    snr_snn: float = 0.0
    snr_ann: float = 0.0
    snn_ann_ratio: float = 1.0   # MSE(SNN)/MSE(ANN)，>1 表示 SNN 更差

    # 稀疏性
    sparsity_snn: float = 0.0    # 系数稀疏度（能量集中度）
    sparsity_ann: float = 0.0

    # 能量比例
    energy_fraction: float = 0.0  # 该子带能量占总能量的比例


@dataclass
class WaveletAnalysisResult:
    """完整的小波分析结果。"""
    n_levels: int
    subbands: List[WaveletSubbandResult] = field(default_factory=list)

    # 按子带类型聚合
    ll_ratio: float = 1.0   # MSE(SNN)/MSE(ANN) 在所有 LL 子带
    lh_ratio: float = 1.0   # 水平高频
    hl_ratio: float = 1.0   # 垂直高频
    hh_ratio: float = 1.0   # 对角高频（最难重建）

    # 关键发现
    worst_subband: str = ""  # SNN 相对 ANN 最差的子带
    best_subband: str = ""   # SNN 相对 ANN 最好的子带


class WaveletDecomposer:
    """
    纯 NumPy 的 2D Haar 小波分析器。

    对 (SNN预测, ANN预测, 目标) 三元组进行多级小波分解，
    在各子带上比较 SNN 和 ANN 的重建质量。

    Args:
        n_levels (int): 小波分解级数（推荐 3-5）。
    """

    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels

    def _haar_2d_one_level(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        一级 2D Haar 小波分解。

        Returns:
            (LL, LH, HL, HH) 四个子带，各为 (H/2, W/2)
        """
        H, W = img.shape
        # 行方向低通/高通
        L = (img[:, 0::2] + img[:, 1::2]) / np.sqrt(2)
        H_ = (img[:, 0::2] - img[:, 1::2]) / np.sqrt(2)
        # 列方向
        LL = (L[0::2, :] + L[1::2, :]) / np.sqrt(2)
        LH = (L[0::2, :] - L[1::2, :]) / np.sqrt(2)
        HL = (H_[0::2, :] + H_[1::2, :]) / np.sqrt(2)
        HH = (H_[0::2, :] - H_[1::2, :]) / np.sqrt(2)
        return LL, LH, HL, HH

    def _decompose(
        self, img: np.ndarray, n_levels: int
    ) -> Dict[str, np.ndarray]:
        """
        多级分解，返回 {subband_name: array}。
        subband_name 格式："{level}_{type}" 如 "1_HH", "2_LL" ...
        """
        subbands = {}
        current = img.copy()
        for level in range(1, n_levels + 1):
            if current.shape[0] < 2 or current.shape[1] < 2:
                break
            LL, LH, HL, HH = self._haar_2d_one_level(current)
            subbands[f"{level}_LH"] = LH
            subbands[f"{level}_HL"] = HL
            subbands[f"{level}_HH"] = HH
            current = LL  # 继续分解低频子带
        subbands[f"{n_levels}_LL"] = current
        return subbands

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """(C, H, W) 或 (H, W) → (H, W) 灰度图。"""
        if img.ndim == 3:
            return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return img

    def _sparsity(self, coeffs: np.ndarray) -> float:
        """
        小波系数稀疏性：Gini 系数。
        0 = 完全均匀；1 = 极稀疏（能量集中在少数系数）。
        """
        x = np.abs(coeffs.flatten())
        x = np.sort(x)
        n = len(x)
        if n == 0 or x.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * idx - n - 1).dot(x) / (n * x.sum()))

    def analyze(
        self,
        pred_snn: np.ndarray,
        pred_ann: np.ndarray,
        target: np.ndarray,
    ) -> WaveletAnalysisResult:
        """
        在所有小波子带上比较 SNN 和 ANN 的重建质量。

        Args:
            pred_snn, pred_ann, target: (C,H,W) 或 (H,W) 图像 [0,1]

        Returns:
            WaveletAnalysisResult
        """
        snn_g   = self._to_gray(pred_snn)
        ann_g   = self._to_gray(pred_ann)
        tgt_g   = self._to_gray(target)

        # 确保尺寸是 2^n 的倍数（裁剪到最近的 2^n_levels 倍数）
        H, W = snn_g.shape
        ph = (H // 2**self.n_levels) * 2**self.n_levels
        pw = (W // 2**self.n_levels) * 2**self.n_levels
        snn_g, ann_g, tgt_g = (
            snn_g[:ph, :pw], ann_g[:ph, :pw], tgt_g[:ph, :pw]
        )

        snn_bands = self._decompose(snn_g, self.n_levels)
        ann_bands = self._decompose(ann_g, self.n_levels)
        tgt_bands = self._decompose(tgt_g, self.n_levels)

        result = WaveletAnalysisResult(n_levels=self.n_levels)
        total_energy = (tgt_g**2).sum() + 1e-12

        ratios_by_type: Dict[str, List[float]] = {"LL":[], "LH":[], "HL":[], "HH":[]}

        for name, tgt_sub in tgt_bands.items():
            level = int(name.split("_")[0])
            stype = name.split("_")[1]
            snn_sub = snn_bands.get(name, tgt_sub)
            ann_sub = ann_bands.get(name, tgt_sub)

            sub_energy = float((tgt_sub**2).sum())
            energy_frac = sub_energy / total_energy

            mse_snn = float(((snn_sub - tgt_sub)**2).mean())
            mse_ann = float(((ann_sub - tgt_sub)**2).mean())

            # SNR（信号=目标能量，噪声=误差能量）
            sig_power = float((tgt_sub**2).mean()) + 1e-12
            snr_snn = 10 * np.log10(sig_power / (mse_snn + 1e-12))
            snr_ann = 10 * np.log10(sig_power / (mse_ann + 1e-12))

            ratio = mse_snn / (mse_ann + 1e-12)

            sub_result = WaveletSubbandResult(
                name=name, level=level, subband_type=stype,
                shape=tuple(tgt_sub.shape),
                mse_snn=mse_snn, mse_ann=mse_ann,
                snr_snn=snr_snn, snr_ann=snr_ann,
                snn_ann_ratio=ratio,
                sparsity_snn=self._sparsity(snn_sub),
                sparsity_ann=self._sparsity(ann_sub),
                energy_fraction=energy_frac,
            )
            result.subbands.append(sub_result)
            ratios_by_type[stype].append(ratio)

        # 聚合
        result.ll_ratio = float(np.mean(ratios_by_type["LL"])) if ratios_by_type["LL"] else 1.0
        result.lh_ratio = float(np.mean(ratios_by_type["LH"])) if ratios_by_type["LH"] else 1.0
        result.hl_ratio = float(np.mean(ratios_by_type["HL"])) if ratios_by_type["HL"] else 1.0
        result.hh_ratio = float(np.mean(ratios_by_type["HH"])) if ratios_by_type["HH"] else 1.0

        # 找最差/最好子带
        if result.subbands:
            worst = max(result.subbands, key=lambda x: x.snn_ann_ratio)
            best  = min(result.subbands, key=lambda x: x.snn_ann_ratio)
            result.worst_subband = worst.name
            result.best_subband  = best.name

        return result

    def print_subband_table(self, result: WaveletAnalysisResult):
        print(f"\n{'='*70}")
        print(f"Wavelet Subband Analysis ({result.n_levels}-level Haar)")
        print(f"{'Subband':<12} {'Type':>4} {'MSE_SNN':>12} {'MSE_ANN':>12} "
              f"{'SNN/ANN':>9} {'EnergyFrac':>11}")
        print("-"*70)
        for sb in sorted(result.subbands, key=lambda x: (x.level, x.subband_type)):
            print(f"{sb.name:<12} {sb.subband_type:>4} {sb.mse_snn:>12.6f} "
                  f"{sb.mse_ann:>12.6f} {sb.snn_ann_ratio:>9.3f} "
                  f"{sb.energy_fraction:>11.4f}")
        print("-"*70)
        print(f"  LL ratio (low-freq): {result.ll_ratio:.3f}  "
              f"HH ratio (high-freq diag): {result.hh_ratio:.3f}")
        print(f"  Worst subband: {result.worst_subband}  |  "
              f"Best subband: {result.best_subband}")
        print(f"{'='*70}")
        print(f"  Interpretation: SNN/ANN ratio < 1.0 = SNN better at this subband")
        print(f"                  ratio >> 1.0 = SNN fails at this subband")
