"""
重建误差频谱分析 (Reconstruction Error Spectrum Analysis).

SNN vs ANN 的误差不仅在幅值上不同，在频谱结构上也不同：
  - ANN 误差：通常是低幅值宽带误差（均匀分布于所有频率）
  - SNN 误差：通常在高频段集中，低频段与 ANN 相近

这种差异揭示了 SNN 的根本局限：
  二值脉冲的频率分辨率受限于 T（时间步数），
  T 步脉冲序列的有效奈奎斯特频率 = 1/(2T)，
  因此 SNN 本质上是一个时间低通滤波器。

分析内容：
  1. 误差功率谱密度 (Error PSD) 对比
  2. 误差频谱形状相似性（SNN 误差是否比 ANN 更"有结构"？）
  3. 频率-误差相关性（哪些频率的误差与退化区域相关）
  4. 误差的各向异性（水平 vs 垂直误差，与雨线方向关系）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ErrorSpectrumStats:
    """误差频谱统计。"""
    model_type: str
    task: str

    # 频谱形状
    psd_curve: np.ndarray = field(default_factory=lambda: np.array([]))  # (N_bins,)
    freq_axis: np.ndarray = field(default_factory=lambda: np.array([]))  # (N_bins,)

    # 关键指标
    spectral_centroid: float = 0.0    # 误差频谱质心（高 = 误差集中于高频）
    spectral_flatness: float = 0.0    # 频谱平坦度（1=白噪声，0=冲激）
    hf_error_ratio: float = 0.0       # 高频误差占总误差的比例
    lf_error_ratio: float = 0.0       # 低频误差比例

    # 各向异性
    h_error: float = 0.0              # 水平方向误差（雨线方向）
    v_error: float = 0.0              # 垂直方向误差
    anisotropy_index: float = 0.0     # |h - v| / (h + v)

    # 结构性
    error_sparsity: float = 0.0       # 误差的空间稀疏性（高 = 局部误差）
    mean_error: float = 0.0
    std_error: float = 0.0


class ErrorSpectrumAnalyzer:
    """
    分析 SNN 和 ANN 的重建误差的频谱特征。

    核心功能：
      - 计算误差 PSD 并对比 SNN/ANN 的频谱形状
      - 测量误差的各向异性（与任务退化方向的关联）
      - 检测误差是否具有"结构性"（与退化模式共振）
    """

    def __init__(self, n_bins: int = 64, hf_cutoff: float = 0.3):
        self.n_bins = n_bins
        self.hf_cutoff = hf_cutoff

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return 0.299*img[0] + 0.587*img[1] + 0.114*img[2]
        return img

    def analyze(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        model_type: str = "snn",
        task: str = "unknown",
    ) -> ErrorSpectrumStats:
        """
        分析单个 (预测, 目标) 对的误差频谱。

        Args:
            pred, target: (C, H, W) 或 (H, W) 图像 [0,1]
        """
        err = self._to_gray(pred) - self._to_gray(target)  # (H, W)
        H, W = err.shape
        mean_err = float(np.abs(err).mean())
        std_err  = float(err.std())

        # ── 1. 功率谱 ──────────────────────────────────────────────────
        fft2 = np.fft.fft2(err)
        psd2 = np.abs(fft2)**2

        fy = np.fft.fftfreq(H)
        fx = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)

        bin_edges = np.linspace(0, R.max(), self.n_bins + 1)
        psd_curve = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            mask = (R >= bin_edges[b]) & (R < bin_edges[b+1])
            if mask.sum() > 0:
                psd_curve[b] = psd2[mask].mean()
        freq_axis = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 归一化
        psd_norm = psd_curve / (psd_curve.sum() + 1e-12)

        # ── 2. 频谱统计 ────────────────────────────────────────────────
        # 质心
        centroid = float(np.average(freq_axis, weights=psd_norm + 1e-12))
        # 平坦度：几何均值/算术均值（SNF, Wiener entropy）
        if (psd_norm > 0).any():
            log_mean = np.exp(np.mean(np.log(psd_norm[psd_norm > 0])))
            arith_mean = psd_norm[psd_norm > 0].mean()
            flatness = float(log_mean / (arith_mean + 1e-12))
        else:
            flatness = 0.0

        hf_mask = freq_axis > self.hf_cutoff
        hf_ratio = float(psd_norm[hf_mask].sum())
        lf_ratio = 1.0 - hf_ratio

        # ── 3. 各向异性 ────────────────────────────────────────────────
        # 水平误差（主要捕捉垂直方向变化，如垂直雨线）
        h_err = float((err**2).mean(axis=1).mean())   # 沿列平均
        v_err = float((err**2).mean(axis=0).mean())   # 沿行平均
        ani = abs(h_err - v_err) / (h_err + v_err + 1e-8)

        # ── 4. 误差稀疏性 ───────────────────────────────────────────────
        err_abs = np.abs(err).flatten()
        err_sorted = np.sort(err_abs)
        n = len(err_sorted)
        idx = np.arange(1, n+1)
        if err_sorted.sum() > 0:
            gini = float((2*idx - n - 1).dot(err_sorted) / (n * err_sorted.sum()))
        else:
            gini = 0.0

        return ErrorSpectrumStats(
            model_type=model_type, task=task,
            psd_curve=psd_curve, freq_axis=freq_axis,
            spectral_centroid=centroid,
            spectral_flatness=flatness,
            hf_error_ratio=hf_ratio,
            lf_error_ratio=lf_ratio,
            h_error=h_err, v_error=v_err,
            anisotropy_index=float(ani),
            error_sparsity=gini,
            mean_error=mean_err, std_error=std_err,
        )

    def compare(
        self,
        pred_dict: Dict[str, np.ndarray],
        target: np.ndarray,
        task: str = "unknown",
    ) -> Dict[str, ErrorSpectrumStats]:
        """对比多个模型的误差频谱。"""
        return {
            mt: self.analyze(pred, target, mt, task)
            for mt, pred in pred_dict.items()
        }

    def print_comparison(self, stats: Dict[str, ErrorSpectrumStats]):
        """打印误差频谱对比表。"""
        print(f"\n{'='*75}")
        task = list(stats.values())[0].task if stats else ""
        print(f"Error Spectrum Analysis (task={task})")
        print(f"{'='*75}")
        print(f"{'Model':<12} {'Centroid':>10} {'Flatness':>10} "
              f"{'HF_ratio':>10} {'Anisotropy':>12} {'Sparsity':>10}")
        print("-"*75)
        for mt, s in stats.items():
            print(f"{mt:<12} {s.spectral_centroid:>10.4f} "
                  f"{s.spectral_flatness:>10.4f} "
                  f"{s.hf_error_ratio:>10.4f} "
                  f"{s.anisotropy_index:>12.4f} "
                  f"{s.error_sparsity:>10.4f}")
        print(f"{'='*75}")
        print("  High HF_ratio = error concentrated in high frequencies")
        print("  High Anisotropy = directional error (rain-aligned)")
        print("  High Sparsity = localized error (sparse degradation handled well)")
