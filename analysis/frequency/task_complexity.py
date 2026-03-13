"""
任务频率复杂度指标 (Task Frequency Complexity Metrics).

关键假设：
  任务所需重建的频率分布越高（越复杂），SNN 越难胜任。
  通过量化"目标图像与输入图像的频率差异"来测量任务复杂度。

5个核心指标：
  1. HFR (High-Frequency Ratio): 目标图中高频能量占比
  2. SFD (Spectral Flux Density): 退化导致的频谱变化量
  3. FSI (Frequency Shift Index): 退化是否移动了频谱中心（超分 > 去噪 > 去雨）
  4. SCSM (Spatial Correlation Sparsity Metric): 退化区域的空间相关性
  5. BFF (Binary Frequency Fidelity): 用二值（脉冲-like）表示能达到的理论上限

与适用性的关联：
  ┌──────────────┬──────┬──────┬──────┬──────┬─────┐
  │ Task         │ HFR  │ SFD  │ FSI  │ SCSM │ BFF │
  ├──────────────┼──────┼──────┼──────┼──────┼─────┤
  │ Deraining    │ High │ Low  │ Low  │ High │ Med │
  │ Denoising    │ Med  │ Med  │ Low  │ Low  │ Low │
  │ SuperRes ×4  │ High │ High │ High │ Med  │ Low │
  │ Dehazing     │ Low  │ Low  │ Low  │ High │ High│
  │ Low-light    │ Low  │ Low  │ High │ Low  │ Low │
  └──────────────┴──────┴──────┴──────┴──────┴─────┘
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrequencyComplexityScore:
    """任务频率复杂度综合得分。"""
    task: str
    hfr: float            # 高频比例 [0,1]
    sfd: float            # 频谱变化密度 [0,1]
    fsi: float            # 频谱偏移指数 [0,1]
    scsm: float           # 空间相关稀疏度 [0,1]
    bff: float            # 二值频率保真度 [0,1]（越高越适合 SNN）

    # 综合得分
    snn_difficulty: float = 0.0   # 对 SNN 的难度 [0,1]（越低越适合 SNN）
    snn_suitability: float = 0.0  # 适合度 = 1 - difficulty [0,1]


class FrequencyComplexity:
    """
    从数据集图像对估计任务的频率复杂度。

    输入：(degraded, clean) 图像对的批次
    输出：FrequencyComplexityScore

    这些指标是**数据驱动**的，不依赖模型输出，
    可以在训练前预估 SNN 对该任务的适用性。
    """

    def __init__(self, hf_cutoff: float = 0.3):
        """
        Args:
            hf_cutoff (float): 高频阈值（归一化频率，0.3 = 30% Nyquist）。
        """
        self.hf_cutoff = hf_cutoff

    def compute(
        self,
        degraded: np.ndarray,   # (N, C, H, W) 批次
        clean: np.ndarray,      # (N, C, H, W) 批次
        task: str = "unknown",
    ) -> FrequencyComplexityScore:
        """
        计算任务的频率复杂度。

        Args:
            degraded: 退化图像批次，[0,1]
            clean:    干净图像批次，[0,1]
            task:     任务名称（用于标签）

        Returns:
            FrequencyComplexityScore
        """
        N = degraded.shape[0]
        hfr_list, sfd_list, fsi_list, scsm_list, bff_list = [], [], [], [], []

        for i in range(N):
            deg_g = self._to_gray(degraded[i])
            cln_g = self._to_gray(clean[i])

            hfr  = self._compute_hfr(cln_g)
            sfd  = self._compute_sfd(deg_g, cln_g)
            fsi  = self._compute_fsi(deg_g, cln_g)
            scsm = self._compute_scsm(deg_g, cln_g)
            bff  = self._compute_bff(cln_g)

            hfr_list.append(hfr)
            sfd_list.append(sfd)
            fsi_list.append(fsi)
            scsm_list.append(scsm)
            bff_list.append(bff)

        score = FrequencyComplexityScore(
            task=task,
            hfr=float(np.mean(hfr_list)),
            sfd=float(np.mean(sfd_list)),
            fsi=float(np.mean(fsi_list)),
            scsm=float(np.mean(scsm_list)),
            bff=float(np.mean(bff_list)),
        )

        # 综合难度得分（高频多、变化大、频偏大 → 难）
        score.snn_difficulty = float(
            0.35 * score.hfr   +
            0.25 * score.sfd   +
            0.20 * score.fsi   +
            0.10 * (1 - score.scsm) +  # 低稀疏度=密集退化=难
            0.10 * (1 - score.bff)     # 低 BFF=不适合二值表示=难
        )
        score.snn_suitability = 1.0 - score.snn_difficulty

        return score

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return 0.299*img[0] + 0.587*img[1] + 0.114*img[2]
        return img

    def _psd(self, img: np.ndarray) -> np.ndarray:
        """计算功率谱（归一化）。"""
        f = np.fft.fft2(img)
        psd = np.abs(f)**2
        return psd / (psd.sum() + 1e-12)

    def _radial_energy(
        self, psd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """返回径向频率和对应能量。"""
        H, W = psd.shape
        fy = np.fft.fftfreq(H)
        fx = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)

        n_bins = 64
        r_max = R.max()
        edges = np.linspace(0, r_max, n_bins + 1)
        energy = np.zeros(n_bins)
        freqs = (edges[:-1] + edges[1:]) / 2

        for b in range(n_bins):
            mask = (R >= edges[b]) & (R < edges[b+1])
            if mask.sum() > 0:
                energy[b] = psd[mask].sum()
        energy /= (energy.sum() + 1e-12)
        return freqs, energy

    def _compute_hfr(self, clean: np.ndarray) -> float:
        """
        HFR: 目标图的高频能量比例。
        HFR = E(f > cutoff) / E(all)
        """
        psd = self._psd(clean)
        freqs, energy = self._radial_energy(psd)
        hf_mask = freqs > self.hf_cutoff
        return float(energy[hf_mask].sum())

    def _compute_sfd(self, deg: np.ndarray, cln: np.ndarray) -> float:
        """
        SFD: 退化导致的频谱变化量。
        SFD = ||PSD(deg) - PSD(cln)||_1 / 2
        范围 [0,1]，越大说明退化在频谱上影响越大。
        """
        psd_d = self._psd(deg)
        psd_c = self._psd(cln)
        return float(np.abs(psd_d - psd_c).sum() / 2.0)

    def _compute_fsi(self, deg: np.ndarray, cln: np.ndarray) -> float:
        """
        FSI: 频谱中心的偏移量（超分 → 输入频谱比输出低，FSI 大）。
        FSI = |centroid(PSD_cln) - centroid(PSD_deg)| / f_nyquist
        """
        _, e_d = self._radial_energy(self._psd(deg))
        _, e_c = self._radial_energy(self._psd(cln))
        freqs = np.linspace(0, 0.5, len(e_d))
        c_d = float(np.average(freqs, weights=e_d + 1e-12))
        c_c = float(np.average(freqs, weights=e_c + 1e-12))
        return min(1.0, abs(c_c - c_d) / 0.5)

    def _compute_scsm(self, deg: np.ndarray, cln: np.ndarray) -> float:
        """
        SCSM: 退化区域的空间稀疏度。
        通过计算 |clean - degraded| 的稀疏性来衡量退化的空间集中程度。
        高稀疏度 → 退化集中在少数区域（如雨线）→ 适合 SNN。
        低稀疏度 → 退化弥漫在整个图像（如噪声）→ 不适合 SNN。
        """
        diff = np.abs(cln - deg).flatten()
        if diff.sum() == 0:
            return 0.0
        # Gini 系数作为稀疏度
        diff_sorted = np.sort(diff)
        n = len(diff_sorted)
        idx = np.arange(1, n + 1)
        gini = float((2 * idx - n - 1).dot(diff_sorted) / (n * diff_sorted.sum()))
        return gini

    def _compute_bff(self, clean: np.ndarray) -> float:
        """
        BFF: 二值频率保真度。
        用二值量化（threshold=mean）近似目标图，计算 PSNR。
        BFF = PSNR(binary_approx, clean) / 50.0（归一化到[0,1]）。
        高 BFF → 目标图易于二值近似 → 适合 SNN 的二值脉冲表示。
        """
        binary = (clean >= clean.mean()).astype(float) * clean.max()
        mse = float(((binary - clean)**2).mean())
        if mse < 1e-12:
            return 1.0
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return min(1.0, max(0.0, psnr / 50.0))

    def compare_tasks(
        self, scores: Dict[str, FrequencyComplexityScore]
    ):
        """打印任务对比表。"""
        print(f"\n{'='*80}")
        print("Task Frequency Complexity → SNN Suitability Predictor")
        print(f"{'='*80}")
        print(f"{'Task':<18} {'HFR':>6} {'SFD':>6} {'FSI':>6} "
              f"{'SCSM':>6} {'BFF':>6} {'Difficulty':>11} {'Suitability':>12}")
        print("-"*80)
        task_order = ["deraining", "denoising", "dehazing", "superresolution", "lowlight"]
        for task in task_order:
            if task not in scores:
                continue
            s = scores[task]
            print(f"{task:<18} {s.hfr:>6.3f} {s.sfd:>6.3f} {s.fsi:>6.3f} "
                  f"{s.scsm:>6.3f} {s.bff:>6.3f} "
                  f"{s.snn_difficulty:>11.3f} {s.snn_suitability:>12.3f}")
        print(f"{'='*80}")
        print("  High Suitability → SNN recommended | Low → ANN/Hybrid preferred")
