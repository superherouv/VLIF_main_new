"""
统一分析流水线 (Unified Analysis Pipeline).

将频域主线和表征主线整合为一个完整的研究流水线。

Pipeline 流程：
  ① 数据准备：加载 (degraded, clean) 图像对
  ② 模型推理：分别收集 SNN/ANN/Hybrid 的
     - 中间层特征（用于表征分析）
     - 膜电位轨迹（用于相空间分析）
     - 脉冲序列（用于 ISI/谱/活性分析）
  ③ 频域分析：
     - FrequencyBandAnalyzer: 频段 PSNR 曲线
     - WaveletDecomposer: 小波子带 SNN/ANN 差异
     - SpikeSpectrumAnalyzer: 脉冲频率指纹
     - ErrorSpectrumAnalyzer: 误差频谱形状
     - FrequencyComplexity: 任务频率复杂度
  ④ 表征分析：
     - CKAAnalyzer: SNN vs ANN 层间相似性
     - MutualInfoAnalyzer: 信息瓶颈分析
     - MembranePhaseSpace: 膜电位动力学
     - ISIAnalyzer: 脉冲编码类型
     - NeuronViability: 神经元健康度
     - ManifoldAnalyzer: 特征流形几何
  ⑤ 综合评分：将所有分析汇总为适用性分数
  ⑥ 报告生成：JSON + ASCII 报告

研究价值：
  这套流水线能回答的核心研究问题：
  "为什么 SNN 在去雨上比低光增强更有优势？"
  答案将从 6 个独立角度给出一致的解释：
    - 频域：去雨的高频误差更小（频域主线）
    - 脉冲谱：去雨的脉冲模式有高频峰值（脉冲指纹）
    - 小波：去雨的 HH 子带 SNN/ANN 比接近 1（小波主线）
    - ISI：去雨的脉冲更稀疏 CV 更高（时间编码）
    - 活性：去雨时神经元活化与雨线位置高度相关（空间活性）
    - 流形：去雨的 SNN 特征有更高的 Fisher 比（类别可分）
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .frequency import (
    FrequencyBandAnalyzer,
    SpikeSpectrumAnalyzer,
    WaveletDecomposer,
    FrequencyComplexity,
    ErrorSpectrumAnalyzer,
)
from .representation import (
    CKAAnalyzer,
    MutualInfoAnalyzer,
    MembranePhaseSpace,
    ISIAnalyzer,
    NeuronViability,
    ManifoldAnalyzer,
)
from .applicability import ApplicabilityAnalyzer


@dataclass
class FullAnalysisReport:
    """完整分析报告的数据结构。"""
    task: str
    n_samples: int = 0

    # ── 频域结果 ──────────────────────────────────────────────
    freq_band_profiles: Dict[str, Any] = field(default_factory=dict)
    wavelet_result: Optional[Any] = None
    spike_spectrum_signature: Dict[str, float] = field(default_factory=dict)
    error_spectrum_stats: Dict[str, Any] = field(default_factory=dict)
    task_freq_complexity: Optional[Any] = None

    # ── 表征结果 ──────────────────────────────────────────────
    cka_matrix_snn_ann: Optional[Any] = None
    within_cka_snn: Optional[Any] = None
    ib_results_snn: List[Any] = field(default_factory=list)
    ib_results_ann: List[Any] = field(default_factory=list)
    phase_space_stats: Dict[str, Any] = field(default_factory=dict)
    isi_stats: Dict[str, Any] = field(default_factory=dict)
    neuron_viability_stats: Dict[str, Any] = field(default_factory=dict)
    manifold_comparison: Dict[str, Any] = field(default_factory=dict)

    # ── 综合适用性 ────────────────────────────────────────────
    applicability_score: float = 0.0
    applicability_level: str = ""
    key_findings: List[str] = field(default_factory=list)


class UnifiedAnalysisPipeline:
    """
    统一分析流水线。

    用法：
        pipeline = UnifiedAnalysisPipeline(task="deraining")

        # 提供图像和模型输出
        pipeline.run(
            degraded=deg_batch,        # (N, C, H, W)
            clean=clean_batch,         # (N, C, H, W)
            pred_snn=snn_outputs,      # (N, C, H, W)
            pred_ann=ann_outputs,      # (N, C, H, W)
            spike_records=spike_dict,  # {layer: (T,B,C,H,W)}
            membrane_records=memb_dict,# {layer: (T,B,C,H,W)}
            features_snn=feat_snn,     # {layer: (N,D)}
            features_ann=feat_ann,     # {layer: (N,D)}
            degradation_labels=labels, # (N,) int
        )

        report = pipeline.get_report()
        pipeline.print_summary(report)
        pipeline.save(report, "experiments/analysis/deraining_full.json")
    """

    def __init__(self, task: str = "deraining"):
        self.task = task
        # 初始化所有分析器
        self.freq_band   = FrequencyBandAnalyzer(n_bands=16)
        self.spike_spec  = SpikeSpectrumAnalyzer()
        self.wavelet     = WaveletDecomposer(n_levels=4)
        self.freq_complex = FrequencyComplexity()
        self.err_spec    = ErrorSpectrumAnalyzer()

        self.cka         = CKAAnalyzer()
        self.mi          = MutualInfoAnalyzer()
        self.phase       = MembranePhaseSpace()
        self.isi         = ISIAnalyzer()
        self.viability   = NeuronViability()
        self.manifold    = ManifoldAnalyzer()

        self._report: Optional[FullAnalysisReport] = None

    def run(
        self,
        degraded: np.ndarray,            # (N, C, H, W)
        clean: np.ndarray,               # (N, C, H, W)
        pred_snn: Optional[np.ndarray],  # (N, C, H, W)
        pred_ann: Optional[np.ndarray],  # (N, C, H, W)
        spike_records: Optional[Dict[str, np.ndarray]] = None,      # {layer: (T,B,C,H,W)}
        membrane_records: Optional[Dict[str, np.ndarray]] = None,    # {layer: (T,B,C,H,W)}
        features_snn: Optional[Dict[str, np.ndarray]] = None,        # {layer: (N,D)}
        features_ann: Optional[Dict[str, np.ndarray]] = None,        # {layer: (N,D)}
        degradation_labels: Optional[np.ndarray] = None,             # (N,) int
        degradation_strengths: Optional[np.ndarray] = None,          # (N,) float
        verbose: bool = True,
    ) -> FullAnalysisReport:
        """
        运行完整分析流水线。
        """
        report = FullAnalysisReport(task=self.task, n_samples=len(degraded))
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Unified Analysis Pipeline: task={self.task}")
            print(f"  N={len(degraded)} samples")
            print(f"{'='*60}")

        # ── 1. 任务频率复杂度（不需要模型输出）────────────────
        if verbose:
            print("[1/7] Computing task frequency complexity...")
        report.task_freq_complexity = self.freq_complex.compute(
            degraded, clean, task=self.task
        )

        # ── 2. 频段 PSNR + 误差频谱 ─────────────────────────
        if verbose:
            print("[2/7] Frequency band PSNR analysis...")
        pred_dict = {}
        if pred_snn is not None:
            pred_dict["snn"] = pred_snn
        if pred_ann is not None:
            pred_dict["ann"] = pred_ann

        if pred_dict and len(degraded) > 0:
            idx = 0  # 分析第一个样本（代表性）
            profiles = {}
            for mt, pred_batch in pred_dict.items():
                profiles[mt] = self.freq_band.analyze_pair(
                    pred_batch[idx], clean[idx], mt, self.task
                )
            report.freq_band_profiles = {
                mt: {
                    "low_freq_psnr": p.low_freq_psnr,
                    "mid_freq_psnr": p.mid_freq_psnr,
                    "high_freq_psnr": p.high_freq_psnr,
                    "critical_freq": p.critical_freq,
                    "hf_lf_ratio": p.hf_lf_ratio,
                    "bands": [(b.freq_range, b.psnr) for b in p.bands],
                }
                for mt, p in profiles.items()
            }

            # 误差频谱（取第一个样本）
            pred_dict_single = {mt: pb[idx] for mt, pb in pred_dict.items()}
            report.error_spectrum_stats = {
                mt: {
                    "spectral_centroid": s.spectral_centroid,
                    "hf_error_ratio": s.hf_error_ratio,
                    "anisotropy_index": s.anisotropy_index,
                    "error_sparsity": s.error_sparsity,
                }
                for mt, s in self.err_spec.compare(pred_dict_single, clean[idx], self.task).items()
            }

        # ── 3. 小波分析 ──────────────────────────────────────
        if verbose:
            print("[3/7] Wavelet subband analysis...")
        if pred_snn is not None and pred_ann is not None and len(degraded) > 0:
            wav_result = self.wavelet.analyze(pred_snn[0], pred_ann[0], clean[0])
            report.wavelet_result = {
                "ll_ratio": wav_result.ll_ratio,
                "lh_ratio": wav_result.lh_ratio,
                "hl_ratio": wav_result.hl_ratio,
                "hh_ratio": wav_result.hh_ratio,
                "worst_subband": wav_result.worst_subband,
                "best_subband": wav_result.best_subband,
            }

        # ── 4. 脉冲频谱分析 ──────────────────────────────────
        if verbose:
            print("[4/7] Spike spectrum analysis...")
        if spike_records:
            all_results = self.spike_spec.analyze(spike_records)
            report.spike_spectrum_signature = self.spike_spec.task_frequency_signature(
                all_results
            )

        # ── 5. 膜电位相空间 + ISI + 神经元活性 ───────────────
        if verbose:
            print("[5/7] Membrane potential phase space & ISI analysis...")
        if membrane_records:
            phase_stats = self.phase.analyze(membrane_records)
            report.phase_space_stats = {
                name: {
                    "mean_v": s.mean_v,
                    "std_v": s.std_v,
                    "threshold_crossing_rate": s.threshold_crossing_rate,
                    "autocorr_tau1": s.autocorr_tau1,
                    "phase_volume": s.phase_volume,
                    "spike_efficiency": s.spike_efficiency,
                }
                for name, s in phase_stats.items()
            }

        if spike_records:
            isi_stats = self.isi.analyze_all_layers(spike_records)
            report.isi_stats = {
                name: {
                    "cv_isi": s.cv_isi,
                    "mean_isi": s.mean_isi,
                    "lv": s.lv,
                    "burst_fraction": s.burst_fraction,
                    "coding_type": s.coding_type,
                    "firing_rate": s.firing_rate,
                }
                for name, s in isi_stats.items()
            }

            # 计算退化掩码（用于神经元活性-退化区域相关性）
            if len(degraded) > 0 and len(clean) > 0:
                diff = np.abs(degraded[0] - clean[0])
                if diff.ndim == 3:
                    diff = diff.mean(0)
                thr = np.percentile(diff, 80)
                degradation_mask = (diff > thr).astype(float)
            else:
                degradation_mask = None

            viability_stats = self.viability.analyze_all(spike_records, degradation_mask)
            health = self.viability.global_health_score(viability_stats)
            report.neuron_viability_stats = {
                "layer_stats": {
                    name: {
                        "silent_fraction": s.silent_fraction,
                        "saturated_fraction": s.saturated_fraction,
                        "effective_fraction": s.effective_fraction,
                        "rate_gini": s.rate_gini,
                        "activation_degradation_corr": s.activation_degradation_corr,
                    }
                    for name, s in viability_stats.items()
                },
                "global_health": health,
            }

        # ── 6. CKA + 流形分析 ────────────────────────────────
        if verbose:
            print("[6/7] CKA & manifold analysis...")
        if features_snn and features_ann:
            cka_mat = self.cka.cross_model_cka(
                features_snn, features_ann, "snn", "ann"
            )
            report.cka_matrix_snn_ann = {
                "diagonal_similarity": cka_mat.diagonal_similarity,
                "mean_cross_similarity": cka_mat.mean_cross_similarity,
                "bottleneck_layer": cka_mat.bottleneck_layer,
                "layer_names": cka_mat.layer_names,
                "matrix": cka_mat.matrix.tolist(),
            }

            within_cka = self.cka.within_model_cka(features_snn, "snn")
            report.within_cka_snn = {
                "diagonal": np.diag(within_cka.matrix).tolist(),
                "bottleneck": within_cka.bottleneck_layer,
            }

            # 流形分析
            if degradation_labels is not None:
                manifold_comp = self.manifold.compare_snn_ann(
                    features_snn, features_ann, degradation_labels
                )
                report.manifold_comparison = {
                    layer: {
                        "snn_effective_rank": snn_s.effective_rank,
                        "ann_effective_rank": ann_s.effective_rank,
                        "snn_fisher": snn_s.fisher_ratio,
                        "ann_fisher": ann_s.fisher_ratio,
                        "procrustes_dist": snn_s.procrustes_distance,
                        "snn_pca_dim_90": snn_s.pca_dim_90,
                        "ann_pca_dim_90": ann_s.pca_dim_90,
                    }
                    for layer, (snn_s, ann_s) in manifold_comp.items()
                }

        # ── 7. 综合适用性评分 ─────────────────────────────────
        if verbose:
            print("[7/7] Computing comprehensive applicability score...")
        report.applicability_score, report.applicability_level, report.key_findings = \
            self._compute_comprehensive_score(report)

        self._report = report
        if verbose:
            self.print_summary(report)
        return report

    def _compute_comprehensive_score(
        self, report: FullAnalysisReport
    ) -> tuple:
        """
        从所有分析结果综合计算适用性评分。

        整合频域和表征两条主线的证据，输出最终评分。
        """
        score_components = {}
        findings = []

        # ── 频域证据 ──────────────────────────────────────────
        if "snn" in report.freq_band_profiles and "ann" in report.freq_band_profiles:
            snn_hf = report.freq_band_profiles["snn"]["high_freq_psnr"]
            ann_hf = report.freq_band_profiles["ann"]["high_freq_psnr"]
            hf_gap = ann_hf - snn_hf
            # 高频差距小 → SNN 适合（高频保真度好）
            freq_score = max(0, 1.0 - hf_gap / 10.0)
            score_components["freq_band"] = freq_score
            if hf_gap < 2.0:
                findings.append(f"✓ SNN high-freq PSNR gap small ({hf_gap:.1f}dB) → frequency-friendly task")
            else:
                findings.append(f"✗ SNN high-freq PSNR gap large ({hf_gap:.1f}dB) → SNN struggles with HF")

        # 小波证据
        if report.wavelet_result:
            hh_ratio = report.wavelet_result["hh_ratio"]
            # HH ratio ≈ 1 → SNN 和 ANN 在对角高频子带表现相同
            wav_score = max(0, 1.0 - abs(hh_ratio - 1.0))
            score_components["wavelet"] = wav_score
            if hh_ratio < 1.5:
                findings.append(f"✓ Wavelet HH ratio={hh_ratio:.2f}: SNN preserves high-freq diagonal")
            else:
                findings.append(f"✗ Wavelet HH ratio={hh_ratio:.2f}: SNN loses high-freq diagonal")

        # 任务频率复杂度
        if report.task_freq_complexity:
            suit = report.task_freq_complexity.snn_suitability
            score_components["task_freq"] = suit
            findings.append(f"  Task signal sparsity (SCSM={report.task_freq_complexity.scsm:.2f}, "
                           f"BFF={report.task_freq_complexity.bff:.2f}) → "
                           f"suitability={suit:.2f}")

        # 脉冲频谱
        if report.spike_spectrum_signature:
            sig = report.spike_spectrum_signature
            # 高空间主频且高突发性 → SNN 在关键位置集中发放（理想）
            burst = sig.get("mean_burstiness", 0)
            spike_score = 0.5 + 0.5 * burst  # 突发性高更好（稀疏事件驱动）
            score_components["spike_spectrum"] = min(1.0, spike_score)

        # ── 表征证据 ──────────────────────────────────────────
        # CKA
        if report.cka_matrix_snn_ann:
            diag_cka = report.cka_matrix_snn_ann["diagonal_similarity"]
            # 适中的 CKA（0.4-0.7）最好：SNN 学到了与 ANN 不同但有效的特征
            # 太低 = SNN 未学到有用特征；太高 = SNN 完全复制 ANN（无 SNN 特有优势）
            if 0.4 <= diag_cka <= 0.7:
                cka_score = 1.0
                findings.append(f"✓ CKA={diag_cka:.2f}: SNN learns distinct but effective representations")
            elif diag_cka > 0.7:
                cka_score = 0.8
                findings.append(f"△ CKA={diag_cka:.2f}: SNN representations similar to ANN (good quality)")
            else:
                cka_score = 0.3
                findings.append(f"✗ CKA={diag_cka:.2f}: SNN representations diverge from ANN")
            score_components["cka"] = cka_score

        # ISI 编码类型
        if report.isi_stats:
            cv_vals = [v.get("cv_isi", 1.0) for v in report.isi_stats.values()]
            mean_cv = float(np.mean(cv_vals))
            # 高 CV（>1）→ 稀疏/突发发放 → 适合稀疏退化（去雨）
            # 低 CV（<0.5）→ 时间编码 → 对某些任务有优势
            isi_score = min(1.0, mean_cv / 1.5)
            score_components["isi"] = isi_score

        # 神经元健康度
        if report.neuron_viability_stats:
            health = report.neuron_viability_stats.get("global_health", {})
            health_score = health.get("health_score", 0.5)
            score_components["neuron_health"] = health_score
            dead_ratio = health.get("dead_neuron_ratio", 0)
            if dead_ratio < 0.1:
                findings.append(f"✓ Neuron health good: only {100*dead_ratio:.0f}% dead neurons")
            elif dead_ratio > 0.3:
                findings.append(f"✗ High dead neuron ratio: {100*dead_ratio:.0f}% → gradient issues")

        # 误差各向异性（与退化方向对齐 → SNN 学到了结构化表征）
        if "snn" in report.error_spectrum_stats:
            aniso = report.error_spectrum_stats["snn"].get("anisotropy_index", 0)
            err_sparse = report.error_spectrum_stats["snn"].get("error_sparsity", 0)
            if aniso > 0.2:
                findings.append(f"✓ Error anisotropy={aniso:.2f}: SNN errors aligned with degradation direction")
            score_components["error_structure"] = 0.5 * aniso + 0.5 * err_sparse

        # ── 综合评分 ──────────────────────────────────────────
        if score_components:
            # 各维度等权重平均
            final_score = float(np.mean(list(score_components.values()))) * 100
        else:
            final_score = 50.0

        if final_score >= 75:
            level = "HIGH"
        elif final_score >= 55:
            level = "MEDIUM"
        elif final_score >= 35:
            level = "LOW"
        else:
            level = "VERY_LOW"

        return final_score, level, findings

    def print_summary(self, report: FullAnalysisReport):
        print(f"\n{'='*70}")
        print(f"UNIFIED ANALYSIS SUMMARY: {report.task.upper()}")
        print(f"{'='*70}")
        print(f"  Applicability Score: {report.applicability_score:.1f}/100  [{report.applicability_level}]")
        print(f"\n  Key Findings:")
        for finding in report.key_findings:
            print(f"    {finding}")

        if report.task_freq_complexity:
            c = report.task_freq_complexity
            print(f"\n  Task Complexity: difficulty={c.snn_difficulty:.3f} "
                  f"suitability={c.snn_suitability:.3f}")

        if report.wavelet_result:
            w = report.wavelet_result
            print(f"  Wavelet: LL={w['ll_ratio']:.2f} LH={w['lh_ratio']:.2f} "
                  f"HL={w['hl_ratio']:.2f} HH={w['hh_ratio']:.2f} "
                  f"(worst={w['worst_subband']})")

        if report.spike_spectrum_signature:
            s = report.spike_spectrum_signature
            print(f"  Spike Spectrum: spatial_peak={s.get('mean_spatial_peak_freq', 0):.4f} "
                  f"burstiness={s.get('mean_burstiness', 0):.4f} "
                  f"1/f_exp={s.get('mean_1f_exponent', 0):.4f}")

        if report.cka_matrix_snn_ann:
            c = report.cka_matrix_snn_ann
            print(f"  CKA SNN-ANN: diagonal={c['diagonal_similarity']:.4f} "
                  f"bottleneck={c['bottleneck_layer']}")

        if report.neuron_viability_stats:
            h = report.neuron_viability_stats.get("global_health", {})
            print(f"  Neuron Health: {100*h.get('effective_neuron_ratio',0):.0f}% effective, "
                  f"{100*h.get('dead_neuron_ratio',0):.0f}% dead")
        print(f"{'='*70}")

    def save(self, report: FullAnalysisReport, path: str):
        """保存完整分析报告为 JSON。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            return obj

        data = {
            "task": report.task,
            "n_samples": report.n_samples,
            "applicability_score": report.applicability_score,
            "applicability_level": report.applicability_level,
            "key_findings": report.key_findings,
            "freq_band_profiles": to_serializable(report.freq_band_profiles),
            "wavelet_result": to_serializable(report.wavelet_result),
            "spike_spectrum_signature": to_serializable(report.spike_spectrum_signature),
            "error_spectrum_stats": to_serializable(report.error_spectrum_stats),
            "task_freq_complexity": {
                "hfr": report.task_freq_complexity.hfr,
                "sfd": report.task_freq_complexity.sfd,
                "fsi": report.task_freq_complexity.fsi,
                "scsm": report.task_freq_complexity.scsm,
                "bff": report.task_freq_complexity.bff,
                "snn_difficulty": report.task_freq_complexity.snn_difficulty,
                "snn_suitability": report.task_freq_complexity.snn_suitability,
            } if report.task_freq_complexity else None,
            "cka_matrix_snn_ann": to_serializable(report.cka_matrix_snn_ann),
            "neuron_viability_stats": to_serializable(report.neuron_viability_stats),
            "isi_stats": to_serializable(report.isi_stats),
            "phase_space_stats": to_serializable(report.phase_space_stats),
            "manifold_comparison": to_serializable(report.manifold_comparison),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Full analysis saved to: {path}")
