"""
特征流形分析 (Feature Manifold Analysis).

研究 SNN 和 ANN 学到的特征表征在高维空间中的几何结构。

关键问题：
  1. SNN 特征是否形成了清晰可分的流形？
     → 不同退化类型的样本在特征空间中的分离度
     → 使用 PCA 降维后的方差解释率（越高越紧凑）

  2. 特征流形的维度：
     → 有效秩（effective rank）= exp(H) 其中 H 是奇异值熵
     → 低有效秩 = 低维流形（SNN 的信息压缩效率）

  3. 任务间流形相似性：
     → 同一 SNN 对不同退化任务的特征是否在不同的子流形上？
     → 若是，说明 SNN 已学到任务特异的脉冲表征

  4. 流形曲率：
     → 特征空间的局部曲率（高曲率 = 非线性、难以分类）
     → 通过 kNN 图的局部维度估计

  5. SNN vs ANN 流形对齐：
     → 用 Procrustes 分析比较两者的流形结构
     → 若高度对齐 → SNN 学到了 ANN 等价的特征
     → 若不对齐 → SNN 发现了不同的表征策略

这些分析回答了"SNN 是如何表征图像退化的？"这个根本问题。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ManifoldStats:
    """特征流形统计。"""
    layer_name: str
    model_type: str
    n_samples: int
    feature_dim: int

    # 维度分析
    pca_dim_90: int = 0             # 解释 90% 方差的 PCA 维数
    effective_rank: float = 0.0     # 有效秩（谱熵）
    participation_ratio: float = 0.0 # 参与比 = (Σλ)²/Σλ²

    # 可分性
    inter_class_dist: float = 0.0   # 不同退化类型间的平均距离
    intra_class_dist: float = 0.0   # 同退化类型内的平均距离
    fisher_ratio: float = 0.0       # Fisher 判别比 = inter/intra

    # 流形紧凑性
    total_variance: float = 0.0
    pca_variance_ratio: List[float] = field(default_factory=list)  # 前 k 个 PC 的方差比

    # Procrustes 对齐（与参考模型的流形对齐度）
    procrustes_distance: float = 0.0   # 0 = 完全对齐，1 = 完全不同


class ManifoldAnalyzer:
    """
    特征流形分析器。

    对 SNN/ANN/Hybrid 的中间层特征进行流形分析，
    揭示不同模型的表征几何结构差异。
    """

    def __init__(self, n_pca_components: int = 50):
        self.n_pca = n_pca_components

    def _pca(self, X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        纯 NumPy PCA。

        Returns:
            (projected, explained_variance_ratio)
        """
        X_c = X - X.mean(0)
        n = min(n_components, min(X_c.shape) - 1)
        try:
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            S = S[:n]
            evr = S**2 / (S**2).sum()
            projected = X_c @ Vt[:n].T
            return projected, evr
        except Exception:
            return X_c[:, :n], np.ones(n) / n

    def _effective_rank(self, singular_values: np.ndarray) -> float:
        """
        有效秩（谱熵）= exp(-Σ p_i log p_i) 其中 p_i = s_i / Σs_i。
        范围 [1, dim]。低有效秩 = 低维紧凑流形。
        """
        s = singular_values
        s = s[s > 0]
        if len(s) == 0:
            return 1.0
        p = s / s.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(np.exp(entropy))

    def _participation_ratio(self, eigenvalues: np.ndarray) -> float:
        """
        参与比 PR = (Σλ_i)² / Σ(λ_i)²。
        高 PR = 方差均匀分布在多个维度（丰富表征）。
        低 PR = 方差集中在少数维度（低维/冗余表征）。
        """
        lam = eigenvalues[eigenvalues > 0]
        if len(lam) == 0:
            return 1.0
        return float(lam.sum()**2 / (lam**2).sum())

    def analyze_layer(
        self,
        features: np.ndarray,             # (N, D) 特征
        labels: Optional[np.ndarray],     # (N,) 退化类型标签
        layer_name: str = "layer",
        model_type: str = "snn",
        reference_features: Optional[np.ndarray] = None,  # 参考模型特征（用于 Procrustes）
    ) -> ManifoldStats:
        """
        分析单层特征的流形结构。

        Args:
            features: (N, D) 特征矩阵（N=样本数，D=特征维度）
            labels: (N,) 退化类型标签（可选，用于类间/类内距离）
            layer_name, model_type: 标签
            reference_features: (N, D) 参考特征（如 ANN 特征，用于 Procrustes 对齐）

        Returns:
            ManifoldStats
        """
        N, D = features.shape

        # ── PCA 分析 ─────────────────────────────────────────────────
        n_comp = min(self.n_pca, N-1, D)
        projected, evr = self._pca(features, n_comp)

        # 解释 90% 方差的维数
        cumvar = np.cumsum(evr)
        dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1

        # 获取奇异值用于有效秩和参与比
        X_c = features - features.mean(0)
        try:
            _, S, _ = np.linalg.svd(X_c, full_matrices=False)
        except Exception:
            S = np.ones(min(N, D))
        eff_rank = self._effective_rank(S)
        part_ratio = self._participation_ratio(S**2)
        total_var = float((S**2).sum())

        # ── 类间/类内可分性 ──────────────────────────────────────────
        inter_dist = intra_dist = fisher_ratio = 0.0
        if labels is not None and len(np.unique(labels)) > 1:
            classes = np.unique(labels)
            class_centers = {c: features[labels == c].mean(0) for c in classes}
            global_center = features.mean(0)

            # 类间距离（各类中心到全局中心的加权距离）
            inter_dists = [
                float(np.linalg.norm(class_centers[c] - global_center))
                for c in classes
            ]
            inter_dist = float(np.mean(inter_dists))

            # 类内距离（各样本到本类中心的平均距离）
            intra_dists = []
            for c in classes:
                mask = labels == c
                if mask.sum() > 1:
                    diff = features[mask] - class_centers[c]
                    intra_dists.append(float(np.linalg.norm(diff, axis=1).mean()))
            intra_dist = float(np.mean(intra_dists)) if intra_dists else 0.0
            fisher_ratio = inter_dist / (intra_dist + 1e-8)

        # ── Procrustes 对齐 ─────────────────────────────────────────
        proc_dist = 0.0
        if reference_features is not None:
            proc_dist = self._procrustes_distance(
                projected[:, :min(n_comp, reference_features.shape[1])],
                reference_features[:N, :min(n_comp, reference_features.shape[1])],
            )

        return ManifoldStats(
            layer_name=layer_name,
            model_type=model_type,
            n_samples=N, feature_dim=D,
            pca_dim_90=dim_90,
            effective_rank=eff_rank,
            participation_ratio=part_ratio,
            inter_class_dist=inter_dist,
            intra_class_dist=intra_dist,
            fisher_ratio=fisher_ratio,
            total_variance=total_var,
            pca_variance_ratio=evr[:min(10, len(evr))].tolist(),
            procrustes_distance=proc_dist,
        )

    def _procrustes_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        正交 Procrustes 距离（流形对齐度）。
        最小化 ||A - B·Q||_F，其中 Q 为正交矩阵。
        返回归一化残差 d = ||A - B·Q*||_F / ||A||_F ∈ [0, 1]。
        """
        min_n = min(A.shape[0], B.shape[0])
        Ac = A[:min_n] - A[:min_n].mean(0)
        Bc = B[:min_n] - B[:min_n].mean(0)

        # 归一化
        norm_A = np.linalg.norm(Ac) + 1e-8
        norm_B = np.linalg.norm(Bc) + 1e-8
        Ac = Ac / norm_A
        Bc = Bc / norm_B

        # SVD-based Procrustes
        try:
            M = Ac.T @ Bc
            U, _, Vt = np.linalg.svd(M)
            Q = Vt.T @ U.T
            residual = np.linalg.norm(Ac - Bc @ Q)
            return float(residual / (np.linalg.norm(Ac) + 1e-8))
        except Exception:
            return 1.0

    def compare_snn_ann(
        self,
        snn_feats: Dict[str, np.ndarray],
        ann_feats: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[ManifoldStats, ManifoldStats]]:
        """
        对比 SNN 和 ANN 各层的流形结构。

        Returns:
            {layer: (snn_stats, ann_stats)}
        """
        results = {}
        for layer in snn_feats:
            if layer not in ann_feats:
                continue
            snn_s = self.analyze_layer(
                snn_feats[layer], labels, layer, "snn",
                reference_features=ann_feats[layer]
            )
            ann_s = self.analyze_layer(
                ann_feats[layer], labels, layer, "ann",
            )
            results[layer] = (snn_s, ann_s)
        return results

    def print_comparison(
        self,
        comparison: Dict[str, Tuple[ManifoldStats, ManifoldStats]]
    ):
        print(f"\n{'='*85}")
        print("Feature Manifold Analysis: SNN vs ANN")
        print(f"{'Layer':<30} | {'SNN':^25} | {'ANN':^25}")
        header2 = f"{'':30} | {'EffRank':>9} {'Fisher':>8} {'PCA90':>6} | "
        header2 += f"{'EffRank':>9} {'Fisher':>8} {'PCA90':>6}"
        print(header2)
        print("-"*85)
        for layer, (snn_s, ann_s) in comparison.items():
            short = layer[:28]
            print(f"{short:<30} | "
                  f"{snn_s.effective_rank:>9.2f} {snn_s.fisher_ratio:>8.3f} "
                  f"{snn_s.pca_dim_90:>6} | "
                  f"{ann_s.effective_rank:>9.2f} {ann_s.fisher_ratio:>8.3f} "
                  f"{ann_s.pca_dim_90:>6}  "
                  f"Procrustes={snn_s.procrustes_distance:.3f}")
        print(f"{'='*85}")
        print("  EffRank: effective dimensionality of feature manifold")
        print("  Fisher: inter/intra-class separability")
        print("  Procrustes: SNN-ANN manifold alignment (0=identical, 1=different)")
