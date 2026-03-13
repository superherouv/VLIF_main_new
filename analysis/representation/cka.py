"""
Centered Kernel Alignment (CKA) 表征相似性分析.

CKA 是比较神经网络层间表征相似性的标准工具（Kornblith et al., 2019）。
在 SNN 研究中的应用：

  1. SNN 各层 vs ANN 对应层：
     量化"相同架构的 SNN 和 ANN 学到了多相似的特征"
     → 高 CKA = SNN 发现了与 ANN 等价的表征
     → 低 CKA = SNN 学到了本质不同的表征（可能更好也可能更差）

  2. SNN 层间 CKA 矩阵：
     揭示特征转换的层次结构
     → 是否存在"脉冲瓶颈"层（与相邻层相似度骤降）？

  3. 同一 SNN 在不同任务上的 CKA：
     去雨和去雾的 SNN 特征是否相似？
     → 相似 = 退化检测有共通的脉冲编码机制
     → 不相似 = 任务特异性脉冲表征

  4. 不同 T 值的 SNN（T=1 vs T=4 vs T=8）的 CKA：
     量化时间步数对特征质量的影响

CKA 定义：
  CKA(X, Y) = HSIC(X, Y) / √(HSIC(X, X) · HSIC(Y, Y))
  其中 HSIC 是 Hilbert-Schmidt 独立性准则
  线性 CKA：kernel = XX^T（特征内积）

  CKA 范围 [0, 1]，1 = 完全相同，0 = 完全无关

参考：
  Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CKAResult:
    """CKA 分析结果。"""
    layer_a: str
    layer_b: str
    cka_score: float            # [0,1]
    hsic_ab: float
    hsic_aa: float
    hsic_bb: float
    n_samples: int


@dataclass
class CKAMatrix:
    """层间 CKA 相似性矩阵。"""
    layer_names: List[str]
    matrix: np.ndarray          # (n_layers, n_layers)
    model_type_a: str
    model_type_b: str           # 同一模型时 = model_type_a

    # 派生指标
    mean_cross_similarity: float = 0.0   # SNN vs ANN 跨模型平均 CKA
    diagonal_similarity: float = 0.0     # 对应层 CKA 均值
    bottleneck_layer: str = ""           # 与其他层相似度最低的层（表征瓶颈）


class CKAAnalyzer:
    """
    CKA 分析器：比较 SNN/ANN/Hybrid 各层的特征表征相似性。

    用法：
        analyzer = CKAAnalyzer()

        # 添加各层特征（前向传播时收集）
        # features: {layer_name: (N_samples, C*H*W) 特征矩阵}
        snn_feats = {"enc_0": ..., "bottleneck": ..., "dec_0": ...}
        ann_feats = {"enc_0": ..., "bottleneck": ..., "dec_0": ...}

        # 计算跨模型 CKA
        matrix = analyzer.cross_model_cka(snn_feats, ann_feats, "snn", "ann")
        analyzer.print_cka_matrix(matrix)

        # 计算单模型层间 CKA
        within = analyzer.within_model_cka(snn_feats, "snn")
    """

    def __init__(self, use_rbf: bool = False, rbf_sigma: float = 1.0):
        """
        Args:
            use_rbf: 若 True 使用 RBF kernel，否则使用线性 kernel。
                     线性 kernel 更快且通常足够。
            rbf_sigma: RBF kernel 的带宽（仅 use_rbf=True 时生效）。
        """
        self.use_rbf = use_rbf
        self.rbf_sigma = rbf_sigma

    def _center(self, K: np.ndarray) -> np.ndarray:
        """双中心化（行列均值归零）。"""
        n = K.shape[0]
        H = np.eye(n) - 1.0 / n
        return H @ K @ H

    def _linear_kernel(self, X: np.ndarray) -> np.ndarray:
        """线性 kernel：K = XX^T。"""
        return X @ X.T

    def _rbf_kernel(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """RBF kernel：K_{ij} = exp(-||x_i - x_j||² / (2σ²))。"""
        sq_dist = np.sum(X**2, axis=1, keepdims=True)
        sq_dist_mat = sq_dist + sq_dist.T - 2 * X @ X.T
        return np.exp(-sq_dist_mat / (2 * sigma**2))

    def _hsic(self, KX: np.ndarray, KY: np.ndarray) -> float:
        """
        无偏 HSIC 估计（Gretton et al. 2005，Song et al. 2012 无偏版本）。
        """
        n = KX.shape[0]
        KX_c = self._center(KX)
        KY_c = self._center(KY)
        return float(np.trace(KX_c @ KY_c) / (n - 1)**2)

    def compute_cka(
        self,
        X: np.ndarray,  # (N, D_x) 特征矩阵
        Y: np.ndarray,  # (N, D_y) 特征矩阵
        layer_a: str = "A",
        layer_b: str = "B",
    ) -> CKAResult:
        """
        计算两组特征的 CKA 相似性。

        Args:
            X: (N, D) 层 A 的特征（N = 样本数，D = 特征维度）
            Y: (N, D) 层 B 的特征（N 必须相同）

        Returns:
            CKAResult
        """
        assert X.shape[0] == Y.shape[0], "样本数必须一致"

        # 标准化（每个特征减均值除标准差）
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

        # Kernel 计算
        if self.use_rbf:
            KX = self._rbf_kernel(X, self.rbf_sigma)
            KY = self._rbf_kernel(Y, self.rbf_sigma)
        else:
            KX = self._linear_kernel(X)
            KY = self._linear_kernel(Y)

        hsic_ab = self._hsic(KX, KY)
        hsic_aa = self._hsic(KX, KX)
        hsic_bb = self._hsic(KY, KY)

        cka = hsic_ab / (np.sqrt(hsic_aa * hsic_bb) + 1e-12)

        return CKAResult(
            layer_a=layer_a, layer_b=layer_b,
            cka_score=float(np.clip(cka, 0.0, 1.0)),
            hsic_ab=float(hsic_ab),
            hsic_aa=float(hsic_aa),
            hsic_bb=float(hsic_bb),
            n_samples=X.shape[0],
        )

    def cross_model_cka(
        self,
        feats_a: Dict[str, np.ndarray],   # {layer: (N,D)} SNN 特征
        feats_b: Dict[str, np.ndarray],   # {layer: (N,D)} ANN 特征
        model_type_a: str = "snn",
        model_type_b: str = "ann",
    ) -> CKAMatrix:
        """
        计算两个模型所有层对之间的 CKA 矩阵。

        Returns:
            CKAMatrix (n_layers_a × n_layers_b)
        """
        layers_a = list(feats_a.keys())
        layers_b = list(feats_b.keys())
        all_layers = layers_a  # 假设层名相同（跨模型对比）

        n = len(all_layers)
        matrix = np.zeros((n, n))

        for i, la in enumerate(layers_a):
            for j, lb in enumerate(layers_b):
                if la in feats_a and lb in feats_b:
                    # 确保样本数一致
                    Xa = feats_a[la]
                    Xb = feats_b[lb]
                    min_n = min(Xa.shape[0], Xb.shape[0])
                    result = self.compute_cka(Xa[:min_n], Xb[:min_n], la, lb)
                    matrix[i, j] = result.cka_score

        cka_mat = CKAMatrix(
            layer_names=all_layers,
            matrix=matrix,
            model_type_a=model_type_a,
            model_type_b=model_type_b,
        )

        # 派生指标
        diag = np.diag(matrix) if n == len(layers_b) else np.array([0.0])
        cka_mat.diagonal_similarity = float(diag.mean())
        cka_mat.mean_cross_similarity = float(matrix.mean())

        # 瓶颈层：与其他层相似度最低的层
        row_means = matrix.mean(axis=1)
        if len(row_means) > 0:
            cka_mat.bottleneck_layer = all_layers[int(np.argmin(row_means))]

        return cka_mat

    def within_model_cka(
        self,
        feats: Dict[str, np.ndarray],
        model_type: str = "snn",
    ) -> CKAMatrix:
        """
        计算同一模型各层之间的 CKA（层间相似度矩阵）。
        揭示特征转换的层次结构和瓶颈位置。
        """
        return self.cross_model_cka(feats, feats, model_type, model_type)

    def print_cka_matrix(self, cka_mat: CKAMatrix):
        """以 ASCII 热图形式打印 CKA 矩阵。"""
        n = len(cka_mat.layer_names)
        M = cka_mat.matrix
        print(f"\n{'='*60}")
        print(f"CKA Matrix: {cka_mat.model_type_a} vs {cka_mat.model_type_b}")
        print(f"Diagonal CKA (corresponding layers): "
              f"{cka_mat.diagonal_similarity:.4f}")
        print(f"Bottleneck layer: {cka_mat.bottleneck_layer}")
        print(f"{'='*60}")

        # 短名
        short = [l.split(".")[-1][:8] for l in cka_mat.layer_names]
        header = f"{'':>10}" + "".join(f"{s:>8}" for s in short)
        print(header)
        for i, (name, row) in enumerate(zip(short, M)):
            cells = ""
            for v in row:
                # ASCII 颜色提示
                if v > 0.8:   c = "██"
                elif v > 0.6: c = "▓▓"
                elif v > 0.4: c = "▒▒"
                elif v > 0.2: c = "░░"
                else:         c = "  "
                cells += f"  {c}  "
            print(f"{name:>10}{cells}  (mean={row.mean():.2f})")
        print(f"{'='*60}")
        print("  ██ >0.8  ▓▓ >0.6  ▒▒ >0.4  ░░ >0.2    <0.2")
