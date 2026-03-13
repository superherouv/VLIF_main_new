"""
互信息分析 (Mutual Information Analysis).

研究问题：
  SNN 各层的特征保留了多少关于"退化类型"和"退化强度"的信息？

应用：
  1. Information Bottleneck（信息瓶颈）分析：
     I(X; Z_l)：层 l 对输入的互信息（保留了多少输入信息）
     I(Z_l; Y)：层 l 对输出标签的互信息（保留了多少任务相关信息）
     理想的层 l 应 minimize I(X; Z_l) while maximize I(Z_l; Y)
     → SNN 的 IB 曲线 vs ANN 有什么不同？

  2. 退化类型分类互信息：
     SNN 特征能多好地区分"雨 vs 雾 vs 噪声"？
     → 高 MI = SNN 的脉冲编码保留了退化类型判别信息
     → 可用 kNN 分类器在中间层特征上测试

  3. 退化强度互信息：
     能从 SNN 中间层特征预测退化强度（雨密度、噪声 σ）？
     → MI(Z_l; sigma_noise) 衡量层 l 的退化强度感知能力

实现：
  - 使用 kNN 估计法（非参数 MI 估计，不需要概率密度假设）
  - MINE (MI Neural Estimator) 的简化版
  - 分类精度作为 MI 的代理指标
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MIResult:
    """互信息分析结果。"""
    layer_name: str
    mi_with_input: float       # I(Z_l; X)，特征对输入的 MI
    mi_with_output: float      # I(Z_l; Y)，特征对目标的 MI
    degradation_mi: float      # I(Z_l; degradation_type)，退化类型 MI
    clf_accuracy: float        # 用 kNN 在层特征上做退化类型分类的精度
    ib_efficiency: float       # MI(Y)/MI(X) = 信息利用效率（高=好）


class MutualInfoAnalyzer:
    """
    互信息分析器。

    使用 kNN 估计法：
      MI(X; Y) ≈ ψ(k) - <ψ(n_x)> - <ψ(n_y)> + ψ(n)
      其中 ψ = digamma 函数，n_x/n_y = 球内 X/Y 邻居数

    简化实现：用 kNN 分类精度作为 MI 的线性代理（精度越高 MI 越大）。
    """

    def __init__(self, k_neighbors: int = 5, n_bins: int = 30):
        self.k = k_neighbors
        self.n_bins = n_bins

    def _knn_classify(
        self,
        features: np.ndarray,   # (N, D)
        labels: np.ndarray,     # (N,) int
        k: int = None,
    ) -> float:
        """
        留一法 kNN 分类精度（作为 MI 的代理）。

        Returns:
            accuracy [0, 1]
        """
        k = k or self.k
        N = features.shape[0]
        if N < k + 1:
            return 0.0

        # 计算所有样本对的距离
        sq = np.sum(features**2, axis=1, keepdims=True)
        dist = sq + sq.T - 2 * features @ features.T  # (N, N)
        np.fill_diagonal(dist, np.inf)

        # k 近邻分类
        correct = 0
        for i in range(N):
            nn_idx = np.argsort(dist[i])[:k]
            nn_labels = labels[nn_idx]
            pred = np.bincount(nn_labels, minlength=int(labels.max())+1).argmax()
            if pred == labels[i]:
                correct += 1
        return correct / N

    def _binned_mi(
        self, x: np.ndarray, y: np.ndarray
    ) -> float:
        """
        基于直方图的 1D 互信息估计。
        用于连续变量（如退化强度）的 MI 估计。
        """
        x_norm = (x - x.min()) / (x.ptp() + 1e-8)
        y_norm = (y - y.min()) / (y.ptp() + 1e-8)
        bins = self.n_bins

        # 联合直方图
        hist_xy, _, _ = np.histogram2d(x_norm, y_norm, bins=bins)
        hist_x = hist_xy.sum(axis=1)
        hist_y = hist_xy.sum(axis=0)
        n = hist_xy.sum()

        if n == 0:
            return 0.0

        # 归一化为概率
        p_xy = hist_xy / n
        p_x  = hist_x / n
        p_y  = hist_y / n

        # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        return float(max(0.0, mi))

    def analyze_layer(
        self,
        features: np.ndarray,          # (N, D) 该层的特征（已展平）
        input_features: np.ndarray,    # (N, D_in) 输入特征
        target_features: np.ndarray,   # (N, D_out) 目标特征
        degradation_labels: np.ndarray,# (N,) int，退化类型标签
        degradation_strengths: Optional[np.ndarray] = None,  # (N,) float
        layer_name: str = "layer",
    ) -> MIResult:
        """
        分析单层的互信息。

        Args:
            features:          当前层特征 (N, D)
            input_features:    输入图像特征 (N, D_in)（展平的像素值）
            target_features:   目标图像特征 (N, D_out)
            degradation_labels: 退化类型标签（0=去雨, 1=去噪, ...）
            degradation_strengths: 退化强度（如噪声 σ）

        Returns:
            MIResult
        """
        # 降维（若特征维度太高，用 PCA 降到 50 维）
        def maybe_pca(X: np.ndarray, d: int = 50) -> np.ndarray:
            if X.shape[1] <= d:
                return X
            # 简单的 PCA via SVD
            X_c = X - X.mean(0)
            _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
            return X_c @ Vt[:d].T

        Z = maybe_pca(features)
        X_in = maybe_pca(input_features)
        Y_out = maybe_pca(target_features)

        # I(Z; X) 和 I(Z; Y) 用分类精度代理
        # 将连续特征二值化（高于均值 = 1，否则 = 0）后用 MI 估计
        def continuous_mi_proxy(A, B):
            """用 A 和 B 的第一主成分计算 1D MI。"""
            a = A @ np.linalg.svd(A, full_matrices=False)[2][0]  # 第一 PC
            b = B @ np.linalg.svd(B, full_matrices=False)[2][0]
            return self._binned_mi(a, b)

        mi_x = continuous_mi_proxy(Z, X_in)
        mi_y = continuous_mi_proxy(Z, Y_out)

        # I(Z; degradation_type) 用分类精度
        clf_acc = self._knn_classify(Z, degradation_labels)
        # 将精度转为 MI 的近似：MI ≈ -log(1 - acc + ε)
        n_classes = len(np.unique(degradation_labels))
        baseline_acc = 1.0 / n_classes  # 随机猜测精度
        if clf_acc > baseline_acc:
            deg_mi = -np.log(1.0 - (clf_acc - baseline_acc) / (1.0 - baseline_acc) + 1e-6)
        else:
            deg_mi = 0.0

        # 信息利用效率
        ib_eff = mi_y / (mi_x + 1e-8)

        return MIResult(
            layer_name=layer_name,
            mi_with_input=float(mi_x),
            mi_with_output=float(mi_y),
            degradation_mi=float(deg_mi),
            clf_accuracy=float(clf_acc),
            ib_efficiency=float(ib_eff),
        )

    def plot_information_plane(
        self,
        results: List[MIResult],
        model_type: str = "snn",
    ) -> Dict[str, List[float]]:
        """
        返回 Information Bottleneck 平面的坐标。
        x = I(Z; X)（压缩）, y = I(Z; Y)（预测）
        理想路径：随层数加深，x 减小 y 增大（压缩输入，保留任务信息）

        Returns:
            {'I_X': [...], 'I_Y': [...], 'layers': [...]}
        """
        return {
            "I_X": [r.mi_with_input for r in results],
            "I_Y": [r.mi_with_output for r in results],
            "layers": [r.layer_name for r in results],
            "model_type": model_type,
        }

    def print_ib_analysis(self, results: List[MIResult]):
        print(f"\n{'='*70}")
        print("Information Bottleneck Analysis")
        print(f"{'Layer':<30} {'I(Z;X)':>10} {'I(Z;Y)':>10} "
              f"{'DegMI':>8} {'ClfAcc':>8} {'IB_Eff':>8}")
        print("-"*70)
        for r in results:
            print(f"{r.layer_name:<30} {r.mi_with_input:>10.4f} "
                  f"{r.mi_with_output:>10.4f} "
                  f"{r.degradation_mi:>8.4f} "
                  f"{r.clf_accuracy:>8.4f} "
                  f"{r.ib_efficiency:>8.4f}")
        print(f"{'='*70}")
        print("  I(Z;X): info retained from input | I(Z;Y): info useful for output")
        print("  High IB_Eff = layer efficiently uses input info for task")
