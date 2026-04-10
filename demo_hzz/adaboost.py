"""
AdaBoost 模块（二分类，信号 +1 / 本底 -1）

α_t = 0.5 · ln((1-ε_t)/ε_t)
w_i ← w_i · exp(-α_t · y_i · h_t(x_i))，归一化
H(x) = sign(Σ α_t · h_t(x))
"""

import math
from decision_tree import build_tree, predict_tree


class AdaBoost:
    def __init__(self, n_estimators=20, max_depth=2):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.estimators   = []
        self.train_errors = []
        self.weak_errors  = []

    def fit(self, X, y):
        n = len(y)
        weights = [1.0 / n] * n

        self._print_theory()
        print("=" * 68)
        print(f"  训练 AdaBoost  T={self.n_estimators}, max_depth={self.max_depth}")
        print("=" * 68)
        print(f"  {'轮次':>4}  {'ε_t':>8}  {'α_t':>9}  集成误差   进度条                  权重范围")
        print("  " + "─" * 66)

        for t in range(self.n_estimators):
            tree    = build_tree(X, y, weights, self.max_depth)
            preds   = predict_tree(tree, X)
            epsilon = sum(w for yi, pi, w in zip(y, preds, weights) if yi != pi)
            epsilon = max(1e-10, min(epsilon, 1 - 1e-10))
            alpha   = 0.5 * math.log((1 - epsilon) / epsilon)

            new_w = [wi * math.exp(-alpha * yi * pi)
                     for yi, pi, wi in zip(y, preds, weights)]
            total   = sum(new_w)
            weights = [w / total for w in new_w]

            self.estimators.append((tree, alpha))
            self.weak_errors.append(epsilon)

            ens_preds = self.predict(X)
            train_err = sum(1 for yi, pi in zip(y, ens_preds) if yi != pi) / n
            self.train_errors.append(train_err)

            self._print_round(t + 1, epsilon, alpha, train_err, weights)

        print("  " + "─" * 66)
        print("  训练完成！")
        print("=" * 68)

    def predict(self, X):
        scores = [0.0] * len(X)
        for tree, alpha in self.estimators:
            for i, p in enumerate(predict_tree(tree, X)):
                scores[i] += alpha * p
        return [1 if s >= 0 else -1 for s in scores]

    # ── 特征重要性（α 加权的特征使用次数）──────────────────────────────
    def feature_importance(self, n_features):
        imp = [0.0] * n_features

        def _walk(node, alpha):
            if node.is_leaf():
                return
            imp[node.feature_idx] += alpha
            _walk(node.left,  alpha)
            _walk(node.right, alpha)

        for tree, alpha in self.estimators:
            _walk(tree, alpha)

        total = sum(imp)
        return [v / total for v in imp] if total else imp

    # ── 打印 ───────────────────────────────────────────────────────────
    def _print_theory(self):
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║              为什么 AdaBoost 比单棵深树泛化能力更强？              ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║  单棵深树的过拟合机制：                                            ║")
        print("║    深度越大 → 叶节点越多 → 每个叶节点训练样本越少                   ║")
        print('║    → 开始学习训练集中的"噪声特征"（eta_lep / delta_phi 等）         ║')
        print("║    → 训练误差 → 0，但测试误差升高                                  ║")
        print("║                                                                    ║")
        print("║  AdaBoost 的泛化优势：                                             ║")
        print("║    每棵弱分类器（浅树）只能捕捉简单的物理规律                        ║")
        print("║    → 无法记忆单个训练事例（方差小）                                 ║")
        print('║    权重更新使后续弱分类器关注"难例"（偏差也小）                      ║')
        print("║    理论保证：训练误差上界 ≤ ∏_t exp(-2γ_t²)（指数下降）            ║")
        print("║                                                                    ║")
        print('║  直觉：10 个只懂一点物理的"弱专家"                                  ║')
        print('║        投票结果 >> 一个死记硬背训练数据的"强记忆者"                  ║')
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    def _print_round(self, t, epsilon, alpha, train_err, weights):
        bar  = "█" * int(train_err * 20) + "░" * (20 - int(train_err * 20))
        qual = "强" if epsilon < 0.35 else ("中" if epsilon < 0.45 else "弱")
        print(f"  {t:>4d}  ε={epsilon:.4f}({qual})  α={alpha:+.5f}  "
              f"{train_err:.4f}  [{bar}]  "
              f"[{min(weights):.5f}, {max(weights):.5f}]")
