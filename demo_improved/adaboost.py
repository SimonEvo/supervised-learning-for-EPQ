"""
AdaBoost 模块（支持二分类与多分类 SAMME）

二分类（n_classes=2）：标准 AdaBoost
  α_t = 0.5 · ln((1-ε_t)/ε_t)
  w_i ← w_i · exp(-α_t · y_i · h_t(x_i))

多分类（n_classes>2）：SAMME 算法（Stagewise Additive Modeling using Multi-class Exponential loss）
  α_t = ln((1-ε_t)/ε_t) + ln(K-1)    ← 多了 ln(K-1) 修正项
  w_i ← w_i · exp(α_t · 1[h_t(x_i) ≠ y_i])
  预测：H(x) = argmax_k Σ_t α_t · 1[h_t(x) = k]

  当 K=2 时，SAMME 退化为标准 AdaBoost（相差常数因子，归一化后等价）。
"""

import math
from decision_tree import build_tree, predict_tree


class AdaBoost:
    """
    AdaBoost 分类器（二分类与多分类统一实现）。

    参数：
      n_estimators : 弱分类器数量 T
      max_depth    : 每棵决策树的最大深度（1 = 决策桩）
      n_classes    : 类别数（2 = 二分类，4 = 四分类等）
    """

    def __init__(self, n_estimators=20, max_depth=1, n_classes=2):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.n_classes    = n_classes
        self.estimators   = []   # [(tree, alpha), ...]
        self.train_errors = []
        self.weak_errors  = []

    # ──────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        n = len(y)
        weights = [1.0 / n] * n

        self._print_theory()

        mode = "二分类" if self.n_classes == 2 else f"{self.n_classes}分类 (SAMME)"
        print("=" * 68)
        print(f"  训练 AdaBoost [{mode}]  T={self.n_estimators}, max_depth={self.max_depth}")
        print("=" * 68)
        print(f"  {'轮次':>4}  {'ε_t':>7}  {'α_t':>8}  {'集成误差':>8}  误差条形图                 权重范围")
        print("  " + "─" * 66)

        for t in range(self.n_estimators):
            # Step 1：用当前权重训练弱分类器
            tree = build_tree(X, y, weights, max_depth=self.max_depth)

            # Step 2：计算加权误差 ε_t
            preds   = predict_tree(tree, X)
            epsilon = sum(w for yi, pi, w in zip(y, preds, weights) if yi != pi)
            epsilon = max(1e-10, min(epsilon, 1 - 1e-10))

            # Step 3：计算弱分类器权重 α_t
            if self.n_classes == 2:
                # 标准 AdaBoost：α = 0.5 ln((1-ε)/ε)
                alpha = 0.5 * math.log((1 - epsilon) / epsilon)
            else:
                # SAMME：α = ln((1-ε)/ε) + ln(K-1)
                K     = self.n_classes
                alpha = math.log((1 - epsilon) / epsilon) + math.log(K - 1)

            # 若 alpha <= 0，该弱分类器不比随机猜测好，跳过
            if alpha <= 0:
                print(f"  轮次 {t+1:>2d}  ε={epsilon:.4f}  α={alpha:+.4f}  ← 弱分类器无效，跳过")
                continue

            # Step 4：更新样本权重
            if self.n_classes == 2:
                # 分对：exp(-α)；分错：exp(+α)
                new_weights = [wi * math.exp(-alpha * yi * pi)
                               for yi, pi, wi in zip(y, preds, weights)]
            else:
                # SAMME：分错样本权重乘以 exp(α)
                new_weights = [wi * math.exp(alpha * (0 if yi == pi else 1))
                               for yi, pi, wi in zip(y, preds, weights)]

            # Step 5：归一化
            total   = sum(new_weights)
            weights = [w / total for w in new_weights]

            self.estimators.append((tree, alpha))
            self.weak_errors.append(epsilon)

            # 计算当前集成训练误差
            ensemble_preds = self.predict(X)
            train_err = sum(1 for yi, pi in zip(y, ensemble_preds) if yi != pi) / n
            self.train_errors.append(train_err)

            self._print_round(t + 1, epsilon, alpha, train_err, weights)

        print("  " + "─" * 66)
        print("  训练完成！")
        print("=" * 68)

    # ──────────────────────────────────────────────────────────────────
    def predict(self, X):
        """
        集成预测：
          二分类：H(x) = sign(Σ α_t · h_t(x))
          多分类：H(x) = argmax_k Σ_t α_t · 1[h_t(x) = k]
        """
        if self.n_classes == 2:
            scores = [0.0] * len(X)
            for tree, alpha in self.estimators:
                preds = predict_tree(tree, X)
                for i, p in enumerate(preds):
                    scores[i] += alpha * p
            return [1 if s >= 0 else -1 for s in scores]
        else:
            K      = self.n_classes
            votes  = [[0.0] * K for _ in range(len(X))]
            for tree, alpha in self.estimators:
                preds = predict_tree(tree, X)
                for i, p in enumerate(preds):
                    votes[i][p] += alpha
            return [max(range(K), key=lambda k: votes[i][k]) for i in range(len(X))]

    # ──────────────────────────────────────────────────────────────────
    def _print_theory(self):
        if self.n_classes == 2:
            self._print_theory_binary()
        else:
            self._print_theory_samme()

    def _print_theory_binary(self):
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                为什么 AdaBoost 会有效？（理论基础）               ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print("║  1. 弱学习器定理                                                  ║")
        print("║     只要每轮加权误差 ε_t < 0.5（比随机猜测稍好），                  ║")
        print("║     AdaBoost 的训练误差就有指数级下降的保证：                        ║")
        print("║                                                                    ║")
        print("║       训练误差 ≤ ∏_t 2√(ε_t(1-ε_t)) = ∏_t exp(-2γ_t²)          ║")
        print('║       其中 γ_t = 0.5 - ε_t  （每轮超越随机猜测的"优势"）           ║')
        print("║                                                                    ║")
        print("║  2. 权重更新的直觉                                                  ║")
        print("║     分对的样本 → 权重降低（已学会，无需再强调）                       ║")
        print("║     分错的样本 → 权重升高（下一轮弱分类器被迫关注这些难例）            ║")
        print("║                                                                    ║")
        print('║  3. α_t 的含义：弱分类器的"话语权"                                  ║')
        print("║     α_t = 0.5 · ln((1-ε_t)/ε_t)                                  ║")
        print("║     ε_t → 0（很准）  → α_t 很大 → 话语权高                         ║")
        print("║     ε_t → 0.5（瞎猜）→ α_t → 0  → 几乎无话语权                    ║")
        print("║                                                                    ║")
        print("║  4. XOR 问题的关键：                                                ║")
        print("║     单棵深度=1的树（决策桩）只能做1次轴对齐切割，                    ║")
        print("║     在 XOR 数据上误差约 0.5。AdaBoost 将多个这样的                  ║")
        print('║     弱分类器按话语权叠加，可以逐渐"拼"出 XOR 边界。                  ║')
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    def _print_theory_samme(self):
        K = self.n_classes
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║           为什么 SAMME 可以做多分类？（理论基础）                  ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  多分类 AdaBoost：SAMME 算法（{K} 类）                               ║")
        print("║                                                                    ║")
        print("║  标准二分类 AdaBoost：                                              ║")
        print("║    α_t = 0.5 · ln((1-ε_t)/ε_t)                                   ║")
        print("║                                                                    ║")
        print("║  SAMME 的修正（多了一项 ln(K-1)）：                                 ║")
        print("║    α_t = ln((1-ε_t)/ε_t) + ln(K-1)                               ║")
        print("║                                                                    ║")
        print(f"║  为什么要加 ln(K-1)？                                               ║")
        print(f"║    K 类问题的随机基线误差 = (K-1)/K ≈ {(K-1)/K:.2f}（不是 0.5）      ║")
        print(f"║    α_t > 0 的条件变为 ε_t < (K-1)/K ≈ {(K-1)/K:.2f}                 ║")
        print(f"║    即只要比均匀随机猜测（准确率 = 1/K = {1/K:.2f}）好就行              ║")
        print("║                                                                    ║")
        print("║  预测：H(x) = argmax_k Σ_t α_t · 1[h_t(x) = k]                  ║")
        print('║    → 对每个类别累加"投给它"的弱分类器权重，取最大值                  ║')
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    def _print_round(self, t, epsilon, alpha, train_err, weights):
        max_w = max(weights)
        min_w = min(weights)
        bar   = "█" * int(train_err * 20) + "░" * (20 - int(train_err * 20))

        if epsilon < 0.15:
            quality = "强"
        elif epsilon < 0.35:
            quality = "中"
        else:
            quality = "弱"

        print(f"  {t:>4d}  ε={epsilon:.4f}({quality})  α={alpha:+.5f}  "
              f"{train_err:.4f}  [{bar}]  [{min_w:.5f}, {max_w:.5f}]")
