from boosted_decision_tree import *

class AdaBoost:
    """
    AdaBoost（Adaptive Boosting）分类器。

    核心思想：
      迭代训练 T 棵弱分类器（浅决策树），每轮：
        1. 用当前样本权重训练弱分类器
        2. 计算加权误差 ε
        3. 根据误差计算该弱分类器的话语权 α
        4. 更新样本权重：被误分的样本权重增大，分对的降低
        5. 重新归一化权重
      最终预测为所有弱分类器按 α 加权投票的结果。

    参数：
      n_estimators : 弱分类器数量 T
      max_depth    : 每棵决策树的最大深度（通常 1~3，即决策桩）
    """

    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.estimators   = []   # 弱分类器列表：(tree, alpha)
        self.train_errors = []   # 记录每轮集成后的训练误差
        self.weak_errors  = []   # 记录每轮弱分类器的加权误差 ε

    def fit(self, X, y):
        """
        训练 AdaBoost。

        AdaBoost 数学公式回顾：
          初始权重：w_i = 1/N
          第 t 轮：
            ε_t = Σ w_i * 1[h_t(x_i) != y_i]   （加权误差）
            α_t = 0.5 * ln((1 - ε_t) / ε_t)    （弱分类器权重）
            w_i <- w_i * exp(-α_t * y_i * h_t(x_i))
            归一化：w_i <- w_i / Σ w_i
        """
        n = len(y)
        # 初始化：每个样本权重相等
        weights = [1.0 / n] * n

        print("=" * 60)
        print(f"  开始训练 AdaBoost  (T={self.n_estimators}, max_depth={self.max_depth})")
        print("=" * 60)

        for t in range(self.n_estimators):
            # Step 1：用当前权重训练一棵弱分类器
            tree = build_tree(X, y, weights, max_depth=self.max_depth)

            # Step 2：计算弱分类器在训练集上的加权误差
            preds = predict_tree(tree, X)
            epsilon = sum(w for yi, pi, w in zip(y, preds, weights) if yi != pi)
            # 防止 epsilon=0 或 epsilon=1 导致 log(0) 的数值问题
            epsilon = max(1e-10, min(epsilon, 1 - 1e-10))

            # Step 3：计算该弱分类器的话语权 α
            alpha = 0.5 * math.log((1 - epsilon) / epsilon)

            # Step 4：更新样本权重
            #   正确分类：w_i * exp(-α)  权重降低
            #   错误分类：w_i * exp(+α)  权重升高（让下一棵树更关注难样本）
            new_weights = []
            for yi, pi, wi in zip(y, preds, weights):
                exponent = -alpha * yi * pi   # y_i * h(x_i)：分对为 +1，分错为 -1
                new_weights.append(wi * math.exp(exponent))

            # Step 5：归一化权重，使其总和为 1
            total = sum(new_weights)
            weights = [w / total for w in new_weights]

            # 保存弱分类器
            self.estimators.append((tree, alpha))
            self.weak_errors.append(epsilon)

            # 计算当前集成的训练误差（用于观察 Boosting 进展）
            ensemble_preds = self.predict(X)
            train_err = sum(1 for yi, pi in zip(y, ensemble_preds) if yi != pi) / n
            self.train_errors.append(train_err)

            # 打印本轮信息
            self._print_round(t + 1, epsilon, alpha, train_err, weights)

        print("=" * 60)
        print("  训练完成！")
        print("=" * 60)

    def predict(self, X):
        """
        集成预测：每棵弱分类器给出带权重的投票，取符号函数决定最终类别。

        H(x) = sign( Σ α_t * h_t(x) )
        """
        scores = [0.0] * len(X)
        for tree, alpha in self.estimators:
            preds = predict_tree(tree, X)
            for i, p in enumerate(preds):
                scores[i] += alpha * p
        # 取符号：正值 -> +1，负值 -> -1
        return [1 if s >= 0 else -1 for s in scores]

    def _print_round(self, t, epsilon, alpha, train_err, weights):
        """打印每轮训练的关键指标，辅助教学观察。"""
        max_w = max(weights)
        min_w = min(weights)
        bar_len = 20
        err_bar = "█" * int(train_err * bar_len) + "░" * (bar_len - int(train_err * bar_len))
        print(f"  轮次 {t:>2d} | ε={epsilon:.4f}  α={alpha:+.4f}  "
              f"训练误差={train_err:.4f} [{err_bar}]  "
              f"权重范围=[{min_w:.5f}, {max_w:.5f}]")
