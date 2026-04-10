"""
可视化模块（改进版）

提供：
  plot_predictions()  - 散点图（支持 XOR 背景 + 二/多分类）
  plot_error_curve()  - 训练误差收敛曲线（matplotlib 折线图）
"""

import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang HK', 'STHeiti', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

# 四分类颜色方案
_COLORS4 = ["#2A7FD4", "#F28C38", "#3DB87E", "#9B59B6"]  # 蓝/橙/绿/紫
_QUAD_BG  = ["#EAF4FB", "#FFF3EC", "#EAFAF1", "#F5EEF8"]


def _xor_true_label(x1, x2, n_classes):
    """根据坐标计算 XOR 真实标签（用于背景着色）。"""
    if abs(x1) < 1e-9 or abs(x2) < 1e-9:
        return 0
    if n_classes == 2:
        return 1 if x1 * x2 > 0 else -1
    else:
        if   x1 > 0 and x2 > 0: return 0
        elif x1 < 0 and x2 > 0: return 1
        elif x1 < 0 and x2 < 0: return 2
        else:                    return 3


def plot_predictions(X, y_true, y_pred, title, n_classes=2, ax=None):
    """
    绘制预测结果散点图：
      正确分类：按类别着色
      误分类：红色 X 标记
      背景：XOR 真实决策区域（棋盘格着色）
      边界：坐标轴虚线（x₁=0, x₂=0）

    参数：
      n_classes : 2 = 二分类（标签 ±1），4 = 四分类（标签 0-3）
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    x_range = 2.15
    res = 300
    xx = np.linspace(-x_range, x_range, res)
    yy = np.linspace(-x_range, x_range, res)

    # 背景：XOR 真实分类区域着色
    if n_classes == 2:
        Z = np.array([[1.0 if xi * yi > 0 else -1.0 for xi in xx] for yi in yy])
        ax.contourf(xx, yy, Z, levels=[-2, 0, 2],
                    colors=["#FFF3EC", "#EAF4FB"], alpha=0.40)
    else:
        Z = np.array([[float(_xor_true_label(xi, yi, 4)) for xi in xx] for yi in yy])
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                    colors=_QUAD_BG, alpha=0.40)

    # 真实决策边界（坐标轴）
    ax.axhline(0, color="#444444", lw=1.5, ls="--", zorder=3,
               label="真实边界（x₁=0, x₂=0）")
    ax.axvline(0, color="#444444", lw=1.5, ls="--", zorder=3)

    # 散点绘制
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

    if n_classes == 2:
        correct_pos = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred) if yt == yp and yt ==  1]
        correct_neg = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred) if yt == yp and yt == -1]
        wrong       = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred) if yt != yp]

        if correct_pos:
            xs, ys = zip(*correct_pos)
            ax.scatter(xs, ys, c="#2A7FD4", s=40, alpha=0.80,
                       edgecolors="#1a5fa0", lw=0.5,
                       label=f"正确 +1（{len(correct_pos)}）", zorder=4)
        if correct_neg:
            xs, ys = zip(*correct_neg)
            ax.scatter(xs, ys, c="#F28C38", s=40, alpha=0.80,
                       edgecolors="#c06010", lw=0.5,
                       label=f"正确 -1（{len(correct_neg)}）", zorder=4)
        if wrong:
            xs, ys = zip(*wrong)
            ax.scatter(xs, ys, c="#E03030", s=70, marker="X", alpha=0.90,
                       edgecolors="#900000", lw=0.5,
                       label=f"误分（{len(wrong)}）", zorder=5)
    else:
        quad_names = ["Q1(+,+)", "Q2(-,+)", "Q3(-,-)", "Q4(+,-)"]
        for k in range(4):
            correct_k = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred) if yt == k and yp == k]
            wrong_k   = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred) if yt == k and yp != k]
            if correct_k:
                xs, ys = zip(*correct_k)
                ax.scatter(xs, ys, c=_COLORS4[k], s=40, alpha=0.80,
                           label=f"类{k} {quad_names[k]} 正确（{len(correct_k)}）", zorder=4)
            if wrong_k:
                xs, ys = zip(*wrong_k)
                ax.scatter(xs, ys, c=_COLORS4[k], s=70, marker="X", alpha=0.90,
                           label=f"类{k} 误分（{len(wrong_k)}）", zorder=5)

    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-x_range, x_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(f"{title}\n准确率 = {acc:.1%}  |  n = {len(y_true)}",
                 fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_error_curve(train_errors, weak_errors, ax=None):
    """
    绘制 AdaBoost 训练误差收敛曲线（matplotlib 版本）：
      蓝色实线：集成训练误差（每轮加入新弱分类器后）
      橙色虚线：弱分类器加权误差 ε_t
      灰色点线：随机猜测基线（ε = 0.5）
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    T      = len(train_errors)
    rounds = list(range(1, T + 1))

    ax.plot(rounds, train_errors, "o-",  color="#2A7FD4", lw=2,   ms=5,
            label="集成训练误差（所有弱分类器合并后）")
    ax.plot(rounds, weak_errors,  "s--", color="#F28C38", lw=1.5, ms=4,
            label="弱分类器加权误差 ε_t（单棵树）")
    ax.axhline(0.5, color="#888888", lw=1, ls=":", label="随机猜测基线（ε = 0.5）")

    ax.set_xlabel("迭代轮次 t", fontsize=12)
    ax.set_ylabel("误差率", fontsize=12)
    ax.set_title('AdaBoost 训练误差收敛曲线\n'
                 '集成误差随轮次增加而下降；弱分类器误差反映当前"难例"的难度',
                 fontsize=11)
    ax.set_ylim(0, min(1.0, max(0.65, max(weak_errors) + 0.08)))
    ax.set_xlim(0.5, T + 0.5)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax
