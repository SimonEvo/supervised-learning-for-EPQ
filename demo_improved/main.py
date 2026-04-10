"""
╔══════════════════════════════════════════════════════════════════════╗
║    Boosted Decision Tree 进阶教学演示 - XOR 棋盘格问题（纯 Python）   ║
║                                                                      ║
║  核心教学目标：                                                        ║
║    1. 理解为何单棵决策树在 XOR 问题上存在根本局限                       ║
║    2. 理解 AdaBoost 为何有效（弱→强学习器的理论保证）                   ║
║    3. 观察调参（T、深度、噪声）对结果的影响                              ║
║                                                                      ║
║  数据集：XOR 棋盘格（蒙特卡洛采样）                                    ║
║    同号象限（第一/三）→ 正类；异号象限（第二/四）→ 负类                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset      import generate_xor_dataset, train_test_split
from decision_tree import (build_tree, predict_tree, accuracy,
                            confusion_matrix, print_confusion_matrix,
                            confusion_matrix_multiclass,
                            print_confusion_matrix_multi,
                            print_tree_structure)
from adaboost     import AdaBoost
from visualisation import plot_predictions, plot_error_curve
import matplotlib.pyplot as plt
import collections


# ══════════════════════════════════════════════════════════════════════
#  可调参数区 ← 建议课上与学生一起修改，观察效果变化
# ══════════════════════════════════════════════════════════════════════

N_SAMPLES   = 600   # 样本数（建议 200~1000）
             #   增大 → 结果更稳定；减小 → 运行更快

NOISE       = 0.10  # 噪声率（0 = 无噪声，0.5 = 完全随机）
             #   试试 0, 0.05, 0.10, 0.20, 0.30

TEST_RATIO  = 0.20  # 测试集比例

N_CLASSES   = 2     # 类别数: 2 = XOR 二分类（±1）
                    #        4 = 四象限多分类（启用 SAMME 算法）

# ── 单棵决策树参数 ──────────────────────────────────────────────────
MAX_DEPTH_SINGLE = 3  # 试试 1, 2, 3
                 #   depth=1：只能做 1 次切割，XOR 上约 50% 准确率
                 #   depth=2：最多 4 个区域，仍受轴对齐限制
                 #   depth=3：8 个区域，接近极限

# ── AdaBoost 参数 ────────────────────────────────────────────────────
N_ESTIMATORS  = 20  # 弱分类器数量 T（试试 5, 10, 20, 50）
              #   增大 → 集成误差继续下降，但边际收益递减
MAX_DEPTH_ADA = 2   # 弱分类器深度
              #   depth=1（决策桩）：在纯 XOR 上误差 ε ≈ 0.5，需要 T=100+ 才收敛
              #   depth=2（推荐）：ε 约 0.3~0.4，T=20 即可看到明显收敛
              #   试试 1/2/3 对比"弱分类器强度"对 Boosting 收敛速度的影响

# ══════════════════════════════════════════════════════════════════════


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    Boosted Decision Tree 进阶教学演示 -  XOR棋盘格问题           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Step 1：生成数据 ───────────────────────────────────────────────
    print("\n【Step 1】生成 XOR 棋盘格数据集（蒙特卡洛均匀采样）")
    if N_CLASSES == 2:
        print("  决策规则：x₁ · x₂ > 0（同号象限）→ +1，否则 → -1")
        print("  真实决策边界：坐标轴（x₁=0 与 x₂=0 两条直线）")
    else:
        print("  决策规则：四个象限分别对应类别 0/1/2/3")
    print(f"  样本数: {N_SAMPLES}  噪声率: {NOISE*100:.0f}%  "
          f"类别数: {N_CLASSES}  测试集: {TEST_RATIO*100:.0f}%")

    X, y = generate_xor_dataset(N_SAMPLES, NOISE, N_CLASSES)
    X_train, y_train, X_test, y_test = train_test_split(X, y, TEST_RATIO)

    cnt = collections.Counter(y_train)
    if N_CLASSES == 2:
        print(f"  训练集: {len(y_train)} 样本  (+1: {cnt[1]}, -1: {cnt[-1]})")
    else:
        dist = "  ".join(f"类{k}: {cnt[k]}" for k in sorted(cnt))
        print(f"  训练集: {len(y_train)} 样本  {dist}")
    print(f"  测试集: {len(y_test)} 样本")

    # ── Step 2：单棵决策树基线 ─────────────────────────────────────────
    print(f"\n【Step 2】单棵决策树基线（max_depth={MAX_DEPTH_SINGLE}）")
    print("  ★ XOR 的关键限制：")
    print('    决策树每次做"轴对齐切割"（平行于坐标轴的直线）。')
    print(f"    深度 {MAX_DEPTH_SINGLE} 的树最多做 {MAX_DEPTH_SINGLE} 次切割，")
    print("    但 XOR 的正负类分布在对角象限，轴对齐切割无法完美分开。")
    print("    → 深度=1 时：任何阈值切割都无法打破 XOR 对称性，准确率约 50%")
    print("    → 深度=2 时：可做 2 次切割，有所改善，但仍受轴对齐限制")

    uniform_w      = [1.0 / len(y_train)] * len(y_train)
    single_tree    = build_tree(X_train, y_train, uniform_w, MAX_DEPTH_SINGLE)
    st_train_preds = predict_tree(single_tree, X_train)
    st_test_preds  = predict_tree(single_tree, X_test)
    st_train_acc   = accuracy(y_train, st_train_preds)
    st_test_acc    = accuracy(y_test,  st_test_preds)
    print(f"\n  训练准确率: {st_train_acc:.4f}   测试准确率: {st_test_acc:.4f}")
    print("\n  决策树结构:")
    print_tree_structure(single_tree, feature_names=["x₁", "x₂"])

    # ── Step 3：AdaBoost 训练 ──────────────────────────────────────────
    mode_str = "SAMME 多分类" if N_CLASSES > 2 else "标准二分类"
    print(f"\n【Step 3】AdaBoost 训练（{mode_str}，T={N_ESTIMATORS}，"
          f"max_depth={MAX_DEPTH_ADA}）")

    model = AdaBoost(N_ESTIMATORS, MAX_DEPTH_ADA, N_CLASSES)
    model.fit(X_train, y_train)

    # ── Step 4：测试集评估与对比 ───────────────────────────────────────
    print("\n【Step 4】测试集评估 & 对比")
    test_preds  = model.predict(X_test)
    ada_test_acc = accuracy(y_test, test_preds)

    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  方法                  训练准确率   测试准确率     │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  单棵决策树 (d={MAX_DEPTH_SINGLE})          "
          f"{st_train_acc:.4f}       {st_test_acc:.4f}        │")
    ada_train_acc = 1 - model.train_errors[-1] if model.train_errors else 0
    print(f"  │  AdaBoost  (T={N_ESTIMATORS:>2d})          "
          f"{ada_train_acc:.4f}       {ada_test_acc:.4f}        │")
    print("  └──────────────────────────────────────────────────┘")
    improvement = (ada_test_acc - st_test_acc) * 100
    print(f"\n  AdaBoost 测试准确率提升: {improvement:+.2f} 个百分点")

    print()
    if N_CLASSES == 2:
        TP, FP, TN, FN = confusion_matrix(y_test, test_preds)
        print("  AdaBoost 混淆矩阵（测试集）：")
        print_confusion_matrix(TP, FP, TN, FN)
    else:
        cm = confusion_matrix_multiclass(y_test, test_preds, N_CLASSES)
        print("  AdaBoost 多分类混淆矩阵（测试集）：")
        print_confusion_matrix_multi(cm, N_CLASSES)

    # ── Step 5：弱分类器权重排行 ───────────────────────────────────────
    print("\n【Step 5】各弱分类器话语权 α 排行（前 10 位）")
    print("  α 越大 → 该轮弱分类器越准确 → 在最终投票中权重越高")
    alphas        = [(i+1, a) for i, (_, a) in enumerate(model.estimators)]
    alphas_sorted = sorted(alphas, key=lambda x: -x[1])
    max_alpha     = max(a for _, a in alphas)
    print("  排名  轮次    α 值       可视化（越长=话语权越高）")
    print("  ─────────────────────────────────────────────────")
    for rank, (t, a) in enumerate(alphas_sorted[:10], 1):
        bar = "▓" * int(a / max_alpha * 28)
        print(f"  #{rank:<3d}  第{t:>2d}轮  {a:+.5f}  {bar}")
    if len(alphas_sorted) > 10:
        print(f"  （共 {len(alphas_sorted)} 个弱分类器，仅显示前 10）")

    # ── Step 6：调参建议 ───────────────────────────────────────────────
    print("\n【Step 6】调参实验建议（修改文件顶部参数后重新运行）")
    print("  ● MAX_DEPTH_SINGLE = 1, 2, 3")
    print("    → 观察单棵树在 XOR 上的准确率极限，感受轴对齐切割的本质限制")
    print("  ● N_ESTIMATORS = 5, 10, 20, 50")
    print('    → 看误差曲线如何随 T 增大而收敛，体会"越多弱分类器越好"的规律')
    print("  ● NOISE = 0, 0.05, 0.10, 0.20, 0.30")
    print("    → 噪声增大时两种方法的准确率如何变化？谁更鲁棒？")
    print("  ● N_CLASSES = 4")
    print("    → 切换 SAMME 多分类模式，观察算法如何推广到 4 个类别")
    print("  ● MAX_DEPTH_ADA = 1, 2, 3")
    print("    → 弱分类器越深收敛越快，但每棵树更复杂；")
    print("       当 depth 很大时，接近单棵深树，Boosting 优势消失")

    # ── 可视化 ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"XOR 棋盘格  {'二分类' if N_CLASSES==2 else '四分类 (SAMME)'}  "
        f"噪声={NOISE*100:.0f}%   "
        f"单棵决策树(d={MAX_DEPTH_SINGLE}) vs AdaBoost(T={N_ESTIMATORS}, d={MAX_DEPTH_ADA})",
        fontsize=13
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    plot_predictions(X_test, y_test, st_test_preds,
                     title=f"单棵决策树（max_depth={MAX_DEPTH_SINGLE}）",
                     n_classes=N_CLASSES, ax=ax1)
    plot_predictions(X_test, y_test, test_preds,
                     title=f"AdaBoost（T={N_ESTIMATORS}, max_depth={MAX_DEPTH_ADA}）",
                     n_classes=N_CLASSES, ax=ax2)
    plot_error_curve(model.train_errors, model.weak_errors, ax=ax3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
