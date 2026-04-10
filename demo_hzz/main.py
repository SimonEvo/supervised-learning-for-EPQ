"""
╔══════════════════════════════════════════════════════════════════════╗
║    BDT Demo 3 - H → τ_lep τ_had  信号/本底分类（纯 Python）          ║
║                                                                      ║
║  核心教学目标：                                                        ║
║    1. 单棵深决策树如何过拟合物理数据（训练好、测试差）                    ║
║    2. 过拟合曲线：depth vs 训练/测试准确率（bias-variance 权衡）         ║
║    3. AdaBoost 的泛化优势与特征重要性                                   ║
║    4. 决策边界：直观展示深树在 m_vis-MET 空间的过复杂边界               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset      import (generate_hzz_dataset, train_test_split,
                           FEATURE_NAMES, FEATURE_UNITS, FEAT_X, FEAT_Y)
from decision_tree import (build_tree, predict_tree, accuracy,
                            confusion_matrix, print_confusion_matrix,
                            print_tree_structure)
from adaboost     import AdaBoost
from visualisation import plot_main_figure
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  可调参数区
# ══════════════════════════════════════════════════════════════════════

N_SAMPLES      = 1000   # 总事例数（信号 + 本底）
SIG_FRACTION   = 0.3   # 信号占比（0.5 = 等量信号本底）
NOISE          = 0.05  # 标签噪声率（模拟重建误差 / mis-ID）
                #   试试 0, 0.05, 0.10, 0.20

TEST_RATIO     = 0.25  # 测试集比例

# ── 过拟合曲线扫描 ────────────────────────────────────────────────────
OVERFIT_MAX_DEPTH = 10  # 单树深度扫描上限（1 → 10）
                  #   增大 → 曲线更完整；减小 → 运行更快

# ── AdaBoost ─────────────────────────────────────────────────────────
N_ESTIMATORS  = 10     # 弱分类器数量 T（试试 5, 10, 20, 50）
MAX_DEPTH_ADA = 3      # 弱分类器深度（建议保持 1~2）

# ── 决策边界展示（三个单树深度 + AdaBoost）────────────────────────────
BOUNDARY_DEPTHS = [2, 5, 10]   # 在图上展示这些深度的单树边界

# ══════════════════════════════════════════════════════════════════════


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    BDT Demo 3 - H → τ_lep τ_had  信号/本底分类              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Step 1：生成数据 ───────────────────────────────────────────────
    print("\n【Step 1】蒙特卡洛生成 H → τ_lep τ_had 数据集")
    print(f"  事例数: {N_SAMPLES}  信号占比: {SIG_FRACTION*100:.0f}%  "
          f"标签噪声: {NOISE*100:.0f}%  测试集: {TEST_RATIO*100:.0f}%")
    print("  特征列表：")
    for i, (n, u) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS)):
        disc = "★ 强判别力" if i in (FEAT_X, FEAT_Y, 1) else "  弱判别力"
        print(f"    [{i}] {n:<12s} {('['+u+']') if u else '':6s}  {disc}")

    X, y = generate_hzz_dataset(N_SAMPLES, SIG_FRACTION, NOISE)
    X_train, y_train, X_test, y_test = train_test_split(X, y, TEST_RATIO)

    n_sig_tr = sum(1 for yi in y_train if yi ==  1)
    n_bkg_tr = sum(1 for yi in y_train if yi == -1)
    print(f"\n  训练集: {len(y_train)} 事例  (信号: {n_sig_tr}, 本底: {n_bkg_tr})")
    print(f"  测试集: {len(y_test)} 事例")

    uniform_w = [1.0 / len(y_train)] * len(y_train)

    # ── Step 2：过拟合曲线 ─────────────────────────────────────────────
    print(f"\n【Step 2】过拟合分析：单树深度 1 → {OVERFIT_MAX_DEPTH}")
    print("  （使用量化阈值加速，每个深度独立训练一棵树）")
    print()
    print(f"  {'深度':>5}  {'训练准确率':>10}  {'测试准确率':>10}  状态")
    print("  " + "─" * 44)

    overfit_depths = list(range(1, OVERFIT_MAX_DEPTH + 1))
    train_accs, test_accs = [], []
    boundary_trees = {}   # depth → tree（给决策边界图用）

    for d in overfit_depths:
        tree      = build_tree(X_train, y_train, uniform_w, d)
        tr_preds  = predict_tree(tree, X_train)
        te_preds  = predict_tree(tree, X_test)
        tr_acc    = accuracy(y_train, tr_preds)
        te_acc    = accuracy(y_test,  te_preds)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        if d in BOUNDARY_DEPTHS:
            boundary_trees[d] = tree

        gap   = tr_acc - te_acc
        state = ("⚠ 过拟合" if gap > 0.10 else
                 ("✓ 欠拟合" if tr_acc < 0.72 else "✓ 平衡"))
        print(f"  depth={d:>2d}   {tr_acc:.4f}      {te_acc:.4f}      {state}")

    best_depth = overfit_depths[test_accs.index(max(test_accs))]
    print(f"\n  单树最优深度: depth={best_depth}  测试准确率: {max(test_accs):.4f}")
    print(f"  depth={OVERFIT_MAX_DEPTH} 时：训练 {train_accs[-1]:.4f}  测试 {test_accs[-1]:.4f}"
          f"  过拟合幅度: {(train_accs[-1]-test_accs[-1])*100:+.1f}%")

    # ── Step 3：AdaBoost 训练 ──────────────────────────────────────────
    print(f"\n【Step 3】AdaBoost 训练 (T={N_ESTIMATORS}, max_depth={MAX_DEPTH_ADA})")
    model = AdaBoost(N_ESTIMATORS, MAX_DEPTH_ADA)
    model.fit(X_train, y_train)

    # ── Step 4：测试集评估 & 对比 ──────────────────────────────────────
    print("\n【Step 4】测试集评估")
    ada_test_preds = model.predict(X_test)
    ada_test_acc   = accuracy(y_test, ada_test_preds)
    ada_train_acc  = 1 - model.train_errors[-1]

    single_best_tree   = build_tree(X_train, y_train, uniform_w, best_depth)
    single_best_preds  = predict_tree(single_best_tree, X_test)
    single_best_acc    = accuracy(y_test, single_best_preds)

    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  方法                     训练准确率   测试准确率      │")
    print("  ├──────────────────────────────────────────────────────┤")
    single_best_train_acc = accuracy(y_train, predict_tree(single_best_tree, X_train))
    print(f"  │  单棵树 depth={best_depth} (最优)        "
          f"{single_best_train_acc:.4f}       {single_best_acc:.4f}         │")
    print(f"  │  单棵树 depth={OVERFIT_MAX_DEPTH} (过拟合)       "
          f"{train_accs[-1]:.4f}       {test_accs[-1]:.4f}         │")
    print(f"  │  AdaBoost (T={N_ESTIMATORS:>2d})             "
          f"{ada_train_acc:.4f}       {ada_test_acc:.4f}         │")
    print("  └──────────────────────────────────────────────────────┘")

    TP, FP, TN, FN = confusion_matrix(y_test, ada_test_preds)
    print("\n  AdaBoost 混淆矩阵（测试集）：")
    print_confusion_matrix(TP, FP, TN, FN)

    # ── Step 5：特征重要性 ─────────────────────────────────────────────
    print("\n【Step 5】AdaBoost 特征重要性（α 加权切割次数）")
    print("  重要性越高 → AdaBoost 越依赖该特征做判别")
    imp = model.feature_importance(len(FEATURE_NAMES))
    max_imp = max(imp)
    for i, (name, unit, v) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, imp)):
        bar  = "▓" * int(v / max_imp * 30)
        mark = "★" if i in (FEAT_X, FEAT_Y, 1) else " "
        print(f"  {mark} [{i}] {name:<12s}  {v:.4f}  {bar}")

    # ── 可视化 ─────────────────────────────────────────────────────────
    print("\n【Step 6】生成可视化图表...")

    # 为决策边界准备预测函数
    boundary_models = []
    for d in BOUNDARY_DEPTHS:
        if d not in boundary_trees:   # 万一参数设置里的深度超过扫描范围
            t = build_tree(X_train, y_train, uniform_w, d)
            boundary_trees[d] = t
        tree = boundary_trees[d]
        te   = accuracy(y_test, predict_tree(tree, X_test))

        # 用闭包捕获 tree（避免 lambda 晚绑定问题）
        def make_pred(t):
            return lambda X: predict_tree(t, X)

        boundary_models.append((
            make_pred(tree),
            f"单棵决策树  depth={d}\n测试 {te:.1%}"
        ))
    # AdaBoost
    boundary_models.append((
        model.predict,
        f"AdaBoost  T={N_ESTIMATORS}  depth={MAX_DEPTH_ADA}\n测试 {ada_test_acc:.1%}"
    ))

    plot_main_figure(
        X_train, y_train, X_test, y_test,
        boundary_models,
        overfit_depths, train_accs, test_accs,
        ada_train_acc, ada_test_acc,
        FEATURE_NAMES, FEATURE_UNITS,
        feat_x=FEAT_X, feat_y=FEAT_Y,
    )
    plt.show()


if __name__ == "__main__":
    main()
