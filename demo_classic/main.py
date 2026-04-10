import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boosted_decision_tree import *
from adaBoost import *
from visualisation import *

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Boosted Decision Tree 教学演示(纯Python）            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. 生成数据
    n_samples = 500     # 随机生成样本量
    noise = 0.15         # 噪声比例，0 为无噪声，所有的数据都正确标注
    test_ratio = 0.2     # 测试集比例

    # 单棵决策树输入：
    max_depth_single = 3 # 单棵决策树最大层数

    # adaboost输入：
    n_estimators = 15    # 训练迭代次数
    max_depth_ada = 2        # 单棵决策树最大层数
    

    print("\n【Step 1】生成合成数据集")
    print("  决策边界:以原点为圆心、半径为 2 的圆(圆内 +1, 圆外 -1)")
    print(f"  样本数: {n_samples}, 噪声率: {noise*100}%, 训练/测试比例: 80/20")
    X, y = generate_dataset(n_samples, noise)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio)
    pos_train = sum(1 for yi in y_train if yi == 1)
    neg_train = len(y_train) - pos_train
    print(f"  训练集:{len(y_train)} 样本  (+1: {pos_train}, -1: {neg_train})")
    print(f"  测试集:{len(y_test)} 样本")

    # 2. 单棵决策树基线
    print("\n"+f"【Step 2】单棵决策树基线(等权重, max_depth={max_depth_single})")
    uniform_w = [1.0 / len(y_train)] * len(y_train)

    single_tree = build_tree(X_train, y_train, uniform_w, max_depth_single)
    st_train_preds = predict_tree(single_tree, X_train)
    st_test_preds  = predict_tree(single_tree, X_test)
    print(f"  训练准确率:{accuracy(y_train, st_train_preds):.4f}")
    print(f"  测试准确率:{accuracy(y_test,  st_test_preds):.4f}")
    print("\n  第一棵决策树结构(depth=3):")
    print_tree_structure(single_tree, feature_names=["x1", "x2"])

    # 3. AdaBoost 训练
    
    print("\n"+f"【Step 3】AdaBoost 训练 (T = {n_estimators}棵决策桩, max_depth = {max_depth_ada})")
    model = AdaBoost(n_estimators, max_depth_ada)
    model.fit(X_train, y_train)

    # 4. 测试集评估
    print("\n【Step 4】测试集评估")
    test_preds = model.predict(X_test)
    test_acc   = accuracy(y_test, test_preds)
    print(f"  AdaBoost 测试准确率:{test_acc:.4f}")
    TP, FP, TN, FN = confusion_matrix(y_test, test_preds)
    print_confusion_matrix(TP, FP, TN, FN)

    # # 5. 误差收敛曲线
    # print("\n【Step 5】误差收敛曲线(ASCII 可视化)")
    # plot_error_curve(model.train_errors, model.weak_errors)

    # 6. 展示第一棵弱分类器结构
    print("\n【Step 6】第一棵弱分类器(决策桩)结构")
    first_tree, first_alpha = model.estimators[0]
    print(f"  话语权 alpha = {first_alpha:.4f}(越大表示该弱分类器越重要)")
    print_tree_structure(first_tree, feature_names=["x1", "x2"])

    # 7. 各弱分类器话语权排行
    print("\n【Step 7】各弱分类器话语权 alpha 排行")
    alphas = [(i + 1, alpha) for i, (_, alpha) in enumerate(model.estimators)]
    alphas_sorted = sorted(alphas, key=lambda x: -x[1])
    print("  排名  轮次   alpha 值     | 可视化")
    print("  ─────────────────────┼─────────────────────────")
    max_alpha = max(a for _, a in alphas)
    for rank, (t, a) in enumerate(alphas_sorted, 1):
        bar = "▓" * int(a / max_alpha * 25)
        print(f"  #{rank:<3d}  轮次{t:>2d}  {a:+.4f}  | {bar}")


    #print_equations()
    
    #预测结果可视化
        # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Single Decision Tree vs AdaBoost  --  Test Set Predictions",
                 fontsize=14)

    plot_predictions(X_test, y_test, st_test_preds,
                     title=f"Single Decision Tree (max_depth={max_depth_single})", ax=axes[0])
    plot_predictions(X_test, y_test, test_preds,
                     title=f"AdaBoost  (T={n_estimators}, max_depth={max_depth_ada})",      ax=axes[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
