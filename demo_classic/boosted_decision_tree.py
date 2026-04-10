"""
╔══════════════════════════════════════════════════════════════════════╗
║           Boosted Decision Tree - 从零实现教学版                       ║
║  本文件不依赖任何第三方 ML 库（无 sklearn / xgboost / lightgbm），        ║
║  仅使用 Python 标准库 + math / random / collections。                  ║
║  算法流程:                                                            ║
║    1. 生成可靠的合成数据集（二分类，带噪声）                               ║
║    2. 实现单棵决策树（CART，基尼系数分裂）                                ║
║    3. 实现 AdaBoost（自适应提升）将弱树组合成强分类器                      ║
║    4. 训练 / 测试 / 可视化每一步的权重与误差变化                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import random
import math
import collections

# ──────────────────────────────────────────────────────────────────────
# 0. 全局随机种子（保证结果可复现）
# ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════
# 第一部分：数据生成
# ══════════════════════════════════════════════════════════════════════

def generate_dataset(n_samples=200, noise=0.15):
    """
    生成二维二分类数据集。

    规则：
      - 特征 x1, x2 均匀分布在 [-3, 3]
      - 真实决策边界：若 x1^2 + x2^2 < 4 则标签为 +1，否则为 -1
        （以原点为圆心、半径为 2 的圆形边界）
      - 以概率 noise 随机翻转标签，模拟真实数据中的噪声

    返回：
      X : list of [x1, x2]   特征矩阵
      y : list of +1 or -1   标签向量（AdaBoost 要求标签为 ±1）
    """
    X, y = [], []
    for _ in range(n_samples):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        # 圆形决策边界
        label = 1 if (x1 ** 2 + x2 ** 2) < 4 else -1
        # 添加噪声
        if random.random() < noise:
            label = -label
        X.append([x1, x2])
        y.append(label)
    return X, y


def train_test_split(X, y, test_ratio=0.25):
    """
    将数据按比例划分为训练集和测试集（简单随机打乱）。
    """
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train = data[:split]
    test  = data[split:]
    X_train, y_train = zip(*train)
    X_test,  y_test  = zip(*test)
    return list(X_train), list(y_train), list(X_test), list(y_test)


# ══════════════════════════════════════════════════════════════════════
# 第二部分：决策树节点 & 构建
# ══════════════════════════════════════════════════════════════════════

class Node:
    """
    决策树的单个节点。

    叶节点：存储最终预测值（+1 或 -1）。
    内部节点：存储分裂特征索引、分裂阈值，以及左右子树指针。
    """
    def __init__(self):
        self.feature_idx = None   # 用于分裂的特征列索引
        self.threshold   = None   # 分裂阈值：x[feature_idx] <= threshold -> 左子树
        self.left        = None   # 左子树（条件成立）
        self.right       = None   # 右子树（条件不成立）
        self.value       = None   # 叶节点的预测值（仅叶节点有效）

    def is_leaf(self):
        return self.value is not None


def weighted_gini(labels, weights):
    """
    计算带权重的基尼不纯度（Weighted Gini Impurity）。

    Gini = 1 - Σ p_k^2，其中 p_k 为第 k 类的加权占比。

    AdaBoost 使用样本权重，所以这里用权重代替简单计数。

    参数：
      labels  : 标签列表（+1 / -1）
      weights : 对应的样本权重列表

    返回：
      float，基尼不纯度（越小越纯）
    """
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    # 统计每个类别的加权频率
    class_weights = collections.defaultdict(float)
    for lbl, w in zip(labels, weights):
        class_weights[lbl] += w
    gini = 1.0 - sum((cw / total_w) ** 2 for cw in class_weights.values())
    return gini


def best_split(X, y, weights):
    """
    遍历所有特征和所有候选阈值，找到使加权基尼增益最大的分裂点。

    策略：
      对每个特征，将该特征的所有唯一值排序后，取相邻两值的中点作为候选阈值。
      计算分裂后左右子节点的加权基尼不纯度之和（加权平均），
      选择使父节点基尼不纯度减少最多的分裂。

    返回：
      (best_feat, best_thresh)  或  (None, None) 如果无法分裂
    """
    n_features = len(X[0])
    parent_gini = weighted_gini(y, weights)
    total_w = sum(weights)

    best_gain   = -1
    best_feat   = None
    best_thresh = None

    for feat in range(n_features):
        # 提取该特征的所有值，并去重排序
        values = sorted(set(x[feat] for x in X))
        # 候选阈值：相邻值的中点
        thresholds = [(values[i] + values[i + 1]) / 2
                      for i in range(len(values) - 1)]

        for thresh in thresholds:
            # 按阈值分组
            left_y,  left_w  = [], []
            right_y, right_w = [], []
            for xi, yi, wi in zip(X, y, weights):
                if xi[feat] <= thresh:
                    left_y.append(yi);  left_w.append(wi)
                else:
                    right_y.append(yi); right_w.append(wi)

            if not left_y or not right_y:
                continue  # 跳过空分裂

            w_left  = sum(left_w)
            w_right = sum(right_w)

            # 加权平均基尼不纯度
            child_gini = (w_left  / total_w) * weighted_gini(left_y,  left_w) + \
                         (w_right / total_w) * weighted_gini(right_y, right_w)

            gain = parent_gini - child_gini

            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh

    return best_feat, best_thresh


def majority_vote(y, weights):
    """
    加权多数投票：返回加权权重之和最大的类别。
    用于确定叶节点的预测值。
    """
    class_weights = collections.defaultdict(float)
    for lbl, w in zip(y, weights):
        class_weights[lbl] += w
    return max(class_weights, key=class_weights.get)


def build_tree(X, y, weights, max_depth, depth=0):
    """
    递归构建 CART 决策树。

    停止条件：
      1. 达到最大深度（max_depth）
      2. 当前节点所有样本属于同一类别
      3. 找不到有效分裂点

    参数：
      X         : 当前节点的样本特征
      y         : 对应标签
      weights   : 对应样本权重（来自 AdaBoost）
      max_depth : 树的最大深度
      depth     : 当前递归深度

    返回：
      Node 对象（树的根节点）
    """
    node = Node()

    # 停止条件 1：到达最大深度，创建叶节点
    if depth >= max_depth:
        node.value = majority_vote(y, weights)
        return node

    # 停止条件 2：纯节点
    if len(set(y)) == 1:
        node.value = y[0]
        return node

    # 寻找最佳分裂
    feat, thresh = best_split(X, y, weights)

    # 停止条件 3：无法分裂
    if feat is None:
        node.value = majority_vote(y, weights)
        return node

    # 按最佳分裂划分数据
    left_X, left_y, left_w    = [], [], []
    right_X, right_y, right_w = [], [], []
    for xi, yi, wi in zip(X, y, weights):
        if xi[feat] <= thresh:
            left_X.append(xi);  left_y.append(yi);  left_w.append(wi)
        else:
            right_X.append(xi); right_y.append(yi); right_w.append(wi)

    # 记录当前节点的分裂信息
    node.feature_idx = feat
    node.threshold   = thresh

    # 递归构建子树
    node.left  = build_tree(left_X,  left_y,  left_w,  max_depth, depth + 1)
    node.right = build_tree(right_X, right_y, right_w, max_depth, depth + 1)

    return node


def predict_single(node, x):
    """
    对单个样本 x 沿树向下遍历，返回叶节点的预测值。
    """
    if node.is_leaf():
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def predict_tree(tree, X):
    """对数据集 X 中的每个样本进行预测，返回预测标签列表。"""
    return [predict_single(tree, x) for x in X]


# ══════════════════════════════════════════════════════════════════════
# 第三部分：AdaBoost 算法
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# 第四部分：评估与可视化
# ══════════════════════════════════════════════════════════════════════

def accuracy(y_true, y_pred):
    """计算准确率。"""
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵（针对二分类 ±1）。
    返回：TP, FP, TN, FN
    """
    TP = sum(1 for a, b in zip(y_true, y_pred) if a == 1  and b == 1)
    FP = sum(1 for a, b in zip(y_true, y_pred) if a == -1 and b == 1)
    TN = sum(1 for a, b in zip(y_true, y_pred) if a == -1 and b == -1)
    FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1  and b == -1)
    return TP, FP, TN, FN


def print_confusion_matrix(TP, FP, TN, FN):
    """以表格形式打印混淆矩阵。"""
    print()
    print("  混淆矩阵（行=真实标签，列=预测标签）：")
    print("  ┌─────────────┬──────────┬──────────┐")
    print("  │             │ 预测 +1  │ 预测 -1  │")
    print("  ├─────────────┼──────────┼──────────┤")
    print(f"  │ 真实 +1     │  TP={TP:>4d}  │  FN={FN:>4d}  │")
    print(f"  │ 真实 -1     │  FP={FP:>4d}  │  TN={TN:>4d}  │")
    print("  └─────────────┴──────────┴──────────┘")
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  精确率 Precision = {precision:.4f}")
    print(f"  召回率 Recall    = {recall:.4f}")
    print(f"  F1 分数          = {f1:.4f}")


def plot_error_curve(train_errors, weak_errors):
    """
    在终端用 ASCII 字符绘制训练误差曲线，直观展示 Boosting 的收敛过程。
    """
    height = 15
    width  = len(train_errors)
    max_e  = max(max(train_errors), max(weak_errors), 0.5)

    print()
    print("  ── AdaBoost 误差收敛曲线 ─────────────────────────────────")
    print(f"  纵轴：误差率 (0 ~ {max_e:.2f})  横轴：迭代轮次")
    print("  ● 集成训练误差   ○ 弱分类器加权误差 ε")
    print()

    # 构建网格
    grid = [['  ' for _ in range(width)] for _ in range(height)]

    for t_idx, (te, we) in enumerate(zip(train_errors, weak_errors)):
        row_te = height - 1 - int(te / max_e * (height - 1))
        row_we = height - 1 - int(we / max_e * (height - 1))
        row_te = max(0, min(height - 1, row_te))
        row_we = max(0, min(height - 1, row_we))
        grid[row_te][t_idx] = '● '
        if grid[row_we][t_idx] == '  ':
            grid[row_we][t_idx] = '○ '
        else:
            grid[row_we][t_idx] = '◉ '

    for r, row in enumerate(grid):
        label_val = max_e * (height - 1 - r) / (height - 1)
        print(f"  {label_val:4.2f} | {''.join(row)}")
    print("       └" + "─" * (width * 2 + 2))
    x_labels = "         "
    for i in range(1, width + 1):
        x_labels += f"{i:<2d}"
    print(x_labels)


def print_tree_structure(node, feature_names=None, depth=0, prefix="根"):
    """
    递归打印决策树结构，用于教学展示单棵弱分类器的内部逻辑。
    """
    indent = "    " * depth
    if node.is_leaf():
        label_str = "+1 (正类)" if node.value == 1 else "-1 (负类)"
        print(f"  {indent}└─[叶] 预测：{label_str}")
    else:
        fname = feature_names[node.feature_idx] if feature_names else f"x{node.feature_idx}"
        print(f"  {indent}[{prefix}] 若 {fname} <= {node.threshold:.4f}：")
        print_tree_structure(node.left,  feature_names, depth + 1, "左 (是)")
        print(f"  {indent}        否：")
        print_tree_structure(node.right, feature_names, depth + 1, "右 (否)")

def print_equations():
    # 8. 核心公式总结
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                  AdaBoost 核心公式速查                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  初始权重  w_i = 1/N                                      ║")
    print("║  加权误差  ε = Σ w_i * 1[h(x_i) != y_i]                 ║")
    print("║  弱分类器权重  alpha = 0.5 * ln((1-ε)/ε)                     ║")
    print("║  权重更新  w_i <- w_i * exp(-alpha * y_i * h(x_i))          ║")
    print("║  归一化    w_i <- w_i / Σ w_i                            ║")
    print("║  最终预测  H(x) = sign( Σ alpha_t * h_t(x) )                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()