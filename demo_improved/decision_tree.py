"""
决策树模块（CART，加权基尼系数）
与 classic demo 算法相同，支持任意标签类型（二分类 ±1 或多分类整数）。
"""

import math
import collections


# ══════════════════════════════════════════════════════════════════════
# 决策树节点与构建
# ══════════════════════════════════════════════════════════════════════

class Node:
    """决策树节点。叶节点存储预测值；内部节点存储分裂特征与阈值。"""
    def __init__(self):
        self.feature_idx = None
        self.threshold   = None
        self.left        = None
        self.right       = None
        self.value       = None

    def is_leaf(self):
        return self.value is not None


def weighted_gini(labels, weights):
    """
    计算带权重的基尼不纯度：Gini = 1 - Σ p_k²
    p_k 为第 k 类的加权占比。AdaBoost 用样本权重代替简单计数。
    """
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    class_w = collections.defaultdict(float)
    for lbl, w in zip(labels, weights):
        class_w[lbl] += w
    return 1.0 - sum((cw / total_w) ** 2 for cw in class_w.values())


def best_split(X, y, weights):
    """
    遍历所有特征与候选阈值（相邻唯一值的中点），
    找到使加权基尼增益最大的分裂点。
    返回 (best_feat, best_thresh) 或 (None, None)。
    """
    n_features   = len(X[0])
    parent_gini  = weighted_gini(y, weights)
    total_w      = sum(weights)
    best_gain    = -1
    best_feat    = None
    best_thresh  = None

    for feat in range(n_features):
        values     = sorted(set(x[feat] for x in X))
        thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

        for thresh in thresholds:
            left_y,  left_w  = [], []
            right_y, right_w = [], []
            for xi, yi, wi in zip(X, y, weights):
                if xi[feat] <= thresh:
                    left_y.append(yi);  left_w.append(wi)
                else:
                    right_y.append(yi); right_w.append(wi)

            if not left_y or not right_y:
                continue

            w_l = sum(left_w)
            w_r = sum(right_w)
            child_gini = (w_l / total_w) * weighted_gini(left_y,  left_w) + \
                         (w_r / total_w) * weighted_gini(right_y, right_w)
            gain = parent_gini - child_gini
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh

    return best_feat, best_thresh


def majority_vote(y, weights):
    """加权多数投票：返回加权权重之和最大的类别。"""
    class_w = collections.defaultdict(float)
    for lbl, w in zip(y, weights):
        class_w[lbl] += w
    return max(class_w, key=class_w.get)


def build_tree(X, y, weights, max_depth, depth=0):
    """
    递归构建 CART 决策树。
    停止条件：① 达到 max_depth  ② 纯节点  ③ 无法分裂
    """
    node = Node()

    if depth >= max_depth:
        node.value = majority_vote(y, weights)
        return node
    if len(set(y)) == 1:
        node.value = y[0]
        return node

    feat, thresh = best_split(X, y, weights)
    if feat is None:
        node.value = majority_vote(y, weights)
        return node

    left_X, left_y, left_w    = [], [], []
    right_X, right_y, right_w = [], [], []
    for xi, yi, wi in zip(X, y, weights):
        if xi[feat] <= thresh:
            left_X.append(xi);  left_y.append(yi);  left_w.append(wi)
        else:
            right_X.append(xi); right_y.append(yi); right_w.append(wi)

    node.feature_idx = feat
    node.threshold   = thresh
    node.left  = build_tree(left_X,  left_y,  left_w,  max_depth, depth+1)
    node.right = build_tree(right_X, right_y, right_w, max_depth, depth+1)
    return node


def predict_single(node, x):
    if node.is_leaf():
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return predict_single(node.left, x)
    return predict_single(node.right, x)


def predict_tree(tree, X):
    return [predict_single(tree, x) for x in X]


# ══════════════════════════════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════════════════════════════

def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """二分类混淆矩阵（标签 ±1）。返回 TP, FP, TN, FN。"""
    TP = sum(1 for a, b in zip(y_true, y_pred) if a == 1  and b == 1)
    FP = sum(1 for a, b in zip(y_true, y_pred) if a == -1 and b == 1)
    TN = sum(1 for a, b in zip(y_true, y_pred) if a == -1 and b == -1)
    FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1  and b == -1)
    return TP, FP, TN, FN


def print_confusion_matrix(TP, FP, TN, FN):
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


def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    """K×K 混淆矩阵。cm[i][j] = 真实为i、预测为j的样本数。"""
    cm = [[0] * n_classes for _ in range(n_classes)]
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1
    return cm


def print_confusion_matrix_multi(cm, n_classes):
    K = n_classes
    header = "  " + "".join(f"  预测{k}" for k in range(K))
    print(header)
    print("  " + "─" * (8 * K + 2))
    for i in range(K):
        row = f"  真实{i} │" + "".join(f"  {cm[i][j]:>4d}  " for j in range(K))
        print(row)
    print()
    for k in range(K):
        tp = cm[k][k]
        fp = sum(cm[i][k] for i in range(K) if i != k)
        fn = sum(cm[k][j] for j in range(K) if j != k)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        print(f"  类{k}：精确率={prec:.4f}  召回率={rec:.4f}  F1={f1:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 树结构打印
# ══════════════════════════════════════════════════════════════════════

def print_tree_structure(node, feature_names=None, depth=0, prefix="根"):
    indent = "    " * depth
    if node.is_leaf():
        lbl = node.value
        if lbl == 1:
            label_str = "+1 (正类)"
        elif lbl == -1:
            label_str = "-1 (负类)"
        else:
            label_str = f"类 {lbl}"
        print(f"  {indent}└─[叶] 预测：{label_str}")
    else:
        fname = feature_names[node.feature_idx] if feature_names else f"x{node.feature_idx}"
        print(f"  {indent}[{prefix}] 若 {fname} <= {node.threshold:.4f}：")
        print_tree_structure(node.left,  feature_names, depth+1, "左 (是)")
        print(f"  {indent}        否：")
        print_tree_structure(node.right, feature_names, depth+1, "右 (否)")
