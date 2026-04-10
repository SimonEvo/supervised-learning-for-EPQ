"""
决策树模块（CART，加权基尼系数）

与 classic / improved demo 算法相同，增加一项加速优化：
  N_THRESH_MAX：每个特征最多使用的候选阈值数。
  连续特征在数百个训练样本时会产生数百个候选阈值，逐一搜索很慢。
  取等间距量化（如 20 个分位数阈值）对准确率影响极小，
  但将 best_split 速度提升约 10-20 倍，使过拟合曲线的多次训练可行。
"""

import collections

N_THRESH_MAX = 20   # 每特征最多候选阈值数（0 = 不限制）


# ══════════════════════════════════════════════════════════════════════
class Node:
    def __init__(self):
        self.feature_idx = None
        self.threshold   = None
        self.left        = None
        self.right       = None
        self.value       = None

    def is_leaf(self):
        return self.value is not None


def weighted_gini(labels, weights):
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    cw = collections.defaultdict(float)
    for l, w in zip(labels, weights):
        cw[l] += w
    return 1.0 - sum((v / total_w) ** 2 for v in cw.values())


def best_split(X, y, weights):
    n_features  = len(X[0])
    parent_gini = weighted_gini(y, weights)
    total_w     = sum(weights)
    best_gain   = -1
    best_feat   = None
    best_thresh = None

    for feat in range(n_features):
        vals   = sorted(set(x[feat] for x in X))
        all_th = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]

        # 等间距量化：若候选阈值过多则降采样
        if N_THRESH_MAX and len(all_th) > N_THRESH_MAX:
            step = max(1, (len(all_th) - 1) // (N_THRESH_MAX - 1))
            thresholds = [all_th[min(i * step, len(all_th) - 1)]
                          for i in range(N_THRESH_MAX)]
        else:
            thresholds = all_th

        for thresh in thresholds:
            ly, lw, ry, rw = [], [], [], []
            for xi, yi, wi in zip(X, y, weights):
                if xi[feat] <= thresh:
                    ly.append(yi); lw.append(wi)
                else:
                    ry.append(yi); rw.append(wi)
            if not ly or not ry:
                continue
            wl, wr = sum(lw), sum(rw)
            child  = (wl / total_w) * weighted_gini(ly, lw) + \
                     (wr / total_w) * weighted_gini(ry, rw)
            gain   = parent_gini - child
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh

    return best_feat, best_thresh


def majority_vote(y, weights):
    cw = collections.defaultdict(float)
    for l, w in zip(y, weights):
        cw[l] += w
    return max(cw, key=cw.get)


def build_tree(X, y, weights, max_depth, depth=0):
    node = Node()
    if depth >= max_depth:
        node.value = majority_vote(y, weights); return node
    if len(set(y)) == 1:
        node.value = y[0]; return node
    feat, thresh = best_split(X, y, weights)
    if feat is None:
        node.value = majority_vote(y, weights); return node

    lX, ly, lw, rX, ry, rw = [], [], [], [], [], []
    for xi, yi, wi in zip(X, y, weights):
        if xi[feat] <= thresh:
            lX.append(xi); ly.append(yi); lw.append(wi)
        else:
            rX.append(xi); ry.append(yi); rw.append(wi)

    node.feature_idx = feat
    node.threshold   = thresh
    node.left  = build_tree(lX, ly, lw, max_depth, depth + 1)
    node.right = build_tree(rX, ry, rw, max_depth, depth + 1)
    return node


def predict_single(node, x):
    if node.is_leaf():
        return node.value
    return predict_single(node.left if x[node.feature_idx] <= node.threshold
                          else node.right, x)


def predict_tree(tree, X):
    return [predict_single(tree, x) for x in X]


# ══════════════════════════════════════════════════════════════════════
def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix(y_true, y_pred):
    TP = sum(1 for a, b in zip(y_true, y_pred) if a ==  1 and b ==  1)
    FP = sum(1 for a, b in zip(y_true, y_pred) if a == -1 and b ==  1)
    TN = sum(1 for a, b in zip(y_true, y_pred) if a == -1 and b == -1)
    FN = sum(1 for a, b in zip(y_true, y_pred) if a ==  1 and b == -1)
    return TP, FP, TN, FN


def print_confusion_matrix(TP, FP, TN, FN):
    print()
    print("  混淆矩阵（行=真实，列=预测）：")
    print("  ┌─────────────┬──────────┬──────────┐")
    print("  │             │ 预测 +1  │ 预测 -1  │")
    print("  ├─────────────┼──────────┼──────────┤")
    print(f"  │ 真实 +1     │  TP={TP:>4d}  │  FN={FN:>4d}  │")
    print(f"  │ 真实 -1     │  FP={FP:>4d}  │  TN={TN:>4d}  │")
    print("  └─────────────┴──────────┴──────────┘")
    prec = TP / (TP + FP) if TP + FP else 0
    rec  = TP / (TP + FN) if TP + FN else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"  Precision = {prec:.4f}   Recall = {rec:.4f}   F1 = {f1:.4f}")


def print_tree_structure(node, feature_names=None, depth=0, prefix="根"):
    indent = "    " * depth
    if node.is_leaf():
        lbl = "+1 (信号)" if node.value == 1 else "-1 (本底)"
        print(f"  {indent}└─[叶] 预测：{lbl}")
    else:
        fname = feature_names[node.feature_idx] if feature_names else f"x{node.feature_idx}"
        print(f"  {indent}[{prefix}] 若 {fname} <= {node.threshold:.3f}：")
        print_tree_structure(node.left,  feature_names, depth + 1, "左 (是)")
        print(f"  {indent}        否：")
        print_tree_structure(node.right, feature_names, depth + 1, "右 (否)")
