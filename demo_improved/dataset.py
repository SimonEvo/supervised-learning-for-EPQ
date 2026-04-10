"""
数据集生成模块 - XOR 棋盘格问题（蒙特卡洛采样）

XOR 决策规则（二分类）：
  特征空间：x₁, x₂ ∈ [-2, 2]，均匀随机采样（蒙特卡洛方法）
  决策边界：坐标轴（x₁=0 与 x₂=0）
    同号（第一/三象限）→ 标签 +1
    异号（第二/四象限）→ 标签 -1

  为什么选 XOR？
    决策树只能做轴对齐切割（水平线或垂直线）。
    一刀切只能把平面分成左/右两半或上/下两半，
    但 XOR 的正负类分布在对角象限，任何单一切割
    最多能分对约 50% 的样本。
    AdaBoost 组合多个这样的弱分类器后，可以逐步
    逼近真实的 XOR 边界——这是 classic demo 中的
    圆形边界所无法如此清晰展示的效果。

四分类扩展（N_CLASSES=4）：
  每个象限对应一个类别（0/1/2/3），展示多分类 SAMME 算法。
"""

import random

SEED = 42


def generate_xor_dataset(n_samples=600, noise=0.10, n_classes=2, x_range=2.0):
    """
    用蒙特卡洛采样生成 XOR 棋盘格数据集。

    参数：
      n_samples : 总样本数
      noise     : 噪声率（0=无噪声，0.5=完全随机翻转）
      n_classes : 2 = XOR 二分类（±1）；4 = 四象限多分类（0/1/2/3）
      x_range   : 特征范围 [-x_range, x_range]

    返回：
      X : list of [x₁, x₂]
      y : list of 标签
    """
    random.seed(SEED)
    X, y = [], []

    while len(X) < n_samples:
        # 蒙特卡洛采样：均匀随机生成特征点
        x1 = random.uniform(-x_range, x_range)
        x2 = random.uniform(-x_range, x_range)

        # 排除坐标轴上的点（边界歧义）
        if abs(x1) < 1e-9 or abs(x2) < 1e-9:
            continue

        if n_classes == 2:
            # 同号 → +1（第一/三象限），异号 → -1（第二/四象限）
            label = 1 if x1 * x2 > 0 else -1
            if random.random() < noise:
                label = -label
        else:  # n_classes == 4
            if   x1 > 0 and x2 > 0: label = 0   # 第一象限
            elif x1 < 0 and x2 > 0: label = 1   # 第二象限
            elif x1 < 0 and x2 < 0: label = 2   # 第三象限
            else:                    label = 3   # 第四象限
            # 噪声：随机切换到相邻象限
            if random.random() < noise:
                label = (label + random.choice([1, 3])) % 4

        X.append([x1, x2])
        y.append(label)

    return X, y


def train_test_split(X, y, test_ratio=0.2):
    """随机划分训练集与测试集。"""
    random.seed(SEED + 1)
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    X_train, y_train = zip(*data[:split])
    X_test,  y_test  = zip(*data[split:])
    return list(X_train), list(y_train), list(X_test), list(y_test)
