"""
数据集生成模块 - Z → l+l- 信号/本底分类

蒙特卡洛生成六个运动学变量：
  pT_l1     [GeV]  主轻子横动量
  pT_l2     [GeV]  次轻子横动量
  MET       [GeV]  缺失横能量（Z 信号极低）
  m_ll      [GeV]  双轻子不变质量（信号峰值 91.2 GeV）
  eta_l1          主轻子赝快度
  delta_phi [rad] 两轻子方位角差

信号  (label=+1)：Z → l+l-（on-peak，91.2 GeV）
本底  (label=-1)：tt̄→双轻子 (40%)  +  WW (25%)  +  W+fake/QCD (35%)

判别力：m_ll / MET  →  强（Z 峰 + 几乎无中微子）
        pT_l1 / delta_phi  →  中
        pT_l2 / eta_l1  →  弱（接近噪声）
深层单棵决策树会学到弱特征上的虚假结构 → 训练误差低、测试误差高（过拟合）
"""

import random
import math

SEED = 42

FEATURE_NAMES = ["pT_l1",  "pT_l2",  "MET",   "m_ll",   "eta_l1", "delta_phi"]
FEATURE_UNITS = ["GeV",    "GeV",    "GeV",   "GeV",    "",       "rad"]
FEAT_X = 3   # m_ll   ← 决策边界可视化 x 轴
FEAT_Y = 2   # MET    ← 决策边界可视化 y 轴


# ──────────────────────────────────────────────────────────────────────
def _signal():
    """Z → l+l- 单事例（on-peak）。"""
    m_ll = -1.0
    while m_ll <= 50.0:
        m_ll = random.gauss(91.2, 4.5)           # Z 峰（含探测器分辨率展宽）

    MET    = random.lognormvariate(1.80, 0.65)   # 极低 MET（无中微子），峰值约 6 GeV
    pT_l1  = random.lognormvariate(3.80, 0.35)   # 峰值约 45 GeV
    pT_l2  = random.lognormvariate(3.70, 0.35)   # 峰值约 40 GeV
    eta_l1 = random.gauss(0.0, 1.0)
    delta_phi = random.uniform(0.5, math.pi)  # 在 lab frame 受 Z pT boost 影响，分布宽泛

    return [pT_l1, pT_l2, MET, m_ll, eta_l1, delta_phi]


def _background():
    """本底事例（tt̄ 双轻子 + WW + W+fake/QCD）。"""
    r = random.random()

    if r < 0.40:
        # tt̄ → dileptonic：高 MET，宽 m_ll（两个来自 W 的中微子）
        m_ll = -1.0
        while m_ll <= 10.0:
            m_ll = random.lognormvariate(4.15, 0.55)  # 宽分布，峰值约 63 GeV
        MET    = random.lognormvariate(4.00, 0.50)    # 高 MET，峰值约 55 GeV
        pT_l1  = random.lognormvariate(3.90, 0.40)    # 较硬（来自 top 衰变链）
        pT_l2  = random.lognormvariate(3.70, 0.45)
        eta_l1 = random.gauss(0.0, 1.30)
        delta_phi = random.uniform(0.3, math.pi)

    elif r < 0.65:
        # WW → dileptonic：中等 MET，m_ll 宽分布
        m_ll = -1.0
        while m_ll <= 10.0:
            m_ll = random.lognormvariate(4.00, 0.55)  # 峰值约 55 GeV
        MET    = random.lognormvariate(3.70, 0.55)    # 中等 MET，峰值约 40 GeV
        pT_l1  = random.lognormvariate(3.70, 0.40)
        pT_l2  = random.lognormvariate(3.50, 0.45)
        eta_l1 = random.gauss(0.0, 1.20)
        delta_phi = random.uniform(0.2, math.pi)

    else:
        # W+fake / QCD：低 MET，宽且低的 m_ll（fake lepton 较软）
        m_ll   = random.lognormvariate(3.90, 0.80)   # 宽且低
        MET    = random.lognormvariate(2.80, 0.70)   # 低 MET
        pT_l1  = random.lognormvariate(3.60, 0.50)
        pT_l2  = random.lognormvariate(3.20, 0.55)   # fake lepton 较软
        eta_l1 = random.gauss(0.0, 1.40)
        delta_phi = random.uniform(0.0, math.pi)

    return [pT_l1, pT_l2, MET, m_ll, eta_l1, delta_phi]


# ──────────────────────────────────────────────────────────────────────
def generate_zll_dataset(n_samples=600, sig_fraction=0.5, noise=0.05):
    """
    生成 Z→l+l- 信号/本底数据集。

    参数：
      n_samples    : 总事例数
      sig_fraction : 信号占比
      noise        : 标签噪声率（模拟重建误差 / mis-ID）

    返回：
      X : list of [pT_l1, pT_l2, MET, m_ll, eta_l1, delta_phi]
      y : list of +1（信号）或 -1（本底）
    """
    random.seed(SEED)
    X, y = [], []

    n_sig = int(n_samples * sig_fraction)
    n_bkg = n_samples - n_sig

    for _ in range(n_sig):
        X.append(_signal())
        label = 1 if random.random() >= noise else -1
        y.append(label)

    for _ in range(n_bkg):
        X.append(_background())
        label = -1 if random.random() >= noise else 1
        y.append(label)

    data = list(zip(X, y))
    random.shuffle(data)
    X, y = zip(*data)
    return list(X), list(y)


def train_test_split(X, y, test_ratio=0.25):
    random.seed(SEED + 1)
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    X_train, y_train = zip(*data[:split])
    X_test,  y_test  = zip(*data[split:])
    return list(X_train), list(y_train), list(X_test), list(y_test)
