"""
数据集生成模块 - H → τ_lep τ_had 信号/本底分类

蒙特卡洛生成六个运动学变量：
  pT_lep   [GeV]  轻子横动量（来自轻子型 tau 衰变）
  pT_had   [GeV]  强子型 tau 喷注横动量
  MET      [GeV]  缺失横能量
  m_vis    [GeV]  可见衰变产物不变质量
  eta_lep         轻子赝快度
  delta_phi [rad] 轻子与强子型 tau 方位角差

信号  (label=+1)：H(125 GeV) → τ_lep τ_had
本底  (label=-1)：Z→ττ (60%)  +  W+jets (25%)  +  QCD (15%)

判别力：m_vis / MET / pT_had  →  强
        pT_lep / eta_lep / delta_phi  →  弱（接近噪声）
深层单棵决策树会学到弱特征上的虚假结构 → 训练误差低、测试误差高（过拟合）
"""

import random
import math

SEED = 42

FEATURE_NAMES = ["pT_lep", "pT_had", "MET",  "m_vis", "eta_lep", "delta_phi"]
FEATURE_UNITS = ["GeV",    "GeV",    "GeV",   "GeV",   "",        "rad"]
FEAT_X = 3   # m_vis   ← 决策边界可视化 x 轴
FEAT_Y = 2   # MET     ← 决策边界可视化 y 轴


# ──────────────────────────────────────────────────────────────────────
def _signal():
    """H(125 GeV) → τ_lep τ_had 单事例。"""
    m_vis = -1.0
    while m_vis <= 5.0:
        m_vis = random.gauss(68.0, 18.0)          # 可见质量峰值约 68 GeV

    MET    = random.lognormvariate(3.65, 0.45)  # 峰值约 38 GeV（双中微子）
    pT_had = random.lognormvariate(3.70, 0.40)  # 峰值约 40 GeV
    pT_lep = random.lognormvariate(3.20, 0.45)  # 峰值约 25 GeV（三体衰变，较软）
    eta_lep   = random.gauss(0.0, 0.95)
    delta_phi = min(abs(random.gauss(math.pi * 0.65, 0.85)), math.pi)

    return [pT_lep, pT_had, MET, m_vis, eta_lep, delta_phi]


def _background():
    """本底事例（Z→ττ 主导 + W+jets + QCD）。"""
    r = random.random()

    if r < 0.60:
        # Z→ττ：运动学与信号最相似，可见质量峰值更低
        m_vis = -1.0
        while m_vis <= 5.0:
            m_vis = random.gauss(52.0, 22.0)
        MET    = random.lognormvariate(3.15, 0.55)  # 峰值约 23 GeV（MET 较少）
        pT_had = random.lognormvariate(3.45, 0.45)
        pT_lep = random.lognormvariate(3.05, 0.50)
        eta_lep   = random.gauss(0.0, 1.15)
        delta_phi = random.uniform(0.3, math.pi)

    elif r < 0.85:
        # W+jets：硬轻子（来自 W），MET 较大（W 中微子）
        m_vis  = random.lognormvariate(4.00, 0.65)
        MET    = random.lognormvariate(3.55, 0.60)
        pT_had = random.lognormvariate(3.30, 0.50)
        pT_lep = random.lognormvariate(3.40, 0.40)  # 来自 W，较硬
        eta_lep   = random.gauss(0.0, 1.25)
        delta_phi = random.uniform(0.2, math.pi)

    else:
        # QCD / fake tau：低 MET，分布宽泛
        m_vis  = random.lognormvariate(3.85, 0.75)
        MET    = random.lognormvariate(2.70, 0.70)  # 较低 MET
        pT_had = random.lognormvariate(3.50, 0.50)
        pT_lep = random.lognormvariate(2.90, 0.60)
        eta_lep   = random.gauss(0.0, 1.40)
        delta_phi = random.uniform(0.0, math.pi)

    return [pT_lep, pT_had, MET, m_vis, eta_lep, delta_phi]


# ──────────────────────────────────────────────────────────────────────
def generate_hzz_dataset(n_samples=600, sig_fraction=0.5, noise=0.05):
    """
    生成 H→τ_lep τ_had 信号/本底数据集。

    参数：
      n_samples    : 总事例数
      sig_fraction : 信号占比
      noise        : 标签噪声率（模拟重建误差 / mis-ID）

    返回：
      X : list of [pT_lep, pT_had, MET, m_vis, eta_lep, delta_phi]
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
