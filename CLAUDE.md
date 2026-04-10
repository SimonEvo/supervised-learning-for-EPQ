# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
python run.py          # 统一入口，顶部 DEMO_CHOICE 切换 "classic" / "improved"
python demo_improved/main.py   # 直接运行某个 demo
python demo_classic/main.py
```

Dependencies:
```bash
pip install matplotlib numpy
```

## Architecture

Three self-contained demos under a shared entry point:

```
bdt_demo/
├── run.py                     # DEMO_CHOICE 参数切换 demo，subprocess 调用对应 main.py
├── demo_classic/              # 原版：圆形决策边界二分类
│   ├── main.py
│   ├── boosted_decision_tree.py  # 数据生成 + 决策树 + 评估
│   ├── adaBoost.py
│   └── visualisation.py
├── demo_improved/             # 进阶版：XOR 棋盘格
│   ├── main.py                # 所有可调参数集中在顶部
│   ├── dataset.py             # MC 采样生成 XOR 数据，支持二分类(±1) / 四分类(0-3)
│   ├── decision_tree.py       # CART 决策树（与 classic 算法相同，新增多分类评估）
│   ├── adaboost.py            # AdaBoost（二分类）+ SAMME（多分类）统一实现
│   └── visualisation.py      # XOR 背景散点图 + matplotlib 误差收敛曲线
└── demo_hzz/                  # 物理版：H→τ_lep τ_had 信号/本底分类
    ├── main.py                # 所有可调参数集中在顶部
    ├── dataset.py             # MC 生成 6 个运动学变量（信号+本底混合）
    ├── decision_tree.py       # CART（含 N_THRESH_MAX 量化加速）
    ├── adaboost.py            # 二分类 AdaBoost + feature_importance()
    └── visualisation.py      # 过拟合曲线 + 决策边界网格 + 特征分布
```

Each demo adds its own directory to `sys.path` at the top of `main.py`, so all internal imports are bare names (e.g. `from decision_tree import ...`).

## Core Algorithms

**Decision Tree** (`decision_tree.py` / `boosted_decision_tree.py`): CART with weighted Gini impurity. Exhaustive threshold search over sorted unique feature values. Stopping: max_depth, pure node, or no valid split. Leaf prediction via weighted majority vote.

**AdaBoost binary** (`adaboost.py:AdaBoost`): standard α = 0.5·ln((1-ε)/ε), weight update exp(-α·y·h(x)).

**SAMME multi-class** (same class, `n_classes > 2`): α = ln((1-ε)/ε) + ln(K-1), weight update exp(α·𝟙[h≠y]), prediction = argmax_k Σ α·𝟙[h=k].

## Key Design Decisions

- No ML libraries — pure Python + standard library for all algorithms; only matplotlib/numpy for plotting.
- All comments and console output in Chinese (educational tool for Chinese-speaking students).
- `run.py` uses `subprocess` to avoid import path conflicts between demos.
- All student-facing tunable parameters are at the top of each `main.py` with inline comments explaining the effect of each parameter.
