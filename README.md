# Supervised Learning — Boosted Decision Trees

A hands-on teaching toolkit for **AdaBoost / Boosted Decision Trees (BDT)**, built from scratch in pure Python (no scikit-learn, no XGBoost).  
Designed for students learning the intersection of machine learning and statistics.

---

## Three Progressive Demos

| Demo | Dataset | Key Concept |
|------|---------|-------------|
| `demo_classic` | Circular boundary (2D) | AdaBoost basics: weight update, α, ensemble prediction |
| `demo_improved` | XOR checkerboard (2D) | Why single trees fail on XOR; weak→strong learner theory |
| `demo_hzz` ⭐ | H → τ_lep τ_had (6 kinematic variables, MC) | **Overfitting**: bias–variance tradeoff, decision boundary grid, feature importance |

---

## Quick Start

```bash
pip install matplotlib numpy
```

Open `run.py`, set `DEMO_CHOICE`, then run:

```bash
python run.py
```

```python
# run.py — change this line
DEMO_CHOICE = "hzz"      # "classic" | "improved" | "hzz"
```

Each demo can also be run directly:

```bash
python demo_hzz/main.py
python demo_improved/main.py
python demo_classic/main.py
```

All tunable parameters are at the top of each `main.py` with inline comments.

---

## Demo Details

### `demo_classic` — AdaBoost Fundamentals

Synthetic 2D dataset with a circular decision boundary. Demonstrates the full AdaBoost training loop: per-round weighted error ε, classifier weight α, sample weight update, and ensemble prediction.

**Key output:** training progress table, confusion matrix, α ranking.

---

### `demo_improved` — XOR and the Weak Learner Theorem

The XOR (checkerboard) problem is provably unsolvable by a single depth-1 decision tree (axis-aligned cuts cannot separate diagonal classes). AdaBoost combines multiple such "useless" stumps into a strong classifier.

**Key output:** side-by-side prediction plots, training error convergence curve.  
**Optional:** set `N_CLASSES = 4` to enable four-class SAMME.

---

### `demo_hzz` — Overfitting on Physical Data ⭐

MC-generated H → τ_lep τ_had signal/background classification using six kinematic variables:

| Feature | Unit | Discriminating power |
|---------|------|----------------------|
| `pT_lep` | GeV | weak |
| `pT_had` | GeV | moderate |
| `MET` | GeV | **strong** |
| `m_vis` | GeV | **strong** |
| `eta_lep` | — | weak (near noise) |
| `delta_phi` | rad | weak |

**Background mixture:** Z→ττ (60 %) + W+jets (25 %) + QCD (15 %)

**Three visualisations in one figure:**

1. **Decision boundary grid** — single tree at depth = 2 / 5 / 10 and AdaBoost, projected onto the `m_vis`–`MET` plane. Deep trees produce jagged, overfit boundaries; AdaBoost is smooth.

2. **Overfitting curve** — training and test accuracy vs. depth (1 → 10). Classic bias–variance plot: test accuracy peaks then drops; AdaBoost sits stably above the single-tree test curve.

3. **Feature distributions** — `m_vis` and `MET` signal/background histograms showing why these variables are informative.

**Console output also includes feature importance** (α-weighted split counts), which shows AdaBoost automatically focuses on `m_vis` > `MET` > `pT_had` and suppresses the noise features.

---

## Algorithm Summary

**Decision Tree (CART):** weighted Gini impurity, exhaustive threshold search, recursive binary splits.

**AdaBoost (binary):**
```
Initialise  w_i = 1/N
For t = 1 … T:
    Train weak classifier h_t with weights w
    ε_t  = Σ w_i · 1[h_t(x_i) ≠ y_i]
    α_t  = 0.5 · ln((1 − ε_t) / ε_t)
    w_i ← w_i · exp(−α_t · y_i · h_t(x_i)),  then normalise
Final:  H(x) = sign(Σ α_t · h_t(x))
```

**SAMME (multi-class, `demo_improved` only):** α_t = ln((1−ε_t)/ε_t) + ln(K−1), prediction = argmax over class votes.

---

## Repository Structure

```
supervised-learning-for-EPQ/
├── run.py
├── demo_classic/
│   ├── main.py
│   ├── boosted_decision_tree.py
│   ├── adaBoost.py
│   └── visualisation.py
├── demo_improved/
│   ├── main.py
│   ├── dataset.py
│   ├── decision_tree.py
│   ├── adaboost.py
│   └── visualisation.py
└── demo_hzz/
    ├── main.py
    ├── dataset.py
    ├── decision_tree.py
    ├── adaboost.py
    └── visualisation.py
```

No ML libraries are used in the core algorithms — everything is implemented from scratch using Python's standard library (`random`, `math`, `collections`). Only `matplotlib` and `numpy` are required for plotting.

---

## Suggested Classroom Progression

1. **`demo_classic`** — introduce AdaBoost mechanics on a clean 2D problem  
2. **`demo_improved`** — motivate why ensembles are theoretically necessary (XOR theorem)  
3. **`demo_hzz`** — connect to real physics; demonstrate overfitting with the bias–variance curve

Students can explore by modifying the parameters at the top of each `main.py`.
