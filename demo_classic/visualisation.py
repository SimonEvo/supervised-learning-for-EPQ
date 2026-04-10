import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang HK', 'STHeiti', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(X, y_true, y_pred, title="AdaBoost Predictions", ax=None):
    """
    Scatter-plot all samples with color encoding:
      Blue   circle  - correctly predicted positive (+1, inside r=2 circle)
      Orange circle  - correctly predicted negative (-1, outside r=2 circle)
      Red    X mark  - misclassified (wrong prediction regardless of true label)

    Also draws:
      - The true decision boundary circle r = 2 (dashed line)
      - A shaded background showing the Bayes-optimal regions

    Parameters
    ----------
    X       : list of [x1, x2]   feature matrix
    y_true  : list of +1 / -1    ground-truth labels
    y_pred  : list of +1 / -1    model predictions
    title   : str                plot title
    ax      : matplotlib Axes    if None, a new figure is created

    Returns
    -------
    ax : the Axes object used for plotting
    """

    # Split samples into three display groups
    correct_pos = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred)
                   if yt == yp and yt ==  1]
    correct_neg = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred)
                   if yt == yp and yt == -1]
    wrong       = [(x[0], x[1]) for x, yt, yp in zip(X, y_true, y_pred)
                   if yt != yp]

    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Shaded background: Bayes-optimal regions based on true boundary
    res = 300
    xx  = np.linspace(-3.2, 3.2, res)
    yy  = np.linspace(-3.2, 3.2, res)
    Z   = np.array([[1 if xi**2 + yi**2 < 4 else -1
                     for xi in xx] for yi in yy])
    ax.contourf(xx, yy, Z,
                levels=[-2, 0, 2],
                colors=["#FFF3EC", "#EAF4FB"],   # light orange / light blue
                alpha=0.45)

    # True decision boundary: circle r = 2
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(2 * np.cos(theta), 2 * np.sin(theta),
            color="#444444", linewidth=1.8, linestyle="--",
            label="True boundary  r = 2", zorder=3)

    # Correctly predicted positive class (blue dots)
    if correct_pos:
        xs, ys = zip(*correct_pos)
        ax.scatter(xs, ys,
                   c="#2A7FD4", s=45, linewidths=0.5,
                   edgecolors="#1a5fa0", alpha=0.85,
                   label=f"Correct  +1  ({len(correct_pos)})",
                   zorder=4)

    # Correctly predicted negative class (orange dots)
    if correct_neg:
        xs, ys = zip(*correct_neg)
        ax.scatter(xs, ys,
                   c="#F28C38", s=45, linewidths=0.5,
                   edgecolors="#c06010", alpha=0.85,
                   label=f"Correct  -1  ({len(correct_neg)})",
                   zorder=4)

    # Misclassified samples: red X markers
    if wrong:
        xs, ys = zip(*wrong)
        ax.scatter(xs, ys,
                   c="#E03030", s=80, marker="X", linewidths=0.5,
                   edgecolors="#900000", alpha=0.95,
                   label=f"Wrong  ({len(wrong)})",
                   zorder=5)

    # Axes decoration
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_title(f"{title}\nAccuracy = {acc:.1%}  |  n = {len(y_true)}",
                 fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)

    # Small annotation in the bottom-right corner
    ax.text(3.1, -3.1,
            "Dashed circle = true boundary r=2",
            fontsize=8, color="#888888",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.75))

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax
