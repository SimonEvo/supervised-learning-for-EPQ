"""
可视化模块（HZZ demo）

plot_main_figure()  - 主图：决策边界网格（上）+ 过拟合曲线（下左）+ 特征分布（下右）
"""

import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang HK', 'STHeiti', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 辅助
# ──────────────────────────────────────────────────────────────────────
_SIG_COLOR  = "#2A7FD4"   # 信号蓝
_BKG_COLOR  = "#F28C38"   # 本底橙
_WRONG_COLOR = "#E03030"  # 误分红


def _accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _draw_boundary(ax, predict_fn, X_train, X_test, y_test,
                   feat_x, feat_y, feat_means,
                   x_range, y_range, title,
                   feature_names=None, feature_units=None):
    """
    在 (feat_x, feat_y) 二维截面画决策边界。
    其余特征固定为训练集均值。
    散点：测试集，正确=圆点，误分=X。
    """
    res = 110
    xx  = np.linspace(*x_range, res)
    yy  = np.linspace(*y_range, res)
    XX, YY = np.meshgrid(xx, yy)

    # 构建网格样本（其余特征取均值）
    grid_X = []
    for fy, fx in zip(YY.ravel().tolist(), XX.ravel().tolist()):
        pt         = list(feat_means)
        pt[feat_x] = fx
        pt[feat_y] = fy
        grid_X.append(pt)

    Z = np.array(predict_fn(grid_X)).reshape(XX.shape)

    ax.contourf(XX, YY, Z, levels=[-2, 0, 2],
                colors=["#FFE4E1", "#E1EEFF"], alpha=0.55)
    ax.contour(XX, YY, Z, levels=[0],
               colors=["#333333"], linewidths=1.4)

    # 测试集散点
    test_preds = predict_fn(X_test)
    for yi, yp, x in zip(y_test, test_preds, X_test):
        fx_val = x[feat_x]
        fy_val = x[feat_y]
        if yi == yp:
            color = _SIG_COLOR if yi == 1 else _BKG_COLOR
            ax.scatter(fx_val, fy_val, c=color, s=22, alpha=0.70,
                       edgecolors="none", zorder=4)
        else:
            ax.scatter(fx_val, fy_val, c=_WRONG_COLOR, s=40,
                       marker="x", linewidths=1.2,
                       alpha=0.85, zorder=5)

    test_acc = _accuracy(y_test, test_preds)
    ax.set_title(f"{title}\n测试准确率 {test_acc:.1%}", fontsize=10, pad=6)
    if feature_names is not None:
        xu = (feature_units[feat_x] if feature_units else "")
        yu = (feature_units[feat_y] if feature_units else "")
        ax.set_xlabel(feature_names[feat_x] + (f" [{xu}]" if xu else ""), fontsize=9)
        ax.set_ylabel(feature_names[feat_y] + (f" [{yu}]" if yu else ""), fontsize=9)
    else:
        ax.set_xlabel(f"特征 {feat_x}", fontsize=9)
        ax.set_ylabel(f"特征 {feat_y}", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, ls="--", lw=0.35, alpha=0.45)

    # 图例（只加一次）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_SIG_COLOR,
               markersize=6, label='信号（正确）'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_BKG_COLOR,
               markersize=6, label='本底（正确）'),
        Line2D([0], [0], marker='x', color=_WRONG_COLOR,
               markersize=6, label='误分类'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right",
              framealpha=0.85)


# ──────────────────────────────────────────────────────────────────────
# 主图
# ──────────────────────────────────────────────────────────────────────
def plot_main_figure(
    X_train, y_train, X_test, y_test,
    boundary_models,          # list of (predict_fn, label_str)
    overfit_depths, train_accs, test_accs,
    ada_train_acc, ada_test_acc,
    feature_names, feature_units,
    feat_x=3, feat_y=2,       # m_ll, MET
    x_range=(20, 160), y_range=(2, 120),
):
    """
    主图布局：
      上行（1 × n_models）: 决策边界 + 测试散点
      下左（大）          : 过拟合曲线（depth vs accuracy）
      下右（2小）         : m_vis / MET 信号本底分布
    """
    n_models = len(boundary_models)
    n_feats  = len(X_train[0])

    # 其余特征取训练集均值
    feat_means = [sum(x[j] for x in X_train) / len(X_train) for j in range(n_feats)]

    fig = plt.figure(figsize=(5 * n_models, 11))
    gs  = gridspec.GridSpec(
        2, n_models,
        figure=fig,
        height_ratios=[1.15, 1.0],
        hspace=0.52, wspace=0.38
    )

    # ── 上行：决策边界 ────────────────────────────────────────────────
    for col, (pred_fn, label) in enumerate(boundary_models):
        ax = fig.add_subplot(gs[0, col])
        _draw_boundary(ax, pred_fn, X_train, X_test, y_test,
                       feat_x, feat_y, feat_means,
                       x_range, y_range, label,
                       feature_names=feature_names,
                       feature_units=feature_units)

    # ── 下左：过拟合曲线（占左侧 n_models-2 列）────────────────────────
    span_ov = max(n_models - 2, 2)
    ax_ov = fig.add_subplot(gs[1, :span_ov])

    ax_ov.plot(overfit_depths, train_accs, "o-",
               color=_SIG_COLOR, lw=2, ms=5,
               label="单棵决策树  训练准确率")
    ax_ov.plot(overfit_depths, test_accs,  "s--",
               color=_BKG_COLOR, lw=2, ms=5,
               label="单棵决策树  测试准确率")
    ax_ov.axhline(ada_train_acc, color=_SIG_COLOR, lw=1.5, ls=":",
                  label=f"AdaBoost 训练  {ada_train_acc:.3f}")
    ax_ov.axhline(ada_test_acc,  color=_BKG_COLOR, lw=1.5, ls=":",
                  label=f"AdaBoost 测试  {ada_test_acc:.3f}")

    best_d   = overfit_depths[test_accs.index(max(test_accs))]
    best_acc = max(test_accs)
    ax_ov.axvline(best_d, color="#888", lw=1, ls="--", alpha=0.7)
    ax_ov.annotate(f"单树最优 depth={best_d}\n测试 {best_acc:.3f}",
                   xy=(best_d, best_acc),
                   xytext=(best_d + 0.8, best_acc - 0.04),
                   fontsize=8, color="#555",
                   arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    ax_ov.set_xlabel("单棵决策树深度 (max_depth)", fontsize=11)
    ax_ov.set_ylabel("准确率", fontsize=11)
    ax_ov.set_title("过拟合曲线：深度增大 → 训练准确率持续上升，测试准确率先升后降",
                    fontsize=10)
    ax_ov.set_ylim(min(min(test_accs), ada_test_acc) - 0.05, 1.02)
    ax_ov.set_xticks(overfit_depths)
    ax_ov.legend(fontsize=8, loc="lower left")
    ax_ov.grid(True, ls="--", lw=0.4, alpha=0.5)

    # ── 下右：m_vis 和 MET 分布（各占 1 列）───────────────────────────
    for plot_idx, feat_idx in enumerate([feat_x, feat_y]):
        ax = fig.add_subplot(gs[1, span_ov + plot_idx])
        sig_vals = [x[feat_idx] for x, yi in zip(X_test, y_test) if yi ==  1]
        bkg_vals = [x[feat_idx] for x, yi in zip(X_test, y_test) if yi == -1]
        unit = feature_units[feat_idx]
        label_x = f"{feature_names[feat_idx]}" + (f" [{unit}]" if unit else "")

        hi, lo = max(sig_vals + bkg_vals), min(sig_vals + bkg_vals)
        bins = np.linspace(lo, hi, 35)
        ax.hist(sig_vals, bins=bins, color=_SIG_COLOR, alpha=0.55,
                label="信号", density=True)
        ax.hist(bkg_vals, bins=bins, color=_BKG_COLOR, alpha=0.55,
                label="本底", density=True)
        ax.set_xlabel(label_x, fontsize=10)
        ax.set_ylabel("归一化计数", fontsize=9)
        disc = "强" if feat_idx in (feat_x, feat_y) else "弱"
        ax.set_title(f"{feature_names[feat_idx]} 分布\n（判别力{disc}）",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", lw=0.35, alpha=0.45)

    fig.suptitle(
        "Z → l+l-  信号/本底分类  |  单棵决策树过拟合 vs AdaBoost 泛化",
        fontsize=12, y=1.005
    )
    plt.tight_layout()
    return fig
