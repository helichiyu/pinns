"""AL-PINNs augmented-Lagrangian loss weighting -- flow-style figure (B/W)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 7.8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.8)
ax.set_aspect("equal")
ax.axis("off")


def box(cx, cy, w, h, text, fontsize=10, fill="white", edge="black", lw=1.2, ls="-"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=lw, linestyle=ls, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def arrow(p1, p2, lw=1.2, color="black", ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, linestyle=ls, color=color, zorder=2))


# title + source
ax.text(6.5, 7.5, "AL-PINNs 增广拉格朗日", ha="center", va="bottom",
        fontsize=14, fontweight="bold")
ax.text(6.5, 7.15, "Son et al., 2023  ·  arXiv:2205.01059", ha="center", va="bottom",
        fontsize=9, color="#666666")

# formula card (compact)
box(6.5, 6.45, 10.2, 0.6,
    r"$L=L_{obj}+\sum_i\left[\lambda_i\,c_i+\frac{\mu_i}{2}\,c_i^{\,2}\right]$",
    fontsize=15)

# column headers
ax.text(1.8, 5.95, "目标 / 约束", ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(4.7, 5.95, "乘子罚项", ha="center", va="center", fontsize=10, fontweight="bold")

# objective + two constraints
box(1.8, 5.3, 2.3, 0.7, "目标：振幅 $L_{obj}$\n（权重 = 1，恒定）", fontsize=9.5)
box(1.8, 4.2, 2.3, 0.6, "约束 1：背景 $c_1$", fontsize=9.5)
box(1.8, 3.3, 2.3, 0.6, "约束 2：占比 $c_2$", fontsize=9.5)

# multiplier penalty gates
box(4.7, 4.2, 2.5, 0.6, r"$\lambda_1 c_1+\frac{\mu_1}{2}c_1^2$", fontsize=9)
box(4.7, 3.3, 2.5, 0.6, r"$\lambda_2 c_2+\frac{\mu_2}{2}c_2^2$", fontsize=9)

# Sigma + L_total
ax.add_patch(Circle((7.4, 4.25), 0.24, facecolor="white", edgecolor="black",
                    linewidth=1.3, zorder=3))
ax.text(7.4, 4.25, r"$\Sigma$", ha="center", va="center", fontsize=11, zorder=4)
box(9.6, 4.25, 1.7, 0.65, r"$L_{total}$", fontsize=12)

# flow arrows
arrow((2.95, 5.3), (7.16, 4.42))                      # objective -> Sigma
arrow((2.95, 4.2), (3.45, 4.2))                       # bg -> mult
arrow((2.95, 3.3), (3.45, 3.3))                       # area -> mult
arrow((5.95, 4.2), (7.16, 4.32))                      # mult bg -> Sigma
arrow((5.95, 3.3), (7.2, 4.12))                       # mult area -> Sigma
arrow((7.64, 4.25), (8.75, 4.25))                     # Sigma -> L_total

# dual-update feedback rail: L_total -> multipliers (lambda adapts)
rail_y = 2.05
ax.plot([9.6, 9.6, 4.7, 4.7], [3.92, rail_y, rail_y, 3.0],
        color="#555555", lw=1.3, ls=(0, (5, 3)), zorder=2, solid_capstyle="round")
arrow((4.7, 3.15), (4.7, 3.4), lw=1.3, color="#555555")
ax.text(7.15, rail_y + 0.13,
        r"对偶更新：$\lambda_i\leftarrow\max(0,\ \lambda_i+\mu_i c_i)$"
        "  ·  每 N 步 / no_grad  ·  由违反 $c_i$ 驱动",
        ha="center", va="bottom", fontsize=9.5, style="italic", color="#444444")

# result banner
box(6.5, 1.15, 11.8, 0.65,
    "关键：λ 由违反驱动（非 1/L）；违反归零则 λ 停止增长 —— 避免无噪声下界的病态",
    fontsize=11, fill="#ECECEC")

# failure note (dashed)
box(6.5, 0.35, 11.8, 0.6, "", edge="#666666", lw=1.2, ls="--")
ax.text(6.5, 0.35,
        "本项目实测：失败。背景残差恒正 → λ_bg 无界增长 → 压过振幅 → 重建塌缩。",
        ha="center", va="center", fontsize=9, color="#444444")

plt.tight_layout()
plt.savefig("al_pinns.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved al_pinns.png")
