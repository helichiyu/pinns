"""Kendall homoscedastic-uncertainty loss weighting -- flow-style figure (B/W)."""

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
ax.text(6.5, 7.5, "Kendall 不确定性加权（同方差不确定性）", ha="center", va="bottom",
        fontsize=14, fontweight="bold")
ax.text(6.5, 7.15, "Kendall et al., 2017  ·  arXiv:1705.07115", ha="center", va="bottom",
        fontsize=9, color="#666666")

# formula card (compact, no long annotation under it)
box(6.5, 6.45, 9.2, 0.6,
    r"$L_{total}=\sum_i\left[\,e^{-s_i}\,L_i\;+\;s_i\,\right]$",
    fontsize=15)

# column headers
ax.text(1.8, 5.95, "损失项", ha="center", va="center", fontsize=10, fontweight="bold")
ax.text(4.0, 5.95, "可学习权重", ha="center", va="center", fontsize=10, fontweight="bold")

# 4 losses + 4 weight gates
labels = ["振幅 $L_{amp}$", "直方图 $L_{hist}$", "背景 $L_{bg}$", "占比 $L_{area}$"]
ys = [5.25, 4.55, 3.85, 3.15]
for y in ys:
    box(1.8, y, 2.0, 0.55, "", fontsize=10)
    box(4.0, y, 1.5, 0.55, r"$e^{-s_i}$", fontsize=11)
    arrow((2.8, y), (3.25, y))                       # loss -> weight
for y in ys:
    arrow((4.75, y), (5.96, 4.2))                    # weight -> Sigma (converge)
# loss labels (drawn after so they sit in the boxes)
for y, lab in zip(ys, labels):
    ax.text(1.8, y, lab, ha="center", va="center", fontsize=9.5, zorder=4)

# Sigma + L_total
ax.add_patch(Circle((6.2, 4.2), 0.24, facecolor="white", edgecolor="black",
                    linewidth=1.3, zorder=3))
ax.text(6.2, 4.2, r"$\Sigma$", ha="center", va="center", fontsize=11, zorder=4)
box(8.5, 4.2, 1.7, 0.65, r"$L_{total}$", fontsize=12)
arrow((6.44, 4.2), (7.65, 4.2))

# regularizer feeds the sum
box(6.2, 2.05, 2.2, 0.6, r"$+\ \sum_i s_i$（正则）", fontsize=9.5, fill="#F2F2F2")
arrow((6.2, 2.35), (6.2, 3.96))
ax.text(7.5, 2.05, "阻止权重→0", ha="left", va="center", fontsize=8.5, color="#666666")

# optimization feedback: s_i tuned jointly -> weight = 1/L_i
ax.plot([8.5, 8.5, 4.0, 4.0], [4.55, 5.55, 5.55, 5.55],
        color="#555555", lw=1.3, ls=(0, (5, 3)), zorder=2)
arrow((4.0, 5.55), (4.0, 5.28), lw=1.3, color="#555555")
ax.text(6.25, 5.72, r"优化 $s_i$（与 $\theta$ 共用 Adam）$\Rightarrow\ e^{-s_i}=\frac{1}{L_i}$",
        ha="center", va="bottom", fontsize=9.5, style="italic", color="#444444")

# result banner
box(6.5, 1.15, 11.8, 0.65,
    "结果：各项加权贡献自动趋近 1，量级自动平衡 —— 免调参",
    fontsize=11, fill="#ECECEC")

# failure note (dashed)
box(6.5, 0.35, 11.8, 0.6, "", edge="#666666", lw=1.2, ls="--")
ax.text(6.5, 0.35,
        "本项目实测：失败。损失无噪声下界（背景项 L→0）→ 不动点 1/L→∞；"
        "s_i 跟不上 L 下降，振幅被牺牲、重建塌缩。",
        ha="center", va="center", fontsize=9, color="#444444")

plt.tight_layout()
plt.savefig("kendall_weighting.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved kendall_weighting.png")
