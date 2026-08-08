"""Current method: relative-share (credibility-ranked) loss weighting -- B/W figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.5)
ax.set_aspect("equal")
ax.axis("off")


def box(cx, cy, w, h, text, fontsize=10, fill="white", edge="black", lw=1.2, ls="-"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=lw, linestyle=ls, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)


# title
ax.text(6.0, 7.15, "当前方法：相对占比（按可信度排定贡献比）", ha="center", va="bottom",
        fontsize=14, fontweight="bold")
ax.text(6.0, 6.78, "（当前所用方法 · 结果待出）", ha="center", va="bottom",
        fontsize=9.5, color="#666666")

# formula card
box(6.0, 6.12, 8.8, 0.55,
    r"$\beta_i=\mathrm{SHARE}_i\times\frac{L_{amp,0}}{L_{i,0}}$",
    fontsize=14)
ax.text(6.0, 5.55, "各项初始加权贡献 = SHARE$_i$ × L$_{amp}$（振幅项的一个固定占比）",
        ha="center", va="center", fontsize=9.5, color="#444444")

# bars: (label, share, credibility, fill)
rows = [("振幅", 1.0, "唯一的硬数据（锚点，权重恒 1）", "#8C8C8C"),
        ("占比", 0.5, "可信标量，抗塌缩主力", "#CCCCCC"),
        ("直方图", 0.30, "统计先验，只约束分布", "#CCCCCC"),
        ("背景", 0.10, "由网络自身轮廓推出，最不可信", "#CCCCCC")]
bar_x0 = 2.75
scale = 3.9
y_centers = [4.85, 4.05, 3.25, 2.45]
bar_h = 0.5

for (lab, share, note, fill), y in zip(rows, y_centers):
    ax.text(2.6, y, lab, ha="right", va="center", fontsize=10.5, zorder=4)
    width = share * scale
    ax.add_patch(Rectangle((bar_x0, y - bar_h / 2), width, bar_h, facecolor=fill,
                           edgecolor="black", linewidth=1.0, zorder=3))
    ax.text(bar_x0 + width + 0.12, y, "{:.2f}".format(share), ha="left", va="center",
            fontsize=10, fontweight="bold", zorder=4)
    ax.text(7.5, y, note, ha="left", va="center", fontsize=9.5, color="#333333",
            zorder=4)

# reference tick: amplitude bar = 100% reference line
ax.plot([bar_x0, bar_x0], [2.45 - 0.45, 4.85 + 0.45], color="#999999",
        linewidth=0.8, zorder=1)
ax.text(bar_x0, 1.95, "0", ha="center", va="top", fontsize=8.5, color="#888888")
ax.text(bar_x0 + scale, 1.95, "振幅 = 1.0（参照）", ha="center", va="top",
        fontsize=8.5, color="#888888")

# bottom note
box(6.0, 1.5, 11.2, 0.6,
    "振幅为锚点（权重恒 1）；其余 β 由首次前向自动标定，贡献占比固定、不随训练变化 → 免调参。",
    fontsize=10, fill="#ECECEC")

# contrast (dashed)
box(6.0, 0.7, 11.2, 0.55, "", edge="#666666", lw=1.2, ls="--")
ax.text(6.0, 0.7,
        "非等权（Kendall）、非自适应乘子（AL-PINNs）—— 按可信度排定的固定相对占比。",
        ha="center", va="center", fontsize=9.5, color="#444444")

plt.tight_layout()
plt.savefig("relative_share.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved relative_share.png")
