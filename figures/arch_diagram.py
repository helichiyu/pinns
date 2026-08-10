"""Draw a clean, paper-style iterative U-Net architecture diagram (black & white)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 6.8))
ax.set_xlim(-1.7, 15.9)
ax.set_ylim(0.35, 7.35)
ax.set_aspect("equal")
ax.axis("off")

# block: name, cx, cy, side, channels, fill
ENC, DEC, BTL, OUTC = "#C9C9C9", "#ECECEC", "#969696", "#FFFFFF"
blocks = [
    ("E1", 1.3, 4.70, 1.30, "32", ENC),
    ("E2", 2.7, 3.80, 0.85, "64", ENC),
    ("E3", 4.1, 3.05, 0.55, "128", ENC),
    ("E4", 5.5, 2.40, 0.36, "256", ENC),
    ("B",  6.9, 1.85, 0.24, "256", BTL),
    ("D4", 8.3, 2.40, 0.36, "128", DEC),
    ("D3", 9.7, 3.05, 0.55, "64", DEC),
    ("D2", 11.1, 3.80, 0.85, "32", DEC),
    ("D1", 12.5, 4.70, 1.30, "32", DEC),
    ("O",  13.9, 4.70, 1.00, "1", OUTC),
]
b = {n: (cx, cy, s, ch, fill) for (n, cx, cy, s, ch, fill) in blocks}
order = ["E1", "E2", "E3", "E4", "B", "D4", "D3", "D2", "D1", "O"]


def edge(name, side):
    cx, cy, s = b[name][0], b[name][1], b[name][2]
    return {"r": (cx + s / 2, cy), "l": (cx - s / 2, cy),
            "t": (cx, cy + s / 2), "b": (cx, cy - s / 2)}[side]


# blocks + channel labels
for n in order:
    cx, cy, s, ch, fill = b[n]
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, facecolor=fill,
                           edgecolor="black", linewidth=1.3, zorder=3))
    ax.text(cx, cy + s / 2 + 0.14, ch, ha="center", va="bottom",
            fontsize=9, zorder=4)

# main flow arrows (right edge -> left edge of next block)
for i in range(len(order) - 1):
    a, c = order[i], order[i + 1]
    ax.add_patch(FancyArrowPatch(edge(a, "r"), edge(c, "l"),
                                 arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color="black", zorder=2))

# skip / concat connections (dashed, encoder -> decoder)
skips = [("E1", "D1", 5.72), ("E2", "D2", 4.58),
         ("E3", "D3", 3.68), ("E4", "D4", 2.93)]
for a, c, y in skips:
    ax.add_patch(FancyArrowPatch((b[a][0] + b[a][2] / 2, y),
                                 (b[c][0] - b[c][2] / 2, y),
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, linestyle=(0, (4, 3)),
                                 color="#555555", zorder=1))
ax.text((b["E1"][0] + b["D1"][0]) / 2, 5.90, "跳跃拼接（skip）",
        ha="center", va="bottom", fontsize=8.5, color="#555555")

# input
ax.text(-0.7, 4.90, "输入", ha="center", va="bottom",
        fontsize=11, fontweight="bold")
ax.text(-0.7, 4.58, "Patterson map", ha="center", va="top", fontsize=9)
ax.text(-0.7, 4.28, "$\\mathrm{IFFT}(|F|^2 \\cdot \\mathrm{mask})$", ha="center", va="top",
        fontsize=8, color="#555555")
ax.add_patch(FancyArrowPatch((-0.15, 4.70), edge("E1", "l"),
                             arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.3, color="black"))

# output + loss
ax.add_patch(FancyArrowPatch(edge("O", "r"), (14.95, 4.70),
                             arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.3, color="black"))
ax.text(15.25, 4.80, "损失", ha="center", va="bottom",
        fontsize=12, fontweight="bold")

# iterative loop: output -> input, rectangular rail over the top
ox = b["O"][0]
otop = b["O"][1] + b["O"][2] / 2
rail_y = 6.30
left_x = -0.3
ax.plot([ox, ox, left_x, left_x], [otop + 0.02, rail_y, rail_y, 5.25],
        color="black", linewidth=1.6, zorder=5, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((left_x, 5.25), (left_x, 4.85),
                             arrowstyle="-|>", mutation_scale=16,
                             linewidth=1.6, color="black", zorder=5))
ax.text((ox + left_x) / 2, rail_y + 0.12,
        "迭代：下一轮输入 = 上一轮输出（detach）",
        ha="center", va="bottom", fontsize=9.5, style="italic")

# title + caption
ax.text(6.85, 6.98, "迭代式 U-Net", ha="center", va="bottom",
        fontsize=13, fontweight="bold")
ax.text(6.85, 0.50,
        "矩形大小代表特征图分辨率　·　块上方数字为通道数　·　"
        "深色为编码器，浅色为解码器",
        ha="center", va="bottom", fontsize=8.5, color="#444444")

plt.tight_layout()
plt.savefig("arch_diagram.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved arch_diagram.png")
