"""Clean, paper-style PINN (Physics-Informed Neural Network) diagram, black & white."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15.5, 7.4))
ax.set_xlim(-0.2, 16.8)
ax.set_ylim(0.8, 8.2)
ax.set_aspect("equal")
ax.axis("off")


def box(cx, cy, w, h, text, fontsize=9.5, fill="white"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fill,
                           edgecolor="black", linewidth=1.2, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return cx, cy, w, h


def arrow(p1, p2, lw=1.2, ls="-", color="black", z=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, linestyle=ls, color=color, zorder=z))


def edge(b, side):
    cx, cy, w, h = b
    return {"r": (cx + w / 2, cy), "l": (cx - w / 2, cy),
            "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}[side]


# ------------------------------------------------------------- main boxes
inp = box(1.0, 4.0, 1.3, 0.9, "输入\n$(x,\\,t)$")
out = box(6.3, 4.0, 1.2, 0.9, "$u(x,\\,t)$")
autodiff = box(8.7, 6.2, 1.9, 0.8, "自动微分", fontsize=9)
residual = box(11.1, 6.2, 2.3, 0.9,
               "PDE 残差\n$u_t + \\mathcal{N}[u] = 0$", fontsize=9)
lpde = box(13.5, 6.2, 1.4, 0.7, "物理损失\n$L_{pde}$", fontsize=9)
data = box(8.7, 1.8, 2.1, 0.9, "边界 / 初始条件\n（已知数据）", fontsize=9)
ldata = box(13.5, 1.8, 1.4, 0.7, "数据损失\n$L_{data}$", fontsize=9)
total = box(15.3, 4.0, 2.0, 1.0, "总损失\n$L = L_{data} + L_{pde}$", fontsize=9.5)

# ------------------------------------------------------------- neural network
col_x = [2.8, 3.5, 4.2, 4.9]
col_n = [3, 5, 5, 2]
node_pos = []
for x, n in zip(col_x, col_n):
    ys = [4.0 + (i - (n - 1) / 2) * 0.5 for i in range(n)]
    col = [(x, y) for y in ys]
    node_pos.append(col)

# fully-connected edges (thin gray)
for c in range(len(col_x) - 1):
    for (x1, y1) in node_pos[c]:
        for (x2, y2) in node_pos[c + 1]:
            ax.plot([x1, x2], [y1, y2], color="#CCCCCC", linewidth=0.5, zorder=1)

# nodes
for col in node_pos:
    for (x, y) in col:
        ax.add_patch(Circle((x, y), 0.12, facecolor="white",
                            edgecolor="black", linewidth=1.0, zorder=3))
ax.text(3.85, 2.5, "神经网络  $u_{\\theta}(x,\\,t)$",
        ha="center", va="center", fontsize=10)

# ------------------------------------------------------------- flow arrows
arrow((1.65, 4.0), (col_x[0] - 0.15, 4.0))                 # input -> net
arrow((col_x[-1] + 0.15, 4.0), edge(out, "l"))              # net -> output
# output branches
arrow(edge(out, "r"), edge(autodiff, "l"))                  # up to autodiff
arrow(edge(out, "r"), edge(data, "l"))                      # down to data
# physics path
arrow(edge(autodiff, "r"), edge(residual, "l"))
arrow(edge(residual, "r"), edge(lpde, "l"))
arrow(edge(lpde, "r"), edge(total, "t"))
# data path
arrow(edge(data, "r"), edge(ldata, "l"))
arrow(edge(ldata, "r"), edge(total, "b"))

# ------------------------------------------------------------- middle detail boxes
# how the differentiation is computed (autodiff)
ux, uy, uw, uh = 10.8, 4.9, 5.2, 1.0
ax.add_patch(Rectangle((ux - uw / 2, uy - uh / 2), uw, uh, facecolor="white",
                       edgecolor="#666666", linewidth=1.0, linestyle="--", zorder=3))
ax.text(ux, uy + uh / 2 - 0.12, "自动微分（autograd）", ha="center", va="top",
        fontsize=9.5, fontweight="bold", zorder=4)
ax.text(ux, uy - 0.05,
        r"$\frac{\partial u}{\partial t},\ \frac{\partial u}{\partial x},\ "
        r"\frac{\partial^2 u}{\partial x^2},\ \dots$",
        ha="center", va="center", fontsize=10, zorder=4)
ax.text(ux, uy - uh / 2 + 0.10, "链式法则对网络求导，无需离散网格",
        ha="center", va="bottom", fontsize=8.5, color="#444444", zorder=4)

# how the losses are computed (MSE)
lx, ly, lw, lh = 10.8, 3.15, 5.2, 1.2
ax.add_patch(Rectangle((lx - lw / 2, ly - lh / 2), lw, lh, facecolor="white",
                       edgecolor="#666666", linewidth=1.0, linestyle="--", zorder=3))
ax.text(lx, ly + lh / 2 - 0.12, "损失（均方误差 MSE）", ha="center", va="top",
        fontsize=9.5, fontweight="bold", zorder=4)
ax.text(lx, ly + 0.10,
        r"$L_{pde}=\frac{1}{N}\sum\left|u_t+\mathcal{N}[u]\right|^2$",
        ha="center", va="center", fontsize=9.5, zorder=4)
ax.text(lx, ly - 0.24,
        r"$L_{data}=\frac{1}{M}\sum\left|u-u_{data}\right|^2$",
        ha="center", va="center", fontsize=9.5, zorder=4)

# ------------------------------------------------------------- backpropagation (dashed rail)
rail_y = 7.35
net_x = 3.85
tl_x = total[0]
ax.plot([tl_x, tl_x, net_x, net_x], [edge(total, "t")[1] + 0.02, rail_y, rail_y, 5.25],
        color="#555555", linewidth=1.3, linestyle=(0, (5, 3)), zorder=2,
        solid_capstyle="round")
arrow((net_x, 5.25), (net_x, 5.03), lw=1.3, color="#555555")
ax.text((tl_x + net_x) / 2, rail_y + 0.16, "反向传播（更新 $\\theta$）",
        ha="center", va="bottom", fontsize=9.5, style="italic", color="#444444")

# ------------------------------------------------------------- title
ax.text(8.0, 7.95, "物理约束神经网络（PINN）",
        ha="center", va="bottom", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("pin_diagram.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved pin_diagram.png")
