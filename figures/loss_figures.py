"""Two explanatory figures for the loss-function slide (black & white, minimal)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# consistent clean style
plt.rcParams.update({
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
})


# ---------------------------------------------------------------- log1p figure
fig, ax = plt.subplots(figsize=(6.6, 5))
x = np.linspace(0, 10, 400)
ax.plot(x, x, color="#999999", linestyle="--", linewidth=1.4, label=r"$y = x$（线性，无压缩）")
ax.plot(x, np.log1p(x), color="black", linewidth=2.2, label=r"$y = \log(1 + x)$")

# two example points: small value (barely moves) and large value (compressed)
for xv, ha, va in [(0.5, "left", "bottom"), (9.0, "right", "top")]:
    yv = np.log1p(xv)
    ax.scatter([xv], [yv], color="black", zorder=5, s=18)
    ax.plot([xv, xv], [0, yv], color="#555555", linestyle=":", linewidth=0.9)
    ax.plot([0, xv], [yv, yv], color="#555555", linestyle=":", linewidth=0.9)
ax.annotate("小值：几乎不变\n(0.5 → 0.40)", xy=(0.5, 0.405), xytext=(1.6, 1.5),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))
ax.annotate("大值：被压缩\n(9 → 2.3)", xy=(9, 2.30), xytext=(6.0, 4.2),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("输入值 x（归一化振幅）")
ax.set_ylabel(r"映射值  $\log(1 + x)$")
ax.set_title("log1p：压缩大值，保留小值", fontsize=11)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
plt.tight_layout()
plt.savefig("fig_log1p.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------- histogram + soft CDF figure
rng = np.random.default_rng(0)
data = np.concatenate([
    rng.beta(2.0, 6.0, size=650) * 0.45,        # background cluster near 0
    rng.beta(5.0, 3.0, size=350) * 0.55 + 0.35,  # object cluster mid-high
])
data = np.clip(data, 0.0, 1.0)

fig, ax1 = plt.subplots(figsize=(7.2, 5))
ax2 = ax1.twinx()

# histogram (density) -- gray bars
counts, edges = np.histogram(data, bins=26, range=(0, 1))
widths = np.diff(edges)
density = counts / counts.sum() / widths
ax1.bar(edges[:-1], density, width=widths, align="edge",
        color="#D8D8D8", edgecolor="#888888", linewidth=0.6, zorder=2,
        label="强度直方图")

# hard CDF (cumulative sum) -- non-differentiable step function
cdf_hard = np.concatenate([[0], np.cumsum(counts) / counts.sum()])
ax2.step(edges, cdf_hard, where="post", color="#777777", linestyle="--",
         linewidth=1.5, zorder=4, label="硬 CDF（阶跃，不可导）")

# soft CDF built from sigmoids (as in losses.py) -- differentiable
centers = np.linspace(0, 1, 400)
softness = 0.02
soft_cdf = (1.0 / (1.0 + np.exp(-(centers[None, :] - data[:, None]) / softness))).mean(axis=0)
ax2.plot(centers, soft_cdf, color="black", linewidth=2.2, zorder=5,
         label="软 CDF（sigmoid 平滑，可导）")

ax1.set_xlim(0, 1)
ax1.set_ylim(0, density.max() * 1.25)
ax2.set_ylim(0, 1)
ax1.set_xlabel("像素强度")
ax1.set_ylabel("密度")
ax2.set_ylabel("累积概率")
ax1.set_title("sigmoid 平滑得到的可导 CDF", fontsize=11)

# combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8.2,
           framealpha=0.92, ncol=1)

# small inset: the sigmoid shape itself
ax_in = ax1.inset_axes([0.62, 0.20, 0.32, 0.32])
xs = np.linspace(-6, 6, 200)
ax_in.plot(xs, 1 / (1 + np.exp(-xs)), color="black", linewidth=1.6)
ax_in.axhline(0.5, color="#BBBBBB", linewidth=0.7, linestyle=":")
ax_in.axvline(0.0, color="#BBBBBB", linewidth=0.7, linestyle=":")
ax_in.set_title("sigmoid", fontsize=8)
ax_in.tick_params(labelsize=6)
ax_in.grid(False)

plt.tight_layout()
plt.savefig("fig_histogram.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("saved fig_log1p.png and fig_histogram.png")
