"""不确定性加权的好处：+s 正则项使权重存在唯一最优解（黑白极简）。

只画一件事：单独的 e^{-s}·L̂ 单调下降、权重无界增长；加上 +s 之后出现
唯一极小点 s* = ln L̂，网络与权重可联合优化，无需人工调权。
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#E4E4E4",
    "grid.linewidth": 0.6,
})

fig, axis = plt.subplots(figsize=(8.2, 5.6))
s = np.linspace(-2.0, 4.0, 600)

axis.plot(s, np.exp(-s), color="#A0A0A0", linestyle="--", linewidth=1.7,
          label=r"$e^{-s_i}\hat{L}_i$：单调下降，无极小点")
axis.plot(s, np.exp(-s) + s, color="black", linewidth=2.6,
          label=r"$e^{-s_i}\hat{L}_i+s_i$：存在唯一极小点")

axis.scatter([0.0], [1.0], color="black", zorder=6, s=55)
axis.annotate("极小点\n" r"$s_i^\ast=\ln\hat{L}_i$，此处 $e^{-s_i}\hat{L}_i=1$",
              xy=(0.0, 1.0), xytext=(0.85, 2.55), fontsize=11.5, ha="left",
              bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                        edgecolor="#BBBBBB", linewidth=0.9),
              arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.1))
axis.annotate(r"权重 $e^{-s_i}$ 无界增长",
              xy=(-1.55, 4.71), xytext=(-1.9, 6.15), fontsize=11, ha="left",
              color="#555555",
              arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.9))

axis.set_xlim(-2.0, 4.0)
axis.set_ylim(0.0, 6.8)
axis.set_xlabel(r"第 $i$ 项的可学习参数 $s_i=\log\sigma_i^2$（越小则权重 $e^{-s_i}$ 越大）",
                fontsize=12)
axis.set_ylabel(r"方括号 $[\,e^{-s_i}\hat{L}_i+s_i\,]$ 的取值", fontsize=12)
axis.set_title("不确定性加权：单项的方括号函数\n"
               r"总损失 $L=\sum_i w_i\,[\,e^{-s_i}\hat{L}_i+s_i\,]$，"
               r"图中取归一化损失 $\hat{L}_i=1$（训练起点）", fontsize=13)
axis.legend(loc="upper center", fontsize=11, framealpha=1.0, edgecolor="#CCCCCC")

fig.tight_layout()
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "uncertainty_objective.png")
fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved " + save_path)

# 自检：谷底应在 s=0、取值为 1（L=1 时）
grid = np.linspace(-2.0, 4.0, 600001)
print("谷底 s = %+.5f（理论 0），取值 %.5f（理论 1）" % (
    grid[np.argmin(np.exp(-grid) + grid)], np.min(np.exp(-grid) + grid)))
