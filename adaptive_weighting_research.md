# 自适应损失权重调研报告（PINNs ↔ 本项目相位恢复）

> 背景：本项目的四项损失（振幅/直方图/背景/轮廓占比）采用固定权重，且不同图片需要重新调参。
> 本报告梳理 PINNs 领域解决"多损失项权重平衡"的经典方案，评估对本项目的适用性，
> 并记录对网络结构本身的优化观察。调研来源为 `C:\Users\23600\Desktop\PINNs` 文件夹中的 10 篇论文。

---

## 一、本项目与 PINNs 的对应关系（结论：高度同构）

两者本质都是**多约束加权优化**，且核心痛点完全一致——损失项量级差异巨大、不同样本需要不同权重。

| PINNs 概念 | 本项目对应 | 性质 |
|---|---|---|
| PDE 残差损失（物理一致性） | 振幅损失（傅里叶域一致性） | 本质目标 / 物理项 |
| 边界条件损失（BC） | 背景损失（轮廓外为 0） | 硬约束 |
| 初始条件损失（IC） | 轮廓占比损失（匹配标量） | 软约束 |
| 数据拟合损失 | 直方图损失（分布拟合） | 统计先验 |

当前 `config.py` 的权重（振幅 `4000`、直方图 `0.2`、背景 `3.0`、占比 `0.5`）正是手动版的"梯度量级平衡"——振幅原始损失量级约 `1e-4`，需要乘约 `1e3~1e4` 才能与其他项抗衡。这正是自适应方法的标准适用场景。

---

## 二、自适应 / 免调参权重方法分类

### 类别 1：梯度平衡类（Gradient-based）

- **Learning Rate Annealing (LRA)** — Wang et al. 2021《Understanding and Mitigating Gradient Flow Pathologies in PINNs》
  - 机制：以数据/物理主项梯度统计量为参考，为各损失项分配独立学习率。
  - 特点：不直接改权重，改有效更新步长。
- **GradNorm** — Chen et al. 2018（多任务学习）
  - 机制：动态调整权重使各损失项对共享参数的梯度范数趋于相等。
  - 本项目当前"按短测损失量级手动平衡"就是 GradNorm 的手动退化版，可平滑升级为自动版。

### 类别 2：不确定性加权（本报告首选）

- **同方差不确定性加权（Homoscedastic / Kendall）** — Kendall et al. 2017《Multi-task Learning Using Uncertainty to Weigh Losses》
  - 机制：每项损失配一个可学习的对数方差 `s_i`，总损失 `L = Σ [exp(-s_i)·L_i + s_i]`。
  - 特点：**5 行代码、完全免调参、对所有图片通用**。详见第四节。

### 类别 3：软注意力 / 可学习权重（对抗训练）

- **SA-PINNs** — McClenny & Braga-Neto 2023《Self-Adaptive Loss Balanced PINNs》
  - 机制：每项配可学习权重，通过 max-min（网络最小化损失、权重最大化聚焦难区）对抗训练。
  - 适用：本项目"前景/背景区域重要性不同"的场景。

### 类别 4：NTK 谱加权（理论支撑）

- **NTK-weighting** — Wang et al. 2022《When and Why PINNs fail to train》
  - 机制：用神经正切核（NTK）的特征值之比确定权重，保证各项收敛速率一致。
  - 特点：理论最严格，但 NTK 计算开销大，对大图（本项目千像素级）不实用。

### 类别 5：约束优化类（理论最干净、近似免调参）

- **AL-PINNs** — Son et al. 2023《Enhanced PINNs with Augmented Lagrangian》（**本文件夹**）
  - 机制：把 BC/IC 当硬约束，用拉格朗日乘子 `λ` 自适应平衡，构成序列 max-min 问题，**有收敛性证明**。
  - 对本项目映射：把**背景 + 占比当约束**（≤容差），**振幅当主目标**，`λ` 自学。
- **hPINN** — Lu et al.《PINNs with Hard Constraints for Inverse Design》（**本文件夹**）
  - 机制：penalty 法 + 增广拉格朗日，处理不等式约束的拓扑优化。
- 注意：上一份 `todo_0806_1.md` 已把"标量增广拉格朗日"列为占比项的后续候选，可作为方案 5 的落点。

### 类别 6：课程学习（不调权重，改问题难度）

- **Curriculum Regularization + Seq2Seq** — Krishnapriyan et al. 2021《Characterizing possible failure modes in PINNs》（**本文件夹**，NeurIPS）
  - 核心发现：PINN 失败不是网络表达能力不足，而是**软约束让损失地形病态、难优化**。
  - 两种解法：(a) 课程正则——从简单约束逐步加难；(b) 序列到序列——不一次性预测整个时空。
  - 对本项目：迭代反馈循环本身已带 seq2seq 味道；"课程"思路=先拟合低频振幅再逐步加高频，可作为未来实验。

---

## 三、各方法对本项目的适用性评估

| 方法 | 免调参程度 | 实现成本 | 风险 | 适配本项目 |
|---|---|---|---|---|
| **Kendall 不确定性**（类别2） | 完全免调参 | 极低（5 行） | 低 | ★ 首选 |
| GradNorm（类别1） | 需设参考比例 | 中 | 低 | 次选 |
| AL-PINNs 增广拉格朗日（类别5） | 近似免调参 | 中高 | 中 | 理论最优，后续 |
| SA-PINns 软注意力（类别3） | 需设正则 | 中 | 中 | 备选 |
| NTK 加权（类别4） | 完全免调参 | 高（计算贵） | 中 | 不推荐（大图） |
| 课程学习（类别6） | 改难度不改权重 | 中 | 中 | 未来实验 |

---

## 四、首选方案详解：同方差不确定性加权（Kendall）

### 4.1 公式

设四项原始损失为 `L_amp, L_hist, L_bg, L_area`，各配一个可学习对数方差 `s_i = log(σ_i²)`：

```
L_total = Σ_i [ exp(-s_i) · L_i + s_i ]
```

- `exp(-s_i)` 即该项的有效权重（始终为正）；
- `+ s_i` 是**关键正则项**：它阻止 `s_i → +∞`（即权重 → 0）。

### 4.2 为什么是"免调参"

对 `s_i` 求导令其为 0：`-exp(-s_i)·L_i + 1 = 0` ⇒ `exp(-s_i) = 1/L_i` ⇒ 有效权重自动等于 `1/L_i`。
也就是说**每项的有效贡献自动收敛到约 `1`，量级自动平衡**——这正是手动调权想达到的效果，但完全自动、对每张图自适应。

验证：本项目振幅原始损失约 `1e-4` ⇒ 方法会自动把振幅权重推到约 `1e4`，与人工试出的 `4000` 同量级。**这就是它能替代手动调参的原因。**

### 4.3 为什么安全（回应 `todo_0806_1.md` 的警告）

上一份 TODO 警告："不要把权重当普通可学参数和 U-Net 一起最小化，非负权重会把 `Σ w_i L_i` 推向 0。"
该警告针对的是**裸权重** `min Σ w_i L_i`（确实会塌缩）。
Kendall 方法多了 `+ s_i` 正则项，**直接化解了这个塌缩风险**：权重无法趋零（趋零会让 `s_i→+∞`、总损失增大）。因此可以安全地用同一个 Adam 一起优化 `s_i` 与网络参数。

### 4.4 初始化（关键实践）

`s_i = 0`（权重=1）会让初始时振幅项（`1e-4`）严重欠权，导致前几百轮重建漂移。
推荐**首步自适应初始化**：训练前做一次前向，取各项原始损失 `L_i_0`，令 `s_i ← log(L_i_0 + ε)`，
使得初始 `exp(-s_i)·L_i ≈ 1`。该初始化**完全自动、不引入任何图片相关超参**。

### 4.5 实施范围

三个 pipeline 的加权行完全一致，统一替换：
- `main.py:150` / `main_123.py:96` / `main_projected_123.py:88`

详见 `todos/todo_0808_1.md`。

---

## 五、网络结构层面的优化观察（未来方向，本次不实施）

> 调研中发现的、**损失权重之外**的可能改进点。仅作记录，本次 Kendall 改动**不触碰网络结构**，
> 以保证可独立验证。是否实施需另行开 TODO。

1. **Sigmoid 输出饱和**（`model.py:35` / `model_projected_123.py:36`）
   - 输出端 `Sigmoid()` 在接近 0/1 时梯度趋零。背景损失把轮廓外推向 0，处于 Sigmoid 近线性区（尚可）；
     但物体亮峰饱和区梯度小，可能拖慢高频细节收敛。
   - 可选：去掉 Sigmoid、改用 `clamp(0,1)` 或 soft-clamp。风险中等（改变输出统计），需对照实验。

2. **跨迭代无梯度传递**（`main.py:154` 等的 `current_input = prediction.detach()`）
   - 当前是纯定点迭代，迭代步之间梯度被截断。
   - PINNs/递归方法有时受益于**截断反向传播（BPTT 数步）**或可学习步长。但 detach 是当前刻意设计，
     改动属独立实验，不并入本次。

3. **谱偏置（Spectral Bias）** — 来自失败模式论文与 PINNs 通识
   - 网络倾向于先学低频。相位恢复中高频细节重要，可能受限。
   - 可选：Fourier-feature 输入映射、SIREN 正弦激活。属较大改动，列为远期方向。

4. **课程正则化** — 来自 `Characterizing possible failure modes in PINNs`
   - 先拟合低频振幅目标，逐步引入高频，缓解病态优化。与本次不确定性加权**正交、可叠加**。

5. **ProjectedUNet 的物理输入**（`model_projected_123.py`）
   - 已采用振幅投影作为第二输入通道，这正是 PINNs"物理约束嵌入网络"的方向，设计合理，保留。

---

## 六、本文件夹论文索引

| 论文 | 与本项目的相关性 | 对应方法 |
|---|---|---|
| AL-PINNs (Augmented Lagrangian) | 高 | 类别5（约束优化/增广拉格朗日） |
| Characterizing possible failure modes in PINNs | 高 | 类别6（课程/seq2seq）+ 病态分析 |
| Physics Constrained Unsupervised DL (PtychoPINN) | **极高（同为相位恢复/CDI）** | 物理约束网络做相位恢复，方法论同源 |
| PINNs with hard constraints (hPINN) | 中高 | 类别5（penalty + 增广拉格朗日） |
| Physics Informed Deep Learning Part I (Raissi 原始) | 中 | PINNs 基础（软约束加权起源） |
| Exploring PINNs (综述) | 中 | 各类方法综述 |
| PINNs for fluid mechanics review | 低-中 | 应用综述 |
| PINNs for PDE-Constrained Optimization and Control | 低 | 最优控制视角 |
| PINNs Forward/Inverse with Limited Noisy Data | 低 | 噪声/少数据 |
| PINNs for transformed geometries and manifolds | 低 | 几何变换 |

> 建议：**PtychoPINN** 值得单独精读——它就是用无监督物理约束网络做相干衍射相位恢复，与本项目设定几乎同源（实空间约束 + 衍射前向模型），可借鉴其约束构造与训练策略。
