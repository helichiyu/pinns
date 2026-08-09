# 等点数直方图调研与实施方案（2026-08-09，2026-08-10 实施）

> 背景：导师建议把轮廓内直方图损失从"等宽分箱"改为"等点数分箱"。本文档记录
> 文献调研、方案确定过程（含一个被否决的中间方案）、最终实现方案、B 取值结论。
>
> **实施状态（2026-08-10）**：已整体替换，`soft_cdf` 与 `masked_histogram_cdf_loss`
> 已删除，不保留新旧对照开关；`HISTOGRAM_BINS` 由 64 改为 256，`HISTOGRAM_SOFTNESS`
> 随 sigmoid 软化一并移除。B 的取值依据见第七节。
>
> 配套记录：本轮同时讨论了"损失权重改可训练"的改动一，结论是**不改**，见末尾附录。

---

## 一、导师建议的实质

当前 `soft_cdf`（`losses.py:57-60`）是**等宽分箱**：在 [0,1] 上取 64 个等距横坐标
（centers），对每个 center 算"轮廓内有多少比例的像素值 ≤ 这里"（用 sigmoid 软化），
得到一条长度 64 的 CDF 曲线，比较源图与输出的两条曲线。

即：**固定横坐标（64 个值），比纵坐标（累计比例）。**

导师建议改为**等点数分箱**：每块包含的像素数相等，块的横坐标宽度随密度变化。

即：**固定纵坐标（每箱点数相等），比横坐标（箱边界 = 分位点值）。**

两者互为反函数——一个固定 x 比 y，一个固定 y 比 x。

---

## 二、文献支撑（蛋白质相位恢复的原生方法）

等点数分箱（quantile / equal-frequency / equal-depth binning）在数据挖掘里是经典方法，
更重要的是：**它正是晶体学相位恢复里 histogram matching 的原生形态**，本项目现在的等宽
CDF 反倒是简化近似版。

1. **Zhang & Main (1990)**, *Histogram matching as a density modification technique for phase
   refinement and extension of macromolecules*——蛋白质晶体学相位恢复的 histogram matching，
   用的就是按秩重映射（sort by intensity → reassign）。是 SQUASH 软件的核心模块之一。
2. **Elser (2003)**, difference map（[arXiv:math/0111080](https://arxiv.org/abs/math/0111080)）——
   明确把 object histogram 列为支撑约束的替代品：
   > "When support constraints are replaced by object histogram or atomicity constraints, the
   > difference map lends itself to crystallographic phase retrieval."
3. **Bricogne / Millane** 系列工作：晶体学 histogram matching 的统计基础都是按秩（rank）形式。

经典实现就是"排序后重映射"——sorting pixels according to intensity, then assigning new
values so that for each intensity exactly hᵢ pixels have that intensity。

**结论：导师提的这种划分方式，是相位恢复里 histogram matching 的原生形态，方向正确。**

---

## 三、被否决的中间方案：全排序逐元素比较

调研中曾考虑把"等点数分箱"推到极致——B = k（像素总数），即"逐元素比较两组排序后的
像素值"：

```
target_sorted = sort(source_values)        # 长度 k
pred_sorted   = sort(pred_values)          # 长度 k
loss = mse(pred_sorted, target_sorted)
```

实测上它很诱人：23 ms（与 soft_cdf 20.7 ms 相当）、无中间显存、无超参、torch.sort 原生
可导、约束最强（损失为 0 ⇔ 轮廓内取值集合完全一致）。

**但它有一个根本性错误：违反相位恢复的问题设定。**

`target_sorted` 要求知道源图轮廓内 k 个像素**每个 rank 上的精确值**，这等于掌握了原图
的完整像素值集合（只是打乱了位置）。在真实相位恢复场景里，原图就是要恢复的目标，根本
拿不到这套值。直接 `sort(source)` 算 target 等于**用答案本身做监督**，指标好看但不可信。

直方图先验的本质是"一个粗糙的分布形态"，信息量就是 B 个数。无论等宽还是等点数，输出
形态都必须是 B 个箱。**全排序（B = k）不是直方图，是原图信息，不可用。** 这不是"哪个
更好"的选择题，是物理设定的硬约束。

---

## 四、确定的方案：B 箱等点数，比较分位点值

等点数分箱定义下每箱点数都相等，所以比的**只能是 B 个分位点的"值"**（不是箱内计数——
比计数是常数，没意义）。

### 4.1 两组分布的形态

target 和 pred 经过处理后，**都是长度为 B 的向量**：

```
target_quantiles = [t_1, t_2, ..., t_B]   # 训练前算一次，缓存，detach 常量
pred_quantiles   = [q_1, q_2, ..., q_B]   # 每轮算，可导
```

每个元素是该 rank（分位点）对应的像素值。比如 `q_3` = "输出轮廓内第 `3k/B` 名亮的那个
像素的值"。

### 4.2 loss 就是普通 MSE

```
loss = (1/B) · Σ_i (q_i − t_i)²
```

与当前 `soft_cdf` 末尾的 `F.mse_loss` 完全同形，只是比较对象从"CDF 曲线上的点"换成
"分位值"。

### 4.3 实现伪代码

```python
# target: computed once before training, cached, detached
target_values   = source[source_mask > 0.5]                 # k pixels
target_sorted   = torch.sort(target_values).values
ranks           = (torch.arange(1, B + 1, device=...) * k // B) - 1  # 0-indexed, length B
target_quantiles = target_sorted[ranks].detach()            # cached

# pred: every iteration
pred_values     = prediction[contour > 0.5]                 # k pixels
pred_sorted     = torch.sort(pred_values).values
pred_quantiles  = pred_sorted[ranks]                        # length B, differentiable
loss            = F.mse_loss(pred_quantiles, target_quantiles)
```

`ranks` 是固定的整数数组，每轮复用；两侧像素数严格相等（`topk_contour` 锁死 k），rank
一一对应，无需插值或长度补偿。

---

## 五、可导性与梯度回流

target 是常量不用管。pred 侧从 `prediction` 到 `q_i` 经过四步，**每一步都原生可导**：

```
prediction (N 个像素)
   │  ① 用 contour mask 选出轮廓内像素（contour 是常量）
   ▼
pred_values (k 个像素)
   │  ② torch.sort 升序
   ▼
pred_sorted (k 个像素，已排好)
   │  ③ 用固定的 rank 索引取 B 个分位值（gather）
   ▼
pred_quantiles (B 个值)  ← 与 target_quantiles 做 MSE
```

- **①选像素**：`prediction[contour>0.5]`，梯度直接回到这些像素位置（与当前
  `masked_histogram_cdf_loss` 第一步相同，`losses.py:69`）。
- **②排序**：`torch.sort(...).values` 原生可导。反向时，对 `sorted[j]` 的梯度按
  `indices[j]` **散射回原始位置**——"排在第 j 名的那个像素"领走这份梯度。
- **③取分位值**：`pred_sorted[ranks]`，固定整数索引，是 `gather`，梯度回到对应 rank
  的元素。
- **④MSE**：标准可导。

### 小例子（k=6, B=3, rank 取第 2/4/6 名 = 0-indexed [1,3,5]）

```
original = [0.3, 0.05, 0.2, 0.15, 0.1, 0.25]   # 图像位置 0..5
sorted   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
indices  = [   1,    4,    3,    2,    5,    0]
pred_quantiles = [sorted[1], sorted[3], sorted[5]] = [0.10, 0.20, 0.30]
target_quantiles = [0.0, 0.2, 0.5]

loss = mean([0.10², 0², 0.20²]) = 0.0167
```

反传 `∂loss/∂q_i = (2/B)(q_i − t_i)`：

| 原位置 | 值   | rank      | 收到梯度  | 含义                |
|--------|------|-----------|-----------|---------------------|
| 0      | 0.30 | 第6(采样) | −0.1333   | 偏高，往下压        |
| 1      | 0.05 | 第1(未采) | 0         | —                   |
| 2      | 0.20 | 第4(采样) | 0         | —                   |
| 3      | 0.15 | 第3(未采) | 0         | —                   |
| 4      | 0.10 | 第2(采样) | +0.0667   | 偏低，往上抬        |
| 5      | 0.25 | 第5(未采) | 0         | —                   |

### 两个关键特性

1. **每轮只有 B 个像素直接收到直方图梯度。** B=64、k≈17 万时，每轮仅 64 个像素被
   直方图 loss 直接拽，其余靠振幅损失和背景损失间接训练。B 越大约束越细，但 target
   先验信息量也越大（要权衡"模拟真实场景"）。
2. **sort 的梯度是离散配对，rank 交换瞬间梯度会跳变。** 两像素值接近时微小扰动可能
   让排序顺序交换，梯度瞬间从一个跳到另一个。这和 `topk_contour` 的离散性同类，本项目
   已在用，Adam 能吸收这种抖动——但理论上它不是处处光滑的（当前 `soft_cdf` 因用 sigmoid
   反而是光滑的，这是两者一个本质差异，实施时需观察收敛是否变抖）。

---

## 六、0 值堆积的重新评估

轮廓内有 **39.1%（567）/ 58.7%（123）** 的像素值恰好为 0（sigma=16 外包络圈进来的分子
间暗缝）。

- 早期曾担心这会让"B 个箱退化"。那是针对"**比较箱内计数**"的误解——等点数定义下比计数
  是常数，没意义。
- 在"比较分位点值"的正确实现下，这一段的行为分两个阶段，且**对本项目是正面的**：
  - **收敛前（大部分训练时间）**：随机相位初始化后输出轮廓内基本没有精确的 0，低 rank
    段 target = 0 而 pred > 0，这批 rank 的梯度全部指向"往下压"。也就是说等点数分箱
    **强制"轮廓内必须有 39% / 59% 的像素归零"**，是一条锐利的硬约束。相比之下
    `soft_cdf` 因 softness=0.02 的 sigmoid 软化，在 x=0 处只能给出一个被抹开的累计
    比例约束，零值边界是模糊的。
  - **收敛后**：target 和 pred 都是 0，那段梯度为 0。但 0 值内部本来就都是 0、无需
    细分，这段不再提供梯度本身是正确的。
- 真正起约束的是"0 值与非零值的**边界 rank**"（即界定 0 值占比）+ 非零值区的分位点
  分布——非零值区会被等点数均匀覆盖整个动态范围，分辨率反而比等宽 soft_cdf 在高值区
  更高。

**结论：0 值堆积不构成反对等点数分箱的理由，导师的方向在本项目里不劣于当前等宽 soft_cdf。**

---

## 七、B 取值结论

**取 B = 256**（`config.HISTOGRAM_BINS`，原为 64）。三条依据，从强到弱：

### 7.1 零值堆积压缩了有效分辨率（本项目特有，最硬的理由）

轮廓内 39.1%（567）/ 58.7%（123）的像素恰好为 0，这段的分位值全是 0，不携带形态信息。
真正描述非零动态范围的箱数要打折：

| B | 567 零值段箱数 | 567 非零段有效箱数 | 123 非零段有效箱数 |
|---|---|---|---|
| 64  | 25  | 39  | **26**  |
| 128 | 50  | 78  | 53  |
| 256 | 100 | 156 | 106 |

B = 64 时 123 图只剩 26 个箱在描述非零分布——比当前 soft_cdf 的 64 个等宽箱（高值区确实
浪费，但低值区不浪费）不见得更细。这是提高 B 最直接的理由。

### 7.2 统计学侧的等概率分箱规则给出 230–466

卡方拟合优度检验用等概率分箱时的经典规则是 Mann–Wald：

```
c ≈ 4 · (2n² / z²_α)^(1/5)
```

代入本项目量级（k ≈ 4.9×10⁴ ~ 1.7×10⁵，α = 0.05，z = 1.645）得 c ≈ 300–470；Williams
后来证明减半到 ~150–230 也不显著损失检验效力。

值得注意的是这个规则的增长率是 n^(2/5)，比等宽分箱那套（Freedman–Diaconis / Scott，
∝ n^(1/3)）**更快**——等点数分箱天生该比等宽用更多箱，因为它不把箱浪费在空白区。

机器学习侧最贴近的类比是 QR-DQN（Dabney et al., 2017，[arXiv:1710.10044](https://arxiv.org/abs/1710.10044)），
它就是用 N 个等概率分位数逼近一个分布，最终取 N = 200。（消融细节未核实原文，仅作量级
参考。）

256 落在这两个区间的交叠处。

### 7.3 上界约束：B 不能趋近 k

B 同时是"先验信息量"的刻度，B → k 就退化成第三节否决的全排序，等于用答案监督。但这个
约束很宽松：B = 256 时 B/k ≈ 0.15%~0.5%，离泄漏差两三个数量级。**统计规则给的 230–466
与"保持先验粗糙"并不冲突，64 是白白留了余量没用。**

### 7.4 晶体学侧待查

Zhang & Main 那类 histogram matching 用的是全像素重映射（B = k 形式），没有"有限 B"的
直接参考；IUCr double-histogram 方法里的 "ten groups" 是按分辨率壳分组，不是直方图分箱，
别混。**但导师提这个方向时应有具体论文依据，晶体学侧的 B 取值文献待后续查证**，可能会
推翻或收窄上面的取值。

---

## 八、实施时的注意点与风险

1. **光滑性差异**：sort+gather 是分段线性、rank 交换处梯度跳变；soft_cdf 用 sigmoid 是
   光滑的。收敛初期（轮廓乱跳）hist 损失曲线可能比现在更抖，需观察。
2. **量级变化**：分位值 MSE 与 CDF MSE 的数值量级不同，但 `CalibratedWeights`（
   `losses.py:101-107`）首步自动标定 β_hist，会自动适配，不用手调。
3. **B 超过 k 时会静默出错**：`ranks` 首个下标变成 −1，负索引在 PyTorch 里会绕回末位，
   拿到的是错的 target 而不是报错。`train.py` 在预计算前加了 `B > k` 的显式校验。
4. **target 的真实场景来源**（远期）：当前 target 从源图算，是研究阶段的"用原图给先验"。
   纯真实场景下 target 应来自物理先验（如分子密度分布的已知形态），从先验 CDF 反推 B 个
   分位点值。这是后续课题，不影响当前实施。

---

## 附录：改动一（损失权重改可训练）的决定——不改

本轮同时讨论了把三项损失的权重改可训练：当前
`L = L_amp + β_hist·L_hist + β_bg·L_bg`（β 由首步标定固定），想改成三项各乘一个可训练
标量 `w_i`、与网络一起最小化。

**结论：不改。** 理由：

- `min_{θ,w} Σ w_i·L_i(θ)` 对 w 的最优解是无界的（w→−∞ 或 w→0），**这是目标函数本身
  决定的，与网络结构无关**。网络的大改动改变的是 L_i 的数值与收敛行为，改变不了目标
  函数对 w 的无界性。
- 若要避免塌缩必须给 w 加约束（非负 + 正则 / 对偶更新 / 梯度归一），那就不是"简单乘
  一个数字"了；且这些机制（Kendall 不确定性加权、AL-PINns、占比配乘子）已被本项目实测
  失败，失败是结构性的（无噪声下界 + 等权目标错误 + 背景残差恒正致 λ 无界）。
- 当前固定 β 本身已是健康自适应：对每张图自动标定初始量级、训练中固定、无反馈回路、
  不震荡、SHARE 是全局常数换图不用调。在拿到有理论支撑的新机制前，先不动。

完整论证与历次失败实测记录见 `adaptive_weighting_research.md`（同目录，尤其第六、七节）。
