# 单模型三损失相位恢复

本项目只训练一个未预训练 UNet。输入由原图的已知傅里叶振幅和随机相位构成；第 0 轮输入为该频域组合的 IFFT，之后每轮将上一轮 UNet 输出 `detach` 后作为下一轮输入。

训练不使用 HIO、RAAR，也不使用原图轮廓的位置或形状作为监督。

## 损失函数

1. 振幅损失：排除直流点和近零振幅点，在归一化对数振幅域计算 MSE，防止低频大数值主导。
2. 轮廓内直方图损失：只在轮廓内比较原图与输出的可微软 CDF 分布。目标分布取自源图轮廓内的像素值（训练前算一次并缓存），预测分布取自输出**自己的**轮廓内像素值——不用源图轮廓的位置，否则等于把轮廓形状当成监督。两侧像素数严格相等，无需长度补偿。
3. 背景损失：轮廓外的输出被约束为 0。轮廓是常量，网络不能通过扩大轮廓规避背景惩罚。

轮廓的算法分两侧：

- **源图侧**（训练前算一次）：高斯模糊 → 减去自身最低基线 → 峰值归一化 → 硬阈值二值化，得到 0/1 的 mask。数出其中 1 的个数记为 `k`，占比 `ρ = k / N` 只用于显示。减基线是为了避免整体亮度把轮廓撑到全图；阈值是相对峰值的比例，不随图片整体亮度漂移。
- **输出侧**（每轮）：高斯模糊 → 取前 `k` 个最亮像素置 1。像素数恒为 `k`，与源图轮廓严格相等。

锁 `k` 而不是锁 `ρ`：后者要经两次浮点取整，两边像素数会差几个。用 topk 的下标散射而不是阈值比较，也是为了这个严格性——并列值在阈值下会多圈几个像素。

mask 的 `requires_grad` 为 `False`，是纯常量，只当选择器用。它本身几乎处处梯度为 0、不可导，但两个消费者都只需要"选中哪些像素"这个信息，梯度经 `prediction` 本身回流到网络。原先的轮廓占比损失需要可导轮廓（它算 `d(contour.mean())/d(pred)`），而在 top-k 构造下占比恒等相等、该损失恒为 0，因此已删除。

图片统一采用反转后的工作表示：黑色背景为 0，物体为亮色。

## 目录结构

```
损失/
  backend/
    config.py          集中式配置参数
    losses.py          三项损失函数 + 两侧轮廓 + CalibratedWeights
    model.py           UNet（单输入）
    visualization.py   matplotlib 出图与 CSV 保存
    train.py           统一训练脚本（命令行参数控制图片/画布扩大/高斯半径）
    server.py          tornado Web 服务（前端 + 实验队列 + WebSocket）
  frontend/
    index.html         界面骨架
    style.css          样式（Apple 风格）
    app.js             实验管理 + WebSocket + canvas 曲线 + 轮廓预览
  images/              数据图片（123.png / 567.png，不入库）
  tests/
    conftest.py        让测试能 import backend 模块
    test_losses.py     轮廓与损失的不变式断言
  results/             训练产物（不入库）
  start.vbs            双击启动 Web 服务
```

## 命令行运行

```powershell
# 567：不扩大画布
D:\anaconda3\envs\use\python.exe backend\train.py --image images/567.png --expand 1 --iterations 3000

# 123：画布扩大 2 倍（物体贴边，需过采样裕量避免 FFT wraparound）
D:\anaconda3\envs\use\python.exe backend\train.py --image images/123.png --expand 2 --iterations 3000

# 自定义高斯半径（控制轮廓提取的模糊尺度）
D:\anaconda3\envs\use\python.exe backend\train.py --image images/567.png --contour-sigma 8 --iterations 3000

# 自定义两项损失的初始贡献占比
D:\anaconda3\envs\use\python.exe backend\train.py --image images/567.png --share-histogram 0.5 --share-background 0.10
```

参数说明：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image` | `config.IMAGE_PATH` | 图片文件名 |
| `--expand` | `1` | 画布扩大倍数；`1`=不扩大，`2`=每维扩大 2 倍。**可为小数**（如 `1.5`）：扩大后的边长向下取整，随后统一 pad 到 16 的倍数（UNet 四级下采样，固定不可调），所以实际倍率会略高于名义值。例如 567（600×302）填 `1.5` 得到 464×912，实际约 1.54×1.52 |
| `--contour-sigma` | `config.CONTOUR_SIGMA` (16) | 高斯模糊半径，控制两侧轮廓提取的尺度（源图与每轮输出都用同一个值） |
| `--contour-threshold` | `config.CONTOUR_THRESHOLD` (0.20) | 源图轮廓的相对峰值阈值。**只在训练前用一次**，用来二值化源图 mask 并数出像素数 `k`；输出侧不用阈值，改按 `k` 取 top-k |
| `--share-histogram` | `config.SHARE_HISTOGRAM` (0.5) | 轮廓内直方图项初始贡献占振幅项的比例 |
| `--share-background` | `config.SHARE_BACKGROUND` (0.10) | 背景项初始贡献占振幅项的比例 |
| `--iterations` | `config.ITERATIONS` (3000) | 训练轮数 |
| `--seed` | 无 | 随机种子 |
| `--cpu` | 关 | 强制 CPU |
| `--log-every` | `config.LOG_EVERY` (20) | 日志间隔 |

结果目录包含：`real_space.png`（原图 / 输出 / 叠加 / 误差）、`spectra.png`（振幅谱）、`support.png`（硬轮廓对比）、`convergence.png`（3×3 共 9 个指标面板）、`history.csv`、`metrics.csv`、`state.pt`。

## Web 前端

双击 `start.vbs` 或手动启动：

```powershell
D:\anaconda3\envs\use\python.exe backend\server.py
```

浏览器打开 `http://localhost:8770`：

- 顶部选择实验组数（1~6），每组一行参数设定：第一行图片（下拉列出 `images/` 里的文件），第二行画布扩大 + 训练轮数，第三行高斯半径 + 轮廓阈值，第四行直方图权重 + 背景权重。右上角「重置」按钮清除全部状态，方便重新配置。
- 参数行右侧实时绘制两条曲线（总损失 + IoU），自动填充行高。
- 卡片底部为结果按钮区，从左到右三个：「轮廓对比图」在训练前显示处理后的原图与二值化 mask 的对比（标题栏标注轮廓占比），训练完成后改为显示 `support.png`（源图轮廓 vs 输出轮廓）；「结果对比图」显示 `real_space.png`；「结果曲线图」显示 `convergence.png`。后两个在训练完成后激活。
- 底部一整行终端输出，带「暂停」「终止」按钮。暂停通过 Windows NT API 挂起子进程，终止会 kill 当前实验并清空队列。
- 点「开始运行」后实验按顺序执行，前一组跑完自动开始下一组。

## 主要参数

集中在 `backend/config.py`：默认训练 3000 轮；轮廓参数为 `sigma=16.0`、阈值为基线校正后峰值的 `0.20`，以形成整个分子的连续概括外包络。离线指标及最终对比图会先消除平移和镜像歧义，再与原图比较。

三项损失用固定权重相加。振幅是唯一的硬数据，权重恒为 1；其余两项的权重由第一次前向按目标贡献比自动标定，训练中保持不变，**不需要按图片重调**：

```
β_i = SHARE_i × 首步振幅损失 / 首步第 i 项损失
```

`SHARE_i` 是该项初始贡献相对振幅项的目标占比，按各约束的可信度排定：轮廓内直方图 `0.5`、背景 `0.10`，即振幅 > 直方图 > 背景。背景取值最小，因为它是由网络自己输出的轮廓推出来的、最不可信，且权重过大会把整幅图压黑而塌缩。换图片时各项损失值都变、算出的 β 跟着变，但贡献占比不变，所以这两个常数是全局的；前端第四行也可以逐组覆盖。

直方图接过了原轮廓占比项的抗塌缩主力位置（原为 `0.5`）：输出全黑时轮廓内是集中在 0 的分布，而源图轮廓内全是中高亮度，CDF 差距被最大化，罚得比占比那个标量约束更狠。施压分工也因此变干净——轮廓内由直方图 + 振幅负责，轮廓外由背景 + 振幅负责，不再有两项在全画布重叠拉扯。

先前曾给占比项配 AL-PINNs 增广拉格朗日乘子 λ（[arXiv:2205.01059](https://arxiv.org/abs/2205.01059)）做梯度上升自适应，但收敛末期 θ 与 λ 的原始-对偶追逐导致占比损失上下振荡，已改回固定权重；该项现已随硬轮廓改动一并删除。方法选型的完整论证、以及先前多次失败（Kendall 不确定性加权、背景配乘子、占比配乘子）的实测记录，见 `adaptive_weighting_research.md`（记录的是历史实验，其中的四损失结构已不是当前实现）。

## 测试

```powershell
D:\anaconda3\envs\use\python.exe -m pytest tests\ -v
```

覆盖轮廓的不变式（两侧 mask 像素数严格相等、top-k 选中的确是最亮像素、mask 是常量）、梯度能经 `prediction` 回流、以及权重标定后各项贡献等于 `SHARE × 振幅损失`。
