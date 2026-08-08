# 单模型四损失相位恢复

本项目只训练一个未预训练 UNet。输入由原图的已知傅里叶振幅和随机相位构成；第 0 轮输入为该频域组合的 IFFT，之后每轮将上一轮 UNet 输出 `detach` 后作为下一轮输入。

训练不使用 HIO、RAAR，也不使用原图轮廓的位置或形状作为监督。

## 损失函数

1. 振幅损失：排除直流点和近零振幅点，在归一化对数振幅域计算 MSE，防止低频大数值主导。
2. 直方图损失：直接比较原图与输出整图的可微软 CDF 分布；不做直方图匹配，也不改变输出图。
3. 背景损失：每轮仅根据当前输出，以固定粗参数执行高斯模糊和软阈值获得动态轮廓；阈值前先减去当前模糊图最低基线，以避免初始输出的整体亮度使轮廓覆盖全图。轮廓外输出被约束为 0。轮廓反传被截断，不能通过扩大轮廓规避背景惩罚。
4. 轮廓占比损失：原图以相同固定参数计算软轮廓均值标量，输出只匹配这个软面积占比，不接收原图轮廓的位置、形状或像素级监督。

图片统一采用反转后的工作表示：黑色背景为 0，物体为亮色。

## 目录结构

```
损失/
  backend/
    config.py          集中式配置参数
    losses.py          四项损失函数 + CalibratedWeights
    model.py           UNet（单输入）
    visualization.py   matplotlib 出图与 CSV 保存
    train.py           统一训练脚本（命令行参数控制图片/画布扩大/高斯半径）
    server.py          tornado Web 服务（前端 + 实验队列 + WebSocket）
  frontend/
    index.html         界面骨架
    style.css          样式（Apple 风格）
    app.js             实验管理 + WebSocket + canvas 曲线 + 轮廓预览
  123.png / 567.png    数据图片（放根目录，不入库）
  results/             训练产物（不入库）
  start.bat            双击启动 Web 服务
```

## 命令行运行

```powershell
# 567：不扩大画布
D:\anaconda3\envs\use\python.exe backend\train.py --image 567.png --expand 1 --iterations 3000

# 123：画布扩大 2 倍（物体贴边，需过采样裕量避免 FFT wraparound）
D:\anaconda3\envs\use\python.exe backend\train.py --image 123.png --expand 2 --iterations 3000

# 自定义高斯半径（控制粗轮廓提取的模糊尺度）
D:\anaconda3\envs\use\python.exe backend\train.py --image 567.png --contour-sigma 8 --iterations 3000
```

参数说明：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image` | `config.IMAGE_PATH` | 图片文件名 |
| `--expand` | `1` | 画布扩大倍数；`1`=不扩大，`2`=每维扩大 2 倍。扩大后统一 pad 到 16 的倍数（UNet 四级下采样，固定不可调） |
| `--contour-sigma` | `config.CONTOUR_SIGMA` (16) | 高斯模糊半径，控制 `dynamic_contour` 提取粗轮廓的尺度 |
| `--contour-threshold` | `config.CONTOUR_THRESHOLD` (0.20) | 轮廓峰值阈值（0~1），越大轮廓越收紧 |
| `--iterations` | `config.ITERATIONS` (3000) | 训练轮数 |
| `--seed` | 无 | 随机种子 |
| `--cpu` | 关 | 强制 CPU |
| `--log-every` | `config.LOG_EVERY` (20) | 日志间隔 |

结果目录包含：`real_space.png`（原图 / 输出 / 叠加 / 误差）、`spectra.png`（振幅谱）、`support.png`（粗轮廓对比）、`convergence.png`（11 个指标面板）、`history.csv`、`metrics.csv`、`state.pt`。

## Web 前端

双击 `start.bat` 或手动启动：

```powershell
D:\anaconda3\envs\use\python.exe backend\server.py
```

浏览器打开 `http://localhost:8770`：

- 顶部选择实验组数（1~6），每组一行参数设定：第一行图片 + 画布扩大，第二行高斯半径 + 轮廓阈值 + 训练轮数。右上角「重置」按钮清除全部状态，方便重新配置。
- 参数行右侧实时绘制两条曲线（总损失 + IoU），自动填充行高。
- 卡片底部为结果按钮区：训练前可点「预览轮廓图」查看源图粗轮廓（加载时显示「请稍候」）；训练完成后该按钮变为「轮廓对比图」，同时激活「效果对比图」。
- 底部一整行终端输出，带「暂停」「终止」按钮。暂停通过 Windows NT API 挂起子进程，终止会 kill 当前实验并清空队列。
- 点「开始运行」后实验按顺序执行，前一组跑完自动开始下一组。

## 主要参数

集中在 `backend/config.py`：默认训练 3000 轮；粗轮廓参数为 `sigma=16.0`、阈值为基线校正后峰值的 `0.20`，以形成整个分子的连续概括外包络。离线指标及最终对比图会先消除平移和镜像歧义，再与原图比较。

四项损失用固定权重相加。振幅是唯一的硬数据，权重恒为 1；其余三项的权重由第一次前向按目标贡献比自动标定，训练中保持不变，**不需要按图片重调**：

```
β_i = SHARE_i × 首步振幅损失 / 首步第 i 项损失
```

`SHARE_i` 是该项初始贡献相对振幅项的目标占比，按各约束的可信度排定（`config.py`）：占比 `0.5`、直方图 `0.30`、背景 `0.10`，即振幅 > 占比 > 直方图 > 背景。背景取值最小，因为它是由网络自己输出的轮廓推出来的、最不可信，且权重过大会把整幅图压黑而塌缩；占比是抗塌缩的主力，故高于直方图。换图片时各项损失值都变、算出的 β 跟着变，但贡献占比不变，所以这三个常数是全局的。

先前曾给占比项配 AL-PINNs 增广拉格朗日乘子 λ（[arXiv:2205.01059](https://arxiv.org/abs/2205.01059)）做梯度上升自适应，但收敛末期 θ 与 λ 的原始-对偶追逐导致占比损失上下振荡，已改回固定权重。方法选型的完整论证、以及先前多次失败（Kendall 不确定性加权、背景配乘子、占比配乘子）的实测记录，见 `adaptive_weighting_research.md`。
