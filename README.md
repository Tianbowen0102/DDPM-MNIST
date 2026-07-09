# DDPM + MNIST：从零实现扩散模型

> 从零构建 DDPM（Denoising Diffusion Probabilistic Models），在 MNIST 上训练无条件图像生成。不依赖 diffusers 库，深入理解扩散原理。

---

## 🎯 做什么？

从纯噪声中"生长"出手写数字。

```
训练：真实数字 → 逐步加噪 → 纯噪声（固定规则，不需要学习）
                      ↓
                U-Net 学会"预测加了什么噪声"
                      ↓
推理：纯噪声 → 逐步去噪 → 生成数字（学到的反向过程）
```

---

## 🧠 方法核心

### 扩散模型的物理直觉

> 一滴墨水滴入水中会逐渐扩散——反过来，如果我们能学会"逆转扩散"，就能从混沌中恢复秩序。

```
前向过程（固定规则，不需要学习）：
  x₀（真实图像）→ x₁ → x₂ → ... → x_T（纯噪声）

反向过程（需要学习）：
  x_T（纯噪声）→ x_{T-1} → ... → x₀（生成图像）
```

### 三个核心公式

**① 前向扩散（闭式解，一步到位）：**

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

**② 训练目标（就是 MSE 回归！）：**

$$\mathcal{L} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

**③ 采样（逐步去噪）：**

$$x_{t-1} = f(x_t, \epsilon_\theta(x_t, t), \alpha_t, \beta_t)$$

### U-Net 架构（590 万参数）

```
Input (1, 28, 28)
  → Encoder L1: ResBlock×2 (28×28, 64ch)
  → Downsample → (14×14, 128ch)
  → Encoder L2: ResBlock×2 + SelfAttention
  → Downsample → (7×7, 256ch)
  → Bottleneck: ResBlock + SelfAttention + ResBlock
  → Upsample → (14×14, 128ch)
  → Decoder L2: [Concat Skip] + ResBlock×3 + SelfAttention
  → Upsample → (28×28, 64ch)
  → Decoder L1: [Concat Skip] + ResBlock×3
  → Output: Conv (1, 28, 28)
```

关键设计：
- **GroupNorm** 替代 BatchNorm（扩散对 batch statistics 敏感）
- **Sinusoidal 时间嵌入** → MLP → 128 维条件注入
- **SelfAttention** 仅在 7×7 和 14×14（计算量与收益的平衡）
- **显式命名** 每层而非动态 ModuleList（避免通道对齐 Bug）
- **输出层零初始化**（训练初期输出≈0，更稳定）

---

## 📊 核心结果

### 训练

| 指标 | 值 |
|------|------|
| GPU | RTX 5060 Laptop (8.5 GB) |
| Epochs | 10 |
| 总耗时 | ~34 分钟 |
| 初始 Loss | 0.0735 |
| 最终 Loss | **0.0227（↓69.1%）** |
| 最大降幅 | Epoch 1→2（↓61%） |

### 采样

| 方法 | 步数 | 质量 | 速度 |
|------|------|------|------|
| **DDPM** | 1000 步 | 高质量 | 基准 |
| **DDIM** | 50 步 | 接近 DDPM | **20× 加速** |

---

## 🏗️ 项目文件

| 文件 | 内容 |
|------|------|
| `train_ddpm.py` | 完整训练 + 采样脚本（420 行，可直接运行） |
| `verify_architecture.py` | 5 项自动化验证（前向/训练步/DDPM/DDIM/EMA） |
| `ddpm_mnist.ipynb` | Jupyter Notebook，交互式学习版 |
| `从零构建DDPM扩散模型指南.md` | 800 行完整操作指南（理论 + 代码 + 排错） |

---

## 🚀 运行

```bash
cd Diffusion

# 验证架构（30 秒，确认 U-Net 正确）
PATH="/c/Users/LENOVO/AppData/Local/Microsoft/WindowsApps:$PATH" python3 verify_architecture.py

# 完整训练 + 生成（~35 分钟）
PATH="/c/Users/LENOVO/AppData/Local/Microsoft/WindowsApps:$PATH" python3 train_ddpm.py
```

---

## 📁 输出

| 文件 | 内容 |
|------|------|
| `outputs/ddpm_mnist.pth` | 训练好的模型权重（含 EMA 版本） |
| `outputs/final_ddim_64.png` | DDIM 50 步生成的 64 张手写数字 |
| `outputs/final_ddpm_16.png` | DDPM 1000 步生成的 16 张手写数字 |
| `outputs/denoising_trajectory.png` | 去噪过程：纯噪声 → 数字的 12 帧轨迹 |
| `outputs/loss_curve.png` | 训练 Loss 曲线 |
| `outputs/samples_epoch_*.png` | 各 epoch 的生成样本 |

---

## 🔑 技术要点

| 技术 | 作用 |
|------|------|
| **β 线性调度** | 噪声从 1e-4 逐步增加到 0.02（T=1000） |
| **闭式前向扩散** | 不需要逐步加噪，一步从 x₀ 算 x_t |
| **EMA 平滑** | 指数移动平均参数的副本，生成质量更好 |
| **DDIM 跳步采样** | 确定性采样，50 步 ≈ 1000 步质量，20× 加速 |
| **GroupNorm** | 替代 BN，对 batch size 不敏感 |
| **时间条件注入** | Sinusoidal → MLP → 逐通道加到 ResBlock |
| **混合精度训练** | float16 GradScaler，省显存 + 加速 |
| **输出零初始化** | 让训练初期输出≈0，Loss 从预测均值噪声开始 |

---

## 🐛 踩坑记录（5 个关键 Bug）

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | U-Net shape mismatch | 动态 ModuleList 通道计算错误 | 显式命名每层 |
| 2 | 设备不匹配 | schedule 在 CPU，索引在 CUDA | `t.cpu()` + `.to(device)` |
| 3 | DataLoader 崩溃 | Windows 多进程限制 | `num_workers=0` |
| 4 | 日志文件为空 | Python 输出缓冲 | `PYTHONUNBUFFERED=1` |
| 5 | Python PATH 混乱 | mingw64 vs WindowsApps | 显式前置 WindowsApps 路径 |

---

## 📖 详细指南

参见同目录下的 [从零构建DDPM扩散模型指南.md](./从零构建DDPM扩散模型指南.md)，含原理讲解、完整代码、验证清单、常见问题排错。
