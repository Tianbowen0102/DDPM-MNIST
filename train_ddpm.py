"""
DDPM MNIST — 完整训练 + 采样脚本
用法: PATH="/c/Users/LENOVO/AppData/Local/Microsoft/WindowsApps:$PATH" python3 train_ddpm.py
"""
import math
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import matplotlib
matplotlib.use("Agg")  # 无头模式，不弹窗
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
T = 1000  # 扩散总步数
EMA_DECAY = 0.995
EMA_UPDATE_AFTER = 100

print(f"设备: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════
# 组件定义
# ═══════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x); h = F.silu(h); h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(h); h = F.silu(h); h = self.conv2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(1, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        return self.to_out(out.reshape(B, C, H, W)) + x


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(), nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.enc1_block1 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.enc1_block2 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        ch2 = base_ch * 2
        self.enc2_block1 = ResBlock(ch2, ch2, time_emb_dim)
        self.enc2_block2 = ResBlock(ch2, ch2, time_emb_dim)
        self.enc2_attn = SelfAttention(ch2)
        self.down2 = nn.Conv2d(ch2, ch2 * 2, 3, stride=2, padding=1)

        mid_ch = base_ch * 4
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_ch, ch2, 3, padding=1),
        )
        self.dec2_block1 = ResBlock(ch2 * 2, ch2, time_emb_dim)
        self.dec2_block2 = ResBlock(ch2, ch2, time_emb_dim)
        self.dec2_block3 = ResBlock(ch2, ch2, time_emb_dim)
        self.dec2_attn = SelfAttention(ch2)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch2, base_ch, 3, padding=1),
        )
        self.dec1_block1 = ResBlock(base_ch * 2, base_ch, time_emb_dim)
        self.dec1_block2 = ResBlock(base_ch, base_ch, time_emb_dim)
        self.dec1_block3 = ResBlock(base_ch, base_ch, time_emb_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(), nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.conv_in(x)
        h = self.enc1_block1(h, t_emb); h = self.enc1_block2(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.enc2_block1(h, t_emb); h = self.enc2_block2(h, t_emb)
        h = self.enc2_attn(h)
        skip2 = h
        h = self.down2(h)
        h = self.mid_block1(h, t_emb); h = self.mid_attn(h); h = self.mid_block2(h, t_emb)
        h = self.up1(h); h = torch.cat([h, skip2], dim=1)
        h = self.dec2_block1(h, t_emb); h = self.dec2_block2(h, t_emb); h = self.dec2_block3(h, t_emb)
        h = self.dec2_attn(h)
        h = self.up2(h); h = torch.cat([h, skip1], dim=1)
        h = self.dec1_block1(h, t_emb); h = self.dec1_block2(h, t_emb); h = self.dec1_block3(h, t_emb)
        return self.conv_out(h)


# ═══════════════════════════════════════════════════════
# 扩散过程定义
# ═══════════════════════════════════════════════════════

betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    s_a = sqrt_alphas_cumprod[t.cpu()].to(x_0.device)
    s_om = sqrt_one_minus_alphas_cumprod[t.cpu()].to(x_0.device)
    while s_a.dim() < x_0.dim():
        s_a = s_a.unsqueeze(-1); s_om = s_om.unsqueeze(-1)
    return s_a * x_0 + s_om * noise


# ═══════════════════════════════════════════════════════
# 采样函数
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def ddpm_sample(model, shape):
    """DDPM 完整 1000 步采样"""
    model.eval()
    x = torch.randn(shape, device=DEVICE)
    for t_val in reversed(range(T)):
        t_batch = torch.full((shape[0],), t_val, device=DEVICE, dtype=torch.long)
        noise_pred = model(x, t_batch)
        x = sqrt_recip_alphas[t_val].to(DEVICE) * (
            x - betas[t_val].to(DEVICE) / sqrt_one_minus_alphas_cumprod[t_val].to(DEVICE) * noise_pred
        )
        if t_val > 0:
            x = x + torch.sqrt(posterior_variance[t_val]).to(DEVICE) * torch.randn_like(x)
    return x


@torch.no_grad()
def ddim_sample(model, shape, ddim_steps=50, init_noise=None):
    """DDIM 加速采样"""
    model.eval()
    step_indices = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long)
    if init_noise is not None:
        x = init_noise.to(DEVICE)
    else:
        x = torch.randn(shape, device=DEVICE)

    for i in range(ddim_steps):
        t = step_indices[i]
        t_batch = torch.full((shape[0],), t, device=DEVICE, dtype=torch.long)
        noise_pred = model(x, t_batch)

        a_t = alphas_cumprod[t].to(DEVICE)
        sqrt_a_t = sqrt_alphas_cumprod[t].to(DEVICE)
        sqrt_om_a_t = sqrt_one_minus_alphas_cumprod[t].to(DEVICE)

        x0_pred = (x - sqrt_om_a_t * noise_pred) / sqrt_a_t
        x0_pred = torch.clamp(x0_pred, -1, 1)

        if i < ddim_steps - 1:
            t_prev = step_indices[i + 1]
        else:
            t_prev = -1
        a_prev = alphas_cumprod[t_prev].to(DEVICE) if t_prev >= 0 else torch.tensor(1.0, device=DEVICE)
        x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * noise_pred
    return x


# ═══════════════════════════════════════════════════════
# 辅助：采样并保存图片
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def sample_and_save(model, filename, n=16, use_ddim=True):
    """生成样本图片并保存"""
    model.eval()
    if use_ddim:
        samples = ddim_sample(model, shape=(n, 1, 28, 28), ddim_steps=50)
    else:
        samples = ddpm_sample(model, shape=(n, 1, 28, 28))
    samples = samples * 0.5 + 0.5
    samples = torch.clamp(samples, 0, 1)
    grid = make_grid(samples, nrow=4, padding=2)
    save_image(grid, OUTPUT_DIR / filename)
    return grid


# ═══════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════

print("\n[数据] 加载 MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True,
)
print(f"  训练样本: {len(train_dataset)}, 批次: {len(train_loader)}")

# ═══════════════════════════════════════════════════════
# 模型初始化
# ═══════════════════════════════════════════════════════

print("\n[模型] 初始化 U-Net...")
model = UNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  参数量: {n_params:,}")

ema_model = deepcopy(model)
ema_model.eval()

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

# ═══════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════

print(f"\n[训练] {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
print(f"  总步数: {EPOCHS * len(train_loader)}")
print(f"  EMA decay: {EMA_DECAY}")
print(f"{'='*55}")

loss_history = []
global_step = 0
t_start = time.time()

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    t_epoch = time.time()

    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        B = imgs.shape[0]

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.long)
                noise = torch.randn_like(imgs)
                x_t = q_sample(imgs, t, noise)
                noise_pred = model(x_t, t)
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.long)
            noise = torch.randn_like(imgs)
            x_t = q_sample(imgs, t, noise)
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

        # EMA 更新
        if global_step >= EMA_UPDATE_AFTER:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data = EMA_DECAY * ema_p.data + (1 - EMA_DECAY) * p.data

        epoch_loss += loss.item()
        global_step += 1

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]

    elapsed = time.time() - t_start
    eta = (elapsed / epoch) * (EPOCHS - epoch)
    bar = "#" * (epoch * 30 // EPOCHS)
    print(f"Epoch {epoch:2d}/{EPOCHS} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
          f"elapsed={elapsed/60:.0f}m | eta={eta/60:.0f}m | {bar}", flush=True)

    # 每 10 轮保存一次样本
    if epoch % 10 == 0 or epoch == 1:
        sample_and_save(ema_model, f"samples_epoch_{epoch:02d}.png", n=16, use_ddim=True)

total_time = time.time() - t_start
print(f"\n训练完成！总耗时: {total_time/60:.1f} min ({total_time:.0f}s)")

# ═══════════════════════════════════════════════════════
# 保存模型
# ═══════════════════════════════════════════════════════

checkpoint = {
    "model": model.state_dict(),
    "ema_model": ema_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": EPOCHS,
    "loss_history": loss_history,
}
torch.save(checkpoint, OUTPUT_DIR / "ddpm_mnist.pth")
print(f"模型已保存: {OUTPUT_DIR / 'ddpm_mnist.pth'}")

# ═══════════════════════════════════════════════════════
# Loss 曲线
# ═══════════════════════════════════════════════════════

plt.figure(figsize=(10, 4))
plt.plot(loss_history, marker="o", markersize=3, linewidth=1)
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=100)
plt.close()
print(f"Loss 曲线已保存: {OUTPUT_DIR / 'loss_curve.png'}")

# ═══════════════════════════════════════════════════════
# 最终生成：DDPM vs DDIM 对比
# ═══════════════════════════════════════════════════════

print("\n[生成] 最终采样...")

# DDIM 快速生成 64 张
print("  DDIM-50 步采样中...")
t0 = time.time()
ddim_samples = ddim_sample(ema_model, shape=(64, 1, 28, 28), ddim_steps=50)
ddim_time = time.time() - t0
ddim_grid = ddim_samples * 0.5 + 0.5
ddim_grid = torch.clamp(ddim_grid, 0, 1)
save_image(make_grid(ddim_grid, nrow=8, padding=2), OUTPUT_DIR / "final_ddim_64.png")
print(f"  DDIM 完成 ({ddim_time:.1f}s) -> {OUTPUT_DIR / 'final_ddim_64.png'}")

# DDPM 完整 1000 步（只生成 16 张，太慢）
print("  DDPM-1000 步采样中...")
t0 = time.time()
ddpm_samples = ddpm_sample(ema_model, shape=(16, 1, 28, 28))
ddpm_time = time.time() - t0
ddpm_grid = ddpm_samples * 0.5 + 0.5
ddpm_grid = torch.clamp(ddpm_grid, 0, 1)
save_image(make_grid(ddpm_grid, nrow=4, padding=2), OUTPUT_DIR / "final_ddpm_16.png")
print(f"  DDPM 完成 ({ddpm_time:.1f}s) -> {OUTPUT_DIR / 'final_ddpm_16.png'}")

# ═══════════════════════════════════════════════════════
# 去噪过程可视化
# ═══════════════════════════════════════════════════════

print("\n[可视化] 去噪过程轨迹...")
model.eval()

@torch.no_grad()
def get_trajectory():
    """记录从纯噪声到图像的 11 帧"""
    frames = []
    x = torch.randn(1, 1, 28, 28, device=DEVICE)
    frames.append(x.cpu())
    step_markers = set(range(0, T, 100))  # 0, 100, 200, ..., 900

    for t_val in reversed(range(T)):
        t_batch = torch.full((1,), t_val, device=DEVICE, dtype=torch.long)
        noise_pred = model(x, t_batch)
        x = sqrt_recip_alphas[t_val].to(DEVICE) * (
            x - betas[t_val].to(DEVICE) / sqrt_one_minus_alphas_cumprod[t_val].to(DEVICE) * noise_pred
        )
        if t_val > 0:
            x = x + torch.sqrt(posterior_variance[t_val]).to(DEVICE) * torch.randn_like(x)
        if t_val in step_markers:
            frames.append(x.cpu())
    frames.append(x.cpu())  # t=0
    return frames

trajectory = get_trajectory()

fig, axes = plt.subplots(1, len(trajectory), figsize=(18, 2.5))
steps_shown = [1000] + list(range(900, -1, -100))
for i, (step, img) in enumerate(zip(steps_shown, trajectory)):
    img_show = img.squeeze() * 0.5 + 0.5
    axes[i].imshow(img_show, cmap="gray")
    axes[i].set_title(f"t={step}" if step > 0 else "t=0", fontsize=8)
    axes[i].axis("off")
plt.suptitle("Reverse Denoising: Pure Noise -> Generated Image", fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "denoising_trajectory.png", dpi=100)
plt.close()
print(f"  去噪轨迹已保存: {OUTPUT_DIR / 'denoising_trajectory.png'}")

# ═══════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════

print(f"\n{'='*55}")
print(f"  全部完成！")
print(f"  初始 Loss: {loss_history[0]:.4f}")
print(f"  最终 Loss: {loss_history[-1]:.4f}")
print(f"  下降幅度: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")
print(f"  总耗时: {total_time/60:.1f} min")
print(f"  DDPM 采样速度: {ddpm_time:.1f}s/16张")
print(f"  DDIM 采样速度: {ddim_time:.1f}s/64张 (加速 {ddpm_time/ddim_time*4:.0f}x)")
print(f"  产出目录: {OUTPUT_DIR}")
print(f"{'='*55}")
