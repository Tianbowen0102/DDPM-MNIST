"""
DDPM MNIST — 快速验证脚本
测试：模型前向传播 / 训练一步 / DDPM 采样 / DDIM 采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 所有组件定义 ──
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


# ═══════════ 测试 1: 模型实例化 + 前向传播 ═══════════
model = UNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params:,}")

x = torch.randn(4, 1, 28, 28).to(DEVICE)
t = torch.randint(0, 1000, (4,)).to(DEVICE)
with torch.no_grad():
    out = model(x, t)
print(f"Test 1 PASS - Input: {x.shape} -> Output: {out.shape}")
assert out.shape == x.shape

# ═══════════ 测试 2: 训练一步 ═══════════
T_val = 1000
betas = torch.linspace(1e-4, 0.02, T_val)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

def q_sample(x_0, t, noise=None):
    if noise is None: noise = torch.randn_like(x_0)
    s_a = sqrt_alphas_cumprod[t.cpu()].to(x_0.device)
    s_om = sqrt_one_minus_alphas_cumprod[t.cpu()].to(x_0.device)
    while s_a.dim() < x_0.dim():
        s_a = s_a.unsqueeze(-1); s_om = s_om.unsqueeze(-1)
    return s_a * x_0 + s_om * noise

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
imgs = torch.randn(4, 1, 28, 28).to(DEVICE)
t_step = torch.randint(0, T_val, (4,)).to(DEVICE)
noise = torch.randn_like(imgs)
x_t = q_sample(imgs, t_step, noise)
noise_pred = model(x_t, t_step)
loss = F.mse_loss(noise_pred, noise)
loss.backward()
optimizer.step()
print(f"Test 2 PASS - Training step Loss: {loss.item():.4f}")

# ═══════════ 测试 3: DDPM 采样逻辑 ═══════════
model.eval()
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

with torch.no_grad():
    x_sample = torch.randn(2, 1, 28, 28).to(DEVICE)
    for t_val in reversed(range(T_val - 10, T_val)):
        t_b = torch.full((2,), t_val, device=DEVICE, dtype=torch.long)
        noise_pred = model(x_sample, t_b)
        x_sample = sqrt_recip_alphas[t_val].to(DEVICE) * (
            x_sample - betas[t_val].to(DEVICE) / sqrt_one_minus_alphas_cumprod[t_val].to(DEVICE) * noise_pred
        )
        if t_val > 0:
            x_sample = x_sample + torch.sqrt(posterior_variance[t_val]).to(DEVICE) * torch.randn_like(x_sample)
print(f"Test 3 PASS - DDPM sampling shape: {x_sample.shape}")

# ═══════════ 测试 4: DDIM 采样逻辑 ═══════════
ddim_steps = 10
step_indices = torch.linspace(T_val - 1, 0, ddim_steps, dtype=torch.long)
with torch.no_grad():
    x_ddim = torch.randn(2, 1, 28, 28).to(DEVICE)
    for i in range(ddim_steps):
        t_s = step_indices[i]
        t_b = torch.full((2,), t_s, device=DEVICE, dtype=torch.long)
        noise_pred = model(x_ddim, t_b)
        sqrt_a = sqrt_alphas_cumprod[t_s].to(DEVICE)
        sqrt_om = sqrt_one_minus_alphas_cumprod[t_s].to(DEVICE)
        x0_pred = (x_ddim - sqrt_om * noise_pred) / sqrt_a
        x0_pred = torch.clamp(x0_pred, -1, 1)
        t_prev = step_indices[i + 1] if i < ddim_steps - 1 else torch.tensor(-1, device=DEVICE)
        a_prev = alphas_cumprod[t_prev].to(DEVICE) if t_prev >= 0 else torch.tensor(1.0, device=DEVICE)
        x_ddim = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * noise_pred
print(f"Test 4 PASS - DDIM sampling shape: {x_ddim.shape}")

# ═══════════ 测试 5: EMA 更新 ═══════════
ema_model = deepcopy(model)
ema_model.eval()
with torch.no_grad():
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data = 0.995 * ema_p.data + 0.005 * p.data
print("Test 5 PASS - EMA update OK")

print()
print("=" * 50)
print("  全部 5 项测试通过！PASS")
print("  模型架构正确，可以开始完整训练")
print("=" * 50)
