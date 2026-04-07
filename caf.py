import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    def __init__(self, groups = 4):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        assert c % self.groups == 0, "channels must be divisible by groups"

        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x

class CAFBlock(nn.Module):
    # Default class labels
    runtime_alpha = 0.5
    runtime_dilation_rates = (2, 3)

    def __init__(self, c):
        super().__init__()
        alpha = CAFBlock.runtime_alpha
        dilations = CAFBlock.runtime_dilation_rates

        # 1) Initialize modules
        self.ln1 = nn.LayerNorm(c)
        self.acfm = ACFM(c)
        self.ln2 = nn.LayerNorm(c)
        self.msnn = MSNN(c, dilation_rates=dilations)

        # 2) Learnable parameters initialized with the configured alpha
        self.alpha1 = nn.Parameter(torch.full((1,), float(alpha)))
        self.alpha2 = nn.Parameter(torch.full((1,), float(alpha)))

    def forward(self, x):
        # ACFM stage
        x_ln = x.permute(0, 2, 3, 1)
        x = x + (self.alpha1 * self.acfm(self.ln1(x_ln).permute(0, 3, 1, 2)))

        # MSNN stage
        x_ln = x.permute(0, 2, 3, 1)
        x = x + (self.alpha2 * self.msnn(self.ln2(x_ln).permute(0, 3, 1, 2)))
        return x

class ACFM(nn.Module):
    """
    Attention and Convolution Fusion Module
    Paper-aligned implementation
    """
    def __init__(self, c, shuffle_groups=4):
        super().__init__()
        self.c = c

        # ---------- Local branch ----------

        # 1x1 conv mixes channel information as weighted sums.
        self.local_conv1 = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # Deterministic channel shuffle enforces cross-group interaction.
        self.shuffle = ChannelShuffle(shuffle_groups)

        # 3x3 depthwise conv captures spatial features efficiently.
        self.local_dwconv = nn.Conv2d(
            c, c, kernel_size=3, padding=1, groups=c, bias=False
        )
        self.local_conv2 = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # ---------- Global branch (channel attention) ----------
        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(
            c * 3, c * 3, kernel_size=3, padding=1, groups=c * 3, bias=False
        )
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=False)

        self.scale = c ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape

        # ===== Local branch =====
        local_out = self.local_conv1(x)
        local_out = self.shuffle(local_out)
        local_out = self.local_dwconv(local_out)
        local_out = self.local_conv2(local_out)

        # ===== Global branch =====
        qkv = self.qkv(x)
        qkv = self.dw_conv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape: (B, C, H*W)
        q = q.view(b, c, -1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)

        # channel-wise attention (C x C)
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)

        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        global_out = torch.matmul(attn, v)
        global_out = global_out.view(b, c, h, w)
        global_out = self.proj(global_out)

        # ===== Fusion =====
        return global_out + local_out

class MSNN(nn.Module):
    """
    Multi-Scale Neural Network
    FFN replacement in CAFBlock
    """
    def __init__(self, c, dilation_rates=(2, 3)):
        super().__init__()
        d1, d2 = dilation_rates

        # Shared channel alignment.
        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # Local path.
        self.local_dwconv = nn.Conv2d(
            c, c, kernel_size=3, padding=1, groups=c, bias=False
        )

        # Global path (dilated convolutions).
        self.dilated_conv1 = nn.Conv2d(
            c, c, kernel_size=3, padding=d1, dilation=d1, bias=False
        )
        self.dilated_conv2 = nn.Conv2d(
            c, c, kernel_size=3, padding=d2, dilation=d2, bias=False
        )

        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)

        # Local branch.
        local_feat = self.local_dwconv(x)
        local_feat = self.act(local_feat)

        # Global branch.
        global_feat = self.dilated_conv1(x) + self.dilated_conv2(x)

        # Gating.
        out = local_feat * global_feat
        out = self.proj(out)
        return out
