# https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/model.py

import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class SpatialSelfAttention(nn.Module):
    """Multi-head spatial self-attention applied over (H,W) for each batch and channel.

    Input shape: (B, C, H, W). We flatten the spatial dims to length L = H*W and
    treat it like a 1D sequence of length L with C channels. We apply GroupNorm,
    then Conv1d to compute q/k/v across the channel dimension for each spatial
    position. The result is projected back and added residually to the input.
    This is a lightweight way to capture long-range dependencies in the bottleneck.
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # normalize channels first (per-channel GN)
        self.norm = Normalize(in_channels)

        # operate in 1d over the spatial length L (flatten H*W): Conv1d expects (B, C, L)
        # qkv produces 3*C channels, which we then split
        self.qkv = torch.nn.Conv1d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape

        # normalize channels
        x_norm = self.norm(x)

        # flatten spatial dims -> (B, C, L)
        x_flat = x_norm.view(b, c, h * w)

        # compute q,k,v -> (B, 3C, L), split to (B, C, L)
        qkv = self.qkv(x_flat)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # reshape for multi-head: (B, heads, head_dim, L)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # compute attention weights: q @ k -> (B, heads, L, L)
        q_ = q.permute(0, 1, 3, 2)  # (B, heads, L, head_dim)
        k_ = k.permute(0, 1, 2, 3)  # (B, heads, head_dim, L)
        attn = torch.matmul(q_, k_)  # (B, heads, L, L)
        attn = attn * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        # attend to values: attn @ v_permuted -> (B, heads, L, head_dim)
        v_ = v.permute(0, 1, 3, 2)  # (B, heads, L, head_dim)
        out = torch.matmul(attn, v_)  # (B, heads, L, head_dim)

        # reshape back -> (B, C, L)
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, heads, head_dim, L)
        out = out.view(b, c, h * w)

        # final projection + residual
        out = self.proj(out)
        out = out.view(b, c, h, w)
        return x + out


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class UNet(nn.Module):
    def __init__(self, *, init_channels, out_channels, channels_multipliers=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 input_img_resolution, cond_dim: int = 4):
        super().__init__()
        self.init_channels = init_channels
        self.temb_ch = self.init_channels*4
        self.num_resolutions = len(channels_multipliers)
        self.num_res_blocks = num_res_blocks
        self.input_img_resolution = input_img_resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.init_channels,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # conditioning vector projection (small vector -> temb space)
        self.cond_dim = cond_dim
        self.cond_proj = None
        if cond_dim is not None and cond_dim > 0:
            self.cond_proj = torch.nn.Linear(cond_dim, self.temb_ch)

        # projection for original observed (speckled) image
        # we flatten x_orig (B, in_channels, H, W) -> (B, in_channels*H*W) then
        # apply a linear projection into the timestep embedding space so the
        # network can condition on the raw observed image when provided.
        self.orig_proj = torch.nn.Linear(in_channels * input_img_resolution * input_img_resolution, self.temb_ch)

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.init_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = input_img_resolution
        in_ch_mult = (1,)+channels_multipliers
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = init_channels*in_ch_mult[i_level]
            block_out = init_channels*channels_multipliers[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        # spatial self-attention over the HxW grid at bottleneck
        self.mid.ssa = SpatialSelfAttention(block_in, num_heads=8)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = init_channels*channels_multipliers[i_level]
            skip_in = init_channels*channels_multipliers[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = init_channels*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, t, cond=None, x_orig=None):
        assert x.shape[2] == x.shape[3] == self.input_img_resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.init_channels)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # if a conditioning vector is supplied, project it into temb space and add
        if cond is not None and self.cond_proj is not None:
            # cond expected shape: (B, cond_dim)
            cproj = self.cond_proj(cond.float())
            temb = temb + cproj

        # if an original observed image is provided, flatten and project to temb
        if x_orig is not None and hasattr(self, "orig_proj") and self.orig_proj is not None:
            # flatten per-sample: (B, C, H, W) -> (B, C*H*W)
            xflat = x_orig.view(x_orig.shape[0], -1).float()
            oproj = self.orig_proj(xflat)
            temb = temb + oproj

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # apply global spatial self-attention at the bottleneck
        h = self.mid.ssa(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h