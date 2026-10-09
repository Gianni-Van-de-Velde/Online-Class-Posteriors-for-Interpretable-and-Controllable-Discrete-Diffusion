# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Licensed under CC BY-NC 4.0, as the original DiT code (see section 2 of LICENSE).
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Adapted to discrete pixels: every pixel is one token (no patching), and the model
# outputs logits over the pixel vocabulary for every channel of every pixel.

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
from transformers import PretrainedConfig


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  DiT config                                   #
#################################################################################

class DiTConfig(PretrainedConfig):
    """Extra keyword arguments are accepted (stored by PretrainedConfig) but not used by the model."""
    model_type = "DiT"

    def __init__(
        self,
        hidden_size: int = 960,
        depth: int = 32,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        frequency_embedding_size: int = 256,  # size of the sinusoidal timestep features
        vocab_size: int = 256,
        input_size: int = 32,                 # image height = width
        in_channels: int = 3,
        num_classes: int = 10,
        with_bias: bool = True,               # bias of the pixel-embedding projection
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.frequency_embedding_size = frequency_embedding_size
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.with_bias = with_bias


#################################################################################
#                  Embedding layers for pixels, timesteps and labels            #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps t ∈ [0, 1] into vectors: sinusoidal features followed by an MLP."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.frequency_embedding_size = config.frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(config.frequency_embedding_size, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """[N] (possibly fractional) timesteps -> [N, dim] sinusoidal features. From GLIDE."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # NB: uses the default max_period=10000 (config.max_period is not used); all released checkpoints rely on this.
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Embeds class labels. Index num_classes is the unconditional (CFG) token."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.embedding_table = nn.Embedding(config.num_classes + 1, config.hidden_size)

    def forward(self, labels, force_drop_ids=None):
        # CFG label dropout is sampled by the trainer and passed in as force_drop_ids.
        if force_drop_ids is not None:
            labels = torch.where(force_drop_ids == 1, self.num_classes, labels)
        return self.embedding_table(labels)


class DiscretePixelEmbed(nn.Module):
    """[B, C, H, W] pixel tokens -> [B, H*W, hidden]: embed each channel separately, concatenate, project."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.pixel_dim = 16  # embedding size per channel
        self.embedding = nn.Embedding(config.vocab_size, self.pixel_dim)
        self.proj = nn.Linear(config.in_channels * self.pixel_dim, config.hidden_size, bias=config.with_bias)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.embedding(x.long())      # [B, H*W, C, pixel_dim]
        return self.proj(x.flatten(2))    # [B, H*W, hidden]


#################################################################################
#                                 Core DiT model                                #
#################################################################################

class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(config.hidden_size, num_heads=config.num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=config.hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiscreteFinalLayer(nn.Module):
    """adaLN-modulated norm followed by a linear map to C * vocab_size logits per pixel."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(config.hidden_size, config.in_channels * config.vocab_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer predicting per-pixel logits over the clean image."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.in_channels = config.in_channels
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size

        # NB: the order in which submodules are created fixes the random initialisation; do not reorder.
        self.x_embedder = DiscretePixelEmbed(config)
        self.t_embedder = TimestepEmbedder(config)
        self.y_embedder = LabelEmbedder(config)
        # Fixed 2D sin-cos positional embedding (frozen parameter, so it is stored in the checkpoints)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.input_size * config.input_size, config.hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.depth)])
        self.final_layer = DiscreteFinalLayer(config)
        self.initialize_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_weights(self):
        # Xavier-uniform for all linear layers, zero biases
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.input_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: every block starts as the identity
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify_logits(self, x):
        """[B, H*W, C*V] -> [B, C, H, W, V]"""
        B, T, _ = x.shape
        H = W = int(T ** 0.5)
        x = x.reshape(B, H, W, self.in_channels, self.vocab_size)
        return x.permute(0, 3, 1, 2, 4)

    def forward(self, x, t, y, force_drop_ids=None):
        """
        x: [B, C, H, W] pixel tokens
        t: [B] noise levels kappa(t) in [0, 1] (equal to the diffusion times for the linear schedule)
        y: [B] class labels (num_classes = unconditional)
        force_drop_ids: optional [B] bool, True replaces the label with the unconditional token
        returns: [B, C, H, W, vocab_size] logits
        """
        x = self.x_embedder(x) + self.pos_embed       # [B, H*W, D]
        c = self.t_embedder(t) + self.y_embedder(y, force_drop_ids)  # [B, D]
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)                    # [B, H*W, C*V]
        return self.unpatchify_logits(x)


#################################################################################
#                  Sine/cosine positional embedding functions                   #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Returns [grid_size*grid_size, embed_dim]."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # half of the dimensions encode the height, the other half the width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)                       # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """pos: positions of size (M,) -> (M, embed_dim)"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega                # (D/2,)

    pos = pos.reshape(-1)                    # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
