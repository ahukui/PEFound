# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock


# =============================================================================
# Custom Transformer blocks — identical to the pretrain script.
# These do NOT have cross-attention, ensuring a 1:1 key match with the
# pretrained encoder checkpoint (zero missing keys).
# =============================================================================

class SABlock(nn.Module):
    """Multi-head self-attention (no cross-attention)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class MLPBlock(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.act(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with self-attention only (no cross-attention).

    Architecture matches the pretrain script exactly:
        x = x + SA(LN(x))
        x = x + MLP(LN(x))

    This ensures a 1:1 key match with pretrained encoder weights —
    no extra cross_attn / norm_cross_attn parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# ViT — uses MONAI's PatchEmbeddingBlock + custom TransformerBlock
# =============================================================================

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    <https://arxiv.org/abs/2010.11929>"

    Uses MONAI's PatchEmbeddingBlock for patch embedding (key remapping handled
    by load_pretrained_vision_tower) and custom TransformerBlock without
    cross-attention to match the pretrained encoder architecture exactly.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.classification = classification

        # MONAI's PatchEmbeddingBlock — key remapping in
        # load_pretrained_vision_tower handles the .0. / .1. index difference
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # Custom TransformerBlock — self-attention only, no cross-attention.
        # Matches the pretrain script's TransformerBlock exactly.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)

        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out


# =============================================================================
# ViT3DTower — wrapper used by LaMed
# =============================================================================

class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)

        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_layer]
        else:
            raise ValueError(f"Unexpected select layer: {self.select_layer}")

        if self.select_feature == "patch":
            image_features = image_features[:, 1:]  # strip cls token
        elif self.select_feature == "cls_patch":
            image_features = image_features  # keep everything
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        return image_features

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size