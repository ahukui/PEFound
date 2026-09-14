"""
Multi-Modal Contrastive + Reconstruction Pretraining (single shared ViT).

Improvements vs the basic version:
  - MAE-style per-patch normalized reconstruction target  (config.norm_pix_loss)
  - DDP-gathered negatives for contrastive loss           (config.gather_loss)
  - I2T / T2I retrieval accuracy returned for monitoring
  - Per-modality random 3D sub-volume masking (replaces whole-modality masking)
  - Vision tower architecture is BIT-FOR-BIT identical to the downstream LaMed
    `vit3d` (SABlock / MLPBlock / TransformerBlock + MONAI PatchEmbeddingBlock),
    so saved `vision_tower.*` keys load with strict=True downstream.

Inputs to forward(...):
  modality_0, modality_1, modality_2 : each (B, 1, D, H, W)
  input_ids, attention_mask          : BERT inputs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    BertModel,
    AutoConfig,
    AutoModel,
)
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock


# ===========================================================================
# Distributed helpers
# ===========================================================================
def _is_dist():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _all_gather_with_grad(x):
    """
    All-gather `x` across DDP ranks, allowing gradient to flow through the
    local slice (other slices are detached). Standard CLIP-style gather.
    """
    if not _is_dist():
        return x
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    gathered[rank] = x  # restore graph for local slice
    return torch.cat(gathered, dim=0)


# ===========================================================================
# Transformer building blocks — IDENTICAL to downstream `vit3d`.
# Don't rename anything in this section, or weight transfer will break.
# ===========================================================================
class SABlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.0, qkv_bias=False):
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

    def forward(self, x):
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
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout1(self.act(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, mlp_dim, num_heads, dropout_rate=0.0, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ===========================================================================
# ViT — module structure mirrors downstream exactly so weight keys transfer 1:1
# ===========================================================================
class ViT(nn.Module):
    def __init__(
        self,
        in_channels,
        img_size,
        patch_size,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        pos_embed="perceptron",
        classification=True,
        dropout_rate=0.0,
        spatial_dims=3,
        qkv_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.classification = classification

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
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        if classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, N+1, H) when classification else (B, N, H)


# ===========================================================================
# Reconstruction decoder — pretraining-only, not transferred downstream.
# ===========================================================================
class ReconDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=4,
        num_heads=12,
        dropout_rate=0.0,
        img_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        out_channels=3,
        num_extra_tokens=1,   # number of non-patch leading tokens (e.g., CLS)
    ):
        super().__init__()
        self.grid_size = tuple(i // p for i, p in zip(img_size, patch_size))
        self.num_patches = int(np.prod(self.grid_size))
        self.num_extra_tokens = num_extra_tokens
        self.num_tokens = self.num_patches + num_extra_tokens
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_vol = patch_size[0] * patch_size[1] * patch_size[2] * out_channels

        # pos_embed sized for the FULL input sequence (CLS + patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias=False)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.patch_vol)

    def forward(self, feats):
        """feats: (B, num_tokens, H)  ->  patch preds (B, num_patches, patch_vol).

        The decoder receives the FULL encoder sequence (CLS + patches). After the
        head, the first `num_extra_tokens` outputs (the CLS-position preds) are
        dropped so we return exactly one prediction per image patch.
        """
        x = feats + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)                              # (B, num_tokens, patch_vol)
        if self.num_extra_tokens > 0:
            x = x[:, self.num_extra_tokens:]          # drop CLS-position outputs
        return x                                       # (B, num_patches, patch_vol)

    def unpatchify(self, x):
        """(B, N, patch_vol) -> (B, C, D, H, W). For visualization only."""
        B = x.shape[0]
        gD, gH, gW = self.grid_size
        pD, pH, pW = self.patch_size
        C = self.out_channels
        x = x.view(B, gD, gH, gW, C, pD, pH, pW)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(B, C, gD * pD, gH * pH, gW * pW)


# ===========================================================================
# Config
# ===========================================================================
class MultiModalPretrainConfig(PretrainedConfig):
    model_type = "multimodal_pretrain"

    def __init__(
        self,
        language_model_name_or_path: str = "bert-base-uncased",
        in_channels: int = 3,                  # 3 modalities stacked as channels
        img_size: tuple = (32, 256, 256),
        patch_size: tuple = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        enc_num_layers: int = 12,
        dec_num_layers: int = 4,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        contrastive_weight: float = 1.0,
        recon_weight: float = 1.0,
        temperature_init: float = 0.07,
        norm_pix_loss: bool = True,            # MAE per-patch target normalization
        gather_loss: bool = True,              # DDP-gathered negatives for InfoNCE
        # ---- Sub-volume masking (per-modality, replaces whole-modality mask) ----
        mask_ratio: float = 0.5,               # fraction of sub-volume blocks masked, per modality
        mask_block_size: tuple = (4,16,16),         # 3D block size; None -> patch_size (per-patch MAE-style)
        recon_on_masked_only: bool = False,    # if True, MSE only on masked patches/modalities
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.in_channels = in_channels
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.qkv_bias = qkv_bias
        self.contrastive_weight = contrastive_weight
        self.recon_weight = recon_weight
        self.temperature_init = temperature_init
        self.norm_pix_loss = norm_pix_loss
        self.gather_loss = gather_loss
        self.mask_ratio = float(mask_ratio)
        self.mask_block_size = (
            tuple(mask_block_size) if mask_block_size is not None else tuple(patch_size)
        )
        self.recon_on_masked_only = bool(recon_on_masked_only)
        super().__init__(**kwargs)


# ===========================================================================
# Model
# ===========================================================================
class MultiModalPretrain(PreTrainedModel):
    config_class = MultiModalPretrainConfig

    def __init__(self, config: MultiModalPretrainConfig):
        super().__init__(config)
        C = config

        # --- Sanity-check mask_block_size against img_size and patch_size ---
        D, H, W = C.img_size
        bD, bH, bW = C.mask_block_size
        pD, pH, pW = C.patch_size
        if not (D % bD == 0 and H % bH == 0 and W % bW == 0):
            raise ValueError(
                f"img_size {C.img_size} must be divisible by mask_block_size "
                f"{C.mask_block_size}"
            )
        if not (bD % pD == 0 and bH % pH == 0 and bW % pW == 0):
            raise ValueError(
                f"mask_block_size {C.mask_block_size} must be a multiple of "
                f"patch_size {C.patch_size} so blocks are patch-aligned"
            )
        if not (0.0 <= C.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1), got {C.mask_ratio}")

        # Submodule MUST be named `vision_tower` so saved keys are
        # `vision_tower.*` and load directly into the downstream ViT3DTower.
        self.vision_tower = ViT(
            in_channels=C.in_channels,
            img_size=C.img_size,
            patch_size=C.patch_size,
            hidden_size=C.hidden_size,
            mlp_dim=C.mlp_dim,
            num_layers=C.enc_num_layers,
            num_heads=C.num_heads,
            pos_embed="perceptron",
            classification=True,
            dropout_rate=C.dropout_rate,
            spatial_dims=3,
            qkv_bias=C.qkv_bias,
        )

        # Text encoder (used only during pretraining).
        # add_pooling_layer=False drops `pooler.dense.{weight,bias}` — we only use
        # last_hidden_state[:, 0], so the pooler would otherwise sit unused and
        # break DDP with `find_unused_parameters=False`.
        self.language_encoder = BertModel.from_pretrained(
            C.language_model_name_or_path, add_pooling_layer=False
        )

        # Contrastive heads
        self.vision_proj = nn.Linear(C.hidden_size, C.hidden_size)
        self.text_proj = nn.Linear(C.hidden_size, C.hidden_size)

        # Reconstruction decoder (single, 3-channel output)
        self.decoder = ReconDecoder(
            hidden_size=C.hidden_size,
            mlp_dim=C.mlp_dim,
            num_layers=C.dec_num_layers,
            num_heads=C.num_heads,
            dropout_rate=C.dropout_rate,
            img_size=C.img_size,
            patch_size=C.patch_size,
            out_channels=C.in_channels,
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / C.temperature_init)
        )

        self.post_init()

    # ---------------------------------------------------------------- helpers
    def encode_text(self, input_ids, attention_mask):
        out = self.language_encoder(input_ids, attention_mask=attention_mask)
        return out["last_hidden_state"][:, 0]

    def patchify(self, imgs):
        """(B, C, D, H, W) -> (B, N, C*pD*pH*pW).

        In the output's last dim, channel index varies SLOWEST: the layout is
        (c, pd, ph, pw) flattened, i.e. all voxels of channel 0, then channel 1, ...
        This ordering matters for `recon_on_masked_only` below.
        """
        B, C, D, H, W = imgs.shape
        pD, pH, pW = self.config.patch_size
        gD, gH, gW = D // pD, H // pH, W // pW
        x = imgs.reshape(B, C, gD, pD, gH, pH, gW, pW)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(B, gD * gH * gW, C * pD * pH * pW)

    # ---------------------------------------------------- sub-volume masking
    def _random_subvolume_mask(self, images):
        """
        Per-modality independent random 3D sub-volume masking.

        For each (sample, modality) pair, the volume is divided into a grid of
        `mask_block_size` blocks and a random `mask_ratio` fraction of those
        blocks is zeroed out. Each modality channel gets its OWN random pattern,
        so the network is forced to use cross-modality context (plus the
        unmasked regions of the same modality) to reconstruct the holes.

        Returns:
            masked_images : (B, C, D, H, W) input with masked blocks zeroed
            patch_keep    : (B, C, N_patches) at the patch grid resolution,
                            1.0 = visible, 0.0 = masked. Used for the optional
                            masked-only reconstruction loss. None if no masking
                            occurred.
        """
        B, C, D, H, W = images.shape
        bD, bH, bW = self.config.mask_block_size
        pD, pH, pW = self.config.patch_size

        gD, gH, gW = D // bD, H // bH, W // bW
        N_blocks = gD * gH * gW
        n_mask = int(round(self.config.mask_ratio * N_blocks))

        if n_mask == 0:
            return images, None

        # Per-(batch, modality) shuffled scores: smallest n_mask entries get masked.
        # Different (b, c) rows get independent random masks.
        noise = torch.rand(B, C, N_blocks, device=images.device)
        _, mask_idx = noise.topk(n_mask, dim=-1, largest=False)

        # block_keep[b, c, i] = 1 if block i is visible for (b, c), else 0
        block_keep = torch.ones(B, C, N_blocks, device=images.device, dtype=images.dtype)
        block_keep.scatter_(-1, mask_idx, 0.0)
        block_keep = block_keep.view(B, C, gD, gH, gW)

        # Upsample block mask to voxel resolution via repeat_interleave
        img_keep = (
            block_keep
            .repeat_interleave(bD, dim=2)
            .repeat_interleave(bH, dim=3)
            .repeat_interleave(bW, dim=4)
        )
        masked_images = images * img_keep

        # Patch-grid keep mask for masked-only recon loss: each block contains
        # (bD/pD)*(bH/pH)*(bW/pW) patches that all share the block's keep value.
        rD, rH, rW = bD // pD, bH // pH, bW // pW
        patch_keep = (
            block_keep
            .repeat_interleave(rD, dim=2)
            .repeat_interleave(rH, dim=3)
            .repeat_interleave(rW, dim=4)
        )  # (B, C, gD_p, gH_p, gW_p)
        patch_keep = patch_keep.flatten(2)  # (B, C, N_patches)

        return masked_images, patch_keep

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        modality_0,
        modality_1,
        modality_2,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        # -------- Stack 3 modalities into one 3-channel volume --------
        images = torch.cat([modality_0, modality_1, modality_2], dim=1)  # (B,3,D,H,W)

        # -------- Per-modality random sub-volume masking (training only) --------
        # Replaces the old whole-modality `mask_idx` behavior: each modality
        # now has random 3D sub-volumes zeroed, independently sampled per
        # (sample, modality). Evaluation sees full inputs.
        patch_keep = None
        if self.training and self.config.mask_ratio > 0:
            enc_in, patch_keep = self._random_subvolume_mask(images)
        else:
            enc_in = images

        # -------- Encode (whole feature shared by BOTH heads) --------
        feats = self.vision_tower(enc_in)        # (B, N+1, H) — CLS + all patches

        # =================== Contrastive loss ===================
        # Mean-pool the WHOLE sequence so img_z reflects the full feature, not
        # just the CLS token.
        img_repr = feats.mean(dim=1)             # (B, H)
        img_z = F.normalize(self.vision_proj(img_repr), dim=-1)
        txt_z = F.normalize(
            self.text_proj(self.encode_text(input_ids, attention_mask)), dim=-1
        )

        scale = self.logit_scale.exp().clamp(max=100.0)
        B = img_z.shape[0]

        if self.config.gather_loss and _is_dist():
            all_img_z = _all_gather_with_grad(img_z)
            all_txt_z = _all_gather_with_grad(txt_z)
            logits_i2t = scale * img_z @ all_txt_z.T   # (B_local, B_global)
            logits_t2i = scale * txt_z @ all_img_z.T
            rank = dist.get_rank()
            labels = torch.arange(B, device=img_z.device) + rank * B
        else:
            logits_i2t = scale * img_z @ txt_z.T       # (B, B)
            logits_t2i = logits_i2t.T
            labels = torch.arange(B, device=img_z.device)

        contrastive_loss = (
            F.cross_entropy(logits_i2t, labels)
            + F.cross_entropy(logits_t2i, labels)
        ) / 2.0

        # Retrieval accuracy (monitoring only)
        with torch.no_grad():
            i2t_acc = (logits_i2t.argmax(dim=-1) == labels).float().mean()
            t2i_acc = (logits_t2i.argmax(dim=-1) == labels).float().mean()

        # =================== Reconstruction loss (MAE-style) ===================
        recon_pred_patches = self.decoder(feats)             # (B, N, patch_vol)
        target_patches = self.patchify(images)               # (B, N, patch_vol)

        if self.config.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var  = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6) ** 0.5

        if self.config.recon_on_masked_only and patch_keep is not None:
            # Loss only on (modality, patch) entries that were MASKED in input.
            # patch_keep: (B, C, N), 1=visible -> mask_weight = 1 - keep.
            # patchify layout: last dim is (c, pd, ph, pw) flattened with c slowest.
            pD, pH, pW = self.config.patch_size
            mask_w = (1.0 - patch_keep).permute(0, 2, 1)        # (B, N, C)
            mask_w = mask_w.repeat_interleave(pD * pH * pW, -1) # (B, N, C*pD*pH*pW)
            sq_err = (recon_pred_patches - target_patches) ** 2
            recon_loss = (sq_err * mask_w).sum() / (mask_w.sum() + 1e-8)
        else:
            recon_loss = F.mse_loss(recon_pred_patches, target_patches)

        # =================== Total ===================
        total_loss = (
            self.config.contrastive_weight * contrastive_loss
            + self.config.recon_weight * recon_loss
        )

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "logit_scale": scale.detach(),
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
        }


# Register for AutoModel usage
AutoConfig.register("multimodal_pretrain", MultiModalPretrainConfig)
AutoModel.register(MultiModalPretrainConfig, MultiModalPretrain)