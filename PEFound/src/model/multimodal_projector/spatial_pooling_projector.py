from torch import nn
import torch
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class ExpertMLP(nn.Module):
    """A single expert MLP."""
    def __init__(self, in_dim, out_dim, layer_num=2):
        super().__init__()
        modules = [nn.Linear(in_dim, out_dim)]
        for _ in range(1, layer_num):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_dim, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class ModalityMoE(nn.Module):
    """
    Mixture-of-Experts for fusing T1, T2, FLAIR modality embeddings.
    Each modality gets a dedicated expert, plus an optional shared expert.
    """
    def __init__(self, in_dim, out_dim, num_modalities=3, layer_num=2,
                 use_shared_expert=True, top_k=2):
        super().__init__()
        self.num_modalities = num_modalities
        self.use_shared_expert = use_shared_expert
        self.top_k = top_k

        # One expert per modality
        self.modality_experts = nn.ModuleList([
            ExpertMLP(in_dim, out_dim, layer_num) for _ in range(num_modalities)
        ])

        if use_shared_expert:
            self.shared_expert = ExpertMLP(in_dim, out_dim, layer_num)
            num_total_experts = num_modalities + 1
        else:
            num_total_experts = num_modalities

        # Gate sees all modalities concatenated: [B*N, 3*D_single]
        self.gate = nn.Linear(in_dim * num_modalities, num_total_experts)
        self.num_total_experts = num_total_experts

    def forward(self, modality_features, gate_input):
        """
        Args:
            modality_features: list of 3 tensors, each [B, N, D_single]
            gate_input: [B, N, 3*D_single] (original concatenated features)
        Returns:
            fused: [B, N, out_dim]
        """
        B, N, _ = modality_features[0].shape

        gate_flat = rearrange(gate_input, "b n d -> (b n) d")
        gate_logits = self.gate(gate_flat)

        # Sparse top-k gating
        if self.top_k < self.num_total_experts:
            topk_vals, topk_ids = torch.topk(gate_logits, self.top_k, dim=-1)
            softmax_vals = F.softmax(topk_vals, dim=-1).to(gate_logits.dtype)  # cast back to match
            gate_scores = torch.zeros_like(gate_logits).scatter_(
                1, topk_ids, softmax_vals
            )
        else:
            gate_scores = F.softmax(gate_logits, dim=-1)

        # Run each modality expert
        expert_outputs = []
        for i in range(self.num_modalities):
            feat_flat = rearrange(modality_features[i], "b n d -> (b n) d")
            expert_outputs.append(self.modality_experts[i](feat_flat))

        if self.use_shared_expert:
            shared_input = torch.stack(modality_features, dim=0).mean(dim=0)
            shared_flat = rearrange(shared_input, "b n d -> (b n) d")
            expert_outputs.append(self.shared_expert(shared_flat))

        # [num_experts, B*N, out_dim] -> [B*N, num_experts, out_dim]
        expert_outputs = torch.stack(expert_outputs, dim=0).permute(1, 0, 2)
        gate_scores = gate_scores.unsqueeze(-1)

        fused = (gate_scores * expert_outputs).sum(dim=1)
        fused = rearrange(fused, "(b n) d -> b n d", b=B)

        return fused


class SpatialPoolingProjector(nn.Module):
    """
    Splits input [B, N, D] along D into 3 equal parts (T1, T2, FLAIR),
    applies spatial pooling per modality, then fuses via MoE.
    
    Input:  [B, N, D]  where D = 3 * D_single (concatenated modality features)
    Output: [B, N_pooled, out_dim]
    """
    def __init__(self, image_size, patch_size, in_dim, out_dim,
                 layer_type, layer_num, pooling_type='spatial', pooling_size=2,
                 num_modalities=3, use_shared_expert=True, moe_top_k=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type

        assert in_dim % num_modalities == 0, \
            f"in_dim ({in_dim}) must be divisible by num_modalities ({num_modalities})"
        self.d_single = in_dim // num_modalities
        self.in_dim = in_dim

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        # Per-modality layer norm
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(self.d_single) for _ in range(num_modalities)
        ])

        # MoE fusion: each expert takes D_single, outputs out_dim
        self.moe = ModalityMoE(
            in_dim=self.d_single,
            out_dim=out_dim,
            num_modalities=num_modalities,
            layer_num=layer_num,
            use_shared_expert=use_shared_expert,
            top_k=moe_top_k,
        )

    def _spatial_pool(self, x, B):
        """Pool a single modality: [B, N, D_single] -> [B, N_pooled, D_single]."""
        d = self.d_single
        if self.pooling_type == 'spatial':
            x = rearrange(x, "b (p1 p2 p3) d -> b d p1 p2 p3",
                          p1=self.num_patches_pre[0],
                          p2=self.num_patches_pre[1],
                          p3=self.num_patches_pre[2])
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            x = rearrange(x, "b d p1 p2 p3 -> b (p1 p2 p3) d")
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1)
            x = F.avg_pool1d(x, kernel_size=self.pooling_size ** 3, stride=self.pooling_size ** 3)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        """
        Args:
            x: [B, N, D] where D = 3 * D_single
        Returns:
            fused: [B, N_pooled, out_dim]
        """
        B = x.shape[0]

        # Split along channel dim: each [B, N, D_single]
        modality_tokens = torch.chunk(x, self.num_modalities, dim=-1)

        # Pool and normalize each modality
        pooled_features = []
        for i, mod_tokens in enumerate(modality_tokens):
            pooled = self._spatial_pool(mod_tokens, B)
            pooled = self.modality_norms[i](pooled)
            pooled_features.append(pooled)

        # Gate input: re-concatenate pooled features along D for gating context
        gate_input = torch.cat(pooled_features, dim=-1)  # [B, N_pooled, 3*D_single]

        # MoE fusion
        fused = self.moe(pooled_features, gate_input)

        return fused

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num