from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector


def load_pretrained_vision_tower(vision_tower, pretrain_path):
    """
    Load pretrained encoder weights into the vision tower, handling key
    mismatches between the custom pretrain PatchEmbeddingBlock and MONAI's
    PatchEmbeddingBlock.

    MONAI's PatchEmbeddingBlock uses:
        patch_embeddings = nn.Sequential(
            nn.LayerNorm(...),          # index 0  (only for "perceptron")
            nn.Linear(...) or Conv3d    # index 1
        )
    The pretrain script uses a bare nn.Linear / nn.Conv3d without Sequential,
    so checkpoint keys lack the ".0." / ".1." index.
    """
    ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
    else:
        state_dict = ckpt

    # ── Step 1: Add "vision_tower." prefix if the model expects it ────────
    model_dict = vision_tower.state_dict()
    model_keys = list(model_dict.keys())
    if (
        len(model_keys) > 0
        and model_keys[0].startswith("vision_tower.")
        and not any(k.startswith("vision_tower.") for k in state_dict.keys())
    ):
        state_dict = {f"vision_tower.{k}": v for k, v in state_dict.items()}

    # ── Step 2: Remap keys for MONAI's Sequential PatchEmbeddingBlock ─────
    #
    # Pretrain key pattern:
    #   "...patch_embedding.patch_embeddings.weight"
    #   "...patch_embedding.patch_embeddings.bias"
    #
    # MONAI key pattern (perceptron mode):
    #   "...patch_embedding.patch_embeddings.0.weight"  (LayerNorm)
    #   "...patch_embedding.patch_embeddings.0.bias"    (LayerNorm)
    #   "...patch_embedding.patch_embeddings.1.weight"  (Linear/Conv)
    #   "...patch_embedding.patch_embeddings.1.bias"    (Linear/Conv)
    #
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        # Only remap the bare patch_embeddings.weight/bias (not already indexed)
        if "patch_embedding.patch_embeddings.weight" in k and ".patch_embeddings.0." not in k and ".patch_embeddings.1." not in k:
            new_k = k.replace(
                "patch_embedding.patch_embeddings.weight",
                "patch_embedding.patch_embeddings.1.weight",
            )
        elif "patch_embedding.patch_embeddings.bias" in k and ".patch_embeddings.0." not in k and ".patch_embeddings.1." not in k:
            new_k = k.replace(
                "patch_embedding.patch_embeddings.bias",
                "patch_embedding.patch_embeddings.1.bias",
            )
        remapped[new_k] = v
    state_dict = remapped

    # ── Step 3: Filter by matching keys and shapes ────────────────────────
    filtered_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
        else:
            skipped.append(k)

    missing, unexpected = vision_tower.load_state_dict(filtered_state_dict, strict=False)

    print(f"Loaded pretrained vision weights from: {pretrain_path}")
    print(f"  Matched keys: {len(filtered_state_dict)}")
    print(f"  Skipped keys: {len(skipped)}")
    if skipped:
        print(f"  Example skipped keys: {skipped[:10]}")
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:20]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    return missing, unexpected


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        if model_args.pretrain_vision_model is not None:
            missing, unexpected = load_pretrained_vision_tower(
                self.vision_tower, model_args.pretrain_vision_model
            )
            print(f"Loaded pretrained vision weights from {model_args.pretrain_vision_model}")
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector"), strict=True
            )


class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """
        Encode images through the vision tower and projector.

        Args:
            images: (B, C, D, H, W) tensor, or list of per-modality tensors
                    that will be concatenated along dim=1.

        Returns:
            image_features: projected features ready for the LLM.
        """
        # ── FIX 1: Concatenate if list/tuple ──────────────────────────────
        if isinstance(images, (list, tuple)):
            images = torch.cat(images, dim=1)  # (B, C, D, H, W)

        # ── FIX 2: Single pass through vision tower ──────────────────────
        # vision_tower returns (B, N, encoder_hidden) patch tokens
        image_features = self.get_model().get_vision_tower()(images)

        # ── FIX 3: Single pass through mm_projector ──────────────────────
        # The projector maps encoder_hidden -> LLM hidden_size
        image_features = self.get_model().mm_projector(image_features)

        return image_features

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        else:
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (
                    inputs_embeds[:, :1, :],
                    image_features,
                    inputs_embeds[:, (image_features.shape[1] + 1) :, :],
                ),
                dim=1,
            )
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )
            embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. "
                    f"Pretrained: {embed_tokens_weight.shape}. "
                    f"Current: {input_embeddings.shape}. "
                    f"Number of new tokens: {num_new_tokens}."
                )