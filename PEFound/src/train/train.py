import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers.trainer_callback import TrainerCallback
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from PEFound.src.dataset.multi_dataset import PEDataset
from PEFound.src.model.language_model import  LamedPhi3ForCausalLM
from PEFound.src.train.lamed_trainer import LaMedTrainer
import json
import shutil


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class BestCheckpointCallback(TrainerCallback):
    """
    Callback to save the best checkpoint based on validation metric.
    Supports: tune_mm_mlp_adapter (pretrain), lora_enable (finetune), and full model.
    """
    def __init__(self, metric_name="eval_loss", mode="min", output_dir=None, 
                 tune_mm_mlp_adapter=False, lora_enable=False):
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.output_dir = output_dir
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.lora_enable = lora_enable
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation. Note: model is accessed via kwargs."""
        if metrics is None:
            return control
        
        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            rank0_print(f"Warning: {self.metric_name} not found in metrics")
            return control
        
        is_best = False
        if self.mode == 'min':
            if current_metric < self.best_metric:
                is_best = True
                self.best_metric = current_metric
        else:
            if current_metric > self.best_metric:
                is_best = True
                self.best_metric = current_metric
        
        if not is_best:
            return control

        rank0_print(f"New best {self.metric_name}: {current_metric:.4f} at step {state.global_step}")
        
        if args.local_rank not in [-1, 0]:
            return control

        # Save best checkpoint info
        best_info = {
            'best_metric': float(self.best_metric),
            'metric_name': self.metric_name,
            'step': state.global_step,
            'epoch': state.epoch,
        }
        best_info_path = os.path.join(args.output_dir, 'best_checkpoint_info.json')
        with open(best_info_path, 'w') as f:
            json.dump(best_info, f, indent=2)

        if self.tune_mm_mlp_adapter:
            # Pretrain mode: save projector weights only
            model = kwargs.get('model', None)
            if model is None:
                rank0_print("Warning: model not available in callback, cannot save projector")
                return control
            
            keys_to_match = ['mm_projector', 'embed_tokens']
            weight_to_save = get_mm_projector_state_maybe_zero_3(
                model.named_parameters(), keys_to_match
            )
            best_projector_path = os.path.join(args.output_dir, 'best_mm_projector.bin')
            torch.save(weight_to_save, best_projector_path)
            rank0_print(f"Best mm_projector saved to: {best_projector_path}")

        elif self.lora_enable:
            # LoRA finetune mode: save full state_dict (includes LoRA weights)
            model = kwargs.get('model', None)
            if model is None:
                rank0_print("Warning: model not available in callback, cannot save LoRA weights")
                return control
            
            best_lora_dir = os.path.join(args.output_dir, 'best_model_lora')
            os.makedirs(best_lora_dir, exist_ok=True)
            
            # Save the full state dict (base + LoRA + lm_head etc.)
            state_dict = model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            torch.save(cpu_state_dict, os.path.join(best_lora_dir, 'model_with_lora.bin'))
            rank0_print(f"Best LoRA model saved to: {best_lora_dir}/model_with_lora.bin")

        else:
            # Full finetune mode: force checkpoint save
            control.should_save = True
            
            checkpoint_dir = f"checkpoint-{state.global_step}"
            best_marker_path = os.path.join(args.output_dir, 'best_checkpoint')
            
            with open(best_marker_path, 'w') as f:
                f.write(checkpoint_dir)
            
            rank0_print(f"Best checkpoint marker set to: {checkpoint_dir}")
        
        return control


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=3)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})


@dataclass
class DataArguments:

    data_root: str = field(default="/your_path_to_data/Data/", metadata={"help": "Root directory for all data."})
    # caption data
    train_data_path: str = field(
        default="/your_path_to_data/Data/training.json",
        metadata={"help": "Path to caption data."},
    )

    val_data_path: str = field(
        default="/your_path_to_data/Data/val.json",
        metadata={"help": "Path to caption data."},
    )
    test_data_path: str = field(
        default="/your_path_to_data/Data/testing.json",
        metadata={"help": "Path to caption data."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = True
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = None
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 0.04
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"

    load_best_model_at_end: bool = field(
        default=False, 
        metadata={"help": "Load best model at the end of training"}
    )
    metric_for_best_model: str = field(
        default="eval_loss", 
        metadata={"help": "Metric to use for best model selection"}
    )
    greater_is_better: bool = field(
        default=False, 
        metadata={"help": "Whether higher metric value is better"}
    )


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    # Handle tuple/list inputs
    if isinstance(labels_ids, (tuple, list)):
        for item in labels_ids:
            if isinstance(item, np.ndarray) and item.ndim >= 1:
                labels_ids = item
                break
        else:
            labels_ids = np.concatenate(labels_ids, axis=0) if isinstance(labels_ids[0], np.ndarray) else np.array(labels_ids)

    if isinstance(pred_ids, (tuple, list)):
        for item in pred_ids:
            if isinstance(item, np.ndarray) and item.ndim >= 1:
                pred_ids = item
                break
        else:
            pred_ids = np.concatenate(pred_ids, axis=0) if isinstance(pred_ids[0], np.ndarray) else np.array(pred_ids)

    # Flatten everything
    labels_flat = labels_ids.reshape(-1)
    preds_flat = pred_ids.reshape(-1)

    # Align lengths: use the shorter one
    min_len = min(len(labels_flat), len(preds_flat))
    labels_flat = labels_flat[:min_len]
    preds_flat = preds_flat[:min_len]

    # Shift: compare preds[:-1] with labels[1:]
    labels_shifted = labels_flat[1:]
    preds_shifted = preds_flat[:-1]

    # Filter out padding (-100)
    valid_mask = labels_shifted != -100
    if valid_mask.sum() == 0:
        return {"accuracy": 0.0}

    filtered_preds = preds_shifted[valid_mask]
    filtered_labels = labels_shifted[valid_mask]
    acc_score = float(np.sum(filtered_preds == filtered_labels)) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    """
    Extract predicted token ids from logits.

    Handles the case where `logits` is a tuple (model returns
    (loss, logits, ...) or (logits, hidden_states, ...)).
    """
    # ── FIX: handle tuple logits ──────────────────────────────────────────
    if isinstance(logits, (tuple, list)):
        # Pick the first tensor that looks like logits (3-D: B, seq_len, vocab)
        for item in logits:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                logits = item
                break
        else:
            # Fallback: take the first element
            logits = logits[0]
    # ─────────────────────────────────────────────────────────────────────

    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_best_checkpoint(output_dir):
    """Load the best checkpoint path"""
    best_marker_path = os.path.join(output_dir, 'best_checkpoint')
    if os.path.exists(best_marker_path):
        with open(best_marker_path, 'r') as f:
            best_checkpoint_name = f.read().strip()
        return os.path.join(output_dir, best_checkpoint_name)
    return None


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self,):
        pass
    def __call__(self, batch: list) -> dict:
        images0, images1, images2, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image0', 'image1', 'image2', 'input_id', 'label', 'attention_mask'))

        images0 = torch.cat([_.unsqueeze(0) for _ in images0], dim=0)
        images1 = torch.cat([_.unsqueeze(0) for _ in images1], dim=0)
        images2 = torch.cat([_.unsqueeze(0) for _ in images2], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            images=[images0, images1, images2],
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return return_dict


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens

    special_token = {"additional_special_tokens": ["<im_patch>"]}
    tokenizer.add_special_tokens(special_token)

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        model = LamedPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
            )

    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)


    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head']]  #'vision_tower', 'mm_projector',#'vision_tower', 'mm_projector', 'embed_tokens', 
            ):
                p.requires_grad = True

        model.print_trainable_parameters()

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)


    train_dataset = PEDataset(data_args, tokenizer, mode='train')
    eval_dataset = PEDataset(data_args, tokenizer, mode='test')

    data_collator = DataCollator()

    mode = "max" if training_args.greater_is_better else "min"
    best_checkpoint_callback = BestCheckpointCallback(
        metric_name=training_args.metric_for_best_model,
        mode=mode,
        output_dir=training_args.output_dir,
        tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
        lora_enable=training_args.lora_enable,
    )


    rank0_print("="*20 + " Training " + "="*20)
    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                            callbacks=[best_checkpoint_callback],
                      )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)

    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.load_best_model_at_end:
        best_checkpoint_path = load_best_checkpoint(training_args.output_dir)        

        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            rank0_print(f"="*20 + " Loading Best Model " + "="*20)
            rank0_print(f"✓ Loading best model from: {best_checkpoint_path}")
            
            # Copy best checkpoint to 'best_model' directory
            best_model_dir = os.path.join(training_args.output_dir, 'best_model')
            if local_rank in [-1, 0]:
                if os.path.exists(best_model_dir):
                    shutil.rmtree(best_model_dir)
                shutil.copytree(best_checkpoint_path, best_model_dir)
                rank0_print(f"✓ Best model saved to: {best_model_dir}")
        else:
            rank0_print("Warning: Best checkpoint not found")

    rank0_print("="*20 + " Training Complete! " + "="*20)



if __name__ == "__main__":
    main()