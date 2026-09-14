"""
Pretraining script for Multi-Modal Contrastive + Reconstruction model.
Single GPU:
  python train.py --epochs 100 --batch_size 8 --precision bf16

Multi-GPU (DDP):
  torchrun --nproc_per_node=4 train.py --epochs 100 --batch_size 8 --precision bf16
"""

import os
import re
import glob
import json
import math
import random
import argparse
import logging
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from model import MultiModalPretrain, MultiModalPretrainConfig
from Dataset import PEDataset
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-modal pretraining (best version)")

    # Data
    p.add_argument("--data_root", type=str,
                   default="./Data/All/")

    p.add_argument("--train_data_path", type=str,
                   default="./Data/training.json")

    p.add_argument("--val_data_path", type=str,
                   default="./Data/val.json")

    p.add_argument("--test_data_path", type=str,
                   default="./testing.json")

    p.add_argument("--max_length", type=int, default=128)

    # Model
    p.add_argument("--language_model", type=str,
                   default="./pretrained_model/bert-base-uncased")
    p.add_argument("--in_channels", type=int, default=3)            # 3 modalities → 3 channels
    p.add_argument("--img_size", type=int, nargs=3, default=[32, 256, 256])
    p.add_argument("--patch_size", type=int, nargs=3, default=[4, 16, 16])
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--mlp_dim", type=int, default=3072)
    p.add_argument("--enc_num_layers", type=int, default=12)
    p.add_argument("--dec_num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--qkv_bias", action="store_true")

    # Loss
    p.add_argument("--contrastive_weight", type=float, default=1.0)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--norm_pix_loss", action="store_true", default=True,
                   help="MAE-style per-patch normalized recon target")
    p.add_argument("--no_norm_pix_loss", action="store_false", dest="norm_pix_loss")
    p.add_argument("--gather_loss", action="store_true", default=True,
                   help="Gather features across DDP ranks for more contrastive negatives")
    p.add_argument("--no_gather_loss", action="store_false", dest="gather_loss")

    # Training
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=18)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--bert_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=float, default=5.0,
                   help="Warmup expressed in epochs; converted to steps internally")
    p.add_argument("--min_lr_ratio", type=float, default=1e-2,
                   help="Floor of cosine schedule as fraction of base LR")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Masking
    p.add_argument("--mask_strategy", type=str, default="per_batch",
                   choices=["per_epoch", "per_batch", "none"])

    # Validation
    p.add_argument("--eval_every", type=int, default=1, help="Run validation every N epochs")

    # Checkpointing
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--keep_last", type=int, default=3, help="Keep last N periodic ckpts")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None) #

    # DDP
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------
def setup_ddp(args):
    if "RANK" in os.environ:
        args.rank       = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0 if torch.cuda.is_available() else -1
        args.distributed = False


def is_main(args):
    return args.rank == 0


def seed_everything(seed, rank):
    s = seed + rank   # different per rank → augmentation diversity
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Param groups (proper weight decay exclusions)
# ---------------------------------------------------------------------------
NO_DECAY_KEYS = ("bias", "cls_token", "pos_embed", "position_embeddings",
                 "logit_scale", "norm.weight", "norm1.weight", "norm2.weight",
                 "LayerNorm.weight")


def _is_no_decay(name, param):
    if param.ndim <= 1:
        return True
    if any(k in name for k in NO_DECAY_KEYS):
        return True
    return False


def build_param_groups(model, base_lr, bert_lr_scale, wd):
    bert_decay, bert_no_decay = [], []
    other_decay, other_no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bert = name.startswith("language_encoder.")
        no_decay = _is_no_decay(name, p)
        if is_bert and no_decay:        bert_no_decay.append(p)
        elif is_bert:                   bert_decay.append(p)
        elif no_decay:                  other_no_decay.append(p)
        else:                           other_decay.append(p)

    bert_lr = base_lr * bert_lr_scale
    return [
        {"params": other_decay,    "lr": base_lr, "weight_decay": wd,  "base_lr": base_lr},
        {"params": other_no_decay, "lr": base_lr, "weight_decay": 0.0, "base_lr": base_lr},
        {"params": bert_decay,     "lr": bert_lr, "weight_decay": wd,  "base_lr": bert_lr},
        {"params": bert_no_decay,  "lr": bert_lr, "weight_decay": 0.0, "base_lr": bert_lr},
    ]


# ---------------------------------------------------------------------------
# Per-step cosine LR with warmup
# ---------------------------------------------------------------------------
def cosine_factor(step, total_steps, warmup_steps, min_ratio):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, factor):
    for pg in optimizer.param_groups:
        pg["lr"] = pg["base_lr"] * factor


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def cleanup_checkpoints(output_dir, keep_last=3):
    pattern = os.path.join(output_dir, "checkpoint_epoch_*.pt")
    files = glob.glob(pattern)

    def epoch_num(path):
        m = re.search(r"checkpoint_epoch_(\d+)\.pt", os.path.basename(path))
        return int(m.group(1)) if m else 0

    files = sorted(files, key=epoch_num)
    for old in files[:-keep_last]:
        try:
            os.remove(old)
            logger.info(f"Removed old checkpoint: {old}")
        except OSError:
            pass


def save_encoder_weights(model, output_dir):
    """Save vision_tower.* weights only — loads into ViT3DTower with strict=True."""
    state = {}
    for name, p in model.named_parameters():
        if name.startswith("vision_tower."):
            state[name] = p.data.cpu()
    for name, b in model.named_buffers():
        if name.startswith("vision_tower."):
            state[name] = b.cpu()
    save_path = os.path.join(output_dir, "encoder_pretrained.pt")
    torch.save(state, save_path)
    logger.info(f"Saved {len(state)} vision_tower tensors to {save_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_ctx_factory, args):
    model.eval()
    sums = {"loss": 0.0, "contr": 0.0, "recon": 0.0, "i2t_acc": 0.0, "t2i_acc": 0.0}
    n = 0
    for batch in loader:
        m0 = batch["image0"].to(device, non_blocking=True)
        m1 = batch["image1"].to(device, non_blocking=True)
        m2 = batch["image2"].to(device, non_blocking=True)
        ids = batch["input_id"].to(device, non_blocking=True)
        am  = batch["attention_mask"].to(device, non_blocking=True)
        with amp_ctx_factory():
            out = model(modality_0=m0, modality_1=m1, modality_2=m2,
                        input_ids=ids, attention_mask=am, mask_idx=None)
        sums["loss"]    += out["loss"].item()
        sums["contr"]   += out["contrastive_loss"].item()
        sums["recon"]   += out["recon_loss"].item()
        sums["i2t_acc"] += out["i2t_acc"].item()
        sums["t2i_acc"] += out["t2i_acc"].item()
        n += 1

    if n == 0:
        return {k: 0.0 for k in sums}
    avg = {k: v / n for k, v in sums.items()}

    # All-reduce mean across DDP ranks
    if args.distributed:
        t = torch.tensor([avg[k] for k in sums], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= args.world_size
        avg = {k: t[i].item() for i, k in enumerate(sums)}

    model.train()
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_ddp(args)
    seed_everything(args.seed, args.rank)

    assert args.in_channels == 3, "Model is built for 3-channel input (3 modalities stacked)."

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    # ---------- Output dir ----------
    if is_main(args):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    # ---------- AMP context ----------
    amp_dtype_map = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = amp_dtype_map[args.precision]
    amp_enabled = amp_dtype is not None
    use_scaler = (amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_scaler)

    def amp_ctx():
        if not amp_enabled:
            return nullcontext()
        return autocast("cuda", dtype=amp_dtype)

    # ---------- Build model ----------
    config = MultiModalPretrainConfig(
        language_model_name_or_path=args.language_model,
        in_channels=args.in_channels,
        img_size=tuple(args.img_size),
        patch_size=tuple(args.patch_size),
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        enc_num_layers=args.enc_num_layers,
        dec_num_layers=args.dec_num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        qkv_bias=args.qkv_bias,
        contrastive_weight=args.contrastive_weight,
        recon_weight=args.recon_weight,
        norm_pix_loss=args.norm_pix_loss,
        gather_loss=args.gather_loss,
    )
    model = MultiModalPretrain(config).to(device)

    if is_main(args):
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model: {n_total:.1f}M total, {n_train:.1f}M trainable")

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
    raw_model = model.module if args.distributed else model

    # ---------- Tokenizer & datasets ----------
    tokenizer = BertTokenizer.from_pretrained(config.language_model_name_or_path)

    train_ds = PEDataset(args=args, tokenizer=tokenizer, mode="train")
    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    # If your PEDataset supports a `mode` arg ("train"/"val"), use it; else
    # construct a separate val args namespace pointing at the val json.
    val_ds   = PEDataset(args=args, tokenizer=tokenizer, mode="val")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if args.distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    # ---------- Optimizer ----------
    param_groups = build_param_groups(raw_model, args.lr, args.bert_lr_scale, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # ---------- Resume ----------
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        if use_scaler and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if is_main(args):
            logger.info(f"Resumed from epoch {start_epoch} (step {global_step})")

    # ---------- TensorBoard ----------
    writer = SummaryWriter(args.output_dir) if is_main(args) else None

    if is_main(args):
        config.save_pretrained(args.output_dir)
        logger.info("=" * 60)
        logger.info(f"Pretraining: {args.epochs} epochs × {steps_per_epoch} steps "
                    f"= {total_steps} total steps; warmup {warmup_steps}")
        logger.info("=" * 60)

    best_val = float("inf")

    # ===================================================================
    # Training loop
    # ===================================================================
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_mask = epoch % 3 if args.mask_strategy == "per_epoch" else None
        model.train()
        accum_total = accum_contr = accum_recon = accum_i2t = accum_t2i = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            m0 = batch["image0"].to(device, non_blocking=True)
            m1 = batch["image1"].to(device, non_blocking=True)
            m2 = batch["image2"].to(device, non_blocking=True)
            ids = batch["input_id"].to(device, non_blocking=True)
            am  = batch["attention_mask"].to(device, non_blocking=True)

            if args.mask_strategy == "per_batch":
                mask_idx = random.randint(0, 2)
            elif args.mask_strategy == "per_epoch":
                mask_idx = epoch_mask
            else:
                mask_idx = None

            do_step = ((step + 1) % args.grad_accum_steps) == 0
            sync_ctx = (model.no_sync()
                        if (args.distributed and not do_step)
                        else nullcontext())

            with sync_ctx:
                with amp_ctx():
                    out = model(modality_0=m0, modality_1=m1, modality_2=m2,
                                input_ids=ids, attention_mask=am, mask_idx=mask_idx)
                    loss = out["loss"] / args.grad_accum_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if do_step:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Per-step LR update
                factor = cosine_factor(global_step, total_steps, warmup_steps, args.min_lr_ratio)
                set_lr(optimizer, factor)
                global_step += 1

            # Accumulate logs
            accum_total += out["loss"].item()
            accum_contr += out["contrastive_loss"].item()
            accum_recon += out["recon_loss"].item()
            accum_i2t   += out["i2t_acc"].item()
            accum_t2i   += out["t2i_acc"].item()
            n_steps += 1

            if is_main(args) and (step + 1) % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                temp = 1.0 / out["logit_scale"].item()
                logger.info(
                    f"Ep[{epoch+1}/{args.epochs}] Step[{step+1}/{len(train_loader)}] "
                    f"loss={accum_total/n_steps:.4f} "
                    f"contr={accum_contr/n_steps:.4f} recon={accum_recon/n_steps:.4f} "
                    f"i2t={accum_i2t/n_steps:.3f} t2i={accum_t2i/n_steps:.3f} "
                    f"temp={temp:.4f} lr={lr_now:.2e} mask={mask_idx}"
                )
                if writer is not None:
                    writer.add_scalar("train/loss",       accum_total/n_steps, global_step)
                    writer.add_scalar("train/contrastive", accum_contr/n_steps, global_step)
                    writer.add_scalar("train/recon",      accum_recon/n_steps, global_step)
                    writer.add_scalar("train/i2t_acc",    accum_i2t/n_steps, global_step)
                    writer.add_scalar("train/t2i_acc",    accum_t2i/n_steps, global_step)
                    writer.add_scalar("train/temperature", temp, global_step)
                    writer.add_scalar("train/lr",         lr_now, global_step)

        # ---- End-of-epoch: flush leftover gradients ----
        if (step + 1) % args.grad_accum_steps != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---- End-of-epoch summary ----
        avg_train = accum_total / max(n_steps, 1)
        if is_main(args):
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}] DONE | "
                f"train_loss={avg_train:.4f} "
                f"contr={accum_contr/max(n_steps,1):.4f} "
                f"recon={accum_recon/max(n_steps,1):.4f}"
            )

        # ---- Validation ----
        if (epoch + 1) % args.eval_every == 0:
            val = evaluate(model, val_loader, device, amp_ctx, args)
            if is_main(args):
                logger.info(
                    f"  VAL: loss={val['loss']:.4f} contr={val['contr']:.4f} "
                    f"recon={val['recon']:.4f} i2t={val['i2t_acc']:.3f} t2i={val['t2i_acc']:.3f}"
                )
                if writer is not None:
                    writer.add_scalar("val/loss",        val["loss"],    epoch + 1)
                    writer.add_scalar("val/contrastive", val["contr"],   epoch + 1)
                    writer.add_scalar("val/recon",       val["recon"],   epoch + 1)
                    writer.add_scalar("val/i2t_acc",     val["i2t_acc"], epoch + 1)
                    writer.add_scalar("val/t2i_acc",     val["t2i_acc"], epoch + 1)

                if val["loss"] < best_val:
                    best_val = val["loss"]
                    best_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "val_loss": best_val,
                        "config": config.to_dict(),
                    }, best_path)
                    save_encoder_weights(raw_model, args.output_dir)  # keep encoder fresh
                    logger.info(f"  ✓ New best (val_loss={best_val:.4f}) — saved best_model.pt + encoder_pretrained.pt")

        # ---- Periodic checkpoint ----
        if is_main(args) and ((epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs):
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_scaler else None,
                "train_loss": avg_train,
                "config": config.to_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            cleanup_checkpoints(args.output_dir, keep_last=args.keep_last)

    # ---- Final: save encoder for downstream ----
    if is_main(args):
        save_encoder_weights(raw_model, args.output_dir)
        if writer is not None:
            writer.close()
        logger.info("Pretraining complete!")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

    #torchrun --nproc_per_node=4 Train.py --epochs 10000 --batch_size 16 --precision bf16