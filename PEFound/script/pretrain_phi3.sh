#!/bin/bash
##sh PEFound/script/pretrain_phi3.sh
# run "accelerate config" first!

export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port 29501 \
    PEFound/src/train/train.py \
    --version v0 \
    --model_name_or_path ./PEFound/pretrained_model/Phi-3-mini-128k-instruct \
    --model_type phi3 \
    --vision_tower vit3d \
    --pretrain_vision_model /Your_Pretraining_path/checkpoints/encoder_pretrained.pt \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --bf16 True \
    --output_dir ./PEFound/output/pretrain\
    --num_train_epochs 50 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
