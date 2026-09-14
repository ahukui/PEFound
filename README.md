# PEFound: Multimodal MRI–Report Pre-training

This repository provides code for multimodal MRI–report pre-training using paired brain MRI examinations and radiology reports.

The model jointly processes three MRI modalities—T1-weighted, T2-weighted, and T2-FLAIR—and optimizes two complementary objectives:

- **Image–text contrastive learning:** aligns visual representations with the corresponding radiology report representations.
- **Multimodal reconstruction:** reconstructs the original MRI volumes from inputs with independently masked 3D sub-volumes.

The implementation supports single-GPU training, distributed multi-GPU training, mixed precision, validation, TensorBoard logging, checkpoint resumption, and vision-encoder export.

> This repository contains the pre-training stage only. The supplied files do not include downstream diagnosis fine-tuning, report-generation fine-tuning, or clinical inference scripts.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Pre-trained language model](#pre-trained-language-model)
- [Training](#training)
- [Monitoring and outputs](#monitoring-and-outputs)
- [Resuming training](#resuming-training)
- [Implementation notes](#implementation-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License and data availability](#license-and-data-availability)

## Requirements

The supplied dependency file specifies:

| Package | Version |
|---|---|
| PyTorch | 2.2.1 + CUDA 11.8 |
| Torchvision | 0.17.1 + CUDA 11.8 |
| Transformers | 4.39.1 |
| MONAI | 1.3.0 |
| NumPy | 1.26.4 |
| SciPy | 1.13.0 |
| SimpleITK | 2.3.1 |
| einops | 0.8.0 |

Use a Python 3.10 environment and an NVIDIA GPU with a driver compatible with the selected PyTorch CUDA build.

GPU memory requirements depend on batch size and precision. Start with a small per-GPU batch size and increase it according to available memory.

> The instructions below are based on source-code inspection. End-to-end training requires the MRI data and BERT weights and has not been verified with the supplied files alone.

## Installation

### 1. Download the repository

Replace the URL and directory name with the actual repository information.

```bash
git clone https://github.com/ahukui/PEFound.git
cd PEFound
```

### 2. Create an environment

```bash
conda create -n pefound python=3.10 -y
conda activate pefound
```

### 3. Install dependencies

Install the CUDA 11.8 PyTorch packages first:

```bash
python -m pip install torch==2.2.1 torchvision==0.17.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install the supplied requirements and TensorBoard:

```bash
python -m pip install -r requirements.txt
python -m pip install tensorboard
```
## Data preparation

### MRI directory structure

Each examination must have its own directory containing three NIfTI files:

```text
Data/
├── training.json
├── val.json
└── All/
    ├── case_0001/
    │   ├── T1.nii.gz
    │   ├── T2.nii.gz
    │   └── T2_Flair.nii.gz
    ├── case_0002/
    │   ├── T1.nii.gz
    │   ├── T2.nii.gz
    │   └── T2_Flair.nii.gz
    └── case_0003/
        ├── T1.nii.gz
        ├── T2.nii.gz
        └── T2_Flair.nii.gz
```

The filenames must match exactly:

| Modality | Required filename |
|---|---|
| T1-weighted MRI | `T1.nii.gz` |
| T2-weighted MRI | `T2.nii.gz` |
| T2-FLAIR MRI | `T2_Flair.nii.gz` |

The modalities should be spatially aligned before training.

### JSON annotations

Both split files must contain a JSON array.

Example `training.json`:

```json
[
  {
    "images": ["case_0001"],
    "dignosis": ["Example impression for case 0001."],
    "report": ["Example radiology findings for case 0001."]
  },
  {
    "images": ["case_0002"],
    "dignosis": ["Example impression for case 0002."],
    "report": ["Example radiology findings for case 0002."]
  }
]
```

Example `val.json`:

```json
[
  {
    "images": ["case_0003"],
    "dignosis": ["Example impression for case 0003."],
    "report": ["Example radiology findings for case 0003."]
  }
]
```

Replace all example text with the corresponding examination-level annotations.

| Field | Required format | Meaning |
|---|---|---|
| `images` | Non-empty list of strings | The first element identifies the case directory relative to `--data_root` |
| `dignosis` | Non-empty list of strings | The first element supplies diagnosis/impression text |
| `report` | Non-empty list of strings | The first element supplies report text |

## Pre-trained language model

By default, the code loads BERT from:

```text
./pretrained_model/bert-base-uncased
```

For offline training, provide a local Hugging Face BERT directory containing:

```text
pretrained_model/bert-base-uncased/
├── config.json
├── model.safetensors          # or pytorch_model.bin
├── vocab.txt
└── tokenizer_config.json
```

Use:

```bash
--language_model ./pretrained_model/bert-base-uncased
```

If network access is available, a model identifier can be supplied instead:

```bash
--language_model bert-base-uncased
```

The code loads both `BertModel` and `BertTokenizer` from this location. The text encoder's hidden dimension must match `--hidden_size`, which defaults to 768.

## Training

The commands below are examples for running the supplied implementation, not a claim about the final paper's experimental settings.

Both training and validation annotation files are required.

### Single-GPU smoke test

Start with one epoch and a small batch size:

```bash
CUDA_VISIBLE_DEVICES=0 python Train.py \
  --data_root ./Data/All \
  --train_data_path ./Data/training.json \
  --val_data_path ./Data/val.json \
  --language_model ./pretrained_model/bert-base-uncased \
  --output_dir ./checkpoints/smoke_test \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 0 \
  --precision fp16 \
  --log_every 1 \
  --eval_every 1 \
  --save_every 1
```

Provide at least two valid training cases for this example. The training loader uses `drop_last=True`, so the dataset must contain at least one complete batch.

A single-GPU batch size of 1 can test reconstruction and data loading, but its contrastive loss has no negative pairs and is therefore uninformative.

### Single-GPU training

```bash
CUDA_VISIBLE_DEVICES=0 python Train.py \
  --data_root ./Data/All \
  --train_data_path ./Data/training.json \
  --val_data_path ./Data/val.json \
  --language_model ./pretrained_model/bert-base-uncased \
  --output_dir ./checkpoints/single_gpu \
  --epochs 100 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --precision bf16 \
  --lr 1.5e-4 \
  --bert_lr_scale 0.1 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --num_workers 4 \
  --eval_every 1 \
  --save_every 10 \
  --seed 42
```

Use `--precision fp16` if the GPU does not support BF16.

### Four-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node=4 \
  Train.py \
  --data_root ./Data/All \
  --train_data_path ./Data/training.json \
  --val_data_path ./Data/val.json \
  --language_model ./pretrained_model/bert-base-uncased \
  --output_dir ./checkpoints/four_gpu \
  --epochs 100 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --precision bf16 \
  --lr 1.5e-4 \
  --bert_lr_scale 0.1 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --num_workers 4 \
  --eval_every 1 \
  --save_every 10 \
  --seed 42
```

`--batch_size` is the batch size **per GPU**.

By default, DDP gathers representations across GPUs for contrastive learning. Gradient accumulation increases the optimization batch size but does not combine contrastive negatives across accumulated microbatches.

For the supplied script, use `--grad_accum_steps 1` initially. If increasing it, ensure the number of training batches per rank is divisible by the accumulation count. The incomplete-window handling requires care, especially under DDP.

### Main arguments

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./Data/All/` | MRI case root |
| `--train_data_path` | `./Data/training.json` | Training annotations |
| `--val_data_path` | `./Data/val.json` | Validation annotations |
| `--max_length` | `128` | Maximum text token length |
| `--img_size` | `32 256 256` | Model input dimensions |
| `--patch_size` | `4 16 16` | ViT patch dimensions |
| `--hidden_size` | `768` | Embedding dimension |
| `--enc_num_layers` | `12` | Vision encoder blocks |
| `--dec_num_layers` | `4` | Reconstruction decoder blocks |
| `--num_heads` | `12` | Attention heads |
| `--epochs` | `10000` | Total epochs |
| `--batch_size` | `18` | Per-GPU batch size |
| `--grad_accum_steps` | `1` | Gradient accumulation count |
| `--lr` | `1.5e-4` | Base learning rate |
| `--bert_lr_scale` | `0.1` | BERT learning-rate multiplier |
| `--weight_decay` | `0.05` | AdamW weight decay |
| `--precision` | `bf16` | `fp32`, `fp16`, or `bf16` |
| `--eval_every` | `1` | Validation interval in epochs |
| `--save_every` | `50` | Periodic checkpoint interval |
| `--keep_last` | `3` | Number of periodic checkpoints retained |
| `--seed` | `42` | Random seed |

List all arguments:

```bash
python Train.py --help
```

### Checkpoint files

```text
checkpoints/single_gpu/
├── config.json
├── best_model.pt
├── encoder_pretrained.pt
├── checkpoint_epoch_10.pt
└── events.out.tfevents.*
```

| File | Contents |
|---|---|
| `config.json` | Model configuration |
| `best_model.pt` | Lowest-validation-loss model within the current run |
| `checkpoint_epoch_<N>.pt` | Model, optimizer, scaler, epoch, and global-step states |
| `encoder_pretrained.pt` | Vision-encoder parameters and buffers |

Periodic checkpoint cleanup removes older periodic files, retaining the latest `--keep_last` files.
