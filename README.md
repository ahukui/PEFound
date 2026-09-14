# Stage-I: Pre-training
This repository provides code for multimodal pre-training using paired brain MRI examinations and radiology reports.
The model jointly processes three MRI modalities—T1-weighted, T2-weighted, and T2-FLAIR—and optimizes two complementary objectives:
- **Image–text contrastive learning:** aligns visual representations with the corresponding radiology report representations.
- **Multimodal reconstruction:** reconstructs the original MRI volumes from inputs with independently masked 3D sub-volumes.
> This repository contains the pre-training stage only. The supplied files do not include downstream diagnosis fine-tuning, report-generation fine-tuning, or clinical inference scripts.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Pre-trained language model](#pre-trained-language-model)
- [Training](#training)
  
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

## Installation

### 1. Download the repository
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
# Stage-II: Training PEFound

## Step-1: projector training
```bash
bash script/pretrain_phi3.sh
```
This freezes the language backbone and encoder, and trains the MoE projector and input token embeddings. Defaults: 50 epochs, learning rate `1e-4`, evaluation every 50 steps.

Key outputs:
- `output/pretrain/best_mm_projector.bin`: projector and input embeddings selected by validation loss.
- `output/pretrain/mm_projector.bin`: final projector and input embeddings.
- `output/pretrain/config.json` and tokenizer files.
- `output/pretrain/run_config.json`: run arguments.

## Step-2: LoRA fine-tuning

Use the same Phi-3 base model and encoder checkpoint as stage 1.

```bash
export PROJECTOR_CKPT="$(pwd)/output/pretrain/best_mm_projector.bin"
bash script/finetune_lora_phi3.sh
```
Key outputs:

- `output/finetune/best_model_lora/`: checkpoint selected by validation loss.
- `output/finetune/model_with_lora.bin`: final full model state, including LoRA and trained non-LoRA parameters.
- `config.json`, `adapter_config.json` and tokenizer files: metadata required for merging.
- `best_checkpoint_info.json`: selected validation loss, step and epoch.

For four GPUs, run either stage with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 BATCH_SIZE=1 GRAD_ACCUM=4 \
  bash script/finetune_lora_phi3.sh
```
# Stage-III:  Merge LoRA weights

```bash
python -m src.utils.merge_lora_weights_and_save_hf_model \
  --checkpoint ./output/finetune/best_model_lora \
  --output_dir ./output/phi3_merged
```

To export final rather than validation-selected weights, use `--checkpoint ./output/finetune`. The merge reads the saved architecture and LoRA configuration, loads the full state strictly, and exports the model plus tokenizer. Allow enough CPU RAM for the complete model and checkpoint during merging.

# Stage-IV: Inference

Generate a report:

```bash
python -m src.infer \
  --model ./output/phi3_merged \
  --case_dir "$DATA_ROOT/All/Patient-003/Exam-001" \
  --task report \
  --output ./output/report.txt
```
Generate a diagnosis:

```bash
python -m src.infer \
  --model ./output/phi3_merged \
  --case_dir "$DATA_ROOT/All/Patient-003/Exam-001" \
  --task diagnosis --max_new_tokens 64
```
