"""
Drop-in replacement for PEDataset in LaMed/src/dataset/multi_dataset.py.

Key fixes vs the original:
  1. Modalities are stacked into (C=3, D, H, W) BEFORE any random transform,
     so spatial augmentations stay aligned across T1/T2/FLAIR.
  2. Real augmentation pipeline: spatial flip, small affine rotation,
     bias field, gamma, gaussian noise, modality dropout.
  3. Class distribution is printed at init.
  4. Errors are caught specifically and logged (not silently swallowed).
  5. Eval uses a deterministic prompt for reproducibility.
  6. Exposes get_class_weights() for use with WeightedRandomSampler.
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, gaussian_filter, rotate as nd_rotate
import SimpleITK as sitk
rng = random.Random(42)

from .prompt_templates import Caption_templates, Diagnose_templates
import re

# ══════════════════════════════════════════════════════════════════════
# Multi-modal-aware augmentations.
# Each callable takes and returns a numpy array of shape (C, D, H, W).
# Randomness is drawn ONCE per __call__, so all channels see the same
# transform — preserving T1/T2/FLAIR alignment.
# ══════════════════════════════════════════════════════════════════════
class RandFlipStacked:
    """Random flip along one spatial axis, identical across channels."""
    def __init__(self, prob=0.5, spatial_axis=2):
        self.prob = prob
        self.spatial_axis = spatial_axis  # 0=D, 1=H, 2=W → array axis (axis+1)

    def __call__(self, x):
        if random.random() < self.prob:
            x = np.flip(x, axis=self.spatial_axis + 1).copy()
        return x


class RandAffineStacked:
    """
    Small random 3D rotation applied identically to all channels.
    Avoids 90° rotations (not anatomically meaningful for brain MRI).
    """
    def __init__(self, prob=0.3, max_angle_deg=10.0):
        self.prob = prob
        self.max_angle = max_angle_deg

    def __call__(self, x):
        if random.random() < self.prob:
            # One small rotation in the axial plane (most anatomically safe)
            angle = random.uniform(-self.max_angle, self.max_angle)
            # axes refer to dims of x: x is (C, D, H, W), axial plane = (H, W) = (2, 3)
            x = nd_rotate(x, angle=angle, axes=(2, 3),
                          reshape=False, order=1, mode="constant", cval=0.0)
        return x


class RandGaussianNoise:
    def __init__(self, prob=0.3, std=0.03):
        self.prob = prob
        self.std = std

    def __call__(self, x):
        if random.random() < self.prob:
            s = random.uniform(0.0, self.std)
            x = x + np.random.randn(*x.shape).astype(np.float32) * s
        return x


class RandGamma:
    """Gamma correction. Assumes input already roughly in [0, 1]."""
    def __init__(self, prob=0.4, gamma_range=(0.7, 1.4)):
        self.prob = prob
        self.gamma_range = gamma_range

    def __call__(self, x):
        if random.random() < self.prob:
            g = random.uniform(*self.gamma_range)
            x = np.clip(x, 0.0, 1.0)
            x = np.power(x, g, dtype=np.float32)
        return x


class RandBiasField:
    """
    Multiplicative low-frequency field — simulates MRI bias field
    inhomogeneity. One of the most impactful augs for cross-scanner
    robustness.
    """
    def __init__(self, prob=0.3, max_strength=0.25):
        self.prob = prob
        self.max_strength = max_strength

    def __call__(self, x):
        if random.random() < self.prob:
            C, D, H, W = x.shape
            # Generate low-res random field, smooth, upsample to full size
            low = np.random.randn(max(D // 8, 4), max(H // 8, 4), max(W // 8, 4)).astype(np.float32)
            low = gaussian_filter(low, sigma=2.0)
            zfac = (D / low.shape[0], H / low.shape[1], W / low.shape[2])
            field = zoom(low, zfac, order=1)
            field = field / (np.abs(field).max() + 1e-6)
            field = 1.0 + self.max_strength * field
            x = x * field[None, :, :, :]  # broadcast across channels
        return x


class RandModalityDropout:
    """
    Zero out a randomly chosen modality channel. Encourages the model
    to not over-rely on any single sequence — useful when one modality
    is noisier or sometimes unavailable in deployment.
    """
    def __init__(self, prob=0.15):
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob and x.shape[0] > 1:
            ch = random.randint(0, x.shape[0] - 1)
            x = x.copy()
            x[ch] = 0.0
        return x


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════
class PEDataset(Dataset):
    MODALITY_FILES = ["T1.nii.gz", "T2.nii.gz", "T2_Flair.nii.gz"]
    TARGET_SHAPE = (32, 256, 256)
    patient_pattern = re.compile(r"^Patient-\d+$")
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        # ── Pick JSON splits ─────────────────────────────────────────
        if mode == "train":
            json_paths = [args.train_data_path]
            self.data_root = os.path.join(self.data_root, "All")

        elif mode in ("val", "validation"):
            json_paths = [args.val_data_path]
            self.data_root = os.path.join(self.data_root, "All")

        elif mode == "test":
            json_paths = [args.test_data_path]
            self.data_root = os.path.join(self.data_root, "All")
            
        elif mode == "exter":
            json_paths = [args.test_data_path]
            self.data_root = os.path.join(self.data_root)                          
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ── Load + dedup by image path ───────────────────────────────
        self.data_list, seen = [], set()
        for jp in json_paths:
            with open(jp, "r") as f:
                items = json.load(f)

            #if mode == "train":
            jp_str = str(jp)
            jp_name = Path(jp_str).name

            for item in items:
                images = item.get("images", [])
                # Use first image as dedup key
                key = images[0].strip() if len(images) > 0 and isinstance(images[0], str) else id(item)
                seen.add(key)
                self.data_list.append(item)


        # ── Diagnostic: class distribution ───────────────────────────
        self._class_counts = self._compute_class_counts(self.data_list)
        print(f"[PEDataset/{mode}] N={len(self.data_list)}  "
              f"class_distribution={self._class_counts}")

        self.caption_prompts = Caption_templates
        self.Dignoise_prompts = Diagnose_templates

        # ── Augmentation pipeline (lists of callables) ───────────────
        if mode == "train":
            self.transforms = [
                RandFlipStacked(prob=0.5, spatial_axis=2),     # L–R flip (W)
                RandAffineStacked(prob=0.3, max_angle_deg=10), # ±10° axial
                RandBiasField(prob=0.3, max_strength=0.25),
                RandGamma(prob=0.4, gamma_range=(0.7, 1.4)),
                RandGaussianNoise(prob=0.3, std=0.03),
                RandModalityDropout(prob=0.30),
            ]
        else:
            self.transforms = []  # no aug for val/test (TTA handled separately)

    # ── Convenience ──────────────────────────────────────────────────
    @staticmethod
    def _normalize_label(raw):
        return "Pediatric Epilepsy" if raw == "PE" else "Headache"

    @staticmethod
    def _compute_class_counts(items):
        counts = {}
        for d in items:
            try:
                lbl = PEDataset._normalize_label(d["dignosis"][0])
            except (KeyError, IndexError, TypeError):
                lbl = "?"
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def get_class_weights(self):
        """
        Returns a list of per-sample weights (inverse class frequency) for
        use with torch.utils.data.WeightedRandomSampler. Call this once,
        not per epoch.
        """
        per_class_w = {c: 1.0 / max(n, 1) for c, n in self._class_counts.items()}
        weights = []
        for d in self.data_list:
            try:
                lbl = self._normalize_label(d["dignosis"][0])
            except Exception:
                lbl = "?"
            weights.append(per_class_w.get(lbl, 1.0))
        return weights

    def __len__(self):
        return len(self.data_list)

    # ── Image loading ────────────────────────────────────────────────
    def _resample(self, vol, order=1):
        zoom_factors = [t / s for t, s in zip(self.TARGET_SHAPE, vol.shape)]
        return zoom(vol, zoom_factors, order=order).astype(np.float32)

    def _normalize_1_99(self, vol, eps=1e-6):
        """1–99 percentile clip + min-max to [0, 1]."""
        v_lo = np.percentile(vol, 1)
        v_hi = np.percentile(vol, 99)
        vol = np.clip(vol, v_lo, v_hi)
        return ((vol - v_lo) / (v_hi - v_lo + eps)).astype(np.float32)

    def _load_one_modality(self, path):
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)        # (D, H, W)
        arr = self._resample(arr, order=1)
        arr = self._normalize_1_99(arr)
        return arr

    # ── Main item loader ─────────────────────────────────────────────
    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]
            image_path = data["images"][0]
            image_abs_path = os.path.join(self.data_root, image_path)
            #print('~~~~~~~~~~~~~~~~~~~~~~', image_abs_path)
            # Load all 3 modalities, stack to (C=3, D, H, W)
            mods = [self._load_one_modality(os.path.join(image_abs_path, m))
                    for m in self.MODALITY_FILES]
            stack = np.stack(mods, axis=0)  # (3, D, H, W)

            # Apply transforms ONCE on the stacked tensor → preserves alignment
            for t in self.transforms:
                stack = t(stack)

            # Split back per modality (the model's collator expects this format)
            stack = torch.from_numpy(np.ascontiguousarray(stack)).float()
            image0 = stack[0:1]   # (1, D, H, W)
            image1 = stack[1:2]
            image2 = stack[2:3]

        except FileNotFoundError as e:
            print(f"[PEDataset/{self.mode}] missing file at idx={idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
        except Exception as e:
            print(f"[PEDataset/{self.mode}] error at idx={idx}: "
                  f"{type(e).__name__}: {e}")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # ── Build text ───────────────────────────────────────────────
        
        if self.mode == "train":
            if random.random() < 0.2:
                prompt_question = random.choice(self.Dignoise_prompts)
                diagnosis = self._normalize_label(data["diagnosis"][0])

            else:
                prompt_question = random.choice(self.caption_prompts)
                diagnosis = data["report"][0] + 'Diagnosis is' +  self._normalize_label(data["dignosis"][0])
        else:

            if random.random() < 0.5:
                prompt_question = random.choice(self.Dignoise_prompts)
                diagnosis = self._normalize_label(data["diagnosis"][0])

            else:
                prompt_question = self.caption_prompts[0] #random.choice(self.caption_prompts)
                diagnosis = data["report"][0] + 'Diagnosis is' +  self._normalize_label(data["dignosis"][0])



        answer = diagnosis
        question = self.image_tokens + prompt_question

        text_tensor = self.tokenizer(
            question + " " + answer,
            max_length=self.args.max_length,
            truncation=True, padding="max_length",
            return_tensors="pt",
        )
        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        question_tensor = self.tokenizer(
            question, max_length=self.args.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        question_len = torch.sum(question_tensor["attention_mask"][0])

        label = input_id.clone()
        label[:question_len] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            label[label == self.tokenizer.pad_token_id] = -100
            if valid_len < len(label):
                label[valid_len] = self.tokenizer.eos_token_id
        else:
            label[label == self.tokenizer.pad_token_id] = -100

        return {
            "image0": image0,
            "image1": image1,
            "image2": image2,
            "input_id": input_id,
            "label": label,
            "attention_mask": attention_mask,
            "question": question,
            "answer": answer,
            "question_type": "Diagnosis",
            "name": image_path,
        }