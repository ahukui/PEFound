import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import SimpleITK as sitk
from pathlib import Path
import random

rng = random.Random(42)


from transforms import (
    Compose,
    RandScaleIntensity,
    RandShiftIntensity,
    ToTensor,
)


class PEDataset(Dataset):
    """
    Dataset for image–report contrastive pretraining.

    Each sample returns:
        - 3 co-registered MRI modalities (T1, T2, T2_Flair), each shape (1, 32, 256, 256)
        - tokenized radiology report (input_ids, attention_mask)
        - the raw report string and case name (for logging / debugging)
    """

    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        if mode == "train":
            json_paths = [args.train_data_path]
        elif mode in ("val", "validation"):
            json_paths = [args.val_data_path]
        elif mode == "test":
            json_paths = [args.test_data_path]
        else:
            raise ValueError(f"Unknown mode: {mode}")


        self.data_list, seen = [], set()
        for jp in json_paths:
            with open(jp, "r") as f:
                items = json.load(f)

            for item in items:
                key = item["images"][0] if item.get("images") else id(item)
                if key in seen:
                    continue
                seen.add(key)
                self.data_list.append(item)

        print(f"[PEDataset] mode={mode} | n={len(self.data_list)} | from {len(json_paths)} json file(s)")

        # Light intensity augmentation only — no spatial flips/rotations,
        # because the 3 modalities must stay co-registered.
        if mode == "train":
            self.transform = Compose([
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
                ToTensor(dtype=torch.float),
            ])
        else:
            self.transform = Compose([ToTensor(dtype=torch.float)])

    def __len__(self):
        return len(self.data_list)

    # ---------------------------------------------------------------------
    # Image preprocessing
    # ---------------------------------------------------------------------
    def resample_numpy(self, volume, target_shape=(32, 256, 256), order=1):
        """Resample a 3D array (D, H, W) to target_shape. order=1 linear, 0 nearest."""
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
        return zoom(volume, zoom_factors, order=order)

    def normalize_1_99(self, volume, eps=1e-6):
        """Robust min–max normalization to [0, 1] using 1st/99th percentiles."""
        v_min = np.percentile(volume, 1)
        v_max = np.percentile(volume, 99)
        volume = np.clip(volume, v_min, v_max)
        volume = (volume - v_min) / (v_max - v_min + eps)
        return volume.astype(np.float32)

    def load_data(self, path):
        image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(image)             # (D, H, W)
        image = self.resample_numpy(image, target_shape=(32, 256, 256), order=1)
        image = self.normalize_1_99(image)
        image = np.expand_dims(image, axis=0)             # (1, D, H, W)
        image = self.transform(image)
        return image

    # ---------------------------------------------------------------------
    # Sample
    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]
            image_path = data["images"][0]
            image_abs_path = os.path.join(self.data_root, image_path)

            image_list = []
            for fname in ["T1.nii.gz", "T2.nii.gz", "T2_Flair.nii.gz"]:
                image_list.append(self.load_data(os.path.join(image_abs_path, fname)))

            if random.random() < 0.5:
                report = data["dignosis"][0] 
            else:
                report = data["report"][0]

            if not isinstance(report, str) or len(report.strip()) == 0:
                raise ValueError(f"Empty report for {image_path}")

        except Exception as e:
            # Failed sample (missing file, unreadable nii, empty report, …)
            # → resample a different index. Avoid infinite recursion in pathological
            #   datasets by capping retries.
            if getattr(self, "_retry_depth", 0) > 10:
                raise RuntimeError(f"Too many failed samples around idx={idx}") from e
            self._retry_depth = getattr(self, "_retry_depth", 0) + 1
            try:
                return self.__getitem__(random.randint(0, len(self.data_list) - 1))
            finally:
                self._retry_depth -= 1

        # ---- Tokenize the report directly (no prompt, no <im_patch>) ----
        text_tensor = self.tokenizer(
            report,
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        return {
            "image0": image_list[0],
            "image1": image_list[1],
            "image2": image_list[2],
            "input_id": input_id,
            "attention_mask": attention_mask,
            "report": report,
            "name": image_path,
        }