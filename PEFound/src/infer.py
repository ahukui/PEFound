"""Generate a diagnosis or report for one T1/T2/FLAIR examination."""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from transformers import AutoTokenizer

from PEFound.src.dataset.prompt_templates import Caption_templates, Diagnose_templates
from PEFound.src.model.language_model import LamedPhi3ForCausalLM


def load_mri(case_dir, target_shape):
    """Use the original PEDataset resizing and 1–99% normalization."""
    volumes = []
    for filename in ("T1.nii.gz", "T2.nii.gz", "T2_Flair.nii.gz"):
        path = Path(case_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing MRI file: {path}")
        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image)
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError(f"Expected a finite 3D volume: {path}")
        factors = [t / s for t, s in zip(target_shape, volume.shape)]
        volume = zoom(volume, factors, order=1).astype(np.float32)
        low = np.percentile(volume, 1)
        high = np.percentile(volume, 99)
        volume = np.clip(volume, low, high)
        volume = ((volume - low) / (high - low + 1e-6)).astype(np.float32)
        volumes.append(volume)
    return np.stack(volumes, axis=0)


def encode_prompt(tokenizer, prompt, num_image_tokens):
    """Match the original model's [prefix][image tokens][instruction] layout."""
    token = "<im_patch>"
    if token not in tokenizer.get_vocab():
        raise ValueError("The saved tokenizer does not contain <im_patch>. Use the merged model's tokenizer.")
    token_id = tokenizer.convert_tokens_to_ids(token)
    ids = tokenizer(token * num_image_tokens + prompt,
                    add_special_tokens=True)["input_ids"]
    # The original architecture replaces positions 1 through num_image_tokens.
    # Some Phi tokenizers do not add a BOS token automatically.
    if ids and ids[0] == token_id:
        if tokenizer.bos_token_id is None:
            raise ValueError("The original model requires one prefix token; this tokenizer has no BOS token.")
        ids = [tokenizer.bos_token_id] + ids
    if (ids.count(token_id) != num_image_tokens
            or ids[1:1 + num_image_tokens] != [token_id] * num_image_tokens):
        raise ValueError("Image tokens must occupy positions 1 through the projector token count.")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Merged Hugging Face model directory with configuration, weights and tokenizer")
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--task", choices=["report", "diagnosis"], default="report")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = LamedPhi3ForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device).eval()
    model.config.use_cache = True
    if model.get_vision_tower() is None:
        raise ValueError("Load the merged multimodal model, not the base Phi-3 model.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    volume = load_mri(args.case_dir, tuple(model.config.image_size))
    images = torch.from_numpy(volume).unsqueeze(0).to(device=args.device, dtype=dtype)
    prompt = args.prompt or (Diagnose_templates[0] if args.task == "diagnosis" else Caption_templates[0])
    ids = encode_prompt(tokenizer, prompt, model.get_model().mm_projector.proj_out_num)
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)

    with torch.inference_mode():
        generated = model.generate(
            images=images,
            inputs=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # The original generate() supplies inputs_embeds, so decode the returned
    # generated sequence directly rather than slicing off len(input_ids).
    text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
