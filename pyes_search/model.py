"""Hugging Face plumbing: load the frozen model, and encode tasks into layouts."""

from __future__ import annotations

from collections.abc import Callable

import torch

from pyes_search.readout import YES_NO, Readout
from pyes_search.layout import GLOBAL_VERIFIER, IMAGE_PLACEHOLDER, TaskLayout, build_layout
from pyes_search.task import FillTask

#: Images are downscaled to at most this many pixels (aspect preserved) before
#: the processor sees them.
MAX_PIXELS = 448 * 448


#: ``tf32``: fp32 weights and activations with TF32 tensor-core matmuls;
#: ``bf16``: bfloat16 weights (used for every number in the post).
PRECISIONS = {"tf32": torch.float32, "bf16": torch.bfloat16}


def set_precision(precision: str) -> torch.dtype:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return PRECISIONS[precision]


def load(
    model_id: str,
    revision: str | None,
    *,
    device: str = "cuda",
    attn: str = "sdpa",
    precision: str = "bf16",
):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = set_precision(precision)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation=attn
    ).to(device)
    model.requires_grad_(False)  # frozen: search only reads the model
    return model, processor


def _load_image(path: str):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(28, int(width * scale)), max(28, int(height * scale))))
    return image


def encoders(processor) -> tuple[Callable, Callable, int]:
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad token")
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode_prefix(task: FillTask, text: str):
        if text.count(IMAGE_PLACEHOLDER) != len(task.images):
            raise ValueError(f"{task.task_id}: image placeholder count mismatch")
        images = [_load_image(path) for path in task.images]
        out = processor(text=[text], images=images, return_tensors="pt", add_special_tokens=False)
        ids = out["input_ids"][0].tolist()
        mm = [1 if token == image_token_id else 0 for token in ids]
        vision = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}
        return ids, mm, vision

    return encode, encode_prefix, pad_id


def make_layout(
    processor,
    max_tokens: int | None = None,
    verifier: str = GLOBAL_VERIFIER,
    readout: Readout = YES_NO,
) -> Callable[[FillTask], TaskLayout]:
    encode, encode_prefix, pad_id = encoders(processor)

    def layout(task: FillTask) -> TaskLayout:
        return build_layout(
            task,
            encode=encode,
            pad_id=pad_id,
            max_tokens=max_tokens,
            verifier=verifier,
            readout=readout,
            encode_prefix=encode_prefix,
        )

    return layout
