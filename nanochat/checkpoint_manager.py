"""Utilities for saving and loading model/optim/state checkpoints."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from nanochat.common import FilePath, get_base_dir, setup_default_logging
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer

if TYPE_CHECKING:
    from nanochat.gpt import GPT, GPTConfig

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)


def log0(message: str):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)


def save_checkpoint(
    checkpoint_dir: FilePath,
    step: int,
    model_data: dict[str, Any],
    optimizer_data: dict[str, Any] | list[Any] | None,
    meta_data: dict[str, Any],
):
    assert int(os.environ.get("RANK", 0)) == 0  # prevent footguns for now
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    # Save the model state (parameters)
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    torch.save(model_data, model_path)
    log0(f"Saved model file to: {model_path}")
    # Save the optimizer state (useful for SFT or any other fine-tuning)
    if optimizer_data is not None:
        optimizer_path = checkpoint_dir / f"optim_{step:06d}.pt"
        torch.save(optimizer_data, optimizer_path)
        log0(f"Saved optimizer file to: {optimizer_path}")
    # Save the metadata dict as json
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")


def load_checkpoint(
    checkpoint_dir: FilePath,
    step: int,
    device: torch.device | str,
    load_optimizer: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | list[Any] | None, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    # Load the model state
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = checkpoint_dir / f"optim_{step:06d}.pt"
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(
    checkpoint_dir: FilePath,
    step: int,
    device: torch.device | str,
    phase: Literal["train", "eval"],
) -> tuple[GPT, RustBPETokenizer, dict[str, Any]]:
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_buffers()
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir: FilePath) -> str:
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in Path(checkpoint_dir).iterdir() if f.is_dir()]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates: list[tuple[int, Path]] = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag.stem)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidate = candidates[0][1]
    else:
        # 2) if that failed, take the most recently updated model:
        candidate = max(model_tags, key=lambda x: x.stat().st_mtime)
    return candidate.name


def find_last_step(checkpoint_dir: FilePath) -> int:
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [*Path(checkpoint_dir).glob("model_*.pt")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(int(f.stem.rsplit("_")[0]) for f in checkpoint_files)


# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure


def load_model_from_dir(
    checkpoints_dir: FilePath,
    device: torch.device | str,
    phase: Literal["train", "eval"],
    model_tag: str | None = None,
    step: int | None = None,
):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = Path(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data


def load_model(
    source: str, *args: Any, **kwargs: Any
) -> tuple[GPT, RustBPETokenizer, dict[str, Any]]:
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = base_dir / model_dir
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
