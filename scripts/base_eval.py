from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import torch
import typer
import yaml

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    FilePath,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    download_file_with_lock,
    get_base_dir,
    print0,
)
from nanochat.core_eval import evaluate_task
from nanochat.report import get_report
from nanochat.tokenizer import HuggingFaceTokenizer

if TYPE_CHECKING:
    from nanochat.gpt import GPT
    from nanochat.tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model


class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""

    def __init__(self, model, max_seq_len: int | None = None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids: list[int]) -> torch.Tensor:
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits


def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def read_eval_metadata(path: FilePath) -> dict[str, float]:
    eval_metadata: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        idx = header.index("Random baseline")
        for row in reader:
            task_name = row[0]
            random_baseline = float(row[idx])
            eval_metadata[task_name] = random_baseline
    return eval_metadata


# -----------------------------------------------------------------------------
# nanoChat specific function dealing with I/O etc.
# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path: str | Path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = base_dir / "eval_bundle"
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = Path(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_model(
    model: GPT,
    tokenizer: RustBPETokenizer,
    device: torch.device,
    max_per_task: int = -1,
) -> dict[str, Any]:
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    TODO: clean up this function, delete the need for all the files
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = base_dir / "eval_bundle"
    # Download the eval bundle to disk (and unzip if needed)
    if not eval_bundle_dir.exists():
        download_path = download_file_with_lock(
            EVAL_BUNDLE_URL,
            "eval_bundle.zip",
        )
        place_eval_bundle(download_path)
    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    eval_meta_data_path = eval_bundle_dir / "eval_meta_data.csv"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]
    eval_metadata = read_eval_metadata(eval_meta_data_path)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(
            f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ",
            end="",
        )

        # Load data for this task
        data_path = data_base_path / task["dataset_uri"]
        with open(data_path, "r", encoding='utf-8') as f:
            data: list[dict[str, str]] = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = eval_metadata[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s"
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {"results": results, "centered_results": centered_results, "core_metric": core_metric}
    return out


# -----------------------------------------------------------------------------
def main(
    hf_path: Annotated[str | None, typer.Option(help="HF model path to evaluate")] = None,
    max_per_task: Annotated[
        int, typer.Option(help="Max. num. of examples per task to eval (-1 = disable)")
    ] = -1,
):
    """Evaluate the CORE metric for a given model.

    Run on a single GPU:
    python -m scripts.base_eval

    Run with torchrun on e.g. 8 GPUs:
    torchrun --nproc_per_node=8 -m scripts.base_eval.py

    The script will print the CORE metric to the console.
    """
    # distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)  # type: ignore
        if device_type == "cuda"
        else nullcontext()
    )

    # Load model and tokenizer from command line or from file system
    if hf_path is not None:
        # atm assume that if a path is given, it's a huggingface model path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path  # just for logging
        model_slug = hf_path.replace("/", "-")  # for the output csv file
    else:
        # load a local model from the file system
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (step {meta['step']})"  # just for logging
        model_slug = f"base_model_{meta['step']:06d}"  # for the output csv file

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(
            model,  # type: ignore
            tokenizer,  # type: ignore
            device,
            max_per_task=max_per_task,
        )

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = base_dir / "base_eval" / f"{model_slug}.csv"
        output_csv_path.parent.mkdir(exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, "w", encoding='utf-8', newline="") as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("=" * 80)
        print0(f"Model: {model_name}")
        print0("=" * 80)
        with open(output_csv_path, "r") as f:
            print0(f.read())

    get_report().log(
        section="Base model evaluation",
        data=[
            {
                "Model": model_name,
                "CORE metric": core_metric,
            },
            centered_results,  # the full table
        ],
    )

    compute_cleanup()


if __name__ == "__main__":
    typer.run(main)
