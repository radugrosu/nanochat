"""The base/pretraining dataset is a set of parquet files.

This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import typer
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Any

import pyarrow.parquet as pq
import requests

from nanochat.common import FilePath, get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet
index_to_filename = "shard_{:05d}.parquet"  # format of the filenames
base_dir = get_base_dir()
DATA_DIR = Path(base_dir, "base_data")
DATA_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def list_parquet_files(data_dir: FilePath | None = None) -> list[Path]:
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    parquet_files = sorted(f for f in data_dir.glob("*.parquet") if not f.name.endswith(".tmp"))
    parquet_paths = [data_dir / f for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split: str, start: int = 0, step: int = 1) -> Iterator[list[str]]:
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):  # type: ignore
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


# -----------------------------------------------------------------------------
def download_single_file(index: int) -> bool:
    """Downloads a single file index, with some backoff"""

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename.format(index)
    filepath = Path(DATA_DIR, filename)
    if filepath.exists():
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            temp_path.rename(filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath.with_suffix(".tmp"), filepath]:
                path.unlink(missing_ok=True)
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def opt(default: Any, msg: str):
    return typer.Option(default, help=msg)


def main(
    num_files: int = opt(-1, "Number of shards to download (default: -1, -1 = all available)"),
    num_workers: int = opt(4, "Number of parallel download workers"),
):
    """Download FineWeb-Edu 100BT dataset shards")"""
    # Logic adapted from original script
    # Note: Ensure MAX_SHARD, DATA_DIR, and download_single_file are defined globally

    num = MAX_SHARD + 1 if num_files == -1 else min(num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))

    print(f"Downloading {len(ids_to_download)} shards using {num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()

    with Pool(processes=num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")


if __name__ == "__main__":
    typer.run(main)
