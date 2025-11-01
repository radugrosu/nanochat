#!/bin/bash

# WANDB_RUN=testrun screen -L -Logfile testrun.log -S testrun bash testrun.sh
export _TYPER_STANDARD_TRACEBACK=1
unset $CONDA_PREFIX
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f ".env" ]; then
            echo "Sourcing .env file..."
            source .env
            echo "WANDB_API_KEY after sourcing: ${WANDB_API_KEY:0:10}..."  # Show first 10 chars
    else
        echo "No .env file found"
    fi
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set"
    exit 1
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# each data shard is ~250M chars
# each shard is ~100MB of text (compressed)
python -m nanochat.dataset -n 1
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 24 is the right number here
python -m nanochat.dataset -n 24 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters. d2 should be around ~60M
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 2 = 1.1B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 1.1B * 4.8 ~= 5.4B chars.
# At 250M chars/shard, this is 5.4B / 250M ~= 22 shards needed for pretraining.
# Round up to 24 for safety. At ~100MB/shard, this downloads ~2.4GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d2 model
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=2 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=1 -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
