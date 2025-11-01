#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/cpu_demo_run.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/
set -e
unset CONDA_PREFIX

export _TYPER_STANDARD_TRACEBACK=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# use uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

# optional wandb
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# install rust and rustbpe
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# wipe the report
python -m nanochat.report reset

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    # train tokenizer on ~1B characters
    python -m nanochat.dataset -n 4
    python -m scripts.tok_train --max-chars=1000000000
    python -m scripts.tok_eval
fi

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=4 \
    --max-seq-len=1024 \
    --device-batch-size=1 \
    --total-batch-size=1024 \
    --eval-every=50 \
    --eval-tokens=4096 \
    --core-metric-every=50 \
    --core-metric-max-per-task=12 \
    --sample-every=50 \
    --num-iterations=50
python -m scripts.base_loss --device-batch-size=1 --split-tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
python -m scripts.mid_train \
    --max-seq-len=1024 \
    --device-batch-size=1 \
    --eval-every=50 \
    --eval-tokens=4096 \
    --total-batch-size=1024 \
    --num-iterations=100
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device-batch-size=1 \
    --target-examples-per-step=4 \
    --num-iterations=100 \
    --eval-steps=4 \
    --eval-metrics-max-problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate