import time
from typing import Annotated, Iterable

import torch
import typer

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.report import get_report
from nanochat.tokenizer import RustBPETokenizer
from scripts.common import config_from_context


def main(
    ctx: typer.Context,
    max_chars: Annotated[
        int, typer.Option(help="Maximum characters to train on (default: 10B)")
    ] = 10_000_000_000,
    doc_cap: Annotated[
        int, typer.Option(help="Maximum characters per document (default: 10,000)")
    ] = 10_000,
    vocab_size: Annotated[
        int, typer.Option(help="Vocabulary size (default: 65536 = 2^16)")
    ] = 65536,
):
    """Train a GPT-4-type BPE tokenizer using the HuggingFace Tokenizers library."""

    print(f"max_chars: {max_chars:,}")
    print(f"doc_cap: {doc_cap:,}")
    print(f"vocab_size: {vocab_size:,}")

    # -----------------------------------------------------------------------------
    # Text iterator

    def text_iterator() -> Iterable[str]:
        """
        1) Flatten the batches into a single iterator
        2) Crop every document to args.doc_cap characters
        3) Break when we've seen args.max_chars characters
        """
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > doc_cap:
                    doc_text = doc_text[:doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > max_chars:
                    return

    text_iter = text_iterator()

    # -----------------------------------------------------------------------------
    # Train the tokenizer
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    # -----------------------------------------------------------------------------
    # Save the tokenizer to disk
    base_dir = get_base_dir()
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer.save(tokenizer_dir)

    # -----------------------------------------------------------------------------
    # Quick inline sanity check
    test_text = """Hello world! This is a test.
    Numbers: 123, 4567, 89
    Contractions: I'm, you're, it's
    Special chars: @#$%^&*()
    Unicode: 你好世界 🌍"""
    encoded: list[int] = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

    # -----------------------------------------------------------------------------
    # One more thing: we wish to cache a mapping from token id to number of bytes of that token
    # for efficient evaluation of bits per byte. Unlike the typical mean loss, this
    # allows us to report a loss that is invariant to the vocab size of the tokenizer.
    # The bits per byte on the validation set is then one of the primary metrics we care about.
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_str in token_strings:
        if token_str in special_set:
            token_bytes.append(0)  # special characters are not counted
        else:
            id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = tokenizer_dir / "token_bytes.pt"
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # Log to report
    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    user_config, _ = config_from_context(ctx)
    get_report().log(
        section="Tokenizer training",
        data=[
            user_config,
            {"train_time": train_time},
            {"num_special_tokens": len(special_set)},
            {
                "token_bytes_min": int(token_bytes_nonzero.min().item()),
                "token_bytes_max": int(token_bytes_nonzero.max().item()),
                "token_bytes_mean": token_bytes_nonzero.mean().item(),
                "token_bytes_std": token_bytes_nonzero.std().item(),
            },
        ],
    )


if __name__ == "__main__":
    typer.run(main)
