from contextlib import nullcontext
from typing import Literal

import torch
import typer
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from scripts.common import opt


def main(
    # Model Loading
    source: str = opt("sft", "Source of the model: sft|mid|rl"),
    model_tag: str | None = opt(None, "Model tag to load"),
    step: int | None = opt(None, "Step to load"),
    # Runtime
    device_type: Literal["cuda", "cpu", "mps", ""] = opt("", "Device type for evaluation. (empty => autodetect)"),
    dtype: Literal["float32", "bfloat16"] = opt("bfloat16", "Model dtype"),
    # Input
    prompt: str = opt("", "Prompt the model, get a single response back"),
    # Generation / Sampling
    max_tokens: int = opt(256, "Max. num. of tokens to generate"),
    temperature: float = opt(0.6, "Temperature for generation"),
    top_k: int = opt(50, "Top-k sampling parameter"),
):
    """Chat with the model.
    Intended to be run single GPU only atm:
        python -m scripts.chat_cli -i mid
    """

    # Init the model and tokenizer
    device_type = autodetect_device_type() if device_type == "" else device_type
    *_, device = compute_init(device_type)
    ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.autocast(
            device_type=device_type,
            dtype=ptdtype,
        )
        if device_type == "cuda"
        else nullcontext()
    )
    model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)

    # Special tokens for the chat state machine
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = (
        tokenizer.encode_special("<|user_start|>"),
        tokenizer.encode_special("<|user_end|>"),
    )
    assistant_start, assistant_end = (
        tokenizer.encode_special("<|assistant_start|>"),
        tokenizer.encode_special("<|assistant_end|>"),
    )

    # Create Engine for efficient generation
    engine = Engine(model, tokenizer)

    print("\nNanoChat Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens = [bos]
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    while True:
        if prompt:
            # Get the prompt from the launch command
            user_input = prompt
        else:
            # Get the prompt interactively from the console
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        # Handle special commands
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        # Add User message to the conversation
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)

        # Kick off the assistant
        conversation_tokens.append(assistant_start)
        response_tokens = []
        print("\nAssistant: ", end="", flush=True)
        with autocast_ctx:
            for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                token = token_column[0]  # pop the batch dimension (num_samples=1)
                response_tokens.append(token)
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)
        print()
        # we have to ensure that the assistant end token is the last token
        # so even if generation ends due to max tokens, we have to append it to the end
        if response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        # In the prompt mode, we only want a single response and exit
        if prompt:
            break


if __name__ == "__main__":
    typer.run(main)
