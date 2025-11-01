from contextlib import nullcontext
from typing import Annotated, Literal

import torch
import typer

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine


def main(
    source: Annotated[str, typer.Option(help="Source of the model: sft|mid|rl")] = "sft",
    model_tag: Annotated[str | None, typer.Option(help="Model tag to load")] = None,
    step: Annotated[int | None, typer.Option(help="Step to load")] = None,
    prompt: Annotated[str, typer.Option(help="Prompt the model, get a single response back")] = "",
    max_tokens: Annotated[int, typer.Option(help="Max. num. of tokens to generate")] = 256,
    temperature: Annotated[float, typer.Option(help="Temperature for generation")] = 0.6,
    top_k: Annotated[int, typer.Option(help="Top-k sampling parameter")] = 50,
    device_type: Annotated[
        Literal["cuda", "cpu", "mps", ""],
        typer.Option(help="Device type for evaluation. (empty => autodetect)"),
    ] = "",
    dtype: Annotated[Literal["float32", "bfloat16"], typer.Option(help="Model dtype")] = "bfloat16",
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
    model, tokenizer, meta = load_model(
        source, device, phase="eval", model_tag=model_tag, step=step
    )

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
            for token_column, token_masks in engine.generate(
                conversation_tokens, **generate_kwargs
            ):
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
