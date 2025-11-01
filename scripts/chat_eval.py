from contextlib import nullcontext
from functools import partial
from typing import Annotated, Literal

import torch
import torch.distributed as dist
import typer

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_dist_info,
    print0,
)
from nanochat.engine import Engine
from nanochat.gpt import GPT
from nanochat.tokenizer import RustBPETokenizer
from scripts.common import config_from_context
from tasks.arc import ARC
from tasks.common import Task
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)


def run_generative_eval(
    task_object: Task,
    tokenizer: RustBPETokenizer,
    model: GPT,
    engine: Engine,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_problems: int | None = None,
):
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(
            f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")

    # Return the accuracy
    return num_passed / total


# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.


def ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def run_categorical_eval(
    task_object: Task,
    tokenizer: RustBPETokenizer,
    model: GPT,
    batch_size: int,
    max_problems: int | None = None,
) -> float:
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()  # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {}  # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [
            tokenizer.render_for_completion(conversation) for conversation in conversations
        ]  # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [
            len(ids) - 1 for ids in prompt_ids
        ]  # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters: list[str] = conversation["letters"]
            letter_ids: list[int] = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id: int = focus_logits.argmax(dim=-1).item()
            predicted_letter: str = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------


def run_chat_eval(
    task_name: str,
    model: GPT,
    tokenizer: RustBPETokenizer,
    engine: Engine,
    batch_size: int = 1,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    max_problems: int | None = None,
) -> float:
    # Create the evaluation object
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object: Task = task_module()
    # Run the evaluation
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object, tokenizer, model, batch_size, max_problems=max_problems
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


# -----------------------------------------------------------------------------
def main(
    ctx: typer.Context,
    source: Annotated[str, typer.Option("-i", "--source", help="Source of the model: sft|mid|rl")],
    task_name: Annotated[
        str | None,
        typer.Option(
            "-a",
            "--task-name",
            help="Task name. Default = all tasks. Use | to split multiple tasks.",
        ),
    ] = None,
    dtype: Annotated[Literal["float32", "bfloat16"], typer.Option("-d", "--dtype")] = "bfloat16",
    temperature: Annotated[float, typer.Option("-t", "--temperature")] = 0.0,
    max_new_tokens: Annotated[int, typer.Option("-m", "--max-new-tokens")] = 512,
    num_samples: Annotated[int, typer.Option("-n", "--num-samples")] = 1,
    top_k: Annotated[int, typer.Option("-k", "--top-k")] = 50,
    batch_size: Annotated[
        int, typer.Option("-b", "--batch-size", help="Batch size for categorical evaluation")
    ] = 8,
    model_tag: Annotated[
        str | None, typer.Option("-g", "--model-tag", help="Model tag to load")
    ] = None,
    step: Annotated[int | None, typer.Option("-s", "--step", help="Step to load")] = None,
    max_problems: Annotated[
        int | None, typer.Option("-x", "--max-problems", help="Max problems to evaluate")
    ] = None,
    device_type: Annotated[
        Literal["cuda", "cpu", "mps", ""],
        typer.Option(
            "--device-type", help="Device type for evaluation: cuda|cpu|mps. empty => autodetect"
        ),
    ] = "",
):
    """Evaluate the Chat model.

    All the generic code lives here, and all the evaluation-specific
    code lives in nanochat directory and is imported from here.

    Example runs:
        python -m scripts.chat_eval -a ARC-Easy
        torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
    """

    device_type = autodetect_device_type() if device_type == "" else device_type
    *_, device = compute_init(device_type)
    ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    model, tokenizer, meta = load_model(
        source, device, phase="eval", model_tag=model_tag, step=step
    )
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    baseline_accuracies = {
        "ARC-Easy": 0.25,  # multiple choice 1 of 4 => 25%
        "ARC-Challenge": 0.25,  # multiple choice 1 of 4 => 25%
        "MMLU": 0.25,  # multiple choice 1 of 4 => 25%
        "GSM8K": 0.0,  # open-ended => 0%
        "HumanEval": 0.0,  # open-ended => 0%
    }
    task_names = all_tasks if task_name is None else task_name.split("|")

    # Run all the task evaluations sequentially
    results: dict[str, float] = {}
    for task_name in task_names:
        with autocast_ctx:
            acc = run_chat_eval(
                task_name,
                model,
                tokenizer,
                engine,
                batch_size=batch_size,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                max_problems=max_problems,
            )
            results[task_name] = acc
            print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # Log to report
    from nanochat.report import get_report

    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}

    user_config, _ = config_from_context(ctx)
    get_report().log(
        section=f"Chat evaluation {source}",
        data=[
            user_config,
            results,
            chatcore_metric_dict,
        ],
    )

    compute_cleanup()


if __name__ == "__main__":
    typer.run(main)
