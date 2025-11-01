import os
from typing import Annotated, Iterable, Literal

from scripts.common import config_from_context

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from contextlib import nullcontext

import torch
import torch.distributed as dist
import typer
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from nanochat.report import get_report
from scripts.chat_eval import run_chat_eval
from tasks.arc import ARC
from tasks.common import Task, TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee


def main(
    ctx: typer.Context,
    run: Annotated[
        str,
        typer.Option(help='wandb run name default ("dummy" is special - we won\'t log to wandb)'),
    ] = "dummy",
    source: Annotated[
        Literal["base", "mid"],
        typer.Option(
            help="Which checkpoint to load the model from (base model or midtrained model)"
        ),
    ] = "mid",
    model_tag: Annotated[
        str | None,
        typer.Option(help="Model tag to load the model from (base model or midtrained model)"),
    ] = None,
    step: Annotated[
        int | None,
        typer.Option(help="Step to load the model from (base model or midtrained model)"),
    ] = None,
    device_type: Annotated[
        Literal["cuda", "cpu", "mps", ""], typer.Option(help="cuda|cpu|mps (empty => autodetect)")
    ] = "",
    dtype: Annotated[str, typer.Option(help="Data type for model weights")] = "bfloat16",
    device_batch_size: Annotated[int, typer.Option(help="Max to avoid OOM")] = 4,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs to train")] = 1,
    num_iterations: Annotated[
        int,
        typer.Option(
            help="Override number of iterations (-1 = disable, use num_epochs to derive it)"
        ),
    ] = -1,
    target_examples_per_step: Annotated[
        int, typer.Option(help="Target number of examples per training step")
    ] = 32,
    unembedding_lr: Annotated[
        float, typer.Option(help="Learning rate for unembedding layer")
    ] = 0.004,
    embedding_lr: Annotated[float, typer.Option(help="Learning rate for embedding layer")] = 0.2,
    matrix_lr: Annotated[float, typer.Option(help="Learning rate for matrix parameters")] = 0.02,
    weight_decay: Annotated[float, typer.Option(help="Weight decay for optimizer")] = 0.0,
    init_lr_frac: Annotated[float, typer.Option(help="Initial learning rate fraction")] = 0.02,
    eval_every: Annotated[int, typer.Option(help="Evaluate model every N steps")] = 100,
    eval_steps: Annotated[int, typer.Option(help="Number of evaluation steps")] = 100,
    eval_metrics_every: Annotated[int, typer.Option(help="Evaluate metrics every N steps")] = 200,
    eval_metrics_max_problems: Annotated[
        int, typer.Option(help="Maximum number of problems for metrics evaluation")
    ] = 1024,
):
    """Finetune a base model to be a chat model.

    Run on one GPU e.g. for debugging:
        python -m scripts.chat_sft

    Or torchrun for training:
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
    """
    user_config, _ = config_from_context(ctx)
    # -----------------------------------------------------------------------------

    # Compute init
    device_type = autodetect_device_type() if device_type == "" else device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # wandb logging init
    use_dummy_wandb = run == "dummy" or not master_process
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)
    )

    # Load the model and tokenizer
    model, tokenizer, meta = load_model(
        source, device, phase="train", model_tag=model_tag, step=step
    )
    engine = Engine(model, tokenizer)  # will be used for inline model evaluation only

    # -----------------------------------------------------------------------------
    # Task data mixture we'll train on
    identity_conversations_filepath = get_base_dir() / "identity_conversations.jsonl"
    train_ds = TaskMixture(
        [
            ARC(subset="ARC-Easy", split="train"),  # 2.3K rows
            ARC(subset="ARC-Challenge", split="train"),  # 1.1K rows
            GSM8K(subset="main", split="train"),  # 8K rows
            SmolTalk(split="train", stop=10_000),  # 10K rows of smoltalk
            CustomJSON(
                filepath=identity_conversations_filepath
            ),  # 1K rows of synthetic identity conversations
            SimpleSpelling(
                size=300, split="train"
            ),  # 300 rows of Simple Spelling (e.g. spell the word 'apple')
            SpellingBee(
                size=300, split="train"
            ),  # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
        ]
    )  # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
    val_ds = SmolTalk(
        split="test"
    )  # general conversations, 24K rows (though we don't actually use all of it)

    # -----------------------------------------------------------------------------
    # DataLoader

    def sft_data_generator(
        dataset: TaskMixture | Task, batch_size: int
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        pad_token_id = tokenizer.encode_special(
            "<|assistant_end|>"
        )  # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss

        # prepares a list of tokenized conversations into a batch and yields
        def collate_and_yield(
            batch: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nrows = len(batch)
            ncols = max(len(ids) for ids, _ in batch) - 1  # seq of n creates inputs/targets of n-1
            inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
            targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index
            for i, (ids, mask) in enumerate(batch):
                n = len(ids)
                ids_tensor = torch.tensor(ids, dtype=torch.long)
                inputs[i, : n - 1] = ids_tensor[:-1]
                # recall -1 is the ignore index, so mask out targets where mask is 0
                row_targets = ids_tensor[1:]
                # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
                mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
                row_targets[mask_tensor == 0] = -1  # mask out targets where mask is 0
                targets[i, : n - 1] = row_targets
            inputs = inputs.to(device)  # move to device
            targets = targets.to(device)
            return inputs, targets

        # iterates over the dataset in epochs, tokenizes
        batch = []
        while True:
            for i in range(ddp_rank, len(dataset), ddp_world_size):
                doc = dataset[i]
                ids, mask = tokenizer.render_conversation(doc)
                batch.append((ids, mask))
                if len(batch) == batch_size:
                    yield collate_and_yield(batch)
                    batch = []

    examples_per_step = device_batch_size * ddp_world_size
    print0(f"Target examples per step: {target_examples_per_step}")
    print0(f"Device batch size: {device_batch_size}")
    print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
    assert target_examples_per_step % examples_per_step == 0, (
        "Target examples per step must be divisible by examples per step"
    )
    grad_accum_steps = target_examples_per_step // examples_per_step
    assert grad_accum_steps > 0
    print0(f"=> Setting grad accum steps: {grad_accum_steps}")

    if num_iterations == -1:
        # derive num_iterations from num_epochs and the size of the dataset
        assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
        num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs

    train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)

    def build_val_loader():
        return sft_data_generator(val_ds, batch_size=device_batch_size)

    # -----------------------------------------------------------------------------
    # Initialize the Optimizer

    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )
    # Set the initial learning rate as a fraction of the base learning rate
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group[
                "lr"
            ]  # save the initial learning so we can decay easily later

    # -----------------------------------------------------------------------------
    # Training loop

    # Learning rate scheduler
    def get_lr_multiplier(it: int) -> float:
        lrm = 1.0 - it / num_iterations
        return lrm

    # Go!
    step = 0
    train_iter = iter(train_loader)
    for step in range(num_iterations):
        last_step = step == num_iterations - 1

        # evaluate the validation loss
        if last_step or step % eval_every == 0:
            model.eval()
            val_iter = iter(build_val_loader())
            losses = []
            for _ in range(eval_steps):
                val_inputs, val_targets = next(val_iter)
                with torch.no_grad(), autocast_ctx:
                    loss = model(val_inputs, val_targets)
                losses.append(loss)
            val_loss = torch.stack(losses).mean()  # average over eval_steps
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)  # average over ranks
            val_loss = val_loss.item()
            print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
            wandb_run.log(
                {
                    "step": step,
                    "val_loss": val_loss,
                }
            )
            model.train()

        # evaluate accuracy of the multiple choice tasks (which are quick to run)
        if last_step or (step > 0 and step % eval_metrics_every == 0):
            model.eval()
            metrics = {}
            with torch.no_grad(), autocast_ctx:
                # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
                metrics["mmlu_acc"] = run_chat_eval(
                    "MMLU",
                    model,
                    tokenizer,
                    engine,
                    batch_size=device_batch_size * 2,
                    max_problems=eval_metrics_max_problems,
                )
                metrics["arc_easy_acc"] = run_chat_eval(
                    "ARC-Easy",
                    model,
                    tokenizer,
                    engine,
                    batch_size=device_batch_size * 2,
                    max_problems=eval_metrics_max_problems,
                )
            metrics_str = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
            print0(f"Step {step:05d} | {metrics_str}")
            wandb_run.log(
                {
                    "step": step,
                    **metrics,
                }
            )
            model.train()

        if last_step:
            break

        # evaluate the gradient
        num_tokens = torch.tensor(
            0, device=device
        )  # the number of "active" tokens of supervision seen
        for _ in range(grad_accum_steps):
            train_inputs, train_targets = next(train_iter)
            with autocast_ctx:
                loss = model(train_inputs, train_targets)
            train_loss = loss.detach()  # for logging
            loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
            loss.backward()  # accumulate the gradient
            num_tokens += (train_targets >= 0).sum()
        if ddp:
            dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)  # sum over ranks

        # learning rate scheduler
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        # step the optimizers
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        # logging
        train_loss_item = train_loss.item()  # type: ignore
        num_tokens_item = num_tokens.item()
        print0(
            f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}"
        )
        wandb_run.log(
            {
                "step": step,
                "lrm": lrm,
                "train_loss": train_loss_item,
                "num_tokens": num_tokens_item,
            }
        )
        step += 1

    # Save the model at the end of the run
    if master_process:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}"  # base the model tag on the depth of the base model
        checkpoint_dir = base_dir / "chatsft_checkpoints" / model_tag
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,  # note: we don't bother to save the optimizer state
            {
                "step": step,
                "val_loss": val_loss,  # type: ignore
                **metrics,  # type: ignore
                "model_config": model_config_kwargs,
            },
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

    # Log to report

    get_report().log(
        section="Chat SFT",
        data=[
            user_config,  # CLI args
            {
                "Training rows": len(train_ds),
                "Number of iterations": num_iterations,
                "Training loss": train_loss_item,  # type: ignore
                "Validation loss": val_loss,  # type: ignore
            },
        ],
    )

    # Cleanup
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    typer.run(main)
