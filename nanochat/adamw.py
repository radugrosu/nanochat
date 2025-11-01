"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""

from typing import cast

import torch
import torch.distributed as dist
from torch.optim.optimizer import ParamsT


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """

    def __init__(
        self,
        param_groups: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), (
            "All params must have grads"
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_scatter_futures: list[torch.Future[torch.Tensor]] = []
        grad_slices: list[torch.Tensor] = []
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            for param in params:
                grad = param.grad
                grad = cast(torch.Tensor, grad)
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()  # type: ignore
                future = cast(torch.Future[torch.Tensor], future)
                reduce_scatter_futures.append(future)
                grad_slices.append(grad_slice)

        future_idx = 0
        all_gather_futures: list[torch.Future[torch.Tensor]] = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = group["params"]
            for base in range(len(params)):
                reduce_scatter_futures[future_idx].wait()
                g_slice = grad_slices[future_idx]
                future_idx += 1
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]
                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                state: dict[str, torch.Tensor] = self.state[p]
                # State init
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()  # type: ignore
                future = cast(torch.Future[torch.Tensor], future)
                all_gather_futures.append(future)

        torch.futures.collect_all(
            all_gather_futures,  # type: ignore
        ).wait()
