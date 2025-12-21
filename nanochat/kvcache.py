from typing import Self

import torch


class KVCache:
    """Works hand-in-hand with the GPT model to maintain the KV cache.

    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int, num_layers: int):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # current position in time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self) -> int:
        return self.pos

    def prefill(self, other: Self):
        """Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"

        layers, kv, batch, heads, seq, head_dim = self.kv_shape
        olayers, okv, obatch, oheads, oseq, ohead_dim = other.kv_shape
        assert layers == olayers, f"Layer count mismatch: {layers} != {olayers}"
        assert kv == okv, f"K/V dim mismatch: {kv} != {okv}"
        assert heads == oheads, f"Head count mismatch: {heads} != {oheads}"
        assert head_dim == ohead_dim, f"Head dim mismatch: {head_dim} != {ohead_dim}"
        assert batch == obatch or obatch == 1, f"Batch size mismatch: {batch} != {obatch}"
        assert seq >= oseq, f"Sequence length mismatch: {seq} < {oseq}"

        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, : other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        _, _, T_add, _ = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024  # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023  # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view
