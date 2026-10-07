"""FlexAttention (compiled Triton flash-style kernel) for the cached verifier.

Two block-sparse masks cover every first-order attention call:

* packed prefixes: causal within each task's document;
* suffix rows: all of the row's own task prefix (``kv < plen[row]``) plus
  causal attention over the row's own suffix (keys laid out as
  ``[prefix padded to Pmax | suffix]``).

Scores are never materialized.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_compiled = None


def _flex():
    global _compiled
    if _compiled is None:
        import torch._dynamo

        # Shapes vary per batch; past the default limit of 8 recompiles, dynamo
        # silently falls back to eager flex (slow, memory-hungry).
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.accumulated_recompile_limit = 4096
        torch._dynamo.config.cache_size_limit = 256
        _compiled = torch.compile(flex_attention, dynamic=True)
    return _compiled


def prefix_mask(cu: torch.Tensor, length: int):
    doc = torch.repeat_interleave(torch.arange(cu.numel() - 1, device=cu.device), cu[1:] - cu[:-1])

    def mask_mod(b, h, q, kv):
        return (doc[q] == doc[kv]) & (q >= kv)

    return create_block_mask(mask_mod, None, None, length, length, device=cu.device)


def suffix_mask(plen: torch.Tensor, pmax: int, length: int):
    def mask_mod(b, h, q, kv):
        in_prefix = kv < plen[b]
        in_suffix = (kv >= pmax) & (kv - pmax <= q)
        return in_prefix | in_suffix

    return create_block_mask(
        mask_mod, plen.numel(), None, length, pmax + length, device=plen.device
    )


def attend(q, k, v, block_mask, scale: float) -> torch.Tensor:
    """q (B, H, Lq, D); k, v (B, Hkv, Lk, D) with grouped-query heads."""

    # Short queries select FlexAttention's decode kernel, whose default tile
    # (BLOCK_M=256) does not divide the 128-token mask blocks; pin compatible tiles.
    return _flex()(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
        kernel_options={"fwd_BLOCK_M": 64, "fwd_BLOCK_N": 64},
    )
