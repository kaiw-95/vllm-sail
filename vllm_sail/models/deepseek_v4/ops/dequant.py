# SPDX-License-Identifier: Apache-2.0
"""Preflight the vector gather before launching any mutating kernel."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

import vllm_sail.envs as envs
from vllm_sail import _custom_ops as ops

logger = init_logger(__name__)


def can_vector_gather(
    out, cache, seq_lens, gather_lens, block_table, block_size, offset, use_fnuz=False
):
    if (
        use_fnuz
        or out.ndim != 3
        or out.shape[-1] != 512
        or cache.ndim != 3
        or cache.shape[1:] != (block_size, 584)
        or seq_lens.ndim != 1
        or seq_lens.shape[0] != out.shape[0]
        or block_table.ndim != 2
        or block_table.shape[0] != out.shape[0]
        or not 0 < out.shape[0] <= 65535
        or block_size <= 0
        or not 0 <= offset <= out.shape[1]
    ):
        return False
    gather = seq_lens if gather_lens is None else gather_lens
    tensors = (
        (out, torch.bfloat16),
        (cache, torch.uint8),
        (seq_lens, torch.int32),
        (gather, torch.int32),
        (block_table, torch.int32),
    )
    return (
        gather.shape == seq_lens.shape
        and all(
            t.is_cuda and t.device == out.device and t.dtype == dtype
            for t, dtype in tensors
        )
        and seq_lens.is_contiguous()
        and gather.is_contiguous()
        and block_table.stride(1) == 1
        and cache.stride(2) == 1
        and out.stride(2) == 1
        and out.stride(1) >= 512
        and out.stride(0) >= out.shape[1] * out.stride(1)
        and out.stride(0) % 8 == 0
        and out.stride(1) % 8 == 0
        and cache.stride(0) >= block_size * 584
        and cache.stride(0) % 16 == 0
        and out.data_ptr() % 16 == 0
        and cache.data_ptr() % 16 == 0
    )


def try_vector_gather(
    out,
    cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
    use_fnuz=False,
    max_gather_len=None,
) -> bool:
    if not envs.VLLM_SAIL_DEEPSEEK_VECTOR_DEQUANT:
        return False
    if not can_vector_gather(
        out, cache, seq_lens, gather_lens, block_table, block_size, offset, use_fnuz
    ):
        return False
    if not ops.has_vector_dequant():
        logger.warning_once(
            "Vector K-cache gather requested but the native op is absent; rebuild vllm-sail. Using the existing gather."
        )
        return False
    bound = out.shape[1] - offset
    if max_gather_len is not None:
        if max_gather_len < 0:
            raise ValueError("max_gather_len must be nonnegative")
        bound = min(bound, max_gather_len)
    # This opaque op has a fake implementation and a declared output mutation.
    # Launch failures propagate; retrying could read partly modified output.
    torch.ops.vllm.ppu_deepseek_dequant_gather_vector(
        out,
        cache,
        seq_lens,
        gather_lens if gather_lens is not None else seq_lens,
        block_table,
        block_size,
        offset,
        bound,
    )
    return True
