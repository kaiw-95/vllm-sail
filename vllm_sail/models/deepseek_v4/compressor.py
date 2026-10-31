# SPDX-License-Identifier: Apache-2.0
"""Bounded M890 policy for V4's existing split C128 compressor.

SGLang's c128_v2.cuh (ccf7e9a558184a01059e2553f97cdb52c67f12c8) splits the
512 columns into eight 64-column tiles before a separate norm/RoPE/store.
vLLM already has this decomposition; reuse it with its own rounding and cache
layout. V4.1's ratio-1/2 compressor is deliberately outside this policy.
"""

import torch
from vllm.platforms import current_platform

from vllm_sail import envs


def enabled() -> bool:
    return (
        envs.VLLM_SAIL_DEEPSEEK_C128_TWO_STAGE
        and current_platform.is_ppu()
        and current_platform.is_device_capability(
            (8, 9), torch.accelerator.current_device_index()
        )
    )


def can_allocate(vllm_config, compress_ratio: int, head_dim: int) -> bool:
    return (
        enabled()
        and compress_ratio == 128
        and head_dim == 512
        and vllm_config.model_config.hf_config.qk_rope_head_dim == 64
        and vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        # One scratch allocation per module cannot be shared by concurrent
        # microbatches. Keep their existing single-pass compressor.
        and not vllm_config.parallel_config.use_ubatching
    )


def can_launch(compressor, state_metadata, state_cache, kv_cache, num_actual) -> bool:
    scratch = compressor._compress_scratch
    return (
        enabled()
        and compressor.compress_ratio == 128
        and compressor.head_dim == 512
        and compressor.rope_head_dim == 64
        and not compressor.overlap
        and not compressor.use_fp4_cache
        and state_metadata.num_decode_tokens is not None
        and scratch is not None
        and scratch.shape[0] >= num_actual
        and scratch.shape[1] == 512
        and scratch.dtype == torch.float32
        and scratch.is_contiguous()
        and state_cache.dtype == torch.float32
        and scratch.device == state_cache.device == kv_cache.device
        and kv_cache.dtype == torch.uint8
        and kv_cache.ndim == 3
        and kv_cache.shape[-1] == 584
        and kv_cache.stride(2) == 1
        and kv_cache.stride(1) == 584
        and kv_cache.stride(0) >= kv_cache.shape[1] * 584
    )
