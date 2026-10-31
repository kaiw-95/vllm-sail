# SPDX-License-Identifier: Apache-2.0
"""Memory planning and optional API discovery for SAIL indexer logits."""

from __future__ import annotations

import functools
import importlib


def logits_rows_per_chunk(
    width: int, budget_mb: int, *, element_size: int = 2, row_alignment: int = 2
) -> int:
    """Budget the library's padded allocation, not only its sliced view."""
    if width <= 0 or budget_mb <= 0 or element_size <= 0 or row_alignment <= 0:
        raise ValueError(
            "Indexer width, logits budget and allocation alignment must be positive."
        )
    aligned_width = (width + 511) // 512 * 512
    rows = budget_mb * 1024 * 1024 // (aligned_width * element_size)
    rows = rows // row_alignment * row_alignment
    if rows == 0:
        raise ValueError(
            "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB cannot hold one aligned indexer tile."
        )
    return rows


@functools.cache
def sparse_mqa_api():
    try:
        api = importlib.import_module("deep_gemm")
    except ImportError:
        return None
    names = (
        "get_sparse_mqa_logits_metadata",
        "get_paged_sparse_mqa_logits_metadata",
        "fp8_fp4_sparse_mqa_logits",
        "fp8_fp4_paged_sparse_mqa_logits",
    )
    return api if all(callable(getattr(api, name, None)) for name in names) else None


def sparse_indexer_supported(config) -> bool:
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    hf, parallel = config.model_config.hf_text_config, config.parallel_config
    return (
        current_platform.is_ppu()
        and current_platform.has_device_capability(89)
        and is_deep_gemm_supported()
        and dsa_indexer_uses_fp4(config)
        and getattr(hf, "index_n_heads", 0) == 32
        and getattr(hf, "index_head_dim", 0) == 128
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and config.kernel_config.sparse_indexer_topk_backend in ("auto", "default")
        and sparse_mqa_api() is not None
    )
