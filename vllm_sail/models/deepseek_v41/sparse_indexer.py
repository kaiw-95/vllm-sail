# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only SAIL MXFP4 indexer with bounded live logits.

Uses main's candidate expansion, cache gather and metadata containers. Unlike
the SM100 path it uses SAIL BF16 top-k, fixed-length flattened decode rows and
the PPU library's schedule builders. Each plan is scored and consumed before
allocating the next tile, for prefill and decode alike.
"""

from __future__ import annotations

import torch
import vllm.envs as envs
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    candidate_blocks_to_sparse_indices,
    pick_sparse_block_kv,
)
from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadataBuilder,
)

from vllm_sail import _custom_ops as ops
from vllm_sail.utils.indexer import logits_rows_per_chunk, sparse_mqa_api


@triton.jit
def _remap(
    cols,
    si,
    ks,
    out,
    stride_cols: tl.constexpr,
    stride_si: tl.constexpr,
    stride_out: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    SBK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    index = tl.load(cols + row * stride_cols + col, mask=col < K, other=-1)
    valid = (index >= 0) & (index < WIDTH)
    block = tl.load(
        si + row * stride_si + tl.maximum(index, 0) // SBK, mask=valid, other=0
    )
    start = tl.load(ks + row) // SBK
    position = (block - start) * SBK + index % SBK
    tl.store(out + row * stride_out + col, tl.where(valid, position, -1), mask=col < K)


def sparse_topk(
    logits, sparse_indices, end, row_ks, sparse_block_kv, topk_tokens, output, columns
):
    rows, width = logits.shape
    zeros = torch.zeros_like(end)
    # Native prefill top-k returns request-relative columns and pads with -1.
    ops.top_k_per_row_prefill_bf16(
        logits,
        zeros,
        end,
        columns,
        rows,
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    _remap[(rows,)](
        columns,
        sparse_indices,
        row_ks,
        output,
        columns.stride(0),
        sparse_indices.stride(0),
        output.stride(0),
        topk_tokens,
        width,
        sparse_block_kv,
        triton.next_power_of_2(topk_tokens),
    )


def _plans(rows, width, cached):
    step = logits_rows_per_chunk(width, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB)
    ranges = [slice(start, min(start + step, rows)) for start in range(0, rows, step)]
    if cached is not None and len(cached) != len(ranges):
        raise ValueError("Sparse indexer schedule does not match the row chunks.")
    return ranges


def sparse_prefill(
    q,
    q_scale,
    k_quant,
    k_scale,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    candidate_blocks,
    candidate_block_size,
    sparse_block_kv,
    topk_tokens,
    topk_indices,
    *,
    sparse_indices,
    end,
    col_indices,
    kernel_metadata=None,
):
    api = sparse_mqa_api()
    if api is None:
        raise RuntimeError("SAIL sparse MQA API disappeared after model construction.")
    result = []
    for plan, tile in enumerate(
        _plans(q.shape[0], sparse_indices.shape[1] * sparse_block_kv, kernel_metadata)
    ):
        si, valid = sparse_indices[tile], end[tile]
        metadata = None if kernel_metadata is None else kernel_metadata[plan]
        if metadata is None:
            candidate_blocks_to_sparse_indices(
                candidate_blocks[tile],
                cu_seqlen_ks[tile],
                cu_seqlen_ke[tile],
                candidate_block_size,
                sparse_block_kv,
                out=(si, valid),
            )
            metadata = api.get_sparse_mqa_logits_metadata(
                cu_seqlen_ks[tile].contiguous(),
                cu_seqlen_ke[tile].contiguous(),
                k_quant.shape[0],
                si,
                q.dtype,
                sparse_block_kv,
                use_unaligned_ks=True,
            )
        logits = api.fp8_fp4_sparse_mqa_logits(
            (q[tile].contiguous(), q_scale[tile].contiguous()),
            (k_quant, k_scale),
            weights[tile].contiguous(),
            metadata,
            si.shape[1],
            sparse_block_kv,
            use_unaligned_ks=True,
        )
        sparse_topk(
            logits,
            si,
            valid,
            cu_seqlen_ks[tile],
            sparse_block_kv,
            topk_tokens,
            topk_indices[tile],
            col_indices[tile],
        )
        del logits
        result.append(metadata)
    return result


def sparse_decode(
    q,
    q_scale,
    kv_cache,
    weights,
    context_lens,
    block_table,
    row_indices,
    candidate_blocks,
    candidate_block_size,
    sparse_block_kv,
    topk_tokens,
    topk_indices,
    *,
    row_ks,
    sparse_indices,
    end,
    col_indices,
    kernel_metadata=None,
):
    api = sparse_mqa_api()
    if api is None:
        raise RuntimeError("SAIL sparse MQA API disappeared after model construction.")
    if kv_cache.stride(0) % 512:
        raise ValueError("SAIL sparse indexer requires 512-byte aligned cache pages.")
    result = []
    for plan, tile in enumerate(
        _plans(q.shape[0], sparse_indices.shape[1] * sparse_block_kv, kernel_metadata)
    ):
        si, valid = sparse_indices[tile], end[tile]
        metadata = None if kernel_metadata is None else kernel_metadata[plan]
        if metadata is None:
            candidate_blocks_to_sparse_indices(
                candidate_blocks[tile],
                row_ks[tile],
                context_lens[tile],
                candidate_block_size,
                sparse_block_kv,
                out=(si, valid),
            )
            metadata = api.get_paged_sparse_mqa_logits_metadata(
                context_lens[tile].contiguous(),
                block_table[tile].contiguous(),
                row_indices[tile].contiguous(),
                kv_cache.shape[1],
                si,
                q.dtype,
                sparse_block_kv,
            )
        logits = api.fp8_fp4_paged_sparse_mqa_logits(
            (q[tile].contiguous(), q_scale[tile].contiguous()),
            kv_cache,
            weights[tile].contiguous(),
            metadata,
            si.shape[1],
            sparse_block_kv,
        )
        sparse_topk(
            logits,
            si,
            valid,
            row_ks[tile],
            sparse_block_kv,
            topk_tokens,
            topk_indices[tile],
            col_indices[tile],
        )
        del logits
        result.append(metadata)
    return result


class PPUSparseMQAIndexer(SparseMQAIndexer):
    """Keep the upstream forward; the two logits entry points dispatch to SAIL."""

    def __new__(cls, *args, **kwargs):
        if not current_platform.is_ppu():
            return SparseMQAIndexer(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        k_cache,
        topk_tokens,
        head_dim,
        max_total_seq_len,
        topk_indices_buffer,
        candidate_blocks,
        candidate_block_size,
    ):
        torch.nn.Module.__init__(self)
        self.k_cache, self.topk_tokens, self.head_dim = k_cache, topk_tokens, head_dim
        self.max_total_seq_len, self.topk_indices_buffer = (
            max_total_seq_len,
            topk_indices_buffer,
        )
        self.candidate_blocks, self.candidate_block_size = (
            candidate_blocks,
            candidate_block_size,
        )


class PPUSparseIndexerMetadataBuilder(DeepseekV41SparseIndexerMetadataBuilder):
    """Reuse dense fixed-length plans and main's sparse row containers."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        DeepseekV32IndexerMetadataBuilder.__init__(
            self, kv_cache_spec, layer_names, vllm_config, device, **kwargs
        )
        hf = vllm_config.model_config.hf_text_config
        block_size, candidates, topk = (
            hf.candidate_block_size,
            hf.candidate_topk_blocks,
            hf.index_topk,
        )
        self.sparse_block_kv = pick_sparse_block_kv(block_size)
        self.num_sparse_blocks = candidates * (block_size // self.sparse_block_kv)
        if (
            candidates <= 0
            or candidates > 2048
            or candidates & (candidates - 1)
            or block_size // self.sparse_block_kv not in (1, 2)
            or not 0 < self.num_sparse_blocks <= 4096
            or self.num_sparse_blocks % 4
            or not 0 < topk <= 2048
        ):
            raise ValueError(
                "SAIL sparse indexer requires power-of-two candidate top-k <=2048, <=4096 sparse blocks and token top-k <=2048."
            )
        if not self.use_flattening:
            raise ValueError(
                "SAIL sparse indexer requires flattened fixed-length decode metadata."
            )
        width = self.num_sparse_blocks * self.sparse_block_kv
        logits_rows_per_chunk(width, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB)
        # The SAIL base chunker already budgets FP4 logits as BF16.
        self.min_split_seq_len = (width + 511) // 512 * 512
        rows = self.arange_buffer.shape[0]
        options = dict(dtype=torch.int32, device=device)
        self.sparse_indices_buffer = torch.zeros(
            (rows, self.num_sparse_blocks), **options
        )
        self.end_buffer = torch.zeros(rows, **options)
        self.col_indices_buffer = torch.zeros((rows, topk), **options)
        self.zero_row_ks = torch.zeros(rows, **options)
