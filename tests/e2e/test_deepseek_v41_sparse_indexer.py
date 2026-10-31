# SPDX-License-Identifier: Apache-2.0
"""Compare candidate-only SAIL scores with dense MXFP4 scores on a device."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


def test_sparse_prefill_preserves_request_relative_unaligned_candidates():
    import deep_gemm
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.models.deepseek_v41.sparse_indexer import sparse_prefill

    with torch.device("cuda"):
        q = torch.randint(-128, 127, (3, 32, 64), dtype=torch.int8)
        k = torch.randint(-128, 127, (64, 64), dtype=torch.int8)
        qs = (
            torch.full((3, 32, 4), 127, dtype=torch.uint8).view(torch.int32).squeeze(-1)
        )
        ks = torch.full((64, 4), 127, dtype=torch.uint8).view(torch.int32).squeeze(-1)
        weights = torch.rand(3, 32, dtype=torch.bfloat16)
        starts = torch.tensor([0, 5, 20], dtype=torch.int32)
        ends = torch.tensor([13, 30, 63], dtype=torch.int32)
        candidates = torch.tensor(
            [[0, 1, -1, -1], [1, 0, -1, -1], [2, 0, -1, -1]], dtype=torch.int32
        )
        out = torch.empty((3, 8), dtype=torch.int32)
        sparse_prefill(
            q,
            qs,
            k,
            ks,
            weights,
            starts,
            ends,
            candidates,
            16,
            8,
            8,
            out,
            sparse_indices=torch.empty((3, 8), dtype=torch.int32),
            end=torch.empty(3, dtype=torch.int32),
            col_indices=torch.empty_like(out),
        )
        dense = deep_gemm.fp8_fp4_mqa_logits(
            (q, qs),
            (k, ks),
            weights,
            starts,
            ends,
            clean_logits=False,
            logits_dtype=torch.bfloat16,
        )
        for row, start in enumerate([0, 5, 20]):
            length = int((ends[row] - starts[row]).item())
            positions = torch.arange(length)
            selected_blocks = candidates[row][candidates[row] >= 0]
            allowed = (positions[:, None] // 16 == selected_blocks[None, :]).any(-1)
            indices = out[row].long()
            assert bool(((indices >= 0) & (indices < length)).all())
            assert bool(allowed[indices].all())
            reference = (
                dense[row, start : start + length][allowed]
                .float()
                .topk(8)
                .values.sort()
                .values
            )
            actual = dense[row, start + indices].float().sort().values
            torch.testing.assert_close(actual, reference, atol=0.5, rtol=0.03)


@pytest.mark.parametrize("length", [0, 7, 1025, 2048, 4096])
def test_bf16_top2048_writes_every_slot(length):
    import torch

    from vllm_sail import _custom_ops as ops

    logits = (
        torch.arange(max(length, 1), device="cuda", dtype=torch.float32)
        .bfloat16()
        .unsqueeze(0)
    )
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, length)
    output = torch.full((1, 2048), -99, device="cuda", dtype=torch.int32)
    ops.top_k_per_row_prefill_bf16(
        logits, starts, ends, output, 1, logits.stride(0), 1, 2048
    )
    assert not bool((output == -99).any())
    valid = output[output >= 0].long()
    assert valid.numel() == min(length, 2048)
    assert valid.unique().numel() == valid.numel()
    if length > 2048:
        torch.testing.assert_close(
            logits[0, valid].float().sort().values,
            logits[0].float().topk(2048).values.sort().values,
        )


def test_bf16_top2048_refines_overflowing_coarse_histogram_bin():
    import torch

    from vllm_sail import _custom_ops as ops

    # Both BF16 values land in the same coarse FP16 bin. Discarding all but
    # the first 2048 ties could lose larger values encountered later.
    logits = torch.ones((1, 8192), device="cuda", dtype=torch.bfloat16)
    logits[:, -2048:] = 1.0078125
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, 8192)
    output = torch.empty((1, 2048), device="cuda", dtype=torch.int32)
    ops.top_k_per_row_prefill_bf16(
        logits, starts, ends, output, 1, logits.stride(0), 1, 2048
    )
    torch.testing.assert_close(
        output[0].sort().values,
        torch.arange(6144, 8192, device="cuda", dtype=torch.int32),
    )
