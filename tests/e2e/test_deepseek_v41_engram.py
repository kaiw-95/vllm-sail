# SPDX-License-Identifier: Apache-2.0
"""Real PPU lookup checks for per-row FP32 Engram scales, including UVA."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("sorted_reads", [False, True])
def test_channelwise_engram_lookup_preserves_fp32_scales(cpu_offload, sorted_reads):
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel

    placement = (
        {"device": "cpu", "pin_memory": True} if cpu_offload else {"device": "cuda"}
    )
    weight = torch.empty(8, 256, dtype=torch.float8_e4m3fn, **placement)
    weight.fill_(2.0)
    scales = torch.empty(8, 1, dtype=torch.float32, **placement)
    scales.copy_(torch.arange(1, 9, dtype=torch.float32).view(8, 1) * 0.3)
    # Keep host allocations alive while the kernel reads their UVA aliases.
    views = (weight, scales)
    if cpu_offload:
        views = tuple(get_accelerator_view_from_cpu_tensor(t) for t in views)
    ids = torch.tensor(
        [[0, 4, 8, 999], [0, 4, 11, -1]], dtype=torch.int64, device="cuda"
    )
    out = torch.empty(2, 3, 256, dtype=torch.bfloat16, device="cuda")
    kernel_ids, dst = ids, ids
    if sorted_reads:
        local = ids[:, 2:] - 8
        local = torch.nn.functional.pad(local, (0, 1), value=-1)
        kernel_ids, dst = local.flatten().sort()
        dst = dst.int()
    _engram_lookup_kernel[(1,)](
        *views,
        kernel_ids,
        dst,
        out,
        8,
        16,
        6,
        ids.stride(0),
        ids.stride(1),
        HEAD_START=2,
        LOCAL_HEADS=3,
        TOTAL_HEADS=4,
        DIM=256,
        QUANT_BLOCK=256,
        BLOCK_R=16,
        GRID=1,
        SORTED=sorted_reads,
        PACKED=False,
        OUT_STRIDE=out.stride(0),
    )
    torch.cuda.synchronize()
    expected = torch.zeros(2, 3, 256, dtype=torch.bfloat16)
    expected[0, 0] = (2.0 * scales[0, 0].cpu()).to(torch.bfloat16)
    expected[1, 0] = (2.0 * scales[3, 0].cpu()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("sorted_reads", [False, True])
def test_packed_e8m0_lookup_keeps_bytes_and_output_stride(sorted_reads):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel

    weight = torch.full((8, 256), 2.0, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.full((8, 8), 127, dtype=torch.uint8, device="cuda")
    ids = torch.tensor([[8, -1], [11, 999]], dtype=torch.int64, device="cuda")
    kernel_ids, dst = ids, ids
    if sorted_reads:
        kernel_ids, dst = (ids - 8).flatten().sort()
        dst = dst.int()
    # Leave a sentinel after each token's two packed head rows.
    out = torch.full((2, 2 * 264 + 16), 253, dtype=torch.uint8, device="cuda")
    _engram_lookup_kernel[(1,)](
        weight,
        scales,
        kernel_ids,
        dst,
        out,
        8,
        16,
        4,
        ids.stride(0),
        ids.stride(1),
        HEAD_START=0,
        LOCAL_HEADS=2,
        TOTAL_HEADS=2,
        DIM=256,
        QUANT_BLOCK=32,
        BLOCK_R=16,
        GRID=1,
        SORTED=sorted_reads,
        PACKED=True,
        OUT_STRIDE=out.stride(0),
    )
    expected = torch.full_like(out, 253)
    for token, source in ((0, 0), (1, 3)):
        expected[token, :256] = weight[source].view(torch.uint8)
        expected[token, 256:264] = scales[source]
        expected[token, 264:528] = 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
