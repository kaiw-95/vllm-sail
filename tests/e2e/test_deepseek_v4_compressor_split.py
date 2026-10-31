# SPDX-License-Identifier: Apache-2.0
"""Future M890 parity/replay check for mixed C128 decode and prefill rows."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("padded_pages", [False, True])
def test_c128_split_matches_single_pass_and_replays(monkeypatch, padded_pages):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
        compress_norm_rope_store_two_stage_triton,
    )
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or not current_platform.is_device_capability(
        (8, 9)
    ):
        pytest.skip("The split policy requires M890")
    monkeypatch.setenv("VLLM_SAIL_DEEPSEEK_C128_TWO_STAGE", "1")
    generator = torch.Generator(device="cuda").manual_seed(128)
    state = torch.randn(4, 64, 1024, device="cuda", generator=generator)
    page_stride = 37440 if padded_pages else 64 * 584
    storage = torch.full((2, page_stride), 177, dtype=torch.uint8, device="cuda")
    reference_storage = storage.clone()
    cache = storage.as_strided((2, 64, 584), (page_stride, 584, 1))
    reference = reference_storage.as_strided((2, 64, 584), (page_stride, 584, 1))
    scratch = torch.empty(4, 512, device="cuda", dtype=torch.float32)
    slots = torch.tensor([0, 64, -1, 2], device="cuda", dtype=torch.int64)
    angles = torch.randn(256, 32, device="cuda", generator=generator)
    args = dict(
        state_cache=state,
        num_actual=4,
        token_to_req_indices=torch.zeros(4, device="cuda", dtype=torch.int32),
        positions=torch.tensor([127, 255, 255, 126], device="cuda", dtype=torch.int64),
        slot_mapping=slots,
        block_table=torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32),
        block_size=64,
        state_width=512,
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
        k_cache_metadata=SimpleNamespace(slot_mapping=slots),
        pdl_kwargs={},
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        overlap=False,
        use_fp4_cache=False,
        rms_norm_weight=torch.ones(512, device="cuda", dtype=torch.bfloat16),
        rms_norm_eps=1e-6,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
    )

    def split():
        compress_norm_rope_store_two_stage_triton(
            **args, kv_cache=cache, num_decode_tokens=1, compress_scratch=scratch
        )

    def unpack(value):
        flat = value.flatten(1)
        data = flat[:, : 64 * 576].reshape(2, 64, 576)[:, 0]
        scale_bytes = flat[:, 64 * 576 :].reshape(2, 64, 8)[:, 0, :7]
        scales = torch.exp2(scale_bytes.float() - 127).repeat_interleave(64, dim=-1)
        nope = data[:, :448].view(torch.float8_e4m3fn).float() * scales
        rope = data[:, 448:576].contiguous().view(torch.bfloat16).float()
        return torch.cat((nope, rope), dim=-1)

    compress_norm_rope_store_triton(**args, kv_cache=reference)
    split()
    torch.testing.assert_close(
        unpack(cache), unpack(reference), rtol=0.03, atol=0.03125
    )
    compressed = state[1, :64, :512]
    scores = state[1, :64, 512:]
    # The remaining first half of position 128..255 lives in state block 3.
    compressed = torch.cat((state[3, :, :512], compressed))
    scores = torch.cat((state[3, :, 512:], scores))
    expected_scratch = (compressed * scores.softmax(dim=0)).sum(dim=0)
    torch.testing.assert_close(scratch[0], expected_scratch, rtol=2e-5, atol=2e-6)
    # Slot 2 is a non-boundary row; padded/invalid slots must stay untouched.
    assert torch.all(storage[:, 576 : 64 * 576] == 177)
    assert torch.all(storage[:, 64 * 584 :] == 177)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        split()
    state.add_(0.1)
    graph.replay()
    compress_norm_rope_store_triton(**args, kv_cache=reference)
    torch.testing.assert_close(
        unpack(cache), unpack(reference), rtol=0.03, atol=0.03125
    )
