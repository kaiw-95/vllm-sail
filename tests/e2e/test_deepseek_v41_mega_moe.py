# SPDX-License-Identifier: Apache-2.0
"""Device checks for the SAIL MegaMoE loader and routing contracts."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


def test_mega_moe_checkpoint_loader_matches_library_transform():
    import torch
    from deep_gemm.mega import transform_weights_for_mega_moe

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm_sail.models.deepseek_v41.mega_moe import PPUDeepGemmMegaMoEExperts

    e, h, i = 2, 512, 128
    with torch.device("cuda"):
        w13 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
        s13 = torch.randint(120, 130, (e, 2 * i, h // 32), dtype=torch.uint8)
        s2 = torch.randint(120, 130, (e, h, i // 32), dtype=torch.uint8)
        expected = transform_weights_for_mega_moe((w13, s13), (w2, s2))
        layer = PPUDeepGemmMegaMoEExperts.__new__(PPUDeepGemmMegaMoEExperts)
        torch.nn.Module.__init__(layer)
        layer.hidden_size, layer.intermediate_size = h, i
        layer._register_weight("w13", (e, 2 * i, h), None)
        layer._register_weight("w2", (e, h, i), None)
        for expert in range(e):
            for shard, start in (("w1", 0), ("w3", i)):
                layer._load_weight(
                    layer.w13_weight[expert],
                    w13[expert, start : start + i],
                    shard,
                    False,
                )
                layer._load_weight(
                    layer.w13_weight_scale[expert],
                    s13[expert, start : start + i],
                    shard,
                    True,
                )
            layer._load_weight(layer.w2_weight[expert], w2[expert], "w2", False)
            layer._load_weight(layer.w2_weight_scale[expert], s2[expert], "w2", True)
        for actual_pair, expected_pair in zip(
            (layer.l1_weights, layer.l2_weights), expected, strict=True
        ):
            for actual, reference in zip(actual_pair, expected_pair, strict=True):
                assert actual.shape == reference.shape
                assert actual.stride() == reference.stride()
                assert torch.equal(
                    actual.contiguous().view(torch.uint8),
                    reference.contiguous().view(torch.uint8),
                )


def test_mega_gate_preserves_hash_image_and_weight_contracts():
    import torch

    from vllm_sail.models.deepseek_v41.mega_moe import mega_gate

    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    bias = torch.randn(8, device="cuda")
    image_bias = torch.randn(8, device="cuda")
    image = torch.tensor([False, True, False], device="cuda")
    fixed = torch.tensor([True, False, False], device="cuda")
    hash_ids = torch.tensor([[5, 2], [4, 1], [0, 3]], device="cuda")
    weights, ids = mega_gate(
        x,
        w,
        2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        ep_rank=0,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image,
        fix_routing_mask=fixed,
        unmapped_topk_idx=hash_ids,
    )
    torch.testing.assert_close(ids[0], hash_ids[0])
    scores = torch.nn.functional.softplus(x.float() @ w.T).sqrt()
    torch.testing.assert_close(
        ids[1], torch.argsort(scores[1] + image_bias, descending=True, stable=True)[:2]
    )
    selected = scores.gather(1, ids)
    torch.testing.assert_close(weights, selected / selected.sum(-1, keepdim=True) * 2.5)
