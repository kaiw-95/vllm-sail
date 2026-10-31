# SPDX-License-Identifier: Apache-2.0
"""Exercise real C128 admission and metadata/split hooks without device deps."""

from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


@pytest.fixture
def c128(modules):
    state = SimpleNamespace(ppu=True, cap=True)
    modules(
        "torch",
        float32="f32",
        uint8="u8",
        accelerator=SimpleNamespace(current_device_index=lambda: 3),
    )
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: state.ppu,
            is_device_capability=lambda cap, device_id: state.cap and device_id == 3,
        ),
    )
    settings = modules("vllm_sail.envs", VLLM_SAIL_DEEPSEEK_C128_TWO_STAGE=True)
    policy = harness.load_patch("vllm_sail/models/deepseek_v4/compressor.py")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(qk_rope_head_dim=64)),
        cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    return policy, state, settings, config


@pytest.mark.parametrize(
    "reason",
    [None, "disabled", "platform", "arch", "rope", "layout", "ubatch", "ratio", "head"],
)
def test_allocation_requires_nonoverlapped_legacy_c128(c128, reason):
    policy, state, settings, config = c128
    ratio, head = 128, 512
    if reason == "disabled":
        settings.VLLM_SAIL_DEEPSEEK_C128_TWO_STAGE = False
    elif reason == "platform":
        state.ppu = False
    elif reason == "arch":
        state.cap = False
    elif reason == "rope":
        config.model_config.hf_config.qk_rope_head_dim = 128
    elif reason == "layout":
        config.cache_config.cache_dtype = "bf16"
    elif reason == "ubatch":
        config.parallel_config.use_ubatching = True
    elif reason == "ratio":
        ratio = 2
    elif reason == "head":
        head = 128
    assert policy.can_allocate(config, ratio, head) is (reason is None)


def test_metadata_and_split_policy_share_real_module_hooks(c128, modules):
    policy, state, settings, _ = c128
    modules("vllm_sail.models.deepseek_v4.compressor", enabled=policy.enabled)
    source = modules(
        "vllm.models.deepseek_v4.compressor",
        _prefer_two_stage_compressor=lambda: "original",
    )
    ops = modules(
        "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
        _pick_compress_num_splits=lambda *args: ("original", args),
    )
    harness.load_patch(
        "vllm_sail/patch/enhancement/models/deepseek_v4_compressor_split.py"
    )
    assert source._prefer_two_stage_compressor() is True
    assert ops._pick_compress_num_splits(129, 128, 512) == 8
    assert ops._pick_compress_num_splits(129, 4, 512) == ("original", (129, 4, 512))
    settings.VLLM_SAIL_DEEPSEEK_C128_TWO_STAGE = False
    assert source._prefer_two_stage_compressor() is False
    assert ops._pick_compress_num_splits(129, 128, 512) == ("original", (129, 128, 512))
    state.ppu = False
    assert source._prefer_two_stage_compressor() == "original"
    assert ops._pick_compress_num_splits(129, 128, 512) == ("original", (129, 128, 512))


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "metadata",
        "scratch",
        "capacity",
        "format",
        "stride",
        "padded",
        "dtype",
        "device",
        "disabled",
    ],
)
def test_launch_needs_owned_scratch_and_metadata(c128, reason):
    policy, _, settings, _ = c128

    def tensor(shape, dtype):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device="cuda:3",
            is_contiguous=lambda: True,
        )

    scratch = tensor((256, 512), "f32")
    compressor = SimpleNamespace(
        _compress_scratch=scratch,
        compress_ratio=128,
        head_dim=512,
        rope_head_dim=64,
        overlap=False,
        use_fp4_cache=False,
    )
    metadata = SimpleNamespace(num_decode_tokens=1)
    state, cache = tensor((4, 64, 1024), "f32"), tensor((2, 64, 584), "u8")
    cache.stride = lambda axis: (64 * 584, 584, 1)[axis]
    if reason == "metadata":
        metadata.num_decode_tokens = None
    elif reason == "scratch":
        compressor._compress_scratch = None
    elif reason == "capacity":
        scratch.shape = (1, 512)
    elif reason == "format":
        cache.shape = (2, 64, 528)
    elif reason == "stride":
        cache.stride = lambda axis: (64 * 584, 583, 1)[axis]
    elif reason == "padded":
        cache.stride = lambda axis: (37440, 584, 1)[axis]
    elif reason == "dtype":
        state.dtype = "bf16"
    elif reason == "device":
        scratch.device = "cuda:0"
    elif reason == "disabled":
        settings.VLLM_SAIL_DEEPSEEK_C128_TWO_STAGE = False
    assert policy.can_launch(compressor, metadata, state, cache, 128) is (
        reason in (None, "padded")
    )
