# SPDX-License-Identifier: Apache-2.0
"""Exercise real shared mHC post dispatch with strict device-free doubles."""

from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


class Tensor:
    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.contiguous = True

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def post_dispatch(modules):
    calls = []
    state = SimpleNamespace(ppu=True, cap=True)

    class Launcher:
        _warming_key = None

        def register_warmup(self, **kwargs):
            return kwargs

        def __call__(self, *args):
            calls.append(("upstream", args))

        def warmup_inputs(self, compile_key):
            # Strict compile-only placeholders intentionally expose no device.
            return dict(
                zip(
                    ("comb_mix", "residual", "post_mix", "layer_input", "out"),
                    (object() for _ in range(5)),
                )
            )

        def launch(self, jit_impl, args):
            assert self._warming_key is not None
            calls.append(("compile", args))

    provider = modules(
        "vllm.model_executor.kernels.mhc.tilelang_kernels",
        MhcPostTileLangKernel=Launcher,
    )
    modules("torch", bfloat16="bf16", float32="fp32")
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: state.ppu,
            is_device_capability=lambda cap, device_id: state.cap and device_id == 3,
        ),
    )
    settings = modules("vllm_sail.envs", VLLM_SAIL_MHC_POST_TILED=True)
    modules(
        "vllm_sail.model_executor.kernels.mhc_post",
        mhc_post_tiled=lambda *args: calls.append(("tiled", args)),
    )
    harness.load_patch("vllm_sail/patch/enhancement/attention/mhc_post.py")
    device = SimpleNamespace(type="cuda", index=3)
    args = [
        Tensor((64, 4, 4), "fp32", device),
        Tensor((64, 4, 5120), "bf16", device),
        Tensor((64, 4), "fp32", device),
        Tensor((64, 5120), "bf16", device),
        Tensor((64, 4, 5120), "bf16", device),
        4,
        5120,
    ]
    return provider.MhcPostTileLangKernel(), args, calls, state, settings


def test_shared_launcher_preserves_warmup_and_tiled_output(post_dispatch):
    launch, args, calls, _, _ = post_dispatch
    assert launch.register_warmup(hidden_size=5120) == {"hidden_size": 5120}
    assert launch(*args) is None
    assert calls == [("tiled", (*args[:5], 5120))]


def test_compile_only_warmup_covers_tiled_and_serial_without_device(post_dispatch):
    launch, _, calls, _, _ = post_dispatch
    key = SimpleNamespace(hidden_size=5120, hc_mult=4)
    launch._warming_key = key
    inputs = launch.warmup_inputs(key)
    args = tuple(
        inputs[name]
        for name in ("comb_mix", "residual", "post_mix", "layer_input", "out")
    )
    launch(*args, 4, 5120)
    assert calls == [("compile", (*args, 5120)), ("upstream", (*args, 4, 5120))]


@pytest.mark.parametrize(
    "reason", ["disabled", "platform", "arch", "layout", "dtype", "size", "post_shape"]
)
def test_unsupported_mhc_post_keeps_original_launcher(post_dispatch, reason):
    launch, args, calls, state, settings = post_dispatch
    if reason == "disabled":
        settings.VLLM_SAIL_MHC_POST_TILED = False
    elif reason == "platform":
        state.ppu = False
    elif reason == "arch":
        state.cap = False
    elif reason == "layout":
        args[1].contiguous = False
    elif reason == "dtype":
        args[0].dtype = "bf16"
    elif reason == "size":
        args[6] = 5136
    elif reason == "post_shape":
        args[2].shape = (64, 4, 1)
    launch(*args)
    assert calls == [("upstream", tuple(args))]
