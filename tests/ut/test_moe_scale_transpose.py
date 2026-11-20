# SPDX-License-Identifier: Apache-2.0
"""Preserve the activation-scale layout selected by the FP8 backend."""

from enum import Enum
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/fused_moe_ppu.py"


def platform(monkeypatch, ppu):
    import sys

    module = ModuleType("vllm.platforms")
    module.current_platform = NS(is_ppu=lambda: ppu, fp8_dtype=lambda: "fp8")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.current_platform


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_quantizer_forwards_new_layout_keyword(monkeypatch, ppu, transpose):
    platform(monkeypatch, ppu)
    expected, tensor, scale = object(), object(), object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    quantize = function(
        PATCH,
        "install._moe_kernel_quantize_input",
        {
            "_upstream_quantize_input": original,
        },
    )
    assert (
        quantize(tensor, scale, "fp8", False, [128, 128], is_scale_transposed=transpose)
        is expected
    )
    assert calls == [
        (
            (tensor, scale, "fp8", False),
            dict(block_shape=[128, 128], is_scale_transposed=transpose),
        )
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize(
    "backend,block,transpose",
    [
        ("FLASHINFER_TRTLLM", [128, 128], True),
        ("FLASHINFER_TRTLLM", [64, 128], False),
        ("TRITON", [128, 128], False),
        ("FLASHINFER_CUTLASS", [128, 128], False),
    ],
)
def test_actual_fp8_config_layout_reaches_the_group_quantizer(
    monkeypatch, upstream_source_root, ppu, backend, block, transpose
):
    device = platform(monkeypatch, ppu)
    oracle_path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/fp8.py"
    )
    utils_path = upstream_source_root / "vllm/model_executor/layers/fused_moe/utils.py"
    backend_type = function(oracle_path, "Fp8MoeBackend", {"Enum": Enum})
    original_config = function(
        oracle_path,
        "make_fp8_moe_quant_config",
        {
            "Fp8MoeBackend": backend_type,
            "fp8_w8a8_moe_quant_config": lambda **kwargs: NS(**kwargs),
        },
    )
    make_config = function(
        "vllm_sail/registry/moe_backends/fp8.py",
        "make_fp8_moe_quant_config",
        {"_upstream_quant_config": original_config},
    )
    alpha, beta = object(), object()
    config = make_config(
        getattr(backend_type, backend),
        object(),
        object(),
        None,
        None,
        block_shape=block,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
    )
    assert config.is_scale_transposed is transpose
    assert config.gemm1_alpha is alpha and config.gemm1_beta is beta
    calls = []
    values = NS(size=lambda axis: 512)
    scales = NS(size=lambda axis: 4)

    def group_quant(tensor, size, **kwargs):
        calls.append((tensor, size, kwargs))
        return values, scales

    namespace = {
        "current_platform": device,
        "per_token_group_quant_fp8": group_quant,
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    function(utils_path, "_fp8_quantize", namespace)
    upstream_quantizer = function(utils_path, "moe_kernel_quantize_input", namespace)
    quantize = function(
        PATCH,
        "install._moe_kernel_quantize_input",
        {"_upstream_quantize_input": upstream_quantizer},
    )
    tensor = object()
    assert quantize(
        tensor,
        None,
        "fp8",
        False,
        block,
        is_scale_transposed=config.is_scale_transposed,
    ) == (values, scales)
    assert calls == [(tensor, 128, {"column_major_scales": transpose})]


@pytest.mark.requires_runtime
@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("backend", ["FLASHINFER_CUTLASS", "HPC"])
def test_real_per_tensor_config_refills_derived_storage(
    monkeypatch, upstream_source_root, ppu, backend
):
    torch = pytest.importorskip("torch")
    platform(monkeypatch, ppu)
    oracle_path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/fp8.py"
    )
    helpers = {"torch": torch}
    utils_path = upstream_source_root / "vllm/model_executor/utils.py"
    replace = function(utils_path, "replace_parameter", helpers)
    refill = function(utils_path, "refill_derived", helpers)
    backend_type = function(oracle_path, "Fp8MoeBackend", {"Enum": Enum})
    original = function(
        oracle_path,
        "make_fp8_moe_quant_config",
        {
            "Fp8MoeBackend": backend_type,
            "replace_parameter": replace,
            "refill_derived": refill,
            "fp8_w8a8_moe_quant_config": lambda **kwargs: NS(**kwargs),
        },
    )
    make_config = function(
        "vllm_sail/registry/moe_backends/fp8.py",
        "make_fp8_moe_quant_config",
        {"_upstream_quant_config": original},
    )
    layer = torch.nn.Module()
    alpha, beta = 1.7, 0.4

    def configure(multiplier):
        return make_config(
            getattr(backend_type, backend),
            torch.tensor([2.0, 3.0]) * multiplier,
            torch.tensor([5.0, 7.0]) * multiplier,
            torch.tensor([0.5, 0.25]) * multiplier,
            torch.tensor([0.125, 0.0625]) * multiplier,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            layer=layer,
        )

    first = configure(1)
    names = ("g1_alphas", "g2_alphas", "_a1_gscale", "_a2_gscale")
    held = [getattr(layer, name) for name in names]
    pointers = [value.data_ptr() for value in held]
    loader = object()
    layer.g1_alphas.weight_loader = loader
    second = configure(2)
    assert [getattr(layer, name).data_ptr() for name in names] == pointers
    assert all(
        getattr(layer, name) is value for name, value in zip(names, held, strict=False)
    )
    assert layer.g1_alphas.weight_loader is loader
    for field, value in zip(
        ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"), held, strict=False
    ):
        assert getattr(first, field) is getattr(second, field) is value
    expected = ([4.0, 3.0], [2.5, 1.75], [1.0, 2.0], [4.0, 8.0])
    for value, values in zip(held, expected, strict=False):
        assert torch.equal(value, torch.tensor(values))
    assert "_a1_gscale" not in layer._parameters
    assert "_a2_gscale" not in layer._buffers
    if ppu:
        assert second.gemm1_alpha == alpha and second.gemm1_beta == beta
