# SPDX-License-Identifier: Apache-2.0
"""Exercise INT8 Python dispatch without importing the device kernel stack.

Compile the two production methods unchanged, with recorded backend calls in
place of torch/vLLM. These tests check routing and layouts, not device numerics.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.support.source import function


@pytest.fixture
def kernel():
    path = (
        Path(__file__).parents[2]
        / "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PPUInt8ScaledMMLinearKernel"
    )
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"apply_weights", "process_weights_after_loading"}
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    environment = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), environment)
    return environment[cls.name](), environment


@pytest.mark.parametrize(
    ("asymmetric", "deepgemm", "acext", "bias", "row_major", "backend"),
    [
        (True, True, True, True, True, "azp"),
        (False, True, True, False, True, "deepgemm"),
        (False, True, True, True, True, "acext"),
        (False, True, False, True, True, "cutlass"),
        (False, False, False, False, False, "cutlass"),
    ],
)
@pytest.mark.parametrize("triton_quant", [False, True])
def test_dispatch_preserves_bias_priority_layout_and_output_shape(
    kernel, asymmetric, deepgemm, acext, bias, row_major, backend, triton_quant
):
    instance, environment = kernel
    x = Mock(shape=(2, 3, 64), dtype="bf16")
    flat = Mock(shape=(6, 64), dtype="bf16")
    x.reshape.return_value = flat
    flat.contiguous.return_value = flat
    output = Mock(shape=(6, 128))
    calls = {
        name: Mock(return_value=output)
        for name in ("azp", "deepgemm", "acext", "cutlass")
    }
    weight = Mock()
    scale, quantized, input_scale = object(), object(), object()
    zero_point = object() if asymmetric else None
    adjustment = object() if asymmetric else None
    bias_tensor = object() if bias else None
    native_quant = Mock(return_value=(quantized, input_scale, zero_point))
    triton = Mock(return_value=(quantized, input_scale))
    environment.update(
        torch=SimpleNamespace(
            int8="int8",
            ops=SimpleNamespace(
                vllm=SimpleNamespace(
                    w8a8_int8_matmul_deepgemm=calls["deepgemm"],
                    w8a8_int8_matmul_acext=calls["acext"],
                )
            ),
        ),
        ops=SimpleNamespace(
            scaled_int8_quant=native_quant,
            cutlass_scaled_mm_azp=calls["azp"],
            cutlass_scaled_mm=calls["cutlass"],
        ),
        ppu_envs=SimpleNamespace(VLLM_SAIL_USE_TRITON_INT8_QUANT=triton_quant),
        per_token_group_quant_int8=triton,
    )
    instance._get_layer_params = lambda layer: (weight, scale, None, None, adjustment)
    instance.use_deepgemm_int8_gemm = deepgemm
    instance.use_acext_int8_gemm = acext
    instance.weight_RowMajor = row_major

    assert instance.apply_weights(object(), x, bias_tensor) is output.view.return_value

    x.reshape.assert_called_once_with(-1, 64)
    output.view.assert_called_once_with(2, 3, 128)
    for name, call in calls.items():
        assert call.call_count == (1 if name == backend else 0)
    args, kwargs = calls[backend].call_args
    assert args[0] is quantized
    assert args[1] is (
        weight.t.return_value if backend == "cutlass" and row_major else weight
    )
    assert kwargs["out_dtype"] == "bf16"
    if backend == "deepgemm":
        assert "bias" not in kwargs
        assert kwargs["scale_x"] is input_scale
        assert kwargs["scale_w"] is scale
    else:
        assert kwargs["bias"] is bias_tensor
        assert kwargs["scale_a"] is input_scale
        assert kwargs["scale_b"] is scale
    if asymmetric:
        assert kwargs["azp_adj"] is adjustment
        assert kwargs["azp"] is zero_point
    assert triton.call_count == int(triton_quant and not asymmetric)
    assert native_quant.call_count == int(not triton_quant or asymmetric)


@pytest.mark.upstream_source
@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
@pytest.mark.parametrize("channelwise", [False, True])
@pytest.mark.parametrize("static_input", [False, True])
def test_int8_initialization_uses_only_int8_registry(
    kernel, monkeypatch, upstream_source_root, cap, channelwise, static_input
):
    _, environment = kernel
    platform = SimpleNamespace(
        _enum="cuda",
        is_ppu=lambda: True,
        get_device_capability=lambda: cap,
    )
    environment["current_platform"] = platform
    Int8Kernel = environment["PPUInt8ScaledMMLinearKernel"]

    def initialize(self, config, layer_param_names):
        self.config, self.layer_param_names = config, layer_param_names

    monkeypatch.setattr(Int8Kernel, "__init__", initialize)
    Int8Kernel.is_supported = classmethod(function(
        "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py",
        "PPUInt8ScaledMMLinearKernel.is_supported", environment))
    Int8Kernel.can_implement = classmethod(lambda cls, config: (True, None))
    for name in ("PPUCutlassFP8ScaledMMLinearKernel", "PPUDeepGemmFP8ScaledMMLinearKernel", "PPUCutlassFp8BlockScaledMMKernel", "PPUDeepGemmFp8BlockScaledMMKernel"):
        environment[name] = type(name, (), {"is_supported": classmethod(lambda *a: (True, None))})
    fp8_probes = []
    for name in (
        "PPUCutlassFP8ScaledMMLinearKernel",
        "PPUDeepGemmFP8ScaledMMLinearKernel",
    ):
        probe = Mock(return_value=(True, None))
        monkeypatch.setattr(environment[name], "is_supported", classmethod(probe))
        fp8_probes.append(probe)
    ppu_module = ModuleType("vllm_sail.model_executor.kernels.linear.scaled_mm.ppu")
    ppu_module.__dict__.update(environment)
    monkeypatch.setitem(
        sys.modules, "vllm_sail.model_executor.kernels.linear.scaled_mm.ppu", ppu_module
    )
    mxfp8_module = ModuleType("vllm_sail.model_executor.kernels.linear.mxfp8")
    mxfp8_module.PPUEmulationMxfp8LinearKernel = type(
        "PPUEmulationMxfp8LinearKernel", (), {}
    )
    monkeypatch.setitem(sys.modules, mxfp8_module.__name__, mxfp8_module)
    plan = dict(
        function(
            "vllm_sail/registry/linear_kernels/__init__.py", "_selection_plan", {}
        )()
    )
    assert plan["int8"] == [Int8Kernel]
    upstream = upstream_source_root / "vllm/model_executor/kernels/linear/__init__.py"
    supported = function(
        upstream,
        "is_supported_and_can_implement_kernel",
        {
            "envs": SimpleNamespace(VLLM_DISABLED_KERNELS=()),
            "current_platform": platform,
        },
    )
    selected_quantizations = []

    def resolve(kernels, kind, *, quantization):
        selected_quantizations.append(quantization)
        return kernels

    choose = function(
        upstream,
        "choose_scaled_mm_linear_kernel",
        {
            "current_platform": platform,
            "is_supported_and_can_implement_kernel": supported,
            "_resolve_backend_kernels": resolve,
        },
    )
    init = function(
        upstream,
        "init_int8_linear_kernel",
        {
            "Int8ScaledMMLinearLayerConfig": SimpleNamespace,
            "choose_scaled_mm_linear_kernel": choose,
            "_POSSIBLE_INT8_KERNELS": {"cuda": plan["int8"]},
            "_POSSIBLE_FP8_KERNELS": {"cuda": plan["fp8"]},
            "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        },
    )
    instance = init(channelwise, static_input, True, "CompressedTensorsW8A8Int8")
    assert type(instance) is Int8Kernel
    assert vars(instance.config) == dict(
        is_channelwise=channelwise,
        is_static_input_scheme=static_input,
        input_symmetric=True,
    )
    assert instance.layer_param_names == [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    assert selected_quantizations == ["int8_w8a8"]
    for probe in fp8_probes:
        probe.assert_not_called()


@pytest.mark.parametrize("backend", [None, "deepgemm", "acext", "triton"])
def test_weight_preparation_keeps_backend_layout(kernel, backend):
    instance, environment = kernel
    weight, scale = Mock(), Mock()
    layer = SimpleNamespace(weight=weight, scale=scale, logical_widths=[128])
    parameters = []

    def replace(layer, name, value):
        parameters.append((name, value))
        setattr(layer, name, value)

    environment.update(
        torch=SimpleNamespace(
            nn=SimpleNamespace(Parameter=lambda data, **kwargs: data)
        ),
        ppu_envs=SimpleNamespace(VLLM_SAIL_DENSE_BACKEND=backend),
        current_platform=SimpleNamespace(is_device_capability=lambda cap: True),
        _get_acext_int8_gemm=lambda: object(),
        is_deep_gemm_supported=lambda: True,
        replace_parameter=replace,
    )
    instance.config = SimpleNamespace(
        is_channelwise=True, is_static_input_scheme=False, input_symmetric=True
    )
    instance.layer_param_names = (
        "weight",
        "scale",
        "input_scale",
        "input_zp",
        "azp_adj",
    )

    instance.process_weights_after_loading(layer)

    row_major = backend != "triton"
    assert instance.weight_RowMajor is row_major
    assert parameters == [
        ("weight", weight.data if row_major else weight.t.return_value.data),
        ("scale", scale.data),
    ]
    assert weight.t.call_count == int(not row_major)
    assert instance.use_acext_int8_gemm is (backend in (None, "acext"))
    assert instance.use_deepgemm_int8_gemm is (backend in (None, "deepgemm"))
