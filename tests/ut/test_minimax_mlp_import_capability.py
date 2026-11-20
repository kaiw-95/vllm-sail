# SPDX-License-Identifier: Apache-2.0
"""Run the whole upstream model's MLP across the NVIDIA import boundary."""

import ast
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function


def load_model(modules, monkeypatch, source, ppu):
    """Configure only external imports; execute the complete real model module."""
    path = source / "vllm/models/minimax_m3/nvidia/model.py"

    def unavailable(self, *args, **kwargs):
        raise AssertionError("Unconfigured external dependency was invoked")

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith(
            ("vllm", "transformers")
        ):
            attrs = {
                alias.name: type(alias.name, (), {"__init__": unavailable})
                for alias in node.names
            }
            modules(node.module, **attrs)
    modules("vllm", _custom_ops=modules("vllm._custom_ops"))
    modules("vllm.envs")

    class Module:
        def __init__(self):
            self.initialized = True

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    tensor = type("Tensor", (), {})
    nn = modules("torch.nn", Module=Module, Parameter=tensor, ModuleList=list)
    modules(
        "torch",
        nn=nn,
        Tensor=tensor,
        dtype=type("dtype", (), {}),
        device=type("device", (), {}),
    )
    modules("vllm.logger", init_logger=lambda name: NS())
    modules(
        "vllm.multimodal",
        MULTIMODAL_REGISTRY=NS(
            register_processor=lambda processor, *, info, dummy_inputs: lambda cls: cls
        ),
    )
    modules(
        "vllm.model_executor.models.utils",
        WeightsMapper=lambda *, orig_to_new_prefix, orig_to_new_substr: NS(),
    )
    modules(
        "vllm.compilation.breakable_cudagraph",
        eager_break_during_capture=lambda fn: fn,
    )
    modules(
        "vllm.platforms", current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
    )
    name = "vllm.models.minimax_m3.nvidia.model"
    modules(name)
    spec = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], "model", provider)
    spec.loader.exec_module(provider)
    return provider


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("threshold", [16, 64])
@pytest.mark.parametrize("autotune", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("run_gemm_rs", [False, True])
def test_registered_mlp_keeps_constructor_and_forward_contract(
    modules,
    monkeypatch,
    upstream_source_root,
    ppu,
    fuse,
    threshold,
    autotune,
    sequence_parallel,
    run_gemm_rs,
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    calls = []
    quant = object()
    gate_up, output, reduced = object(), object(), object()
    activated = NS(shape=(64, 16))

    def all_gather(x):
        calls.append(("all-gather", x))
        return x

    def reduce_scatter(x):
        assert x is output
        calls.append(("reduce-scatter", x))
        return reduced

    provider.sp_all_gather = all_gather
    provider.sp_reduce_scatter = reduce_scatter
    gemm_rs = NS(can_run=lambda linear: True)
    modules(
        "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar",
        get_gemm_rs_ar=lambda: gemm_rs,
    )

    class Linear:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            calls.append(("linear", self))

        def __call__(self, x):
            calls.append(("linear-forward", self, x))
            return (
                gate_up
                if len(self.args) == 2 and isinstance(self.args[1], list)
                else output
            ), None

    class Activation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, x):
            assert x is gate_up
            calls.append(("act", x))
            return activated

    provider.MergedColumnParallelLinear = provider.RowParallelLinear = Linear
    provider.SiluAndMulWithClamp = Activation
    provider.envs = NS(VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=threshold)
    provider.is_flashinfer_autotuning = lambda: autotune
    config = NS(
        hidden_size=8,
        hidden_act="swigluoai",
        swiglu_limit=7,
        swiglu_alpha=1.702,
        swiglu_beta=1,
    )
    name = "vllm.models.minimax_m3.nvidia.ops"

    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name or fullname.startswith(name + "."):
                raise ModuleNotFoundError("NVIDIA-only dense import", name="quack")
            return None

    def capable(*args):
        calls.append(("capability", args))
        return fuse

    def fused(x, linear, act):
        calls.append(("fused-forward", x, linear, act))
        return activated

    if ppu:
        monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
        with pytest.raises(ModuleNotFoundError, match="NVIDIA-only"):
            provider.MiniMaxM3MLP(config, 16)
        calls.clear()
    else:
        dense = modules(
            name + ".dense_gemm1_mxfp8",
            can_fuse_gemm1=capable,
            MIN_TOKENS=32,
            fused_gemm1=fused,
        )
        modules(name, dense_gemm1_mxfp8=dense)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    disable = threshold == 64
    mlp = provider.MiniMaxM3MLP(
        config,
        16,
        quant,
        not disable,
        disable,
        use_sequence_parallel=sequence_parallel,
        run_gemm_rs=run_gemm_rs,
        prefix="layers.0.mlp",
    )
    assert mlp.use_sequence_parallel is sequence_parallel
    assert mlp.gemm_rs is (
        gemm_rs if sequence_parallel and run_gemm_rs and not ppu else None
    )
    assert mlp.initialized
    assert mlp.gate_up_proj.args == (8, [16, 16])
    assert mlp.gate_up_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        disable_tp=disable,
        prefix="layers.0.mlp.gate_up_proj",
    )
    assert mlp.down_proj.args == (16, 8)
    assert mlp.down_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        reduce_results=not disable,
        disable_tp=disable,
        prefix="layers.0.mlp.down_proj",
    )
    assert mlp.act_fn.kwargs == dict(swiglu_limit=7, alpha=1.702, beta=1)
    assert mlp.fused_gemm1_min_tokens == (
        None if ppu or not fuse else max(32, threshold)
    )
    assert len([c for c in calls if c[0] == "capability"]) == (0 if ppu else 1)
    calls.clear()
    x = NS(shape=(64, 8))
    assert mlp(x) is (reduced if sequence_parallel else output)
    expected = []
    if sequence_parallel:
        expected.append(("all-gather", x))
    if not ppu and fuse and threshold < 64 and not autotune:
        expected.extend(
            [
                ("fused-forward", x, mlp.gate_up_proj, mlp.act_fn),
                ("linear-forward", mlp.down_proj, activated),
            ]
        )
    else:
        expected.extend(
            [
                ("linear-forward", mlp.gate_up_proj, x),
                ("act", gate_up),
                ("linear-forward", mlp.down_proj, activated),
            ]
        )
    if sequence_parallel:
        expected.append(("reduce-scatter", output))
    assert calls == expected


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
def test_registered_mlp_retains_activation_validation(
    modules, monkeypatch, upstream_source_root, ppu
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    provider.MergedColumnParallelLinear = provider.RowParallelLinear = lambda *a, **k: (
        object()
    )
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    with pytest.raises(ValueError, match="Only swigluoai is supported"):
        provider.MiniMaxM3MLP(NS(hidden_size=8, hidden_act="unsupported"), 16)


@pytest.mark.upstream_source
def test_complete_mlp_constructor_preserves_upstream_body(upstream_source_root):
    local = definition(ROOT / "vllm_sail/models/minimax_m3_mlp.py", "init_mlp")
    upstream = definition(
        upstream_source_root / "vllm/models/minimax_m3/nvidia/model.py",
        "MiniMaxM3MLP.__init__",
    )
    local.name = upstream.name
    # 显式 super 是独立绑定 body 的要求，其余只还原两个 PPU 导入门控。
    assert ast.unparse(local.body[0]) == "super(MiniMaxM3MLP, self).__init__()"
    local.body[0] = ast.parse("super().__init__()").body[0]
    regions = [node for node in local.body if isinstance(node, ast.If)]
    assert len(regions) == 3
    import_guard, capability_guard = regions[1:]
    assert ast.unparse(import_guard.test) == "not current_platform.is_ppu()"
    assert len(import_guard.body) == 1 and not import_guard.orelse
    local.body[local.body.index(import_guard)] = import_guard.body[0]
    assert isinstance(capability_guard.test, ast.BoolOp)
    assert isinstance(capability_guard.test.op, ast.And)
    assert len(capability_guard.test.values) == 2
    assert (
        ast.unparse(capability_guard.test.values[0]) == "not current_platform.is_ppu()"
    )
    capability_guard.test = capability_guard.test.values[1]
    assert ast.dump(local) == ast.dump(upstream)


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tokens", [512, 2048])
def test_gemm_rs_initialization_keeps_collective_fallback(
    modules, monkeypatch, upstream_source_root, ppu, sequence_parallel, enabled, tokens
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    calls = []
    provider.envs = NS(VLLM_ENABLE_GEMM_RS=enabled)
    provider.logger = NS(info_once=lambda *args: None)
    config = NS(
        scheduler_config=NS(max_num_batched_tokens=tokens),
        model_config=NS(hf_text_config=NS(hidden_size=3584)),
    )

    def initialize(config, **kwargs):
        calls.append(kwargs)
        return True

    gemm_module = "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar"
    if ppu:

        class Forbidden(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == gemm_module:
                    raise ModuleNotFoundError("NVIDIA-only GEMM-RS", name="cutlass")
                return None

        monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
        if sequence_parallel and enabled and tokens >= 1024:
            with pytest.raises(ModuleNotFoundError, match="NVIDIA-only GEMM-RS"):
                provider._maybe_init_gemm_rs(config, sequence_parallel)
    else:
        modules(gemm_module, maybe_init_gemm_rs_ar=initialize)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    eligible = not ppu and sequence_parallel and enabled and tokens >= 1024
    assert provider._maybe_init_gemm_rs(config, sequence_parallel) is eligible
    assert calls == ([dict(N=3584, all_reduce=False)] if eligible else [])


@pytest.mark.upstream_source
@pytest.mark.parametrize("residual_present", [False, True])
@pytest.mark.parametrize("dimensions", [2, 3])
def test_ppu_norm_retains_unquantized_identity_without_nvidia_import(
    modules, monkeypatch, upstream_source_root, residual_present, dimensions
):
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    calls = []
    x = NS(dim=lambda: dimensions)
    residual = NS(updated=False) if residual_present else None
    output, weight = object(), object()
    instance = object.__new__(provider.MiniMAXGemmaRMSNorm)
    instance.weight, instance.variance_epsilon = weight, 1e-6

    def plain(x_arg, w_arg, eps):
        assert (x_arg, w_arg, eps) == (x, weight, 1e-6)
        calls.append("plain")
        return output

    def fused(x_arg, residual_arg, w_arg, eps):
        assert x_arg is x and residual_arg is residual
        assert w_arg is weight and eps == 1e-6
        residual_arg.updated = True
        calls.append("fused")

    modules("flashinfer.norm", gemma_rmsnorm=plain, gemma_fused_add_rmsnorm=fused)

    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("vllm.models.minimax_m3.nvidia.ops"):
                raise ModuleNotFoundError("NVIDIA-only norm import", name="quack")
            return None

    monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
    if dimensions == 2:
        with pytest.raises(ModuleNotFoundError, match="NVIDIA-only norm"):
            instance.forward(x, residual)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    result = instance.forward(x, residual)
    if residual_present:
        assert result[0] is x and result[1] is residual and residual.updated
    else:
        assert result is output
    assert calls == (["fused"] if residual_present else ["plain"])
    with pytest.raises(ValueError, match="128x4 MXFP8"):
        instance.forward(x, residual, mxfp8=True)
    assert calls == (["fused"] if residual_present else ["plain"])


@pytest.mark.upstream_source
def test_non_ppu_norm_retains_real_quantized_provider(
    modules, monkeypatch, upstream_source_root
):
    provider = load_model(modules, monkeypatch, upstream_source_root, False)
    instance = object.__new__(provider.MiniMAXGemmaRMSNorm)
    instance.weight, instance.variance_epsilon = object(), 1e-6
    x = NS(dim=lambda: 2, dtype=object(), shape=(3, 8))
    data, scale, residual = object(), object(), object()
    calls = []

    def norm(x_arg, w_arg, eps, **kwargs):
        calls.append((x_arg, w_arg, eps, kwargs))
        return data, scale

    modules("vllm.models.minimax_m3.nvidia.ops.norm_mxfp8", gemma_rmsnorm_mxfp8=norm)
    provider.QuantizedActivation = lambda **kwargs: NS(**kwargs)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    result, actual_residual = instance.forward(x, residual, mxfp8=True)
    assert actual_residual is residual
    assert result.data is data and result.scale is scale
    assert result.orig_shape == x.shape and result.orig_dtype is x.dtype
    assert result.quant_key is provider.kMxfp8Dynamic
    assert calls == [(x, instance.weight, 1e-6, dict(residual=residual, mxfp8=True))]


@pytest.mark.upstream_source
def test_ppu_mxfp8_emulation_does_not_request_swizzled_norm(
    modules, monkeypatch, upstream_source_root
):
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    for path, name in [
        (
            ROOT / "vllm_sail/model_executor/kernels/linear/mxfp8.py",
            "PPUEmulationMxfp8LinearKernel",
        ),
        (
            upstream_source_root
            / "vllm/model_executor/kernels/linear/mxfp8/emulation.py",
            "EmulationMxfp8LinearKernel",
        ),
    ]:
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name == "input_quant_key"
            for node in definition(path, name).body
        )
    input_key = function(
        upstream_source_root
        / "vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py",
        "Mxfp8LinearKernel.input_quant_key",
        {},
    )
    linear = NS(quant_method=NS(kernel=object()))
    provider.get_input_quant_key = lambda layer: input_key(layer.quant_method.kernel)
    assert provider._consumes_mxfp8_128x4(linear) is False
