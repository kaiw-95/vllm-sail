# SPDX-License-Identifier: Apache-2.0
"""SAIL MegaGate regression coverage."""

import ast
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function
from vllm_sail import envs

DEEP_GEMM = "vllm_sail/utils/deep_gemm.py"
INDEXER = "vllm_sail/attention/ops/mla_sparse.py"
RUNNER = "vllm_sail/model_executor/layers/fused_moe/mega_gate.py"
PATCH = "vllm_sail/patch/enhancement/models/deepseek_v41_mega_gate.py"


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)

    def to(self, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self


def _api_namespace(symbol):
    calls = []

    def api(*args, **kwargs):
        calls.append((args, kwargs))
        return "logits"

    return {
        "torch": NS(
            bfloat16="bf16", float32="fp32", int64="int64",
            empty=lambda shape, dtype, device: Tensor(shape, dtype),
        ),
        "_lazy_init": lambda: None,
        "_missing": lambda: pytest.fail("unexpected missing API"),
        symbol: api,
    }, calls


def test_mega_gate_preserves_vllm_result_order_and_logical_ids():
    ns, calls = _api_namespace("_bf16_mega_gate_impl")
    x, weight = Tensor((3, 256)), Tensor((128, 256))
    x.device = "cuda"
    weights, ids = function(DEEP_GEMM, "bf16_mega_gate", ns)(
        x, weight, 8, routed_scaling_factor=2.5
    )
    assert weights.dtype == "fp32" and ids.dtype == "int64"
    args, kwargs = calls[0]
    assert args == (x, weight, 8)
    assert kwargs["out"] == (ids, weights)
    assert kwargs["routed_scaling_factor"] == 2.5
    assert not kwargs["use_shared_as_routed"]
    assert "to_physical_map" not in kwargs


def test_factory_installs_once_and_preserves_other_models(modules, patch_utils_module, monkeypatch):
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return kwargs

    config = NS(model_config=NS(hf_config=NS(model_type="deepseek_v41")))
    modules("vllm.config", get_current_vllm_config=lambda: config)
    modules("vllm.models.deepseek_v4.nvidia.model", FusedMoEFactory=factory)
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: True))
    runner = type("Runner", (), {})
    modules("vllm_sail.model_executor.layers.fused_moe.mega_gate", PPUMegaGateMoERunner=runner)
    monkeypatch.setenv("VLLM_SAIL_DEEPGEMM_MEGA_GATE", "1")
    modules("vllm_sail", envs=envs)
    patch = load_patch(PATCH)
    target = "vllm.models.deepseek_v4.nvidia.model.FusedMoEFactory"
    assert patch_utils_module.original_of(patch.fused_moe_factory, target) is factory
    assert patch.fused_moe_factory()["runner_cls"] is runner
    config.model_config.hf_config.model_type = "deepseek_v4"
    assert patch.fused_moe_factory() == {}
    with pytest.raises(RuntimeError, match="already"):
        load_patch(PATCH)


@pytest.mark.upstream_source
def test_routing_postprocess_order_matches_the_pin(upstream_source_root):
    base = definition(upstream_source_root / (
        "vllm/model_executor/layers/fused_moe/router/base_router.py"
    ), "BaseRouter._select_experts")
    expected = ("capture_fn", "_apply_eplb_mapping", "_convert_indices_dtype")

    def calls(node):
        return [n.func.attr for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]

    # lineno order, unlike AST traversal order, includes calls in if bodies.
    def ordered(node):
        return [n.func.attr for n in sorted(ast.walk(node), key=lambda n: getattr(n, "lineno", 0))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in expected]

    assert ordered(base) == list(expected)
    assert ordered(definition(ROOT / RUNNER, "_postprocess_routing")) == list(expected)
    outer = definition(upstream_source_root / (
        "vllm/model_executor/layers/fused_moe/router/fused_moe_router.py"
    ), "FusedMoERouter.select_experts")
    assert "_select_experts" in calls(outer)
    assert any(isinstance(n, ast.Attribute) and n.attr == "_routing_replay_out"
               for n in ast.walk(outer))
