# SPDX-License-Identifier: Apache-2.0
"""Exercise the complete upstream warmup and installed PPU import boundary."""

import contextlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import function
from tests.ut.test_ppu_kernel_capabilities import complete_warmup_contract


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("preloaded", [False, True])
def test_real_warmup_keeps_applicable_work_and_guards_msa(
    modules, monkeypatch, upstream_source_root, ppu, local, jit, preloaded
):
    calls = []

    def unavailable(*args, **kwargs):
        raise AssertionError("Unconfigured external warmup was called")

    imports = {
        "attention_warmup": ["mixed_batch_attention_warmup"],
        "b12x_warmup": ["b12x_warmup"],
        "cutedsl_warmup": ["cutedsl_warmup"],
        "deep_gemm_warmup": ["deep_gemm_warmup"],
        "flashinfer_autotune_cache": [
            "resolve_flashinfer_autotune_v2_root",
            "resolve_rank_flashinfer_autotune_file",
            "use_flashinfer_autotune_v2",
        ],
        "flashinfer_sparse_mla_warmup": [
            "autotune_hisparse_flashinfer_attention",
            "deepseek_v4_sparse_mla_attention_warmup",
            "flashinfer_sparse_mla_decode_autotune_warmup",
        ],
        "kimi_k3_triton_warmup": ["kimi_k3_triton_warmup"],
        "mamba_triton_warmup": ["mamba_triton_warmup"],
        "qwen4_exp_qsa_warmup": ["qwen4_exp_qsa_triton_warmup"],
        "qwen_triton_warmup": ["qwen_triton_warmup"],
        "qwen_vl_triton_warmup": ["qwen_vl_triton_warmup"],
        "replayssm_warmup": ["replayssm_autotune_warmup"],
        "watermark_sample_warmup": ["watermark_sample_warmup"],
    }
    for name, names in imports.items():
        modules(
            "vllm.model_executor.warmup." + name, **dict.fromkeys(names, unavailable)
        )
    logger = NS(
        info=lambda *a: None,
        info_once=lambda *a: None,
        warning=lambda *a: None,
        exception=lambda *a: None,
    )
    modules("vllm.logger", init_logger=lambda _: logger)
    modules("torch", nn=NS(Module=object), Tensor=object)
    envs = modules(
        "vllm.envs",
        VLLM_DEEP_GEMM_WARMUP="full",
        VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=False,
    )
    modules("vllm.utils.deep_gemm", is_deep_gemm_supported=lambda: False)
    modules("vllm.utils.flashinfer", has_flashinfer=lambda: False)
    platform = NS(is_ppu=lambda: ppu, has_device_capability=lambda cap: cap <= 89)
    modules("vllm.platforms", current_platform=platform)
    name = "vllm.model_executor.warmup.kernel_warmup"
    modules(name)
    spec = importlib.util.spec_from_file_location(
        name, upstream_source_root / "vllm/model_executor/warmup/kernel_warmup.py"
    )
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    spec.loader.exec_module(provider)
    original = provider.kernel_warmup

    class Kernel:
        def compile_many(self, keys):
            assert keys == [((), {})]
            calls.append("triton")

    registry = NS(
        _registrations={Kernel(): [((), {})]}, activate=lambda: contextlib.nullcontext()
    )
    worker = NS(
        model_runner=NS(
            jit_warmup_registry=registry, is_last_pp_rank=True, is_pooling_model=False
        ),
        vllm_config=NS(
            kernel_config=NS(enable_jit_warmup=jit, enable_cutedsl_warmup=True)
        ),
        get_model=lambda: "model",
        scheduler_config=NS(max_num_batched_tokens=64),
    )
    complete_warmup_contract(modules, provider, worker, platform)
    provider.envs = envs

    def registry_warmup():
        calls.append("registry")
        for kernel, keys in registry._registrations.items():
            kernel.compile_many(keys)

    registry.warmup = registry_warmup
    for callback, label in [
        ("qwen_triton_warmup", "qwen"),
        ("qwen_vl_triton_warmup", "vl"),
        ("mamba_triton_warmup", "mamba"),
        ("kimi_k3_triton_warmup", "kimi"),
        ("watermark_sample_warmup", "watermark"),
        ("qwen4_exp_qsa_triton_warmup", "qsa"),
        ("_warmup_gemm_rs_ar", "gemm_rs_ar"),
        ("flashinfer_sparse_mla_decode_autotune_warmup", "sparse_decode"),
        ("deepseek_v4_sparse_mla_attention_warmup", "dsv4"),
        ("b12x_warmup", "b12x"),
    ]:

        def spy(*args, _label=label):
            calls.append(_label)

        setattr(provider, callback, spy)
    modules(
        "vllm.model_executor.warmup.step5_dsa_warmup",
        step5_dsa_warmup=lambda worker: calls.append("step5"),
    )
    provider._warmup_skinny_gemm = lambda model: calls.append("skinny")
    provider.cutedsl_warmup = lambda: calls.append("legacy")
    modules(
        "vllm.model_executor.warmup.cutedsl_warmup",
        cutedsl_warmup=provider.cutedsl_warmup,
    )
    modules(
        "vllm.model_executor.warmup.jit_warmup_cutedsl_helper",
        VllmCuTeDSLJitKernel=type("CuTeKernel", (), {}),
    )
    modules(
        "vllm.v1.sample.ops.topk_topp_sampler",
        register_top_k_top_p_warmups=lambda: calls.append("sampler"),
    )
    modules(
        "vllm_sail.envs",
        VLLM_SAIL_MOE_BACKEND="",
        VLLM_SAIL_DENSE_BACKEND="",
        VLLM_SAIL_DENSE_BF16_DEEPGEMM=False,
    )
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    modules(
        "vllm_sail.model_executor.warmup.deep_gemm_warmup",
        deep_gemm_warmup=lambda model, tokens: calls.append(
            ("ppu_deepgemm", model, tokens)
        ),
    )
    msa = "vllm.model_executor.warmup.minimax_m3_msa_warmup"

    class ForbiddenMSA(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == msa:
                calls.append("msa-import")
                raise ModuleNotFoundError("NVIDIA-only warmup dependency", name="quack")
            return None

    if ppu:
        monkeypatch.delitem(sys.modules, msa, raising=False)
        monkeypatch.setattr(sys, "meta_path", [ForbiddenMSA(), *sys.meta_path])
        if not local:
            with pytest.raises(ModuleNotFoundError, match="NVIDIA-only"):
                original(worker, process_local_only=local)
            assert calls[-1] == "msa-import"
            calls.clear()
    else:
        modules(msa, minimax_m3_msa_warmup=lambda worker: calls.append("msa"))
    consumer_name = "vllm.v1.worker.gpu_worker"
    if preloaded:
        consumer = modules(consumer_name, kernel_warmup=original)
    load_patch("vllm_sail/patch/enhancement/kernel_warmup.py")
    if not preloaded:
        consumer = modules(consumer_name, kernel_warmup=provider.kernel_warmup)
    assert consumer.kernel_warmup is provider.kernel_warmup
    assert consumer.kernel_warmup(worker, process_local_only=local) is None
    expected = ["sampler"] if ppu else []
    if jit:
        expected += ["registry", "triton"]
    expected += ["qwen", "vl", "mamba"]
    if jit:
        expected += ["kimi", "watermark", "qsa"]
    expected += ["gemm_rs_ar"]
    if not ppu:
        expected += ["skinny", "legacy"]
    if not local:
        expected += ["sparse_decode", "dsv4", "step5", "b12x"]
        expected += [("ppu_deepgemm", "model", 64)] if ppu else ["msa"]
    assert calls == expected
    assert worker.vllm_config.kernel_config.enable_jit_warmup is jit
    assert worker.vllm_config.kernel_config.enable_cutedsl_warmup is True
    if ppu and not local:
        no_ppu_gemm = expected[:-1]
        envs.VLLM_DEEP_GEMM_WARMUP = "skip"
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == no_ppu_gemm
        envs.VLLM_DEEP_GEMM_WARMUP = "full"
        support = sys.modules["vllm_sail.utils.deep_gemm"]
        support.is_deep_gemm_supported = lambda: False
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == no_ppu_gemm
        support.is_deep_gemm_supported = lambda: True
        selectors = sys.modules["vllm_sail.envs"]
        selectors.VLLM_SAIL_MOE_BACKEND = "triton"
        selectors.VLLM_SAIL_DENSE_BACKEND = "triton"
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == no_ppu_gemm
        selectors.VLLM_SAIL_DENSE_BF16_DEEPGEMM = True
        calls.clear()
        consumer.kernel_warmup(worker)
        assert calls == expected
    elif not ppu and not local:
        error = RuntimeError("MSA warmup failed")

        def failed_msa(worker):
            raise error

        sys.modules[msa].minimax_m3_msa_warmup = failed_msa
        with pytest.raises(RuntimeError) as raised:
            consumer.kernel_warmup(worker)
        assert raised.value is error


@pytest.mark.upstream_source
@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize(
    "backends", [[], ["OTHER"], ["FLASHINFER", "OTHER"], ["FLASHINFER"]]
)
@pytest.mark.parametrize("remote_eligible", [0, 1])
@pytest.mark.parametrize("pooling", [False, True])
def test_flashinfer_warmup_coordinates_all_pp_ranks(
    modules, upstream_source_root, v2, backends, remote_eligible, pooling
):
    """使用真实 helper 验证 V2 的 any 和旧 runner 的 all 集体决策。"""
    calls = []
    collective_inputs = []

    class Eligibility:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    def tensor(values, *, dtype, device):
        assert dtype == "int32" and device == "cpu"
        return Eligibility(values[0])

    def reduce(eligibility):
        collective_inputs.append(eligibility.item())
        eligibility.value += remote_eligible

    modules(
        "vllm.distributed.parallel_state",
        get_pp_group=lambda: NS(world_size=2, cpu_all_reduce_=reduce),
    )
    warmup = function(
        upstream_source_root / "vllm/model_executor/warmup/kernel_warmup.py",
        "_warmup_flashinfer_attention",
        dict(
            torch=NS(tensor=tensor, int32="int32"),
            logger=NS(info_once=lambda *args: None),
            mixed_batch_attention_warmup=lambda worker: calls.append(worker),
        ),
    )
    groups = [NS(backend=NS(get_name=lambda name=name: name)) for name in backends]
    worker = NS(
        use_v2_model_runner=v2,
        model_runner=NS(
            is_pooling_model=pooling, attn_groups=[groups] if groups else []
        ),
    )
    warmup(worker)
    eligible = (
        not pooling
        and bool(backends)
        and (any if v2 else all)(name == "FLASHINFER" for name in backends)
    )
    assert collective_inputs == [int(eligible)]
    count = int(eligible) + remote_eligible
    assert calls == ([worker] if (count > 0 if v2 else count == 2) else [])
