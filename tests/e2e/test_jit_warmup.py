# SPDX-License-Identifier: Apache-2.0
"""Real registry/Triton checks for the 810E warmup exception boundary."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

pytestmark = [pytest.mark.ppu, pytest.mark.cap80]


@triton.jit
def copy_kernel(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets))


@triton.jit
def broken_kernel(X, Y, BLOCK: tl.constexpr):
    tl.static_assert(BLOCK == 999, "warmup-regression-sentinel")
    offsets = tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets))


@pytest.fixture
def runtime():
    from vllm.platforms import current_platform

    import vllm_sail

    vllm_sail.register_out_of_tree()
    assert current_platform.is_ppu()
    assert torch.cuda.get_device_capability() == (8, 0)


def make_registry(kernel, dtype):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        VllmTritonJitKernel,
    )

    config = object()

    class Owner(VllmTritonJitKernel):
        def get_warmup_keys(self, *, vllm_config):
            assert vllm_config is config
            return [16, 32]

        def warmup_inputs(self, key):
            return dict(
                X=TritonWarmupTensor(dtype),
                Y=TritonWarmupTensor(torch.bfloat16),
                BLOCK=key,
                grid=(1,),
            )

        def compile(self, key):
            self.kernel.warmup(**self.warmup_inputs(key))

    owner = Owner()
    owner.kernel = kernel
    registry = JitWarmupRegistry(config)
    with registry.activate():
        owner.register_warmup()
    worker = SimpleNamespace(model_runner=SimpleNamespace(jit_warmup_registry=registry))
    return worker, registry, owner


def test_signature_fp8_async_warmup_is_local_to_registry(runtime, monkeypatch):
    from triton.compiler.errors import CompilationError

    from vllm_sail.patch.enhancement.kernel_warmup import (
        _exclude_sm80_fp8_registrations,
    )

    monkeypatch.setenv("VLLM_TRITON_JIT_WARMUP_NUM_THREADS", "2")
    worker, registry, fp8 = make_registry(copy_kernel, torch.float8_e4m3fn)
    original_compile_many = fp8.compile_many
    _, supported_registry, supported = make_registry(copy_kernel, torch.bfloat16)
    # Exercise explicit registration alongside implicit config injection.
    with registry.activate():
        supported.register_warmup(vllm_config=supported_registry.vllm_config)

    for _ in range(2):
        _exclude_sm80_fp8_registrations(worker)
        registry.warmup()
        with registry.activate():
            fp8.register_warmup()
        assert len(registry) == 2
    assert fp8.supports_parallel_compile
    assert fp8.compile_many == original_compile_many

    x = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(x)
    copy_kernel[(1,)](x, output, BLOCK=16)
    torch.testing.assert_close(output, x)
    # Optional warmup tolerance must not modify the runtime owner itself.
    with pytest.raises(CompilationError, match="warmup-regression-sentinel"):
        broken_kernel[(1,)](x, output, BLOCK=16)


def test_unrelated_async_compiler_error_is_not_suppressed(runtime, monkeypatch):
    from triton.compiler.errors import CompilationError

    from vllm_sail.patch.enhancement.kernel_warmup import (
        _exclude_sm80_fp8_registrations,
    )

    monkeypatch.setenv("VLLM_TRITON_JIT_WARMUP_NUM_THREADS", "2")
    worker, registry, _ = make_registry(broken_kernel, torch.bfloat16)
    _exclude_sm80_fp8_registrations(worker)
    with pytest.raises(CompilationError, match="warmup-regression-sentinel"):
        registry.warmup()
