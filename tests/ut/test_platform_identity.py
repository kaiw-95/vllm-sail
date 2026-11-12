# SPDX-License-Identifier: Apache-2.0
"""PPU identity must be available before the general-plugin patch lifecycle."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


def _load_platform(monkeypatch):
    cuda_enum = object()

    class NvmlCudaPlatform:
        @classmethod
        def pre_register_and_update(cls, parser=None):
            pass

        @classmethod
        def get_hisparse_backend_cls(cls):
            return "vllm.v1.hisparse.backend.CUDAHiSparseBackend"

        def __getattr__(self, name):
            # vLLM's fallback returns None for unknown platform capabilities.
            return None

        def is_cuda(self):
            return self._enum is cuda_enum

    modules = {
        "torch": types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda value: None)
            )
        ),
        "vllm.logger": types.SimpleNamespace(init_logger=logging.getLogger),
        "vllm.platforms.cuda": types.SimpleNamespace(NvmlCudaPlatform=NvmlCudaPlatform),
        "vllm.platforms.interface": types.SimpleNamespace(
            DeviceCapability=tuple, PlatformEnum=types.SimpleNamespace(CUDA=cuda_enum)
        ),
        "vllm.v1.attention.backends.registry": types.SimpleNamespace(
            AttentionBackendEnum=object
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "vllm_sail/platform.py"
    spec = importlib.util.spec_from_file_location("_ppu_identity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, NvmlCudaPlatform, cuda_enum


def test_ppu_identity_exists_before_general_patches(monkeypatch):
    module, NvmlCudaPlatform, cuda_enum = _load_platform(monkeypatch)

    platform = module.PPUPlatform()
    assert platform.is_ppu() is True
    assert module.PPUPlatform.is_ppu() is True
    assert platform.is_cuda() is True
    assert NvmlCudaPlatform.get_hisparse_backend_cls() is not None
    assert platform.get_hisparse_backend_cls() is None
    assert module.PPUPlatform.get_hisparse_backend_cls() is None
    # Installing a predicate on the upstream base later must not change PPU
    # identity or become a prerequisite for it.
    NvmlCudaPlatform.is_ppu = lambda self: False
    assert platform.is_ppu() is True


@pytest.mark.parametrize("is_810e", [False, True])
@pytest.mark.parametrize("requested", [None, "cython", "tvm_ffi", "auto"])
def test_810e_tilelang_default_preserves_explicit_selection(
    monkeypatch, is_810e, requested
):
    import os

    module, base, _ = _load_platform(monkeypatch)
    calls = []
    monkeypatch.setattr(
        base,
        "pre_register_and_update",
        classmethod(lambda cls, parser: calls.append(parser)),
    )
    monkeypatch.setattr(
        module.PPUPlatform,
        "is_device_capability",
        classmethod(lambda cls, capability: is_810e and capability == (8, 0)),
        raising=False,
    )
    # Track cleanup even when the hook creates a previously absent variable.
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", requested or "")
    if requested is None:
        monkeypatch.delenv("TILELANG_EXECUTION_BACKEND")
    parser = object()
    module.PPUPlatform.pre_register_and_update(parser)
    module.PPUPlatform.pre_register_and_update(parser)
    expected = "cython" if is_810e and requested is None else requested
    actual = os.environ.get("TILELANG_EXECUTION_BACKEND")
    assert actual == expected
    assert calls == [parser, parser]
