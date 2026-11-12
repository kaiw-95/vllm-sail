# SPDX-License-Identifier: Apache-2.0
"""MHC must participate in graph capture on Zhenwu 810E."""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap80]


def test_mhc_graph_replay_recomputes_after_input_change(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU device")
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or not current_platform.is_device_capability(
        (8, 0)
    ):
        pytest.skip("requires Zhenwu 810E")

    # A fresh process tests the platform default before any kernel is cached.
    # Run outside the checkout to exercise the installed native wheel.
    env = dict(os.environ)
    env.pop("TILELANG_EXECUTION_BACKEND", None)
    env.pop("PYTHONPATH", None)
    env["VLLM_PLUGINS"] = "ppu"
    script = textwrap.dedent("""
        import torch
        from vllm.platforms import current_platform
        import vllm_sail

        torch.set_num_threads(4)
        current_platform.pre_register_and_update()
        vllm_sail.register_out_of_tree()
        from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang
        from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

        generator = torch.Generator().manual_seed(42)
        tokens, hidden = 33, 5120
        x = torch.randn(tokens, hidden, generator=generator).bfloat16()
        residual = x.unsqueeze(1).expand(-1, 4, -1).contiguous()
        fn = torch.randn(24, hidden, generator=generator) * 0.02
        scale = torch.tensor([0.5, 0.25, 1.0])
        base = torch.randn(24, generator=generator)
        args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
        gpu_args = tuple(a.cuda() if isinstance(a, torch.Tensor) else a for a in args)
        norm = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        x_gpu = x.cuda()

        def forward():
            return mhc_pre_delayed_tilelang(*gpu_args, x=x_gpu, norm_weight=norm)

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = forward()
        torch.cuda.synchronize()
        before = outputs[0].clone()
        gpu_args[3].add_(0.7)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(outputs[0], before), "MHC output was stale after replay"
        expected = mhc_pre_delayed_torch(
            residual, fn, scale, base + 0.7, *args[4:], x=x
        )
        for index in (0, 1, 3):
            torch.testing.assert_close(
                outputs[index].cpu(), expected[index], atol=1e-3, rtol=1e-3
            )
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
