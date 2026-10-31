# SPDX-License-Identifier: Apache-2.0
"""Drive real Engram scheduling policy without a driver or optional packages."""

from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


@pytest.mark.parametrize(
    "ppu,minimum,expected",
    [(True, 0, None), (True, 1, 1), (True, 32768, 32768), (False, 1, None)],
)
def test_host_sort_is_explicit_and_preserves_inline_policy(
    modules, ppu, minimum, expected
):
    calls = []
    device = object()

    def original(actual_device):
        assert actual_device is device
        calls.append(actual_device)
        return None, 8192

    provider = modules(
        "vllm.models.deepseek_v41.common.engram",
        _engram_lookup_thresholds=original,
    )
    settings = modules("vllm_sail.envs", VLLM_SAIL_ENGRAM_HOST_SORT_MIN_ROWS=minimum)
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    harness.load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_engram_host.py")
    assert provider._engram_lookup_thresholds(device) == (expected, 8192)
    assert calls == [device]
    settings.VLLM_SAIL_ENGRAM_HOST_SORT_MIN_ROWS = 0
    assert provider._engram_lookup_thresholds(device) == (None, 8192)
