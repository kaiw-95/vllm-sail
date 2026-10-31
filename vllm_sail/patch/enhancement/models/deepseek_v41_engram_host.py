# SPDX-License-Identifier: Apache-2.0
"""Opt-in address-ordered host-table reads using vLLM's sorted gather.

SGLang 3193f83c4395f31a450c7c5364eb491a77e06389 demonstrates the scheduling
policy for host tables without ATS. Keep it explicitly configurable instead of
requiring CUDA driver Python bindings or assuming SAIL's host-page-table mode.
"""

from vllm.models.deepseek_v41.common import engram as _engram
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.patch.utils import patch

_upstream_thresholds = _engram._engram_lookup_thresholds


@patch(
    "vllm.models.deepseek_v41.common.engram",
    "_engram_lookup_thresholds",
    reason="PPU host-table gathers can reuse table-order reads to reduce translation misses.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The upstream Engram lookup exposes platform-specific scheduling thresholds.",
)
def _engram_lookup_thresholds(device):
    if current_platform.is_ppu():
        minimum = envs.VLLM_SAIL_ENGRAM_HOST_SORT_MIN_ROWS
        if minimum:
            # lookup() itself gates this to CPU offload without packed THP
            # storage and scatters each sorted row to its original token/head.
            # Keep the independent inline-prefetch policy unchanged.
            _, inline_minimum = _upstream_thresholds(device)
            return minimum, inline_minimum
    return _upstream_thresholds(device)
