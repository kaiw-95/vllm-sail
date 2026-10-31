# SPDX-License-Identifier: Apache-2.0
"""Keep metadata and column-split selection aligned for opt-in M890 C128."""

from vllm.models.deepseek_v4 import compressor as _compressor
from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache as _ops
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v4.compressor import enabled
from vllm_sail.patch.utils import patch

_upstream_preference = _compressor._prefer_two_stage_compressor
_upstream_splits = _ops._pick_compress_num_splits


@patch(
    "vllm.models.deepseek_v4.compressor",
    "_prefer_two_stage_compressor",
    reason="Opt-in M890 C128 compression needs decode/prefill metadata for its two-stage launcher.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="Upstream compressor metadata and construction accept a PPU split-kernel policy.",
)
def _prefer_two_stage_compressor():
    if current_platform.is_ppu():
        return enabled()
    return _upstream_preference()


@patch(
    "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache",
    "_pick_compress_num_splits",
    reason="M890 C128 uses eight 64-column tiles without the ROCm device-zero SM-count heuristic.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The upstream split-compressor selector accepts per-platform tile policies.",
)
def _pick_compress_num_splits(num_actual, compress_ratio, head_dim):
    if enabled() and compress_ratio == 128 and head_dim == 512:
        return 8
    return _upstream_splits(num_actual, compress_ratio, head_dim)
