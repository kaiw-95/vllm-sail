# SPDX-License-Identifier: Apache-2.0
"""Dispatch V4/V4.1 legacy packed-cache gathers to the optional vector op."""

import sys

from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v4.ops.dequant import try_vector_gather
from vllm_sail.patch.utils import PATCH_MARKER, patch

_V41 = "vllm.models.deepseek_v41.common.ops.cache_utils"
_NAME = "dequantize_and_gather_k_cache"
_REASON = "Warp-per-token PPU vector reads reduce packed K-cache gather requests while retaining the existing fallback for other records and layouts."
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = "The packed MLA cache gather supports out-of-tree kernel registration."


@patch(
    _V41, _NAME, reason=_REASON, affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN
)
def gather_v41(
    out,
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
    use_fnuz=False,
    max_gather_len=None,
):
    if current_platform.is_ppu() and try_vector_gather(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz,
        max_gather_len,
    ):
        return None
    return getattr(gather_v41, PATCH_MARKER)[f"{_V41}.{_NAME}"](
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=use_fnuz,
        max_gather_len=max_gather_len,
    )


for provider, replacement, consumers in (
    (
        _V41,
        gather_v41,
        (
            "vllm.models.deepseek_v41.common.ops",
            "vllm.models.deepseek_v41.nvidia.flashmla",
        ),
    ),
):
    original = getattr(replacement, PATCH_MARKER)[f"{provider}.{_NAME}"]
    for name in consumers:
        consumer = sys.modules.get(name)
        if consumer is not None and getattr(consumer, _NAME, None) is original:
            patch(
                name,
                _NAME,
                reason=_REASON,
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
            )(replacement)
