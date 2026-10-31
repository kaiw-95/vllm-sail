# SPDX-License-Identifier: Apache-2.0
"""Connect the V4.1 sparse indexer to SAIL metadata, logits and BF16 top-k."""

from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.sparse_indexer import (
    PPUSparseIndexerMetadataBuilder,
    PPUSparseMQAIndexer,
    sparse_decode,
    sparse_prefill,
)
from vllm_sail.patch.utils import PATCH_MARKER, patch

_LAYER = "vllm.model_executor.layers.sparse_mqa_indexer"
_BACKEND = "vllm.v1.attention.backends.mla.sparse_indexer"
_DENSE = "vllm.v1.attention.backends.mla.indexer"
_REASON = "SAIL MXFP4 sparse logits use PPU schedule metadata, fixed-length flattened rows and native BF16 top-k instead of SM100 DeepSelect."
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = "The V4.1 sparse indexer exposes a platform backend registry for metadata, logits and selection."


@patch(
    _BACKEND,
    "DeepseekV41SparseIndexerBackend.get_builder_cls",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def get_builder_cls():
    if current_platform.is_ppu():
        return PPUSparseIndexerMetadataBuilder
    original = getattr(get_builder_cls, PATCH_MARKER)[
        f"{_BACKEND}.DeepseekV41SparseIndexerBackend.get_builder_cls"
    ]
    return original()


@patch(
    _DENSE,
    "_use_flattening",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def use_flattening(vllm_config):
    if current_platform.is_ppu() and vllm_config.attention_config.indexer_sparse_logits:
        return True
    return getattr(use_flattening, PATCH_MARKER)[f"{_DENSE}._use_flattening"](
        vllm_config
    )


@patch(
    _LAYER,
    "sparse_mqa_logits_prefill_chunk",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def prefill(*args, **kwargs):
    if current_platform.is_ppu():
        return sparse_prefill(*args, **kwargs)
    return getattr(prefill, PATCH_MARKER)[f"{_LAYER}.sparse_mqa_logits_prefill_chunk"](
        *args, **kwargs
    )


@patch(
    _LAYER,
    "sparse_mqa_logits_paged_decode",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def decode(*args, **kwargs):
    if current_platform.is_ppu():
        return sparse_decode(*args, **kwargs)
    return getattr(decode, PATCH_MARKER)[f"{_LAYER}.sparse_mqa_logits_paged_decode"](
        *args, **kwargs
    )


patch(
    "vllm.models.deepseek_v41.attention",
    "DeepseekV4Indexer.mqa_cls",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)(PPUSparseMQAIndexer)
