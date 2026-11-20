# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual registered INT8 guard and unchanged fp8 delegation."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def kpool(modules):
    def load(*, fp8=False):
        calls = []
        modules(
            "torch",
            float32="fp32",
            ops=NS(
                vllm=NS(
                    ppu_sparse_attn_indexer_kpool=lambda *args: calls.append(args)
                    or "int8"
                )
            ),
        )
        modules(
            "vllm.platforms",
            current_platform=NS(is_ppu=lambda: True, supports_fp8=lambda: fp8),
        )
        modules("vllm.utils.torch_utils", _encode_layer_name=lambda name: name)
        modules("vllm_sail.ops")

        class Indexer:
            def forward_native(self, *args, **kwargs):
                raise AssertionError("INT8 redirect must reach the registered PPU path")

            def forward_cuda(
                self,
                hidden_states,
                q_quant,
                k,
                weights,
                *,
                gate_score=None,
                compress_ape=None,
                index_kpool=1,
                positions=None,
                topk_backend="auto",
            ):
                return "upstream-fp8", self.dcp_world_size, topk_backend

        modules(
            "vllm.models.glm5next.nvidia.sparse_indexer",
            SparseAttnIndexerKpool=Indexer,
            fp8_fp4_mqa_logits=lambda *args, **kwargs: "prefill",
            fp8_fp4_paged_mqa_logits=lambda *args, **kwargs: "decode",
        )
        for name in (
            "vllm.models.glm5next.sparse_indexer",
            "vllm.models.glm5next.common.attention",
        ):
            modules(name, SparseAttnIndexerKpool=Indexer)
        patch = load_patch(
            "vllm_sail/patch/enhancement/attention/sparse_attn_indexer_kpool.py"
        )
        assert Indexer.forward_ppu is patch.forward_ppu
        return Indexer, calls

    return load


@pytest.mark.parametrize("method", ["forward_ppu", "forward_native"])
def test_int8_dcp_fails_before_accessing_or_mutating_cache(kpool, method):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 2
    with pytest.raises(NotImplementedError, match="decode context parallel size 1"):
        getattr(layer, method)(*[object() for _ in range(4)])
    assert calls == []


@pytest.mark.parametrize("method", ["forward_ppu", "forward_native"])
@pytest.mark.parametrize("requested,producer", [("fp32", "bf16"), ("bf16", "fp32")])
def test_int8_dtype_mismatch_fails_before_cache_access(
    kpool, method, requested, producer
):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 1
    layer.logits_dtype = requested
    with pytest.raises(ValueError, match="must match the producer weights dtype"):
        getattr(layer, method)(object(), object(), object(), NS(dtype=producer))
    assert calls == []


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_int8_matching_dtype_preserves_inputs_and_tail_coordinates(kpool, dtype):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 1
    layer.logits_dtype = dtype
    layer.k_cache = NS(prefix="indexer", kv_cache=object())
    layer.tail_cache = NS(prefix="tail", kv_cache=object())
    for field in (
        "quant_block_size",
        "scale_fmt",
        "topk_tokens",
        "head_dim",
        "max_pool_len",
        "max_total_seq_len",
        "topk_indices_buffer",
        "skip_k_cache_insert",
        "use_fp4_cache",
    ):
        setattr(layer, field, object())
    hidden, q, scale, k, weights = (
        object(),
        object(),
        object(),
        object(),
        NS(dtype=dtype),
    )
    gate, ape, positions = object(), object(), object()
    assert (
        layer.forward_ppu(
            hidden,
            (q, scale),
            k,
            weights,
            gate_score=gate,
            compress_ape=ape,
            positions=positions,
            index_kpool=2,
        )
        == "int8"
    )
    assert calls == [
        (
            hidden,
            "indexer",
            layer.k_cache.kv_cache,
            q,
            scale,
            k,
            weights,
            layer.quant_block_size,
            layer.scale_fmt,
            layer.topk_tokens,
            layer.head_dim,
            layer.max_pool_len,
            layer.max_total_seq_len,
            layer.topk_indices_buffer,
            layer.skip_k_cache_insert,
            layer.use_fp4_cache,
            gate,
            ape,
            2,
            positions,
            layer.tail_cache.kv_cache,
            "tail",
        )
    ]


def test_fp8_dcp_keeps_upstream_dispatch(kpool):
    cls, calls = kpool(fp8=True)
    layer = cls()
    layer.dcp_world_size = 2
    assert layer.forward_ppu(*[object() for _ in range(4)], topk_backend="triton") == (
        "upstream-fp8",
        2,
        "triton",
    )
    assert calls == []
