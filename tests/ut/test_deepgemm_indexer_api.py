# SPDX-License-Identifier: Apache-2.0
"""SAIL DeepGEMM indexer API regression coverage; not yet executed."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

DEEP_GEMM = "vllm_sail/utils/deep_gemm.py"
INDEXER = "vllm_sail/attention/ops/mla_sparse.py"


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)

    def to(self, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self


def _api_namespace(symbol):
    calls = []

    def api(*args, **kwargs):
        calls.append((args, kwargs))
        return "logits"

    return {
        "torch": NS(
            bfloat16="bf16", float32="fp32", int64="int64",
            empty=lambda shape, dtype, device: Tensor(shape, dtype),
        ),
        "_lazy_init": lambda: None,
        "_missing": lambda: pytest.fail("unexpected missing API"),
        symbol: api,
    }, calls


def test_weighted_mxfp4_uses_relative_column_api():
    ns, calls = _api_namespace("_fp8_fp4_mqa_logits_impl")
    fn = function(DEEP_GEMM, "fp8_fp4_mqa_logits", ns)
    q, kv = (Tensor((2, 32, 64)), object()), (Tensor((257, 64)), object())
    weights, starts, ends = Tensor((2, 32), "fp32"), object(), object()
    assert fn(q, kv, weights, starts, ends, False) == "logits"
    assert calls == [((q, kv, weights, starts, ends), {"max_seqlen_k": 257})]
    assert weights.dtype == "bf16"
    with pytest.raises(ValueError, match="BF16"):
        fn(q, kv, weights, starts, ends, False, logits_dtype="fp32")
    assert len(calls) == 1


def test_paged_weighted_mxfp4_forwards_the_same_request_ids():
    ns, calls = _api_namespace("_fp8_fp4_paged_mqa_logits_impl")
    fn = function(DEEP_GEMM, "fp8_fp4_paged_mqa_logits", ns)
    ids, contexts, tables, schedule = object(), object(), object(), object()
    q, cache, weights = object(), object(), Tensor((6, 32), "fp32")
    assert fn(q, cache, weights, contexts, tables, schedule, 8192, False,
              indices=ids) == "logits"
    assert calls == [(
        (q, cache, weights, contexts, tables, schedule, 8192), {"indices": ids}
    )]


@pytest.mark.parametrize("contexts,lengths,expected", [
    ([[9, 10], [19, 20]], [2, 1], [[9], [10], [20], [0]]),
    # C2 contexts must be shifted after compression, without a token offset.
    ([[4, 5], [9, 10]], [2, 1], [[4], [5], [10], [0]]),
])
def test_ragged_verify_keeps_request_contexts(contexts, lengths, expected):
    torch = pytest.importorskip("torch")
    fn = function(INDEXER, "_paged_decode_rows", {"torch": torch})
    lens, tables, ids = fn(
        torch.tensor(contexts, dtype=torch.int32),
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32), 2, None,
    )
    assert lens.tolist() == expected
    assert tables.tolist() == [[11, 12], [11, 12], [21, 22], [21, 22]]
    assert ids.tolist() == [0, 0, 1, -4]


def test_upstream_flattened_mtp_recovers_true_request_ownership():
    torch = pytest.importorskip("torch")
    fn = function(INDEXER, "_paged_decode_rows", {"torch": torch})
    _, _, ids = fn(
        torch.tensor([[8], [9], [10], [7], [9], [10]], dtype=torch.int32),
        torch.zeros((6, 2), dtype=torch.int32),
        torch.ones(6, dtype=torch.int32), 1, None,
        torch.tensor([3, 1, 2], dtype=torch.int32),
    )
    assert ids.tolist() == [0, 0, 0, 1, 2, 2]
