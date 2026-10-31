# SPDX-License-Identifier: Apache-2.0
"""Budget the actual padded SAIL logits allocation using only the stdlib."""

import pytest

from vllm_sail.utils.indexer import logits_rows_per_chunk


@pytest.mark.parametrize("width", [1, 511, 512, 513, 32768, 131072, 1048576])
@pytest.mark.parametrize("size,alignment", [(2, 2), (2, 4), (4, 1), (4, 6)])
def test_aligned_tiles_fit_and_make_progress(width, size, alignment):
    rows = logits_rows_per_chunk(width, 64, element_size=size, row_alignment=alignment)
    row_bytes = -(-width // 512) * 512 * size
    assert rows > 0 and rows % alignment == 0
    assert rows * row_bytes <= 64 * 1024 * 1024
    assert (rows + alignment) * row_bytes > 64 * 1024 * 1024


@pytest.mark.parametrize("width,budget", [(0, 1), (1, 0), (1048576, 1)])
def test_invalid_or_undersized_budget_fails_before_launch(width, budget):
    with pytest.raises(ValueError):
        logits_rows_per_chunk(width, budget)
