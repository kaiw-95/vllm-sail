# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hidden-tiled mHC post, preserving the upstream elementwise operation order.

The SAIL scheduling policy follows SGLang's 6ad06a858bc76d4fff1594f9b8573f69405f4544
(``kernels/ops/layernorm/mhc.py``). Only CTA ownership changes: each hidden
tile independently computes the same four-stream post mix as vLLM's kernel.
"""

from vllm.tilelang_utils import T, tilelang, tilelang_jit


@tilelang_jit
def mhc_post_tiled(
    comb,
    residual,
    post,
    layer_input,
    out,
    hidden: int,
    block_h: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    comb: T.Tensor((n, 4, 4), T.float32)  # type: ignore[no-redef, valid-type]
    residual: T.Tensor((n, 4, hidden), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post: T.Tensor((n, 4), T.float32)  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor((n, hidden), T.bfloat16)  # type: ignore[no-redef, valid-type]
    out: T.Tensor((n, 4, hidden), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, hidden // block_h, threads=128) as (token, tile):
        residual_shared = T.alloc_shared((4, block_h), T.bfloat16)
        input_shared = T.alloc_shared(block_h, T.bfloat16)
        result = T.alloc_fragment((4, block_h), T.float32)
        residual_local = T.alloc_fragment((4, block_h), T.float32)
        input_local = T.alloc_fragment(block_h, T.float32)
        comb_local = T.alloc_fragment((4, 4), T.float32)
        post_local = T.alloc_fragment(4, T.float32)
        T.copy(comb[token, 0, 0], comb_local)
        T.copy(post[token, 0], post_local)
        T.copy(residual[token, 0, tile * block_h], residual_shared)
        T.copy(layer_input[token, tile * block_h], input_shared)
        T.copy(residual_shared, residual_local)
        T.copy(input_shared, input_local)
        for hco, col in T.Parallel(4, block_h):
            result[hco, col] = post_local[hco] * input_local[col]
            for hci in T.vectorized(4):
                result[hco, col] += comb_local[hci, hco] * residual_local[hci, col]
        T.copy(result, out[token, 0, tile * block_h])
