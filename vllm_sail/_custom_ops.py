# SPDX-License-Identifier: Apache-2.0
"""Python wrappers for the plugin's compiled extensions.

Phase-1 decision D2 (spec 1.8): PPU ops live under the plugin-private
namespaces ``torch.ops._ppu_C`` / ``torch.ops._ppu_moe_C`` — never under
upstream's ``_C`` / ``_moe_C`` — so the plugin installs cleanly beside a
stock vLLM wheel and can never shadow an upstream op. PPU code calls these
wrappers directly; there is deliberately **no** patch of upstream's
``vllm._custom_ops``.

The fork's only ``vllm/_custom_ops.py`` addition, ``ep_scatter_2_cuda``,
calls ``torch.ops._moe_C``; here it calls ``torch.ops._ppu_moe_C``.

The extensions only exist when the package was built against a PPU SDK
(``PPU_SDK`` set; see ``setup.py``). Importing this module always works —
on a CPU-only install the wrappers raise an actionable error on first call.
"""

from __future__ import annotations

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_ext_state: dict[str, bool | None] = {"_C": None, "_moe_C": None}


def _ext_loaded(name: str) -> bool:
    """Whether the named extension imported, cached per process.

    Importing registers the ops into their ``torch.ops._ppu_*`` namespaces
    as a static-initializer side effect of the shared library.
    """
    loaded = _ext_state[name]
    if loaded is None:
        try:
            importlib.import_module(f"vllm_sail.{name}")
            loaded = True
        except ImportError as exc:
            logger.info(
                "vllm_sail.%s unavailable (%s); built without PPU_SDK?", name, exc
            )
            loaded = False
        _ext_state[name] = loaded
    return bool(loaded)


def is_available() -> bool:
    """Whether any compiled PPU extension is loaded or loadable."""
    return _ext_loaded("_C") or _ext_loaded("_moe_C")


def has_vector_dequant() -> bool:
    """Permit a source checkout to fall back when paired with an older wheel."""
    return _ext_loaded("_C") and hasattr(
        torch.ops._ppu_C, "deepseek_dequant_gather_vector"
    )


def deepseek_dequant_gather_vector(
    out, cache, seq_lens, gather_lens, block_table, page_size, offset, max_gather_len
):
    _require("_C", "deepseek_dequant_gather_vector")
    torch.ops._ppu_C.deepseek_dequant_gather_vector(
        out,
        cache,
        seq_lens,
        gather_lens,
        block_table,
        page_size,
        offset,
        max_gather_len,
    )


def _require(name: str, op: str) -> None:
    if not _ext_loaded(name):
        raise RuntimeError(
            f"torch.ops._ppu{name}.{op} is unavailable: vllm_sail.{name} was "
            "not built. Rebuild vllm-sail with PPU_SDK set (see setup.py); "
            "CPU-only installs cannot run PPU kernels."
        )


def ep_scatter_2_cuda(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor | None,
    recv_topk_ids: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_index: torch.Tensor,
    output_tensor_scale: torch.Tensor | None,
    with_scale: bool,
) -> None:
    """Scatter EP-received tokens into expert-contiguous layout (fork's
    ``vllm._custom_ops.ep_scatter_2_cuda``, on the plugin namespace)."""
    _require("_moe_C", "ep_scatter_2_cuda")
    torch.ops._ppu_moe_C.ep_scatter_2_cuda(
        recv_x,
        recv_x_scale,
        recv_topk_ids,
        expert_start_loc,
        output_tensor,
        output_index,
        output_tensor_scale,
        with_scale,
    )


def top_k_per_row_prefill_bf16(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    top_k: int,
) -> None:
    """BF16 top-k prefill sampler kernel (fork's ``sampler.cu`` addition,
    on the plugin namespace)."""
    _require("_C", "top_k_per_row_prefill_bf16")
    torch.ops._ppu_C.top_k_per_row_prefill_bf16(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
