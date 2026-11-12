# SPDX-License-Identifier: Apache-2.0
"""The optional, precompiled BF16 DeepSelect backend for SAIL indexers.

The wheel API follows sglang-for-sail b76ad8286bde. Never import the CUDA
JIT backend on PPU, and never round FP32 scores to make them fit this wheel.
"""

import functools

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_sail import envs

logger = init_logger(__name__)


@functools.cache
def _wheel():
    try:
        from deep_select import interface
    except ModuleNotFoundError as exc:
        if exc.name in ("deep_select", "deep_select.interface"):
            return None
        raise
    if not callable(getattr(interface, "topk", None)) or not callable(
        getattr(interface, "get_stride_requirement", None)
    ):
        return None
    return interface


def can_use_deep_select(logits: torch.Tensor, topk: int) -> bool:
    return (
        current_platform.is_ppu()
        and current_platform.has_device_capability(89)
        and envs.VLLM_SAIL_USE_DEEPSELECT
        and logits.dtype == torch.bfloat16
        and logits.ndim == 2
        and 0 < topk <= 4096
        and 0 < logits.shape[1] < 2**23
        and _wheel() is not None
    )


def deep_select_topk(
    logits: torch.Tensor,
    topk: int,
    end: torch.Tensor,
    output_idx: torch.Tensor,
) -> None:
    """Write local, -1-padded indices; accept arbitrary caller row strides.

    Rows already start at their query's first K column. Page-table conversion
    remains with vLLM's attention backend, after this local-index selection.
    """
    if logits.dtype != torch.bfloat16:
        raise TypeError("SAIL DeepSelect requires BF16 logits; FP32 must use FP32 top-k")
    if not can_use_deep_select(logits, topk):
        raise RuntimeError("SAIL DeepSelect wheel, device or top-k shape is unsupported")
    rows, width = logits.shape
    if output_idx.dtype != torch.int32 or output_idx.shape != (rows, topk):
        raise ValueError("DeepSelect output must have shape [rows, topk] and dtype int32")
    if end.numel() != rows:
        raise ValueError("DeepSelect needs one exclusive end per query row")
    interface = _wheel()
    run_bytes, _ = interface.get_stride_requirement()
    # The wheel requires run-aligned row strides, including narrow views.
    if logits.stride(1) != 1 or logits.stride(0) * logits.element_size() % run_bytes:
        step = run_bytes // logits.element_size()
        padded = (width + step - 1) // step * step
        scores = torch.empty((rows, padded), dtype=logits.dtype, device=logits.device)
        scores[:, :width].copy_(logits)
        logits = scores[:, :width]
    ends = end.reshape(-1).to(torch.int32).clamp(0, width).contiguous()
    # Allocate through the wheel: a caller's k/stride need not meet its 32B
    # output alignment. In particular, preserve k instead of rounding it up.
    _, indices = interface.topk(
        logits,
        topk,
        sorted=False,
        end=ends,
        indices_type=torch.int32,
        sorted_index=False,
        output_idx=None,
        output_idx_offset=None,
        idx_oob_fill_value=-1,
        return_value=False,
    )
    output_idx.copy_(indices)


def warn_deep_select_fallback() -> None:
    logger.info_once(
        "DeepSelect fallback: using the existing top-k without score rounding; "
        "check the wheel, dtype, device, "
        "shape and VLLM_SAIL_USE_DEEPSELECT."
    )
