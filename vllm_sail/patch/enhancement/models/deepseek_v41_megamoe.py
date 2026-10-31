# SPDX-License-Identifier: Apache-2.0
"""Select SAIL's W4A4 implementation at the native DeepSeek MoE seam."""

from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.mega_moe import PPUDeepGemmMegaMoEExperts, mega_gate
from vllm_sail.patch.utils import PATCH_MARKER, patch
from vllm_sail.utils.deep_gemm_mega import check_mega_moe_supported

_EXPERTS = "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe"
_MODEL = "vllm.models.deepseek_v4.nvidia.model"
_REASON = "SAIL MegaMoE uses MXFP4 dispatch and PPU-specific expert scale layouts instead of the SM100 FP8 API."
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = "The native DeepSeek MegaMoE path accepts an out-of-tree expert backend."


@patch(
    _EXPERTS,
    "check_deep_gemm_mega_moe_supported",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def check_supported(device, hidden_size, intermediate_size):
    if current_platform.is_ppu():
        return check_mega_moe_supported(device, hidden_size, intermediate_size)
    original = getattr(check_supported, PATCH_MARKER)[
        f"{_EXPERTS}.check_deep_gemm_mega_moe_supported"
    ]
    return original(device, hidden_size, intermediate_size)


@patch(
    _MODEL,
    "DeepGemmMegaMoEExperts",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def make_experts(*args, **kwargs):
    if current_platform.is_ppu():
        return PPUDeepGemmMegaMoEExperts(*args, **kwargs)
    original = getattr(make_experts, PATCH_MARKER)[f"{_MODEL}.DeepGemmMegaMoEExperts"]
    return original(*args, **kwargs)


@patch(
    _MODEL,
    "bf16_mega_gate",
    reason="SAIL's W4A4 MegaMoE API does not export the SM100 MegaGate; preserve sqrtsoftplus, hash and image routing on PPU.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="SAIL DeepGEMM provides a compatible MegaGate implementation.",
)
def route_experts(*args, **kwargs):
    if current_platform.is_ppu():
        return mega_gate(*args, **kwargs)
    original = getattr(route_experts, PATCH_MARKER)[f"{_MODEL}.bf16_mega_gate"]
    return original(*args, **kwargs)
