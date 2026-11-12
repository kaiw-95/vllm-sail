# SPDX-License-Identifier: Apache-2.0
"""Inject SAIL routing via the existing FusedMoEFactory runner extension."""

from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v4.nvidia import model
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.patch.utils import patch

_factory = model.FusedMoEFactory


@patch(
    "vllm.models.deepseek_v4.nvidia.model",
    "FusedMoEFactory",
    reason=(
        "V4.1 inherits the V4 factory call without a runner selector. Supply "
        "SAIL's optional MegaGate runner through runner_cls, keeping vLLM's "
        "expert kernels, shared experts and loader ownership."
    ),
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="DeepSeek V4.1 exposes a platform hook for FusedMoEFactory runner_cls.",
)
def fused_moe_factory(*args, **kwargs):
    if current_platform.is_ppu() and envs.VLLM_SAIL_DEEPGEMM_MEGA_GATE:
        config = get_current_vllm_config()
        if (
            config.model_config.hf_config.model_type == "deepseek_v41"
            and kwargs.get("runner_cls") is None
        ):
            from vllm_sail.model_executor.layers.fused_moe.mega_gate import (
                PPUMegaGateMoERunner,
            )

            kwargs["runner_cls"] = PPUMegaGateMoERunner
    return _factory(*args, **kwargs)
