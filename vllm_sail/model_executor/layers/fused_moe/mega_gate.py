# SPDX-License-Identifier: Apache-2.0
"""SAIL MegaGate routing in vLLM's modular MoE runner.

The BF16 gate GEMM and sqrtsoftplus top-k come from DeepGEMM-for-sail
4869f1e3344a. Dispatch exclusions follow sglang-for-sail 01a98013bb83.
Expert computation, reduction and shared-expert overlap remain vLLM's.
"""

import math

import torch
import vllm.envs as vllm_envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.utils.deep_gemm import (
    bf16_mega_gate,
    has_mega_gate,
    is_deep_gemm_supported,
)

logger = init_logger(__name__)
_MAX_ROWS = 2048


def _postprocess_routing(router, topk_ids, indices_dtype):
    """Match BaseRouter._select_experts and FusedMoERouter.select_experts.

    Keep this order covered by the pinned-source contract test when upgrading
    vLLM: logical capture precedes EPLB, replay records the converted IDs.
    """
    if router.capture_fn is not None:
        router.capture_fn(topk_ids)
    topk_ids = router._apply_eplb_mapping(topk_ids)
    topk_ids = router._convert_indices_dtype(topk_ids, indices_dtype)
    if router._routing_replay_out is not None:
        router._routing_replay_out[: topk_ids.shape[0]].copy_(topk_ids.to(torch.int16))
    return topk_ids


class PPUMegaGateMoERunner(MoERunner):
    """Opt-in V4.1 runner; unsupported forwards delegate without changing state."""

    def _fallback_reason(self, x, input_ids):
        if (
            not current_platform.is_ppu()
            or not current_platform.has_device_capability(89)
        ):
            return "requires PPU SM89+"
        if not is_deep_gemm_supported() or not has_mega_gate():
            return "DeepGEMM bf16_mega_gate is unavailable or disabled"
        if not is_forward_context_available():
            return "forward metadata is unavailable"
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            return "decode metadata is unavailable"
        batches = [m for m in metadata.values() if hasattr(m, "num_prefill_tokens")]
        if (
            not batches
            or any(m.num_prefill_tokens > 0 for m in batches)
            or not any(getattr(m, "num_decode_tokens", 0) > 0 for m in batches)
        ):
            return "only decode and target verification use MegaGate"
        if (
            self.enable_dbo
            or self.moe_config.is_sequence_parallel
            or self.moe_config.pcp_size != 1
            or self.do_naive_dispatch_combine
        ):
            return "DBO, sequence parallelism, PCP or naive dispatch is active"
        if (
            self.is_monolithic
            or type(self.router) is not FusedTopKBiasRouter
            or self.gate is None
            or self._fse_fuse_gate
            or self.routed_input_transform is not None
            or self.routed_output_transform is not None
        ):
            return "requires a modular V4.1 biased router with a separate gate"
        router = self.router
        if (
            router.scoring_func != "sqrtsoftplus"
            or not router.renormalize
            or router.num_fused_shared_experts
            or router._hash_indices_table is not None
            or vllm_envs.VLLM_BATCH_INVARIANT
            or not math.isfinite(router.routed_scaling_factor)
        ):
            return "routing semantics are unsupported"
        weight = self.gate.weight
        if (
            x.ndim != 2
            or not 0 < x.shape[0] <= _MAX_ROWS
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or x.shape[1] != weight.shape[1]
            or x.shape[1] == 0
            or x.shape[1] % 256
            or not 0 < weight.shape[0] <= 512
            or weight.shape[0] % 4
            or not 0 < router.top_k <= min(32, weight.shape[0])
            or weight.device != x.device
            or weight.data_ptr() % 16
            or x.data_ptr() % 16
        ):
            return "BF16 gate shape, alignment or row limit is unsupported"
        for bias in (router.e_score_correction_bias, router.bias_vl):
            if bias is not None and (
                bias.dtype != torch.float32
                or bias.shape != (weight.shape[0],)
                or not bias.is_contiguous()
                or bias.device != x.device
            ):
                return "gate bias must be contiguous FP32 on the gate device"
        if router.bias_vl is not None and (
            input_ids is None
            or input_ids.shape != (x.shape[0],)
            or input_ids.device != x.device
        ):
            return "vision routing needs one input ID per row"
        return None

    def _forward_impl(
        self, hidden_states, router_logits, shared_experts_input, input_ids=None
    ):
        if not envs.VLLM_SAIL_DEEPGEMM_MEGA_GATE or hidden_states.shape[0] == 0:
            return super()._forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        # Like MoERunner, initialize the modular quant method before inspecting
        # its execution contract. No model or forward-global routing cache.
        self.routed_experts._ensure_moe_quant_config_init()
        reason = self._fallback_reason(hidden_states, input_ids)
        if reason is not None:
            logger.info_once("MegaGate fallback: %s; using the regular router.", reason)
            return super()._forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )

        overlapping = False
        if self._shared_experts is not None:
            overlapping = self._shared_experts.maybe_forward_async(shared_experts_input)
        with self._sequence_parallel_context():
            self._maybe_apply_shared_experts(
                shared_experts_input, SharedExpertsOrder.NO_OVERLAP
            )
            router = self.router
            router._validate_eplb_state()
            image_mask = None
            if router.bias_vl is not None:
                lo = router.image_sentinel_lo
                image_mask = (input_ids >= lo) & (input_ids < lo + 5) & (lo > 0)
            padding = get_forward_context().is_padding
            active = None
            if vllm_envs.VLLM_MOE_SKIP_PADDING and padding is not None:
                active = (~padding[: hidden_states.shape[0]]).contiguous()
            topk_weights, topk_ids = bf16_mega_gate(
                hidden_states,
                self.gate.weight,
                router.top_k,
                routed_scaling_factor=router.routed_scaling_factor,
                bias=router.e_score_correction_bias,
                image_bias=router.bias_vl,
                image_token_mask=image_mask,
                mask=active,
            )
            topk_ids = _postprocess_routing(
                router, topk_ids, self._quant_method.topk_indices_dtype
            )
            fused_out = self.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=self._shared_experts,
                shared_experts_input=shared_experts_input,
            )
            if overlapping:
                self._shared_experts.wait()
            shared_out = (
                self._shared_experts.output if self._shared_experts is not None else None
            )
            return self._maybe_combine(shared_out, fused_out)
