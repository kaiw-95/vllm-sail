# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAIL W4A4 MegaMoE under vLLM's native expert and EPLB interfaces.

The checkpoint loader writes the layout documented by DeepGEMM-for-sail
``deep_gemm.mega.transform_weights_for_mega_moe`` directly: interleaved
gate/up weights, K-major GEMM1 scale pairs, N-major GEMM2 scale pairs.
No forward-time weight conversion or duplicate expert bank is needed.
"""

from __future__ import annotations

import torch
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm_sail.utils.deep_gemm_mega import mega_moe_api


def mega_gate(
    x,
    weight,
    num_topk,
    *,
    scoring_func,
    routed_scaling_factor,
    ep_rank,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    unmapped_topk_idx=None,
):
    """Preserve DeepSeek routing when SAIL has no SM100 MegaGate kernel."""
    if scoring_func != "sqrtsoftplus":
        raise ValueError("DeepSeek SAIL MegaMoE requires sqrtsoftplus routing.")
    logits = torch.nn.functional.linear(x.float(), weight.float())
    scores = torch.nn.functional.softplus(logits).sqrt()
    selection = scores if bias is None else scores + bias
    if image_bias is not None:
        if image_token_mask is None:
            raise ValueError("Image routing bias requires the prepared token mask.")
        selection = torch.where(
            image_token_mask[:, None], scores + image_bias, selection
        )
    ids = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :num_topk]
    if fix_routing_mask is not None:
        if unmapped_topk_idx is None:
            raise ValueError("Hash routing mask requires prepared expert IDs.")
        ids = torch.where(fix_routing_mask[:, None], unmapped_topk_idx, ids)
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * routed_scaling_factor
    return weights.contiguous(), ids.to(torch.int64).contiguous()


class PPUDeepGemmMegaMoEExperts(DeepGemmMegaMoEExperts):
    """Retain vLLM routing/loading; dispatch BF16 tokens as MXFP4 on PPU."""

    _symm_buffers: dict[tuple, object] = {}

    def __init__(self, *args, weight_format, activation, **kwargs):
        if weight_format.name != "mxfp4":
            raise ValueError("SAIL MegaMoE supports MXFP4 expert checkpoints only.")
        if (
            activation.name != "swiglu"
            or activation.alpha != 1.0
            or activation.beta != 0.0
        ):
            raise ValueError("V4.1 SAIL MegaMoE requires standard SwiGLU activation.")
        super().__init__(
            *args, weight_format=weight_format, activation=activation, **kwargs
        )

    def _fusable_shared_experts(self, shared_experts):
        # SAIL's complete call fuses only routed experts. The caller retains
        # its ordinary shared MLP and its TP/sequence-parallel reduction rules.
        return 0, None

    def _register_weight(self, name, shape, spec):
        experts, n, k = shape
        self.register_parameter(
            f"{name}_weight",
            torch.nn.Parameter(
                torch.zeros(experts, n, k // 2, dtype=torch.uint8), requires_grad=False
            ),
        )
        if name == "w13":
            scales = torch.zeros(experts, n * k // 64, dtype=torch.uint16)
        else:
            scales = torch.zeros(experts, k // 64, n, dtype=torch.uint16).transpose(
                1, 2
            )
        self.register_parameter(
            f"{name}_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )

    def _load_weight(self, dst, src, shard_id, is_scale):
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if not is_scale:
            src = src.view(torch.uint8)
            rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
            if src.shape != (rows, dst.shape[1]):
                raise ValueError(f"Invalid {shard_id} packed weight shape: {src.shape}")
            dst, src = self._kernel_layout_shard(dst, src, shard_id, False)
        else:
            if src.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise ValueError(
                    "MegaMoE requires raw UE8M0 scales, not NVFP4 E4M3 scales."
                )
            rows, k = (
                (self.hidden_size, self.intermediate_size)
                if shard_id == "w2"
                else (self.intermediate_size, self.hidden_size)
            )
            if src.shape != (rows, k // 32):
                raise ValueError(f"Invalid {shard_id} UE8M0 scale shape: {src.shape}")
            src = src.view(torch.uint8).contiguous().view(torch.uint16)
            if shard_id != "w2":
                # [N/256,K/64,256], where N rows are gate/up-interleaved by 8.
                half = 0 if shard_id == "w1" else 1
                dst = dst.view(-1, k // 64, 16, 2, 8)[:, :, :, half, :]
                dst = dst.permute(0, 2, 3, 1)
                src = src.reshape(-1, 16, 8, k // 64)
        dst.copy_(src)

    def get_symm_buffer(self):
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.cuda.current_device(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.activation.name,
        )
        if key not in self._symm_buffers:
            self._symm_buffers[key] = mega_moe_api().get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="fp4xfp4",
                activation=self.activation.name,
            )
        return self._symm_buffers[key]

    def forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
        tokens = hidden_states.shape[0]
        if tokens > self.max_num_tokens:
            raise ValueError(
                f"MegaMoE token count {tokens} exceeds capacity {self.max_num_tokens}."
            )
        self._synchronize_first_launch()
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        if is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                topk_ids = torch.where(padding[:tokens, None], -1, topk_ids)
        state = self.eplb_state
        if state.logical_to_physical_map is not None:
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=state.expert_load_view,
                logical_to_physical_map=state.logical_to_physical_map,
                logical_replica_count=state.logical_replica_count,
                record_enabled=state.should_record_tensor,
                num_unpadded_tokens=(
                    state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
                    if state.num_unpadded_tokens_tensors is not None
                    else None
                ),
            )
        api, buffer = mega_moe_api(), self.get_symm_buffer()
        api.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids.to(torch.int64).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            buffer,
            num_tokens=tokens,
        )
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        # Every EP rank participates, including ranks with zero local tokens.
        api.fp4_fp4_mega_moe(
            output,
            self.l1_weights,
            self.l2_weights,
            buffer,
            recipe=(1, 1, 32),
            activation=self.activation.name,
            activation_clamp=self.activation.clamp,
            fast_math=self.fast_math,
        )
        if shared_experts is not None:
            output += shared_experts(hidden_states)
        return output
