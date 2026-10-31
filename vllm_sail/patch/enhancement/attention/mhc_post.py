# SPDX-License-Identifier: Apache-2.0
"""Select hidden-tiled post for bounded M890 batches at the shared launcher."""

import torch
from vllm.model_executor.kernels.mhc import tilelang_kernels as _kernels
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.patch.utils import patch

_upstream_post = _kernels.MhcPostTileLangKernel.__call__
_upstream_warmup_inputs = _kernels.MhcPostTileLangKernel.warmup_inputs


@patch(
    "vllm.model_executor.kernels.mhc.tilelang_kernels",
    "MhcPostTileLangKernel.warmup_inputs",
    reason="The opt-in tiled mHC specialization must participate in compile-only JIT warmup.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="Upstream mHC warmup registers platform-specific post specializations.",
)
def warmup_inputs(self, compile_key):
    inputs = _upstream_warmup_inputs(self, compile_key)
    if (
        self._warming_key is not None
        and envs.VLLM_SAIL_MHC_POST_TILED
        and current_platform.is_ppu()
        and compile_key.hc_mult == 4
        and compile_key.hidden_size > 1024
        and compile_key.hidden_size % 1024 == 0
    ):
        from vllm_sail.model_executor.kernels.mhc_post import mhc_post_tiled

        # self.launch compiles without allocating/launching while warming.
        # Token count is dynamic, so the original one-row placeholders warm
        # the same specialization that serves runtime batches of 64..256.
        self.launch(
            mhc_post_tiled,
            tuple(
                inputs[name]
                for name in ("comb_mix", "residual", "post_mix", "layer_input", "out")
            )
            + (compile_key.hidden_size,),
        )
    return inputs


@patch(
    "vllm.model_executor.kernels.mhc.tilelang_kernels",
    "MhcPostTileLangKernel.__call__",
    reason="M890 small-batch mHC post needs hidden-tile CTAs to expose parallelism.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The upstream mHC post launcher offers PPU hidden-tile scheduling.",
)
def mhc_post(
    self, comb_mix, residual, post_mix, layer_input, out, hc_mult, hidden_size
):
    # Provider-level dispatch reaches standalone post, delayed fallbacks and
    # V4.1/DSpark's by-value launcher imports without rebinding those functions.
    if self._warming_key is not None:
        # Compile-only placeholders deliberately have no device or storage.
        return _upstream_post(
            self, comb_mix, residual, post_mix, layer_input, out, hc_mult, hidden_size
        )
    if (
        envs.VLLM_SAIL_MHC_POST_TILED
        and current_platform.is_ppu()
        and residual.device.type == "cuda"
        and current_platform.is_device_capability((8, 9), residual.device.index)
        and hc_mult == 4
        and 64 <= residual.shape[0] <= 256
        and hidden_size > 1024
        and hidden_size % 1024 == 0
        and residual.shape == (layer_input.shape[0], 4, hidden_size)
        and layer_input.shape == (residual.shape[0], hidden_size)
        and out.shape == residual.shape
        and comb_mix.shape == (residual.shape[0], 4, 4)
        and post_mix.shape == (residual.shape[0], 4)
        and all(
            tensor.is_contiguous()
            and tensor.device == residual.device
            and tensor.dtype == dtype
            for tensor, dtype in (
                (comb_mix, torch.float32),
                (residual, torch.bfloat16),
                (post_mix, torch.float32),
                (layer_input, torch.bfloat16),
                (out, torch.bfloat16),
            )
        )
    ):
        from vllm_sail.model_executor.kernels.mhc_post import mhc_post_tiled

        mhc_post_tiled(comb_mix, residual, post_mix, layer_input, out, hidden_size)
        return
    return _upstream_post(
        self, comb_mix, residual, post_mix, layer_input, out, hc_mult, hidden_size
    )
