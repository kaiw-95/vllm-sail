# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash draft quantization names in the text-only namespace."""

from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.glm5next.common.mtp import Glm5NextMTP as _UpstreamGlm5NextMTP


class Glm5NextMTP(_UpstreamGlm5NextMTP):
    """Map quantization layer lists before the upstream draft builds its layers."""

    # The target may already have mapped checkpoint names into its text tower.
    # The draft's construction prefixes are model.layers.N.*, without the
    # mtp_block attribute used by named_parameters() and the weight loader.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "language_model.model.": "model.",
        }
    )
