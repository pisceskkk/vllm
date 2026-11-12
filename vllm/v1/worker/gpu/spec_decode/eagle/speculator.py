# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class EagleSpeculator(TargetDependentARSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        if (
            speculative_config.method == "eagle3"
            and not speculative_config.draft_model_config.use_mla
            and vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            # Replicate GQA draft KV across DCP ranks. The runner derives
            # dcp_sharded=False from the draft's effective DCP size.
            vllm_config = copy.copy(vllm_config)
            vllm_config.parallel_config = replace(
                vllm_config.parallel_config, decode_context_parallel_size=1
            )
        super().__init__(vllm_config, device)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)
