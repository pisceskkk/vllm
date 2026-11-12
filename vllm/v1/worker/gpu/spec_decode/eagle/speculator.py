# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class EagleSpeculator(TargetDependentARSpeculator):
    @property
    def dcp_sharded(self) -> bool:
        if (
            self.method == "eagle3"
            and not self.draft_model_config.use_mla
            and self.vllm_config.parallel_config.prefill_context_parallel_size == 1
        ):
            return False
        return super().dcp_sharded

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)
