# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable token exchange for head-sharded recurrent layers."""

from dataclasses import dataclass

import torch

from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass
class MambaPCPMetadata:
    restore_indices: torch.Tensor
    restore_mask: torch.Tensor
    local_indices: torch.Tensor
    local_mask: torch.Tensor


def mamba_pcp_gather(
    hidden_states: torch.Tensor, pcp_size: int, layer_name: str
) -> torch.Tensor:
    gathered = get_pcp_group().all_gather(hidden_states.contiguous(), dim=0)
    metadata = get_forward_context().attn_metadata
    if metadata is None:
        return gathered
    assert isinstance(metadata, dict)
    if layer_name not in metadata:
        return gathered
    pcp = getattr(metadata[layer_name], "pcp", None)
    if pcp is None:
        return gathered
    mask_shape = (-1,) + (1,) * (hidden_states.ndim - 1)
    return gathered.index_select(0, pcp.restore_indices).masked_fill(
        ~pcp.restore_mask.view(mask_shape), 0
    )


def mamba_pcp_gather_fake(
    hidden_states: torch.Tensor, pcp_size: int, layer_name: str
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0] * pcp_size, *hidden_states.shape[1:])
    )


def mamba_pcp_scatter(
    output: torch.Tensor, pcp_size: int, layer_name: str
) -> torch.Tensor:
    output = get_pcp_group().all_reduce(output)
    return mamba_pcp_select(output, pcp_size, layer_name)


def mamba_pcp_select(
    output: torch.Tensor, pcp_size: int, layer_name: str
) -> torch.Tensor:
    metadata = get_forward_context().attn_metadata
    if metadata is None:
        return output[: output.shape[0] // pcp_size].clone()
    assert isinstance(metadata, dict)
    if layer_name not in metadata:
        return output[: output.shape[0] // pcp_size].clone()
    pcp = getattr(metadata[layer_name], "pcp", None)
    if pcp is None:
        return output[: output.shape[0] // pcp_size].clone()
    mask_shape = (-1,) + (1,) * (output.ndim - 1)
    return output.index_select(0, pcp.local_indices).masked_fill(
        ~pcp.local_mask.view(mask_shape), 0
    )


def mamba_pcp_scatter_fake(
    output: torch.Tensor, pcp_size: int, layer_name: str
) -> torch.Tensor:
    return output.new_empty((output.shape[0] // pcp_size, *output.shape[1:]))


direct_register_custom_op(
    op_name="mamba_pcp_gather",
    op_func=mamba_pcp_gather,
    mutates_args=[],
    fake_impl=mamba_pcp_gather_fake,
)
direct_register_custom_op(
    op_name="mamba_pcp_scatter",
    op_func=mamba_pcp_scatter,
    mutates_args=[],
    fake_impl=mamba_pcp_scatter_fake,
)
direct_register_custom_op(
    op_name="mamba_pcp_select",
    op_func=mamba_pcp_select,
    mutates_args=[],
    fake_impl=mamba_pcp_scatter_fake,
)
