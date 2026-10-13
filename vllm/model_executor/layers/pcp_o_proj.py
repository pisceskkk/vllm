# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA O-Proj with its weight sharded over TP x PCP ranks.

Standard TP shards the O-Proj input (head) dimension over T TP ranks. With
``--enable-pcp-o-proj-tp`` each TP shard is split again over the P PCP ranks,
so rank ``(t, p)`` keeps input shard ``t * P + p`` of ``T * P`` and stores
``1 / P`` of the standard TP weight. The PCP peers of a TP rank thus hold
adjacent head slices of that TP rank's heads.

Every forward runs in one of three modes, chosen identically on all PCP ranks
from the pre-partition (global) batch:

* ``GATHER_WEIGHT`` (any prefill in the global batch): attention starts an
  asynchronous all-gather of the TP weight shard over PCP before its own
  compute (see ``pcp_tp_weight``). O-Proj then projects the local tokens with
  the TP-sized weight, exactly like the standard TP O-Proj.
* ``EXCHANGE_HEADS`` (decode-only, requests sharded over PCP ranks): an
  all-to-all trades token sharding for head sharding, every rank projects all
  PCP ranks' tokens with its own shard, and a reduce-scatter sums the partial
  projections and returns each token to its owner.
* ``LOCAL_HEADS`` (decode-only, every PCP rank holds every request): every
  rank projects its head slice of the shared tokens and the partial
  projections are all-reduced over PCP.

All modes leave the TP reduction to ``reduce_results`` exactly like
``RowParallelLinear``.
"""

import enum

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.distributed import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pcp_tp_weight import (
    TPWeight,
    build_tp_weight,
    gather_raw_tp_shard,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import set_default_torch_dtype


class OProjMode(enum.Enum):
    GATHER_WEIGHT = enum.auto()
    EXCHANGE_HEADS = enum.auto()
    LOCAL_HEADS = enum.auto()


class PCPOProjRowParallelLinear(RowParallelLinear):
    """Row-parallel O-Proj whose weight is sharded over TP x PCP ranks.

    The owning attention layer must call :meth:`prefetch_weights` once per
    forward before calling this layer.

    ``prefill_weight_gather`` controls whether the TP-sized layer and its
    shared full/staging buffers are built at load time. Models enable it
    exactly when PCP is enabled; without PCP the local shard already is the TP
    shard and the layer behaves like ``RowParallelLinear``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        prefill_weight_gather: bool,
    ) -> None:
        if bias:
            raise ValueError("PCP O-Proj TP requires bias=False.")
        if not input_is_parallel:
            raise ValueError("PCP O-Proj TP requires input_is_parallel=True.")

        self.pcp_group = get_pcp_group()
        self.pcp_rank = self.pcp_group.rank_in_group
        self.pcp_size = self.pcp_group.world_size
        if prefill_weight_gather != (self.pcp_size > 1):
            raise ValueError(
                "prefill_weight_gather must be enabled exactly when PCP is "
                f"enabled; got {prefill_weight_gather} with PCP size "
                f"{self.pcp_size}."
            )
        self.prefill_weight_gather = prefill_weight_gather
        parallel_config = get_current_vllm_config().parallel_config
        self.shard_decode_requests = parallel_config.pcp_shard_decode_requests

        # RowParallelLinear's tp_rank/tp_size describe the TP x PCP weight
        # shard; the output is still reduced over the real TP group only.
        tp_rank = get_tensor_model_parallel_rank()
        self.o_proj_tp_size = get_tensor_model_parallel_world_size()
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            tp_rank=tp_rank * self.pcp_size + self.pcp_rank,
            tp_size=self.o_proj_tp_size * self.pcp_size,
        )

        # The standard TP O-Proj that prefill runs is built at load time with
        # the same quant method configured for the TP-sized input, under the
        # config, dtype, and device of this construction.
        self._tp_layer_args = dict(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            params_dtype=self.params_dtype,
            reduce_results=False,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=False,
            tp_rank=tp_rank,
            tp_size=self.o_proj_tp_size,
        )
        self._vllm_config = get_current_vllm_config()
        self._default_dtype = torch.get_default_dtype()
        self._tp_layer: RowParallelLinear | None = None
        self._tp_weight: TPWeight | None = None
        self._mode: OProjMode | None = None

    def on_weights_loaded(self) -> None:
        """Called by the model loader before the quant method runs.

        Builds the TP-sized layer and fills it with the raw TP shard. The
        layer is kept out of the module tree (``object.__setattr__``) so that
        weight loading and other model passes skip it.
        """
        if not self.prefill_weight_gather:
            return
        device = next(self.parameters()).device
        with (
            set_current_vllm_config(self._vllm_config),
            set_default_torch_dtype(self._default_dtype),
            torch.device(device),
        ):
            tp_layer = RowParallelLinear(**self._tp_layer_args)
        gather_raw_tp_shard(self, tp_layer, self.pcp_group)
        object.__setattr__(self, "_tp_layer", tp_layer)

    def on_weights_processed(self) -> None:
        """Called by the model loader after the quant method has run."""
        if not self.prefill_weight_gather:
            return
        tp_layer = self._tp_layer
        if tp_layer is None:
            raise RuntimeError(f"{self.prefix}: on_weights_loaded() was not called.")
        with set_current_vllm_config(self._vllm_config):
            tp_layer.quant_method.process_weights_after_loading(tp_layer)
        self._tp_weight = build_tp_weight(self, tp_layer, self.pcp_group, "o_proj")
        # The TP weight owns the TP-sized layer if it uses it at all.
        self._tp_layer = None

    @property
    def tp_weight(self) -> TPWeight:
        if self._tp_weight is None:
            raise RuntimeError(f"{self.prefix}: weights have not been processed.")
        return self._tp_weight

    def prefetch_weights(self) -> None:
        """Select this forward's mode and start the weight gather for prefill.

        Attention calls this before its own compute so that the gather
        overlaps it.
        """
        if self._mode is not None:
            raise RuntimeError("PCP O-Proj prefetched twice without a forward.")
        has_prefill = get_forward_context().global_has_prefill
        if has_prefill is None:
            raise RuntimeError("PCP O-Proj TP requires the global batch type.")

        if self.pcp_size == 1:
            # The local shard is the TP shard and every rank holds its tokens.
            self._mode = OProjMode.LOCAL_HEADS
        elif has_prefill:
            self.tp_weight.start()
            self._mode = OProjMode.GATHER_WEIGHT
        elif self.shard_decode_requests:
            self._mode = OProjMode.EXCHANGE_HEADS
        else:
            self._mode = OProjMode.LOCAL_HEADS

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        mode, self._mode = self._mode, None
        if mode is None:
            raise RuntimeError(
                "Attention must call o_proj.prefetch_weights() before O-Proj."
            )
        expected_shape = (
            input_.shape[0],
            self.pcp_size * self.input_size_per_partition,
        )
        if input_.shape != expected_shape:
            if mode is OProjMode.GATHER_WEIGHT:
                self.tp_weight.wait()
            raise ValueError(
                f"PCP O-Proj expects input shape {expected_shape}, "
                f"got {tuple(input_.shape)}."
            )

        if mode is OProjMode.GATHER_WEIGHT:
            output = self.tp_weight.apply(input_)
        elif mode is OProjMode.EXCHANGE_HEADS:
            output = self._forward_exchange_heads(input_)
        else:
            output = self._forward_local_heads(input_)

        if self.reduce_results and self.o_proj_tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return (output, None) if self.return_bias else output

    def _forward_exchange_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Project every PCP rank's tokens with the local head shard.

        PCP pads every rank's batch to the same token count T, so the
        all-to-all sends input slice q of the local tokens to peer q and
        receives this rank's slice of all P * T tokens in rank order. The
        reduce-scatter then sums the P partial projections of each token and
        returns them to the owning rank.
        """
        num_tokens = x.shape[0]
        send = (
            x.reshape(num_tokens, self.pcp_size, self.input_size_per_partition)
            .transpose(0, 1)
            .contiguous()
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.pcp_group.device_group)
        partial = self.quant_method.apply(
            self, recv.view(-1, self.input_size_per_partition)
        )
        return self.pcp_group.reduce_scatter(partial, dim=0)

    def _forward_local_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Project the shared tokens with the local head shard."""
        shard_size = self.input_size_per_partition
        x_local = x.narrow(-1, self.pcp_rank * shard_size, shard_size).contiguous()
        return self.pcp_group.all_reduce(self.quant_method.apply(self, x_local))

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, pcp_size={self.pcp_size}"
