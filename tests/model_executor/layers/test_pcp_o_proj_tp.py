# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP O-Proj TP across thread-simulated PCP ranks.

Every simulated rank owns a real ``PCPOProjRowParallelLinear`` that loads its
TP x PCP shard through the layer's weight loader and runs its quantization
method. Only the collectives are simulated, by exchanging tensors between the
rank threads, so each mode is checked end to end against a TP-only O-Proj.
"""

import contextlib
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers import pcp_o_proj, pcp_tp_weight
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    RowParallelLinear,
)
from vllm.model_executor.layers.pcp_o_proj import (
    OProjMode,
    PCPOProjRowParallelLinear,
)
from vllm.model_executor.layers.pcp_tp_weight import (
    AssembledTPWeight,
    PeerShardTPWeight,
    build_tp_weight,
    gather_raw_tp_shard,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.utils.torch_utils import set_default_torch_dtype

PCP_SIZE = 2
NUM_TOKENS = 5


class _Collectives:
    """Tensor exchange between the rank threads of one simulated group."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=120)
        self._values: list[Any] = [None] * world_size

    def exchange(self, rank: int, value: torch.Tensor) -> list[torch.Tensor]:
        self._values[rank] = value.clone()
        self.barrier.wait()
        values = list(self._values)
        self.barrier.wait()
        return values


class _FakeGroup:
    """GroupCoordinator stand-in for one simulated rank."""

    def __init__(self, collectives: _Collectives, rank: int) -> None:
        self.collectives = collectives
        self.rank_in_group = rank
        self.world_size = collectives.world_size
        self.device_group = self

    def exchange(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        return self.collectives.exchange(self.rank_in_group, tensor)

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cat(self.exchange(tensor), dim)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.exchange(tensor)).sum(0)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        total = self.all_reduce(tensor)
        return total.chunk(self.world_size, dim)[self.rank_in_group].contiguous()


class _DoneWork:
    def wait(self) -> None:
        pass


def _all_gather_into_tensor(output, input_, group, async_op=False):
    output.copy_(torch.cat(group.exchange(input_)).view_as(output))
    return _DoneWork()


def _all_to_all_single(output, input_, group):
    received = [
        sent.chunk(group.world_size)[group.rank_in_group]
        for sent in group.exchange(input_)
    ]
    output.copy_(torch.cat(received))


class _PerThreadBuffers(threading.local):
    """Gives every simulated rank its own receive buffers."""

    def __init__(self) -> None:
        self.buffers: dict = {}

    def get(self, key):
        return self.buffers.get(key)

    def __setitem__(self, key, value) -> None:
        self.buffers[key] = value


def _run_ranks(collectives: _Collectives, fn: Callable[[int], Any]) -> list[Any]:
    results: list[Any] = [None] * collectives.world_size
    errors: list[BaseException] = []

    def run(rank: int) -> None:
        try:
            results[rank] = fn(rank)
        except BaseException as e:
            errors.append(e)
            collectives.barrier.abort()

    threads = [
        threading.Thread(target=run, args=(rank,))
        for rank in range(collectives.world_size)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


@pytest.fixture
def simulated_pcp(monkeypatch, default_vllm_config):
    """Route PCP O-Proj collectives through thread-simulated PCP ranks."""
    # Actual ranks have independent default dtypes. Serialize the temporary
    # dtype switch used to construct the TP layer in these simulated ranks.
    dtype_lock = threading.Lock()

    @contextlib.contextmanager
    def rank_default_dtype(dtype):
        with dtype_lock, set_default_torch_dtype(dtype):
            yield

    monkeypatch.setattr(pcp_o_proj, "set_default_torch_dtype", rank_default_dtype)
    fake_dist = SimpleNamespace(
        all_gather_into_tensor=_all_gather_into_tensor,
        all_to_all_single=_all_to_all_single,
        Work=_DoneWork,
    )
    monkeypatch.setattr(pcp_o_proj, "dist", fake_dist)
    monkeypatch.setattr(pcp_tp_weight, "dist", fake_dist)
    monkeypatch.setattr(pcp_tp_weight, "_SHARED_BUFFERS", _PerThreadBuffers())
    for module in (pcp_o_proj, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    return monkeypatch


def _set_global_has_prefill(monkeypatch, has_prefill: bool | None) -> None:
    context = SimpleNamespace(global_has_prefill=has_prefill)
    monkeypatch.setattr(pcp_o_proj, "get_forward_context", lambda: context)


def _load_raw(layer: torch.nn.Module, checkpoint: dict[str, torch.Tensor]) -> None:
    for name, tensor in checkpoint.items():
        param = getattr(layer, name)
        param.weight_loader(param, tensor.to(param.device))


def _load(layer: torch.nn.Module, checkpoint: dict[str, torch.Tensor]) -> None:
    _load_raw(layer, checkpoint)
    layer.quant_method.process_weights_after_loading(layer)


def _finish_loading(layer: PCPOProjRowParallelLinear) -> None:
    """The model loader's post-load sequence for one PCP O-Proj layer."""
    layer.on_weights_loaded()
    layer.quant_method.process_weights_after_loading(layer)
    layer.on_weights_processed()


def _build_pcp_layers(
    monkeypatch,
    collectives: _Collectives,
    checkpoint: dict[str, torch.Tensor],
    *,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    device: str,
    quant_config=None,
    shard_decode_requests: bool = True,
) -> list[PCPOProjRowParallelLinear]:
    parallel_config = SimpleNamespace(pcp_shard_decode_requests=shard_decode_requests)
    monkeypatch.setattr(
        pcp_o_proj,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=parallel_config),
    )
    # The stub config is not a VllmConfig; keep the fixture's real one current.
    monkeypatch.setattr(
        pcp_o_proj, "set_current_vllm_config", lambda config: contextlib.nullcontext()
    )
    layers = []
    for rank in range(collectives.world_size):
        group = _FakeGroup(collectives, rank)
        monkeypatch.setattr(pcp_o_proj, "get_pcp_group", lambda group=group: group)
        with torch.device(device), set_default_torch_dtype(params_dtype):
            layer = PCPOProjRowParallelLinear(
                input_size,
                output_size,
                params_dtype=params_dtype,
                quant_config=quant_config,
                reduce_results=False,
                prefix="o_proj",
                prefill_weight_gather=collectives.world_size > 1,
            )
        _load_raw(layer, checkpoint)
        layers.append(layer)
    _run_ranks(collectives, lambda rank: _finish_loading(layers[rank]))
    return layers


def _build_row_parallel_layer(
    checkpoint: dict[str, torch.Tensor],
    *,
    tp_rank: int,
    tp_size: int,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    device: str,
    quant_config=None,
) -> RowParallelLinear:
    with torch.device(device), set_default_torch_dtype(params_dtype):
        layer = RowParallelLinear(
            input_size,
            output_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
    _load(layer, checkpoint)
    return layer


def _relative_error(output: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((output.float() - expected).norm() / expected.norm()).item()


# (has_prefill, shard_decode_requests, expected mode)
MODES = [
    pytest.param(True, True, OProjMode.GATHER_WEIGHT, id="prefill"),
    pytest.param(False, True, OProjMode.EXCHANGE_HEADS, id="decode_sharded"),
    pytest.param(False, False, OProjMode.LOCAL_HEADS, id="decode_replicated"),
]
MODE_ARGS = "has_prefill,shard_decode_requests,mode"


def _check_modes_match_tp_o_proj(
    monkeypatch,
    checkpoint: dict[str, torch.Tensor],
    *,
    has_prefill: bool,
    shard_decode_requests: bool,
    mode: OProjMode,
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    device: str,
    quant_config=None,
    expect_assembled: bool = True,
    rtol: float = 1e-5,
    tp_rtol: float = 1e-5,
) -> None:
    collectives = _Collectives(PCP_SIZE)
    layer_args = dict(
        input_size=input_size,
        output_size=output_size,
        params_dtype=params_dtype,
        device=device,
        quant_config=quant_config,
    )
    layers = _build_pcp_layers(
        monkeypatch,
        collectives,
        checkpoint,
        shard_decode_requests=shard_decode_requests,
        **layer_args,
    )
    strategy = AssembledTPWeight if expect_assembled else PeerShardTPWeight
    assert all(isinstance(layer.tp_weight, strategy) for layer in layers)

    # Decode, and prefill without an assembled TP layer, compute standard
    # row-parallel TP over all TP x PCP shards. With dynamic activation
    # quantization every input slice gets its own scale, so this differs
    # slightly from the single TP-sized O-Proj that assembled prefill runs.
    shards = [
        _build_row_parallel_layer(
            checkpoint, tp_rank=rank, tp_size=PCP_SIZE, **layer_args
        )
        for rank in range(PCP_SIZE)
    ]
    tp_layer = _build_row_parallel_layer(checkpoint, tp_rank=0, tp_size=1, **layer_args)

    def row_parallel_reference(x: torch.Tensor) -> torch.Tensor:
        x_shards = x.split(input_size // PCP_SIZE, dim=-1)
        partials = [shard(x_s.contiguous())[0] for shard, x_s in zip(shards, x_shards)]
        return torch.stack(partials).sum(0)

    # Replicated decode gives every rank the same tokens.
    replicated = mode is OProjMode.LOCAL_HEADS
    inputs = [
        torch.randn(NUM_TOKENS, input_size, dtype=params_dtype, device=device)
        for _ in range(PCP_SIZE)
    ]
    if replicated:
        inputs = [inputs[0]] * PCP_SIZE

    _set_global_has_prefill(monkeypatch, has_prefill)

    def forward(rank: int) -> torch.Tensor:
        layer = layers[rank]
        layer.prefetch_weights()
        assert layer._mode is mode
        output, output_bias = layer(inputs[rank])
        assert output_bias is None
        return output

    def tp_reference(x: torch.Tensor) -> torch.Tensor:
        return tp_layer(x)[0]

    gathers_assembled = mode is OProjMode.GATHER_WEIGHT and expect_assembled
    exact_reference = tp_reference if gathers_assembled else row_parallel_reference
    for x, output in zip(inputs, _run_ranks(collectives, forward)):
        assert _relative_error(output, exact_reference(x)) <= rtol
        assert _relative_error(output, tp_reference(x)) <= tp_rtol


@pytest.mark.parametrize(MODE_ARGS, MODES)
def test_unquantized_modes_match_tp_o_proj(
    simulated_pcp, has_prefill, shard_decode_requests, mode
):
    input_size, output_size = 8, 6
    checkpoint = {"weight": torch.randn(output_size, input_size)}
    _check_modes_match_tp_o_proj(
        simulated_pcp,
        checkpoint,
        has_prefill=has_prefill,
        shard_decode_requests=shard_decode_requests,
        mode=mode,
        input_size=input_size,
        output_size=output_size,
        params_dtype=torch.float32,
        device="cpu",
    )


class _ScrambledLinearMethod(LinearMethodBase):
    """Post-load layout that no layout-aware weight gather could know about.

    The weight is rescaled by a per-shard scale, permuted along the input axis
    and stored as a transposed view. The scale is a buffer and the inverse
    permutation is a plain tensor attribute.
    """

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        weight = torch.empty(
            sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype
        )
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=weight,
                input_dim=1,
                output_dim=0,
                weight_loader=extra_weight_attrs["weight_loader"],
            ),
        )

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        scale = weight.abs().amax()
        generator = torch.Generator().manual_seed(0)
        perm = torch.randperm(weight.shape[1], generator=generator)
        scrambled = (weight / scale)[:, perm].t().contiguous().t()
        layer.weight = torch.nn.Parameter(scrambled, requires_grad=False)
        layer.register_buffer("scale", scale)
        layer.inverse_perm = torch.argsort(perm)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight[:, layer.inverse_perm] * layer.scale, bias)


class _ScrambledQuantConfig(SimpleNamespace):
    def get_quant_method(self, layer, prefix):
        return _ScrambledLinearMethod()


@pytest.mark.parametrize(MODE_ARGS, MODES)
def test_peer_gather_is_layout_agnostic(
    simulated_pcp,
    monkeypatch,
    has_prefill,
    shard_decode_requests,
    mode,
):
    from vllm.model_executor.layers import linear as linear_module

    monkeypatch.setattr(
        linear_module,
        "resolve_quant_method",
        lambda config, layer, prefix: config.get_quant_method(layer, prefix),
    )
    input_size, output_size = 8, 6
    checkpoint = {"weight": torch.randn(output_size, input_size)}
    _check_modes_match_tp_o_proj(
        simulated_pcp,
        checkpoint,
        has_prefill=has_prefill,
        shard_decode_requests=shard_decode_requests,
        mode=mode,
        input_size=input_size,
        output_size=output_size,
        params_dtype=torch.float32,
        device="cpu",
        quant_config=_ScrambledQuantConfig(),
        expect_assembled=False,
    )


def _fp8_block_checkpoint(output_size, input_size):
    weight = 0.5 * torch.randn(output_size, input_size)
    scale_inv = 0.005 + 0.01 * torch.rand(output_size // 128, input_size // 128)
    return {"weight": weight.to(torch.float8_e4m3fn), "weight_scale_inv": scale_inv}


def _fp8_tensor_checkpoint(output_size, input_size):
    return {
        "weight": (0.5 * torch.randn(output_size, input_size)).to(torch.float8_e4m3fn),
        "weight_scale": torch.tensor(0.01),
    }


def _gptq_int4_checkpoint(output_size, input_size, group_size=128):
    num_groups = input_size // group_size
    return {
        "qweight": torch.randint(
            -(2**31), 2**31 - 1, (input_size // 8, output_size), dtype=torch.int32
        ),
        "scales": (0.005 + 0.005 * torch.rand(num_groups, output_size)).half(),
        # Symmetric GPTQ stores zero point 8 in every nibble.
        "qzeros": torch.full(
            (num_groups, output_size // 8), 0x88888888 - 2**32, dtype=torch.int32
        ),
    }


def _fp8_block_config():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    return Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])


def _fp8_tensor_config():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    return Fp8Config(is_checkpoint_fp8_serialized=True)


def _gptq_int4_config():
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    return AutoGPTQConfig(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA kernels")
@pytest.mark.parametrize(MODE_ARGS, MODES)
@pytest.mark.parametrize(
    "make_config,make_checkpoint,params_dtype",
    [
        pytest.param(None, None, torch.bfloat16, id="bf16"),
        pytest.param(_fp8_block_config, _fp8_block_checkpoint, torch.bfloat16,
                     id="fp8_block"),
        pytest.param(_fp8_tensor_config, _fp8_tensor_checkpoint, torch.bfloat16,
                     id="fp8_tensor"),
        pytest.param(_gptq_int4_config, _gptq_int4_checkpoint, torch.float16,
                     id="gptq_marlin_int4"),
    ],
)  # fmt: skip
def test_quantized_modes_match_tp_o_proj(
    simulated_pcp,
    dist_init,
    default_vllm_config,
    make_config,
    make_checkpoint,
    params_dtype,
    has_prefill,
    shard_decode_requests,
    mode,
):
    default_vllm_config.model_config = SimpleNamespace(
        dtype=params_dtype, hf_text_config=SimpleNamespace(model_type=None)
    )
    input_size, output_size = 1024, 256
    if make_checkpoint is None:
        checkpoint = {
            "weight": torch.randn(output_size, input_size, dtype=params_dtype)
        }
    else:
        checkpoint = make_checkpoint(output_size, input_size)
    _check_modes_match_tp_o_proj(
        simulated_pcp,
        checkpoint,
        has_prefill=has_prefill,
        shard_decode_requests=shard_decode_requests,
        mode=mode,
        input_size=input_size,
        output_size=output_size,
        params_dtype=params_dtype,
        device="cuda",
        quant_config=None if make_config is None else make_config(),
        rtol=1e-3,
        tp_rtol=5e-2,
    )


def test_grouped_o_proj_gathers_column_and_row_shards(simulated_pcp):
    """DeepSeek-V4 style O-Proj: grouped ``wo_a`` followed by ``wo_b``.

    ``wo_a`` is column-parallel over head groups and ``wo_b`` is row-parallel
    over the grouped low-rank features. Sharding both over PCP and gathering
    them reproduces the unsharded projection with TP-sized weights.
    """
    num_groups, group_dim, rank_dim, hidden = 4, 6, 3, 5
    wo_a_weight = torch.randn(num_groups * rank_dim, group_dim)
    wo_b_weight = torch.randn(hidden, num_groups * rank_dim)
    collectives = _Collectives(PCP_SIZE)

    def build(cls, input_size, output_size, rank, tp_size):
        return cls(input_size, output_size, bias=False, params_dtype=torch.float32,
                   tp_rank=rank, tp_size=tp_size)  # fmt: skip

    shapes = {
        "wo_a": (ColumnParallelLinear, group_dim, num_groups * rank_dim, wo_a_weight),
        "wo_b": (RowParallelLinear, num_groups * rank_dim, hidden, wo_b_weight),
    }
    local_layers = []
    for rank in range(PCP_SIZE):
        layers = {}
        for name, (cls, input_size, output_size, weight) in shapes.items():
            layers[name] = build(cls, input_size, output_size, rank, PCP_SIZE)
            _load_raw(layers[name], {"weight": weight})
        local_layers.append(layers)

    def load(rank: int) -> dict:
        group = _FakeGroup(collectives, rank)
        tp_weights = {}
        for name, (cls, input_size, output_size, _) in shapes.items():
            local = local_layers[rank][name]
            tp_layer = build(cls, input_size, output_size, 0, 1)
            gather_raw_tp_shard(local, tp_layer, group)
            local.quant_method.process_weights_after_loading(local)
            tp_layer.quant_method.process_weights_after_loading(tp_layer)
            tp_weights[name] = build_tp_weight(local, tp_layer, group, name)
            assert isinstance(tp_weights[name], AssembledTPWeight)
        return tp_weights

    tp_weights = _run_ranks(collectives, load)
    inputs = [torch.randn(NUM_TOKENS, num_groups, group_dim) for _ in range(PCP_SIZE)]

    def prefill(rank: int) -> torch.Tensor:
        wo_a, wo_b = tp_weights[rank]["wo_a"], tp_weights[rank]["wo_b"]
        wo_a.start()
        wo_b.start()
        wo_a.wait()
        grouped = wo_a.tp_layer.weight.view(num_groups, rank_dim, group_dim)
        z = torch.einsum("tgd,grd->tgr", inputs[rank], grouped).flatten(1)
        return wo_b.apply(z)

    grouped_weight = wo_a_weight.view(num_groups, rank_dim, group_dim)
    for x, output in zip(inputs, _run_ranks(collectives, prefill)):
        z = torch.einsum("tgd,grd->tgr", x, grouped_weight).flatten(1)
        torch.testing.assert_close(output, F.linear(z, wo_b_weight))


def test_forward_requires_prefetch(simulated_pcp):
    collectives = _Collectives(PCP_SIZE)
    layers = _build_pcp_layers(
        simulated_pcp,
        collectives,
        {"weight": torch.randn(3, 4)},
        input_size=4,
        output_size=3,
        params_dtype=torch.float32,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="prefetch_weights"):
        layers[0](torch.randn(2, 4))


def test_prefetch_requires_global_batch_type(simulated_pcp):
    collectives = _Collectives(PCP_SIZE)
    layers = _build_pcp_layers(
        simulated_pcp,
        collectives,
        {"weight": torch.randn(3, 4)},
        input_size=4,
        output_size=3,
        params_dtype=torch.float32,
        device="cpu",
    )
    _set_global_has_prefill(simulated_pcp, None)
    with pytest.raises(RuntimeError, match="global batch type"):
        layers[0].prefetch_weights()


def test_wrong_input_width_drains_pending_gather(simulated_pcp):
    collectives = _Collectives(PCP_SIZE)
    layers = _build_pcp_layers(
        simulated_pcp,
        collectives,
        {"weight": torch.randn(3, 4)},
        input_size=4,
        output_size=3,
        params_dtype=torch.float32,
        device="cpu",
    )
    _set_global_has_prefill(simulated_pcp, True)

    def forward(rank: int) -> None:
        layers[rank].prefetch_weights()
        with pytest.raises(ValueError, match="expects input shape"):
            layers[rank](torch.randn(2, 2))
        # Nothing is pending, so a new gather can start.
        layers[rank].tp_weight.start()
        layers[rank].tp_weight.wait()

    _run_ranks(collectives, forward)


def test_method_holding_tensors_is_rejected(simulated_pcp):
    class _TableMethod(_ScrambledLinearMethod):
        def process_weights_after_loading(self, layer):
            self.table = torch.zeros(1)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    layer.quant_method = _TableMethod()
    layer.quant_method.process_weights_after_loading(layer)
    with pytest.raises(ValueError, match="keeps tensors"):
        PeerShardTPWeight(layer, _FakeGroup(_Collectives(1), 0), "test")


@pytest.mark.parametrize("has_prefill", [True, False])
def test_without_pcp_builds_no_tp_weight(simulated_pcp, has_prefill):
    """With PCP disabled the model turns the prefill weight gather off: no
    TP-sized layer or shared buffers exist and O-Proj is a plain TP O-Proj."""
    collectives = _Collectives(1)
    weight = torch.randn(3, 4)
    (layer,) = _build_pcp_layers(
        simulated_pcp,
        collectives,
        {"weight": weight},
        input_size=4,
        output_size=3,
        params_dtype=torch.float32,
        device="cpu",
    )
    assert layer._tp_layer is None and layer._tp_weight is None
    _set_global_has_prefill(simulated_pcp, has_prefill)
    x = torch.randn(NUM_TOKENS, 4)

    def forward(rank: int) -> torch.Tensor:
        layer.prefetch_weights()
        assert layer._mode is OProjMode.LOCAL_HEADS
        return layer(x)[0]

    (output,) = _run_ranks(collectives, forward)
    torch.testing.assert_close(output, F.linear(x, weight))


def test_prefill_weight_gather_must_follow_pcp(simulated_pcp):
    simulated_pcp.setattr(
        pcp_o_proj, "get_pcp_group", lambda: _FakeGroup(_Collectives(PCP_SIZE), 0)
    )
    with pytest.raises(ValueError, match="exactly when PCP is enabled"):
        PCPOProjRowParallelLinear(4, 3, prefill_weight_gather=False)
