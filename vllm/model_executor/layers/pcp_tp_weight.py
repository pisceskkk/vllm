# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild a TP-sized linear layer from weight shards spread over a PCP group.

PCP O-Proj TP keeps ``1 / P`` of the TP weight shard on each PCP rank, but a
prefill step needs the whole TP shard. :func:`build_tp_weight` prepares one of
two strategies once, after the weights are processed:

* :class:`AssembledTPWeight` (preferred) owns a hidden ``tp_layer`` with the
  same quantization method, built for the TP-sized input. Its post-load tensors
  live in buffers shared by all layers. Each prefill all-gathers the local
  post-load tensors straight into them, or into a staging buffer followed by
  one copy on a side stream, so prefill runs one ordinary TP-sized GEMM.

  Which tensors concatenate along which dimension is not declared by the
  quantization methods. At load time the raw checkpoint shards are gathered
  once, ``tp_layer`` processes them, and every post-load tensor is checked bit
  for bit to be either the concatenation of the ranks' local tensors along one
  dimension or identical to the local tensor.

* :class:`PeerShardTPWeight` (fallback) is used when a tensor fails that check,
  e.g. block scales swizzled per shard. It gathers every rank's post-load
  tensors as opaque bytes and sums the per-shard projections. That is correct
  for any row-parallel method but runs ``P`` smaller GEMMs.
"""

import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# Layers gather one at a time, so all layers that need a buffer of the same
# shape share it. Entries live as long as some layer references them.
_SHARED_BUFFERS: "WeakValueDictionary[tuple, torch.Tensor]" = WeakValueDictionary()

_REPLICATED = -1
_UNSUPPORTED = -2

_side_stream: torch.Stream | None = None


class TPWeight(Protocol):
    """TP-sized projection of a layer whose weight is also sharded over PCP."""

    def start(self) -> None:
        """Launch the asynchronous weight gather."""

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Wait for the gather and project ``x`` with the TP-sized weight."""

    def wait(self) -> None:
        """Wait for the started gather (``apply`` waits implicitly)."""


def build_tp_weight(
    local_layer: torch.nn.Module,
    tp_layer: torch.nn.Module,
    group: GroupCoordinator,
    buffer_key: str,
) -> TPWeight:
    """Prefer the assembled TP-sized layer; fall back to peer shards."""
    assembled = AssembledTPWeight.build(local_layer, tp_layer, group, buffer_key)
    if isinstance(assembled, AssembledTPWeight):
        return assembled
    logger.warning_once(
        "PCP O-Proj TP sums per-shard projections for %s because %s; "
        "prefill O-Proj is slower than a single TP-sized GEMM.",
        type(local_layer.quant_method).__name__,
        assembled,
    )
    return PeerShardTPWeight(local_layer, group, buffer_key)


def gather_raw_tp_shard(
    local_layer: torch.nn.Module, tp_layer: torch.nn.Module, group: GroupCoordinator
) -> None:
    """Fill ``tp_layer`` with the raw (pre-processing) TP shard.

    A raw parameter that is ``world_size`` times larger along one dimension is
    the concatenation of the ranks' shards along it, which is how the weight
    loader sharded it; the others are copied as they are.
    """
    local = dict(local_layer.named_parameters(recurse=False))
    for name, target in tp_layer.named_parameters(recurse=False):
        source = local[name].data
        if source.shape != target.shape:
            dim = _sharded_dim(source.shape, target.shape, group.world_size)
            if dim is None:
                raise ValueError(
                    f"Cannot rebuild {name!r} of shape {tuple(target.shape)} "
                    f"from PCP shards of shape {tuple(source.shape)}."
                )
            source = _all_gather_along(group, source, dim)
        target.data.copy_(source)


def _sharded_dim(
    shard_shape: torch.Size, full_shape: torch.Size, world_size: int
) -> int | None:
    """The single dimension along which ``full_shape`` holds every shard."""
    if len(shard_shape) != len(full_shape):
        return None
    for dim, (shard, full) in enumerate(zip(shard_shape, full_shape)):
        expected = list(shard_shape)
        expected[dim] = shard * world_size
        if list(full_shape) == expected and shard != full:
            return dim
    return None


# --- Assembled TP-sized layer ----------------------------------------------


@dataclass
class _ConcatPart:
    """One post-load tensor that is the concatenation of the ranks' tensors."""

    full: torch.Tensor  # shared buffer in the tp_layer layout
    send: torch.Tensor  # local tensor, viewed in storage order (contiguous)
    recv: torch.Tensor  # [world_size, *send.shape], contiguous
    # When recv is a staging buffer: `full` in storage order, which is the
    # concatenation of the rows of recv along `concat_dim`.
    assemble_into: torch.Tensor | None
    concat_dim: int


class AssembledTPWeight:
    """Runs prefill through a TP-sized layer refilled by all-gathers."""

    def __init__(
        self,
        tp_layer: torch.nn.Module,
        group: GroupCoordinator,
        parts: list[_ConcatPart],
    ) -> None:
        self.tp_layer = tp_layer
        self.group = group
        self._parts = parts
        device = parts[0].send.device if parts else torch.device("cpu")
        needs_assembly = any(p.assemble_into is not None for p in parts)
        self._side_stream = (
            _get_side_stream(device)
            if needs_assembly and device.type != "cpu"
            else None
        )
        self._works: list[dist.Work] = []
        self._ready: torch.Event | None = None

    @classmethod
    def build(
        cls,
        local_layer: torch.nn.Module,
        tp_layer: torch.nn.Module,
        group: GroupCoordinator,
        buffer_key: str,
    ) -> "AssembledTPWeight | str":
        """Match ``tp_layer``'s processed tensors with the local ones.

        Returns the gather, or why the post-load layout is not a
        concatenation of the local tensors.
        """
        _check_method_holds_no_tensors(tp_layer)
        local = _layer_tensors(local_layer)
        full = _layer_tensors(tp_layer)
        names = sorted(full)
        relations = [_relation(local.get(name), full[name], group) for name in names]
        # All ranks must agree so that they issue the same collectives.
        device = next(iter(full.values())).device
        codes = torch.tensor(relations, dtype=torch.int64, device=device)
        all_codes = group.all_gather(codes, dim=0).view(group.world_size, -1)
        if sorted(local) != names:
            return f"the TP-sized layer has tensors {names}, the local {sorted(local)}"
        for index, (name, relation) in enumerate(zip(names, relations)):
            if relation == _UNSUPPORTED or not bool(
                (all_codes[:, index] == relation).all()
            ):
                return f"post-load tensor {name!r} is not a concatenation of shards"

        parts: dict[str, _ConcatPart] = {}
        for name, relation in zip(names, relations):
            if relation == _REPLICATED:
                continue
            part = _plan_concat_part(
                local[name], full[name], relation, group.world_size, (buffer_key, name)
            )
            if part is None:
                return f"post-load tensor {name!r} is not dense in memory"
            parts[name] = part

        # Point tp_layer at the shared buffers, freeing its load-time tensors.
        for name in names:
            tensor = parts[name].full if name in parts else local[name]
            _set_layer_tensor(tp_layer, name, tensor, like=full[name])
        return cls(tp_layer, group, list(parts.values()))

    def start(self) -> None:
        if self._works or self._ready is not None:
            raise RuntimeError("A TP weight gather is already pending.")
        self._works = [
            dist.all_gather_into_tensor(
                part.recv.view(-1),
                part.send.reshape(-1),
                group=self.group.device_group,
                async_op=True,
            )
            for part in self._parts
        ]
        if self._side_stream is None:
            return
        # Assemble on a side stream so that the copy also overlaps attention.
        # The shared buffers may still be read by the previous layer's GEMM.
        self._side_stream.wait_stream(torch.accelerator.current_stream())
        with _use_stream(self._side_stream):
            self._wait_for_gathers()
            self._ready = torch.Event()
            self._ready.record(self._side_stream)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        self.wait()
        return self.tp_layer.quant_method.apply(self.tp_layer, x)

    def wait(self) -> None:
        self._wait_for_gathers()
        if self._ready is not None:
            torch.accelerator.current_stream().wait_event(self._ready)
            self._ready = None

    def _wait_for_gathers(self) -> None:
        if not self._works:
            return
        for work in self._works:
            work.wait()
        self._works = []
        for part in self._parts:
            if part.assemble_into is not None:
                # The batched concatenation kernel is several times faster
                # than a strided copy into the interleaved layout.
                torch.cat(
                    part.recv.unbind(0), dim=part.concat_dim, out=part.assemble_into
                )


def _relation(
    local: torch.Tensor | None, full: torch.Tensor, group: GroupCoordinator
) -> int:
    """``_REPLICATED`` if ``full`` equals ``local``; ``d`` if this rank's slice
    of ``full`` along ``d`` equals ``local``; else ``_UNSUPPORTED``."""
    if local is None or local.dtype != full.dtype or local.dim() != full.dim():
        return _UNSUPPORTED
    if local.shape == full.shape:
        return _REPLICATED if _same_bytes(local, full) else _UNSUPPORTED
    dim = _sharded_dim(local.shape, full.shape, group.world_size)
    if dim is None:
        return _UNSUPPORTED
    size = local.shape[dim]
    own_slice = full.narrow(dim, group.rank_in_group * size, size)
    return dim if _same_bytes(own_slice, local) else _UNSUPPORTED


def _plan_concat_part(
    local: torch.Tensor,
    full: torch.Tensor,
    dim: int,
    world_size: int,
    buffer_key: tuple,
) -> _ConcatPart | None:
    """Plan gathering ``full = cat(local tensors, dim)`` in storage order."""
    order = _dense_order(local)
    if order is None or _dense_order(full) != order:
        return None
    send = local.permute(order)
    dense_dim = order.index(dim)
    full_buffer = _shared_buffer(
        (*buffer_key, "full", full.shape, full.stride(), full.dtype),
        lambda: torch.empty_strided(
            full.shape, full.stride(), dtype=full.dtype, device=full.device
        ),
    )
    full_dense = full_buffer.permute(order)
    if dense_dim == 0:
        # Each rank's chunk is contiguous in the final layout: gather in place.
        by_rank = full_dense.unflatten(0, (world_size, send.shape[0]))
        return _ConcatPart(full_buffer, send, by_rank, None, dense_dim)
    staging = _shared_buffer(
        (*buffer_key, "staging", send.shape, send.dtype),
        lambda: torch.empty(
            (world_size, *send.shape), dtype=send.dtype, device=send.device
        ),
    )
    return _ConcatPart(full_buffer, send, staging, full_dense, dense_dim)


# --- Fallback: sum of per-shard projections --------------------------------

# Byte alignment of every tensor inside a fallback blob.
_ALIGNMENT_BYTES = 256


@dataclass(frozen=True)
class _TensorSlot:
    """Location and strided layout of one layer tensor inside a blob."""

    name: str
    offset: int
    storage_numel: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def span_in(self, blob: torch.Tensor) -> torch.Tensor:
        nbytes = self.storage_numel * self.dtype.itemsize
        return blob[self.offset : self.offset + nbytes].view(self.dtype)

    def view_in(self, blob: torch.Tensor) -> torch.Tensor:
        return self.span_in(blob).as_strided(self.shape, self.stride)


class PeerShardTPWeight:
    """Sums the projections of every PCP peer's opaque post-load shard.

    Every rank packs all of its post-load tensors into one byte blob, so the
    layout does not matter: running the layer with peer ``q``'s tensors
    computes what rank ``q`` computes with its own shard.
    """

    def __init__(
        self, layer: torch.nn.Module, group: GroupCoordinator, buffer_key: str
    ) -> None:
        _check_method_holds_no_tensors(layer)
        local = _layer_tensors(layer)
        slots = []
        offset = 0
        for name, tensor in local.items():
            storage_numel = 1 + sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride())
            )
            slots.append(
                _TensorSlot(
                    name,
                    offset,
                    storage_numel,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
            )
            offset += storage_numel * tensor.itemsize
            offset = round_up(offset, _ALIGNMENT_BYTES)
        device = next(iter(local.values())).device
        layout = [(s.name, str(s.dtype), s.shape, s.stride) for s in slots]
        digest = torch.tensor([zlib.crc32(repr(layout).encode())], device=device)
        if not torch.equal(
            group.all_gather(digest, dim=0), digest.expand(group.world_size)
        ):
            raise RuntimeError(
                f"Post-load layouts differ across PCP ranks; local: {layout}."
            )

        self.layer = layer
        self.group = group
        self.rank = group.rank_in_group
        self._slots = slots
        self._recv = _shared_buffer(
            (buffer_key, "blob", device, group.world_size, offset),
            lambda: torch.empty(
                (group.world_size, offset), dtype=torch.uint8, device=device
            ),
        )
        self._peer_tensors = [
            local
            if peer == self.rank
            else {
                s.name: _like(s.view_in(self._recv[peer]), local[s.name]) for s in slots
            }
            for peer in range(group.world_size)
        ]
        self._work: dist.Work | None = None

    def start(self) -> None:
        if self._work is not None:
            raise RuntimeError("A TP weight gather is already pending.")
        local = self._peer_tensors[self.rank]
        own_blob = self._recv[self.rank]
        for slot in self._slots:
            span = local[slot.name].as_strided((slot.storage_numel,), (1,))
            slot.span_in(own_blob).copy_(span)
        # In-place all-gather: the input is this rank's row of the output.
        self._work = dist.all_gather_into_tensor(
            self._recv.view(-1),
            own_blob,
            group=self.group.device_group,
            async_op=True,
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        self.wait()
        shard_size = x.shape[-1] // self.group.world_size
        output: torch.Tensor | None = None
        for peer, x_peer in enumerate(x.split(shard_size, dim=-1)):
            with self._use_peer(peer):
                partial = self.layer.quant_method.apply(self.layer, x_peer.contiguous())
            output = partial if output is None else output.add_(partial)
        assert output is not None
        return output

    def wait(self) -> None:
        if self._work is not None:
            self._work.wait()
            self._work = None

    @contextmanager
    def _use_peer(self, peer: int) -> Iterator[None]:
        if peer == self.rank:
            yield
            return
        for name, tensor in self._peer_tensors[peer].items():
            setattr(self.layer, name, tensor)
        try:
            yield
        finally:
            for name, tensor in self._peer_tensors[self.rank].items():
                setattr(self.layer, name, tensor)


# --- Helpers ----------------------------------------------------------------


def _shared_buffer(key: tuple, make: Callable[[], torch.Tensor]) -> torch.Tensor:
    buffer = _SHARED_BUFFERS.get(key)
    if buffer is None:
        buffer = make()
        _SHARED_BUFFERS[key] = buffer
    return buffer


def _get_side_stream(device: torch.device) -> torch.Stream:
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.Stream(device=device)
    return _side_stream


@contextmanager
def _use_stream(stream: torch.Stream) -> Iterator[None]:
    previous = torch.accelerator.current_stream()
    torch.accelerator.set_stream(stream)
    try:
        yield
    finally:
        torch.accelerator.set_stream(previous)


def _layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every non-empty tensor the layer owns directly."""
    tensors: dict[str, torch.Tensor] = {}
    tensors.update(layer.named_parameters(recurse=False))
    tensors.update(layer.named_buffers(recurse=False))
    tensors.update(
        (name, value)
        for name, value in vars(layer).items()
        if isinstance(value, torch.Tensor)
    )
    return {name: t for name, t in tensors.items() if t.numel() > 0}


def _check_method_holds_no_tensors(layer: torch.nn.Module) -> None:
    """Fail closed if weight data may live outside the layer's attributes."""
    method = getattr(layer, "quant_method", None)
    for owner in (method, getattr(method, "kernel", None)):
        hidden = [
            name
            for name, value in getattr(owner, "__dict__", {}).items()
            if isinstance(value, torch.Tensor)
        ]
        if hidden:
            raise ValueError(
                f"{type(owner).__name__} keeps tensors {hidden} outside layer "
                f"{getattr(layer, 'prefix', '')!r}, so its weight shards "
                "cannot be gathered."
            )


def _like(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Wrap ``tensor`` as a Parameter with ``like``'s attributes if needed."""
    if not isinstance(like, torch.nn.Parameter) or isinstance(
        tensor, torch.nn.Parameter
    ):
        return tensor
    param = torch.nn.Parameter(tensor, requires_grad=False)
    param.__dict__.update(like.__dict__)
    return param


def _set_layer_tensor(
    layer: torch.nn.Module, name: str, tensor: torch.Tensor, like: torch.Tensor
) -> None:
    setattr(layer, name, _like(tensor, like))


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )


def _all_gather_along(
    group: GroupCoordinator, tensor: torch.Tensor, dim: int
) -> torch.Tensor:
    """Concatenate every rank's ``tensor`` along ``dim``, for any dtype."""
    leading = tensor.movedim(dim, 0).contiguous()
    gathered = group.all_gather(leading.reshape(-1).view(torch.uint8), dim=0)
    shape = (group.world_size * leading.shape[0], *leading.shape[1:])
    return gathered.view(tensor.dtype).view(shape).movedim(0, dim)


def _dense_order(tensor: torch.Tensor) -> list[int] | None:
    """Dimension order in which ``tensor`` is a contiguous view, if any."""
    order = sorted(range(tensor.dim()), key=lambda d: -tensor.stride(d))
    return order if tensor.permute(order).is_contiguous() else None
