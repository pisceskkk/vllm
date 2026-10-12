# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Borrow a linear layer's processed weight shards from its group peers.

A layer whose weight is sharded more finely than its computation needs can run
its peers' shards instead of reassembling a larger weight. Every rank packs all
tensors of its layer, in whatever layout the quantization method left them
after ``process_weights_after_loading``, into one byte blob. A single
all-gather then gives each rank every peer's blob, and
:meth:`PeerWeightGather.use_peer` points the layer at peer ``q``'s tensors.

Running the layer's quantization method under ``use_peer(q)`` computes exactly
what rank ``q`` computes with its own shard. This needs no knowledge of the
post-load layout (swizzled or packed scales, repacked weights, per-shard
requantization) and therefore works for every linear quantization method.
"""

import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.utils.math_utils import round_up

# Byte alignment of every tensor inside a blob, enough for vectorized and
# TMA-based GEMM kernels.
_ALIGNMENT_BYTES = 256

# Layers that gather one at a time share a receive buffer per buffer key.
_RECV_BUFFERS: "WeakValueDictionary[tuple, torch.Tensor]" = WeakValueDictionary()


def _storage_numel(tensor: torch.Tensor) -> int:
    """Number of storage elements spanned by a (possibly strided) tensor."""
    return 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )


def _storage_span(tensor: torch.Tensor) -> torch.Tensor:
    """1-D view of the storage elements spanned by ``tensor``."""
    return tensor.as_strided((_storage_numel(tensor),), (1,))


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
                f"{getattr(layer, 'prefix', '')!r}, so its peer weight shards "
                "cannot be gathered."
            )


def _plan_slots(tensors: dict[str, torch.Tensor]) -> tuple[list[_TensorSlot], int]:
    slots = []
    offset = 0
    for name, tensor in tensors.items():
        slot = _TensorSlot(
            name=name,
            offset=offset,
            storage_numel=_storage_numel(tensor),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
        )
        slots.append(slot)
        offset = round_up(
            offset + slot.storage_numel * slot.dtype.itemsize, _ALIGNMENT_BYTES
        )
    return slots, offset


def _check_same_layout_on_all_ranks(
    group: GroupCoordinator, slots: list[_TensorSlot], device: torch.device
) -> None:
    layout = [(s.name, str(s.dtype), s.shape, s.stride) for s in slots]
    digest = torch.tensor([zlib.crc32(repr(layout).encode())], device=device)
    digests = group.all_gather(digest, dim=0)
    if not torch.equal(digests, digest.expand(group.world_size)):
        raise RuntimeError(
            "Peer weight shards have different post-load layouts across ranks; "
            f"local layout: {layout}."
        )


def _as_layer_tensor(view: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Wrap ``view`` so it can replace ``like`` as a layer attribute."""
    if not isinstance(like, torch.nn.Parameter):
        return view
    param = torch.nn.Parameter(view, requires_grad=False)
    param.__dict__.update(like.__dict__)
    return param


class PeerWeightGather:
    """Gathers a layer's processed tensors from every rank of ``group``.

    Build it after the layer's quantization method has processed the weights.
    Per use, call :meth:`start` to launch the asynchronous all-gather,
    :meth:`wait` before compute, then run the layer under :meth:`use_peer`.
    """

    def __init__(
        self, layer: torch.nn.Module, group: GroupCoordinator, buffer_key: str
    ) -> None:
        _check_method_holds_no_tensors(layer)
        local = _layer_tensors(layer)
        if not local:
            raise ValueError(f"Layer {getattr(layer, 'prefix', '')!r} has no tensors.")
        slots, blob_nbytes = _plan_slots(local)
        device = next(iter(local.values())).device
        _check_same_layout_on_all_ranks(group, slots, device)

        self.layer = layer
        self.group = group
        self.rank = group.rank_in_group
        self._slots = slots

        # Row q holds rank q's blob. Layers gather one at a time, so all
        # layers with the same key and blob size share one buffer.
        recv_key = (buffer_key, device, group.world_size, blob_nbytes)
        recv = _RECV_BUFFERS.get(recv_key)
        if recv is None:
            recv = torch.empty(
                (group.world_size, blob_nbytes), dtype=torch.uint8, device=device
            )
            _RECV_BUFFERS[recv_key] = recv
        self._recv = recv

        self._peer_tensors = [
            local
            if peer == self.rank
            else {
                s.name: _as_layer_tensor(s.view_in(recv[peer]), local[s.name])
                for s in slots
            }
            for peer in range(group.world_size)
        ]
        self._work: dist.Work | None = None

    @property
    def pending(self) -> bool:
        return self._work is not None

    def start(self) -> None:
        """Pack the local tensors and launch the asynchronous all-gather."""
        if self._work is not None:
            raise RuntimeError("A peer weight gather is already pending.")
        local = self._peer_tensors[self.rank]
        own_blob = self._recv[self.rank]
        for slot in self._slots:
            slot.span_in(own_blob).copy_(_storage_span(local[slot.name]))
        # In-place all-gather: the input is this rank's row of the output.
        self._work = dist.all_gather_into_tensor(
            self._recv.view(-1),
            own_blob,
            group=self.group.device_group,
            async_op=True,
        )

    def wait(self) -> None:
        if self._work is None:
            raise RuntimeError("No peer weight gather is pending.")
        work, self._work = self._work, None
        work.wait()

    @contextmanager
    def use_peer(self, peer: int) -> Iterator[None]:
        """Temporarily point the layer's tensors at ``peer``'s shard."""
        if peer == self.rank:
            yield
            return
        if self._work is not None:
            raise RuntimeError("wait() for the peer weight gather before use.")
        self._install(self._peer_tensors[peer])
        try:
            yield
        finally:
            self._install(self._peer_tensors[self.rank])

    def _install(self, tensors: dict[str, torch.Tensor]) -> None:
        for name, tensor in tensors.items():
            setattr(self.layer, name, tensor)
