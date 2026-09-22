# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.platforms import current_platform

_pcp_kv_streams: dict[int, torch.cuda.Stream] = {}


def _get_pcp_kv_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _pcp_kv_streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream()
        _pcp_kv_streams[device_index] = stream
    return stream


CacheUpdate = Callable[[tuple[torch.Tensor, ...], torch.Tensor], None]


class PCPAsyncCacheUpdate:
    """Overlap sharded-decode cache replication with this layer's attention.

    Prefill rows are gathered and written on the current stream.  The rank's
    own decode rows are also written before attention because its queries read
    those rows.  Only peer decode rows are gathered and written on the PCP KV
    stream; :meth:`finish` joins that stream after attention has completed.
    Pure decode batches stay on the synchronous path because they have no
    prefill work with which to overlap the peer-decode collective.
    """

    def __init__(self) -> None:
        self._ready_event: torch.cuda.Event | None = None
        self._done_event: torch.cuda.Event | None = None
        self._pending = False
        self._pending_tensors: tuple[torch.Tensor, ...] = ()
        self._pending_slot_mapping: torch.Tensor | None = None

    @property
    def pending(self) -> bool:
        return self._pending

    def start(
        self,
        tensors: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        num_decode_tokens: int,
        tokens_per_rank: tuple[int, ...] | None,
        decode_tokens_per_rank: tuple[int, ...] | None,
        update_cache: CacheUpdate,
    ) -> bool:
        """Start an asynchronous peer-decode replication, if it is safe.

        Returns ``False`` when metadata or the execution mode cannot support
        the split operation; callers must then use the synchronous path.
        """
        if (
            not current_platform.is_cuda()
            or BreakableCUDAGraphCapture.is_active()
            or torch.cuda.is_current_stream_capturing()
            or tokens_per_rank is None
            or decode_tokens_per_rank is None
        ):
            return False
        if self.pending:
            raise RuntimeError("Previous PCP decode cache update is still pending.")

        pcp_group = get_pcp_group()
        pcp_size = pcp_group.world_size
        pcp_rank = pcp_group.rank_in_group
        if len(tokens_per_rank) != pcp_size or len(decode_tokens_per_rank) != pcp_size:
            return False
        if any(
            decode_tokens < 0 or decode_tokens > total_tokens
            for total_tokens, decode_tokens in zip(
                tokens_per_rank, decode_tokens_per_rank
            )
        ):
            return False

        local_num_tokens = tensors[0].shape[0]
        if not all(tensor.shape[0] == local_num_tokens for tensor in tensors):
            return False
        if (
            max(tokens_per_rank) > local_num_tokens
            or decode_tokens_per_rank[pcp_rank] != num_decode_tokens
            or slot_mapping.numel() < pcp_size * local_num_tokens
        ):
            return False
        if sum(decode_tokens_per_rank) == 0:
            return False

        rank_slot_mappings = slot_mapping[: pcp_size * local_num_tokens].view(
            pcp_size, local_num_tokens
        )
        prefill_sizes = [
            total_tokens - decode_tokens
            for total_tokens, decode_tokens in zip(
                tokens_per_rank, decode_tokens_per_rank
            )
        ]
        if sum(prefill_sizes) == 0:
            return False
        local_decode_tokens = decode_tokens_per_rank[pcp_rank]

        # Preserve the existing blocking semantics for prefills. Variable-size
        # gather avoids retransmitting rank padding or sharded decode rows.
        if sum(prefill_sizes) > 0:
            local_prefill_end = tokens_per_rank[pcp_rank]
            local_prefill_tensors = [
                tensor[local_decode_tokens:local_prefill_end].contiguous()
                for tensor in tensors
            ]
            gathered_prefills = pcp_group.all_gatherv(
                local_prefill_tensors, dim=0, sizes=prefill_sizes
            )
            prefill_slots = torch.cat(
                [
                    rank_slot_mappings[rank, decode_tokens:total_tokens]
                    for rank, (total_tokens, decode_tokens) in enumerate(
                        zip(tokens_per_rank, decode_tokens_per_rank)
                    )
                ]
            )
            update_cache(tuple(gathered_prefills), prefill_slots)

        # The owner must publish its own current-token KV before attention.
        if local_decode_tokens > 0:
            update_cache(
                tuple(tensor[:local_decode_tokens] for tensor in tensors),
                rank_slot_mappings[pcp_rank, :local_decode_tokens],
            )

        current_stream = torch.cuda.current_stream(device=tensors[0].device)
        if self._ready_event is None:
            self._ready_event = torch.cuda.Event()
            self._done_event = torch.cuda.Event()
        ready_event = self._ready_event
        done_event = self._done_event
        assert done_event is not None
        ready_event.record(current_stream)
        pcp_stream = _get_pcp_kv_stream(tensors[0].device)

        # Keep inputs alive until the auxiliary stream has consumed them.
        for tensor in tensors:
            tensor.record_stream(pcp_stream)
        slot_mapping.record_stream(pcp_stream)

        with torch.cuda.stream(pcp_stream):
            pcp_stream.wait_event(ready_event)
            local_decodes = [
                tensor[:local_decode_tokens].contiguous() for tensor in tensors
            ]
            gathered_decodes = pcp_group.all_gatherv(
                local_decodes, dim=0, sizes=list(decode_tokens_per_rank)
            )

            decode_offsets = [0]
            for size in decode_tokens_per_rank:
                decode_offsets.append(decode_offsets[-1] + size)
            local_start = decode_offsets[pcp_rank]
            local_end = decode_offsets[pcp_rank + 1]
            remote_decode_count = decode_offsets[-1] - local_decode_tokens
            if remote_decode_count > 0:
                remote_decodes = tuple(
                    torch.cat((tensor[:local_start], tensor[local_end:]), dim=0)
                    for tensor in gathered_decodes
                )
                remote_slots = torch.cat(
                    [
                        rank_slot_mappings[rank, :decode_tokens]
                        for rank, decode_tokens in enumerate(decode_tokens_per_rank)
                        if rank != pcp_rank and decode_tokens > 0
                    ]
                )
                update_cache(remote_decodes, remote_slots)
            done_event.record(pcp_stream)

        self._pending = True
        self._pending_tensors = tensors
        self._pending_slot_mapping = slot_mapping
        return True

    def finish(self, wait: bool = True) -> None:
        """Join, or clear after a later same-stream update has been joined."""
        if not self._pending:
            return
        done_event = self._done_event
        assert done_event is not None
        if wait:
            torch.cuda.current_stream().wait_event(done_event)
        self._pending = False
        self._pending_tensors = ()
        self._pending_slot_mapping = None


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    shard_decode_requests: bool = False,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Gather PCP cache inputs while preserving replicated KV-cache state.

    PCP-only execution shards decode requests across ranks. In that mode each
    rank must gather the other owners' decode KV as well as partitioned prefill
    KV so every PCP rank retains a complete cache replica. DCP execution keeps
    decode requests replicated and uses the legacy prefill-only gather.
    """
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    pcp_group = get_pcp_group()
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if shard_decode_requests:
        gathered_inputs = tuple(
            pcp_group.all_gather(tensor.contiguous(), dim=0) for tensor in tensors
        )
        return gathered_inputs, gathered_slot_mapping

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    gathered_prefills = tuple(
        pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
        for tensor in tensors
    )
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
    pcp_shard_decode_requests: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
        pcp_shard_decode_requests,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
    pcp_shard_decode_requests: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens, pcp_shard_decode_requests
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
