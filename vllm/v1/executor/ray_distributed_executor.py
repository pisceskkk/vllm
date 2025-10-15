# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from typing import Optional, Union, Generator
import copy

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.ray_distributed_executor import (  # noqa
    RayDistributedExecutor as RayDistributedExecutorV0)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData, NewRequestData
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class FutureWrapper(Future):
    """A wrapper around Ray output reference to meet the interface
    of .execute_model(): The top level (core busy loop) expects .result() api 
    to block and return a single output.
    
    If aggregator is provided, the outputs from all workers are aggregated upon 
    the result() call. If not only the first worker's output is returned.
    """

    def __init__(self, refs, aggregator: Optional[KVOutputAggregator] = None):
        super().__init__()
        self.refs = refs
        self.aggregator = aggregator

    def result(self, timeout=None):
        if timeout is not None:
            raise NotImplementedError("timeout is not supported")

        if self.aggregator is None:
            return merge_refs_results(self.refs)

        outputs = [ref.get() for ref in self.refs]
        return self.aggregator.aggregate(outputs, output_rank=0)


class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    """Ray distributed executor using Ray Compiled Graphs."""

    supports_pp: bool = True

    def _init_executor(self) -> None:
        super()._init_executor()

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Execute the model on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        # refs = self.forward_dag.execute(scheduler_output)  # type: ignore
        refs = []
        for scheduler_output_with_signle_req in split_scheduler_output(scheduler_output):
            refs.extend(self.forward_dag.execute(scheduler_output_with_signle_req))  # type: ignore

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if self.max_concurrent_batches == 1:
                return refs[0].get()

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs)

        # Get output from all workers when connector is present
        if self.max_concurrent_batches == 1:
            # Block and get results from all workers
            outputs = [ref.get() for ref in refs]
            return self.kv_output_aggregator.aggregate(outputs)

        # Return a future that will aggregate outputs from all workers
        return FutureWrapper(refs, self.kv_output_aggregator)

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self._run_workers("reinitialize_distributed", reconfig_request)
        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            self.shutdown()
        return


def split_scheduler_output_with_multichunks(
    scheduler_output: SchedulerOutput) -> Generator[SchedulerOutput, None, None]:
    assert scheduler_output.grammar_bitmask is None, \
        f"Please implement the relevant logic for not-None" \
        f"grammar_bitmask {scheduler_output.grammar_bitmask}"
    assert scheduler_output.kv_connector_metadata is None, \
        f"Please implement the relevant logic for not-None" \
        f"kv_connector_metadata {scheduler_output.kv_connector_metadata}"
    for req in scheduler_output.scheduled_new_reqs:
        new_req = slice_new_request(req, 0, req.chunks_list[0])
        yield get_new_scheduler_output(scheduler_output=scheduler_output,
                                       req_ids=[req.req_id],
                                       scheduled_new_reqs=[new_req],
                                       num_scheduled_tokens={req.req_id:req.chunks_list[0]})
        num_computed_tokens = req.num_computed_tokens + req.chunks_list[0]
        for chunk_idx, chunk_size in enumerate(req.chunks_list[1:], start=1):
            yield get_new_scheduler_output(scheduler_output=scheduler_output,
                                           req_ids=[req.req_id],
                                           scheduled_cached_reqs=CachedRequestData(
                                               req_ids=[req.req_id],
                                               resumed_from_preemption=[False],
                                               new_token_ids=[req.all_token_ids[num_computed_tokens:num_computed_tokens+chunk_size]],
                                               new_block_ids=[()],
                                               num_computed_tokens=[num_computed_tokens],
                                               chunks_lists=[req.chunks_list[chunk_idx:]],
                                               ),
                                           num_scheduled_tokens={req.req_id:chunk_size})
            num_computed_tokens += chunk_size


def slice_new_request(request: NewRequestData, start_id:int, end_id:int) -> NewRequestData:
    new_req = copy.deepcopy(request)
    new_req.prompt_token_ids = new_req.prompt_token_ids[start_id:end_id]
    new_req.mm_kwargs = new_req.mm_kwargs[start_id:end_id]
    new_req.mm_hashes = new_req.mm_hashes[start_id:end_id]
    new_req.mm_positions = new_req.mm_positions[start_id:end_id]
    block_size = len(new_req.block_ids[0])
    new_req.block_ids = new_req.block_ids[start_id//block_size:(end_id+block_size-1)//block_size]
    return new_req


def split_scheduler_output(scheduler_output: SchedulerOutput) -> Generator[SchedulerOutput, None, None]:
    assert scheduler_output.grammar_bitmask is None, \
        f"Please implement the relevant logic for not-None" \
        f"grammar_bitmask {scheduler_output.grammar_bitmask}"
    assert scheduler_output.kv_connector_metadata is None, \
        f"Please implement the relevant logic for not-None" \
        f"kv_connector_metadata {scheduler_output.kv_connector_metadata}"
    for req in scheduler_output.scheduled_new_reqs:
        yield get_new_scheduler_output(scheduler_output=scheduler_output,
                                       req_ids=[req.req_id],
                                       scheduled_new_reqs=[req])
    decoding_reqs_list = []
    for cached_req in split_cached_request_data(scheduler_output.scheduled_cached_reqs):
        req_id = cached_req.req_ids[0]
        if scheduler_output.num_scheduled_tokens[req_id] == 1:
            decoding_reqs_list.append(cached_req)
            continue
        yield get_new_scheduler_output(scheduler_output=scheduler_output,
                                       req_ids=[req_id],
                                       scheduled_cached_reqs=cached_req)
    if len(decoding_reqs_list) != 0:
        cached_reqs = merge_cached_request_data(decoding_reqs_list)
        yield get_new_scheduler_output(scheduler_output=scheduler_output,
                                       req_ids=cached_reqs.req_ids,
                                       scheduled_cached_reqs=cached_reqs)

def get_new_scheduler_output(scheduler_output: SchedulerOutput, 
                             req_ids: list[str],
                             scheduled_new_reqs: list[NewRequestData]=[],
                             scheduled_cached_reqs: CachedRequestData=CachedRequestData.make_empty(), 
                             num_scheduled_tokens: Optional[dict[str, int]] = None,
                             ) -> SchedulerOutput:
    if num_scheduled_tokens is None:
        num_scheduled_tokens={
            req_id:scheduler_output.num_scheduled_tokens[req_id]
                for req_id in req_ids
        }
    return SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=scheduled_cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            num_common_prefix_blocks=scheduler_output.num_common_prefix_blocks,
            scheduled_spec_decode_tokens=scheduler_output.scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduler_output.scheduled_encoder_inputs,
            finished_req_ids=scheduler_output.finished_req_ids,
            free_encoder_mm_hashes=scheduler_output.free_encoder_mm_hashes,
            structured_output_request_ids=scheduler_output.structured_output_request_ids,
            grammar_bitmask=scheduler_output.grammar_bitmask,
            kv_connector_metadata=scheduler_output.kv_connector_metadata
        )

def split_cached_request_data(cached_reqs:CachedRequestData) -> Generator[CachedRequestData, None, None]:
    for i in range(cached_reqs.num_reqs):
        yield CachedRequestData(
            req_ids=cached_reqs.req_ids[i:i+1],
            resumed_from_preemption=cached_reqs.resumed_from_preemption[i:i+1],
            new_token_ids=cached_reqs.new_token_ids[i:i+1],
            new_block_ids=cached_reqs.new_block_ids[i:i+1],
            num_computed_tokens=cached_reqs.num_computed_tokens[i:i+1],
            chunks_lists=cached_reqs.chunks_lists[i:i+1],
        )

def merge_cached_request_data(cached_reqs_list: list[CachedRequestData]) -> CachedRequestData:
    req_ids = []
    resumed_from_preemption = []
    new_token_ids = []
    new_block_ids = []
    num_computed_tokens = []
    chunks_lists = []
    for req in cached_reqs_list:
        req_ids.extend(req.req_ids)
        resumed_from_preemption.extend(req.resumed_from_preemption)
        new_token_ids.extend(req.new_token_ids)
        new_block_ids.extend(req.new_block_ids)
        num_computed_tokens.extend(req.num_computed_tokens)
        chunks_lists.extend(req.chunks_lists)
    return CachedRequestData(
        req_ids=req_ids,
        resumed_from_preemption=resumed_from_preemption,
        new_token_ids=new_token_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=num_computed_tokens,
        chunks_lists=chunks_lists,
    )

def merge_refs_results(refs):
    outputs = [ref.get() for ref in refs]
    output = outputs[0]
    for o in outputs[1:]:
        for req, idx in o.req_id_to_index.items():
            output.req_id_to_index[req] = idx + len(output.req_ids)
        output.req_ids.extend(o.req_ids)
        output.sampled_token_ids.extend(o.sampled_token_ids)
        if output.logprobs is not None and o.logprobs is not None:
            output.logprobs.logprob_token_ids.extend(o.logprobs.logprob_token_ids)
            output.logprobs.logprobs.extend(o.logprobs.logprobs)
            output.logprobs.sampled_token_ranks.extend(o.logprobs.sampled_token_ranks)
        elif o.logprobs is not None:
            output.logprobs = o.logprobs
        output.prompt_logprobs_dict.update(o.prompt_logprobs_dict)
        assert output.kv_connector_output is None, "Please implement kv_connector_output merge for CPP"
        if output.num_nans_in_logits is not None and o.num_nans_in_logits is not None:
            output.num_nans_in_logits.update(o.num_nans_in_logits)
        elif o.num_nans_in_logits is not None:
            output.num_nans_in_logits = o.num_nans_in_logits
    return output