# PCP O-Proj Tensor Parallelism

Implementation baseline: `vllm-project/vllm upstream/main@bfa7e5b7af`.

## Background

The fine-grained O-Proj tensor parallelism in vllm-ascend further shards the
O-Proj input dimension across additional ranks beyond TP, reducing the resident
weight on each rank. In the DSA-CP implementation, the runtime path depends on
the batch type: decode uses the finer-grained weight shards and reduces their
partial sums, while a batch containing prefill first gathers the original
TP-sized weight shard and then runs the standard O-Proj computation.

vLLM already has independent TP, PCP, and DCP process groups, but the generic
`RowParallelLinear` shards weights only across TP ranks. PCP O-Proj tensor parallelism uses PCP as a
second O-Proj weight-sharding axis and provides an equivalent dynamic path under
the name `pcp_o_proj_tp`.

## Supported configurations

- `prefill_context_parallel_size > 1`.
- `decode_context_parallel_size = 1`.
- Only DeepSeek V2/V3/V3.2, GLM-4 MoE Lite MLA, and the GLM-5 family are
  supported.
  GLM-4 MoE Lite attention inherits the DeepSeekV2 MLA path. GLM-5.2 reuses the
  NVIDIA DeepSeekV3.2 DSA path. Both reuse the same explicit prefetch and
  O-Proj implementation.
- O-Proj may use any linear quantization method. The weight gather treats the
  post-load tensors as opaque (see below), so no per-method adapter exists.
- `bias=False` is required.
- Only eager execution is supported. CUDA Graph, `torch.compile`, and DBO are
  not supported yet.
- LoRA is not supported yet.
- CPU weight offloading is not supported yet.
- The attention module must call `o_proj.prefetch_weights()` before the main
  attention computation.

Configuration validation and linear-layer construction checks enforce all of
these restrictions and fail fast instead of silently ignoring the feature.

## Weight layout

Let the O-Proj weight be `W[N, K]`, the TP size be `T`, and the PCP size be `P`.
Standard row parallelism keeps the following shard on TP rank `t`:

```text
W_tp[t] = W[:, t*K/T : (t+1)*K/T]
```

The feature further shards each TP shard across PCP rank `p`:

```text
W_local[t,p] = W[:, (t*P+p)*K/(T*P) : (t*P+p+1)*K/(T*P)]
```

The checkpoint loader uses flattened rank `t*P+p` and world size `T*P`, so no
checkpoint format change is required. The resident O-Proj weight on each rank
is `1/P` of the standard TP layout, and the PCP peers of a TP rank hold
adjacent head slices of that TP rank's heads.

## Peer weight gather

`PeerWeightGather` (`vllm/model_executor/layers/peer_weight_gather.py`) never
reassembles a TP-sized weight. Instead, a rank borrows the shards of its PCP
peers and runs the layer once per shard:

```text
Y_t = sum_q apply(W_local[t,q], X_t[..., q-th input slice])
```

This is the same identity that makes TP row parallelism correct, so it holds
for every linear quantization method. Each peer shard is used exactly as its
owner processed it: swizzled or packed scales, Marlin/Machete repacking,
TRT-LLM shuffles, K padding, per-shard requantized scales, and kernels
configured for the local K all stay valid. The kernel always sees the local
shape it was built and warmed up for, so no layer metadata is switched either.

Mechanics:

1. After the quantization method's `process_weights_after_loading`, the model
   loader calls `layer.on_weights_processed()`. The layer collects every
   non-empty tensor it owns (parameters, buffers, and plain tensor
   attributes), plans one 256-byte-aligned byte blob that preserves each
   tensor's strides, and checks with one small all-gather that all PCP ranks
   produced the same layout. Quantization methods or kernels that keep tensors
   outside the layer are rejected, because such weight data would not be
   gathered.
2. `start()` copies the local tensors into this rank's row of a shared
   `[P, blob_bytes]` receive buffer and launches one asynchronous in-place
   `all_gather_into_tensor` per layer. The collective sees raw bytes, so FP8,
   FP4, and packed integer formats need no special datatype.
3. `use_peer(q)` temporarily points the layer's tensor attributes at strided
   views of row `q` and restores the local tensors afterwards. The views are
   built once, so swapping allocates nothing.

## Runtime modes

The model runner records whether the logical, pre-partition batch contains
any prefill before PCP splits its tokens, and passes it through
`ForwardContext.global_has_prefill`. `prefetch_weights()` uses this flag to
select one of three modes, identically on every PCP rank.

### Prefill or mixed batches: `PREFILL`

The attention module starts the asynchronous peer weight gather after its
Q/KV normalization, so the gather overlaps the remaining query projection,
RoPE, indexer, and main attention. O-Proj waits for it, then projects the
local tokens with all `P` shards and accumulates the partial results. No PCP
output reduction is needed because different PCP ranks process different
tokens. The existing TP reduction is unchanged.

### Decode-only batches with sharded requests: `DECODE_SHARDED`

With `DCP = 1`, PCP assigns each decode request to one PCP rank
(`ParallelConfig.pcp_shard_decode_requests`). PCP pads every rank's batch to
the same token count `T_pad`, so the head slices can be exchanged with
equal-size collectives:

```text
X_t[T_pad, K/T] --view--> [T_pad, P, K/(T*P)] --transpose--> [P, T_pad, K/(T*P)]
--PCP all-to-all--> [P*T_pad, K/(T*P)]   # this rank's head slice of all tokens
--local GEMM-->     [P*T_pad, N]
--PCP reduce-scatter--> [T_pad, N]       # summed over head slices, own tokens
```

### Decode-only batches with replicated requests: `DECODE_REPLICATED`

If every PCP rank holds every decode request, each rank projects its head
slice of the shared tokens and the partial results are all-reduced over PCP:

```text
Y_t = PCP-AllReduce(X_t[..., p-th input slice] @ W_local[t,p]^T)
```

In all modes, `Y = TP-AllReduce(Y_t)` follows when `reduce_results=True`.
DeepSeek-V3.2 O-Proj uses `reduce_results=False`; its fused TP
all-reduce/RMSNorm remains responsible for the TP reduction.

## Buffer lifecycle

The receive buffer is allocated during weight post-processing, before
profiling and KV-cache sizing, and is shared by all O-Proj layers with the
same blob size. Its size equals one TP-sized O-Proj shard plus alignment.
Layers execute sequentially and every layer waits for its gather before the
next layer's attention starts the next one, so a single buffer suffices. DBO is
disabled to prevent two concurrent microbatches from sharing it. The decode
modes allocate only activation-sized temporaries.

## Explicit triggering and correctness constraints

Both the generic MLA wrapper and the modular DeepSeek-V3.2 attention path call
`prefetch_weights()` after their Q/KV normalization and before the remaining
query projection and main attention computation.
`PCPOProjRowParallelLinear.forward()` requires that call before every
invocation, and `prefetch_weights()` requires the global batch type instead of
inferring it from rank-local metadata. This guarantees that:

- every PCP rank issues the same collectives in the same order;
- decode-only batches do not issue a weight gather;
- O-Proj does not read peer tensors before the asynchronous gather has
  completed; and
- dummy, profile, and warmup runs, which are unpartitioned decode-only batches
  on every rank, take a decode mode.

## DeepSeek-V4 grouped O-Proj

DeepSeek-V4 projects attention output through a grouped `wo_a`
(column-parallel over head groups, applied as a batched GEMM) followed by a
row-parallel `wo_b`. Sharding the head groups over PCP shards `wo_a` on its
output axis and `wo_b` on its input axis. `PeerWeightGather` is layer-agnostic,
so both layers are gathered the same way: in `PREFILL` mode each peer's `wo_a`
groups produce that peer's slice of the low-rank features, and each peer's
`wo_b` shard is accumulated as above. The unit tests cover this composition.
DeepSeek-V4 attention does not yet support PCP on CUDA, so the model is not
wired to the feature.

## Configuration

The feature adds the following CLI/configuration option:

```text
--enable-pcp-o-proj-tp
```

Example:

```bash
vllm serve <model> \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 2 \
  --decode-context-parallel-size 1 \
  --enable-pcp-o-proj-tp \
  --enforce-eager \
  --dtype bfloat16
```

## Validation results

Subsections marked "previous implementation" were measured with the earlier
design, which reassembled a TP-sized weight from representation-specific
layouts and replicated decode requests over PCP.

### GLM-5.2 cropped execution

On October 12, 2026, commit `81334e6294` was validated with the same four-layer
GLM-5.2 crop, topology, and workload as the previous-implementation run below
(TP1/PCP2/DCP1/EP2 on two H20 GPUs, BF16 O-Proj, eager execution, 128-token
scheduler budget, six requests). Decode requests were sharded over PCP in both
arms. Both arms used synchronous scheduling so that they executed the same 32
batches; with the default asynchronous scheduling, batch composition depends
on timing and differed by one or two steps between arms.

Every O-Proj call was compared with a full-weight BF16 GEMM on its actual
input. Worst-case errors across both ranks of the enabled arm:

| Phase | Calls across ranks | Relative L2 error | Max error / max reference magnitude |
| --- | ---: | ---: | ---: |
| prefill | 16 | 0.003074 | 0.007752 |
| mixed | 96 | 0.003110 | 0.006711 |
| decode | 144 | 0.003130 | 0.007634 |

All outputs were finite and within the predeclared 0.01 relative L2 and 0.02
scaled maximum error bounds. Each rank ran 56 `PREFILL` and 72
`DECODE_SHARDED` O-Proj calls. Prefill is no longer bit-identical to the
oracle because it sums two BF16 partial projections. Five of six generated
continuations matched the disabled arm; the sixth diverged at its fifth
generated token.

Resident O-Proj parameters per rank halved from `[6144, 16384]` to
`[6144, 8192]` per layer. Tensor allocation before installing the GEMM oracle
was 11.0743 GiB/rank disabled and 10.8868 GiB/rank enabled (the previous
implementation: 11.2618 GiB/rank enabled).

Thread-simulated two-rank unit tests run real kernels on H20 for BF16, FP8
block (CUTLASS), FP8 per-tensor (CUTLASS, transposed weight view), and
GPTQ-Marlin int4 O-Proj layers in all three modes. Every output matches
standard row parallelism over the TP x PCP shards bit for bit. The tests also
cover an opaque per-shard-rescaled, permuted, strided layout and the
DeepSeek-V4 grouped `wo_a`/`wo_b` composition.

### GLM-5.2 cropped execution (previous implementation)

On October 12, 2026, code commit `eb04a10709` was validated on two H20 GPUs
using a four-layer crop of GLM-5.2-FP8, TP1/PCP2/DCP1/EP2, eager execution,
FP8 KV cache, and a fixed 1 GiB KV cache. Original attention dimensions and
all 256 routed experts were retained. Only O-Proj weights were dequantized to
BF16 with the checkpoint's 128x128 block scales and excluded from FP8 weight
quantization. Both feature settings used this same checkpoint.

Each setting completed six requests: two 32-token prompts with eight output
tokens, followed by 32/600/48/720-token prompts with twelve output tokens.
Chunked prefill used a 128-token scheduler budget to exercise mixed batches.
Every O-Proj call was compared with a full-weight BF16 GEMM on its actual input.
The enabled arm produced the following worst-case errors across both ranks:

| Phase | Calls across ranks | Relative L2 error | Max error / max reference magnitude |
| --- | ---: | ---: | ---: |
| prefill | 16 | 0.000000 | 0.000000 |
| mixed | 96 | 0.000000 | 0.000000 |
| decode | 160 | 0.003212 | 0.007752 |

All outputs were finite. The predeclared bounds were 0.01 relative L2 and 0.02
scaled maximum error. Five of six generated continuations matched the baseline
exactly; the different BF16 reduction order can change an autoregressive path.
The crop validates execution and per-layer numerical error, rather than
full-model dataset accuracy or serving performance.

Resident O-Proj parameters decreased from 768 MiB to 384 MiB per rank across
four layers. Ready tensor allocations before installing the GEMM oracle were
11.0743 GiB/rank with the feature disabled and 11.2618 GiB/rank with it enabled.
The reconstruction/staging buffers therefore offset the parameter saving in
this crop; parameter residency and total serving memory remain separate metrics.

The run also covered dummy startup execution and the requirement to replicate
decode queries when PCP O-Proj output reduction is enabled. Sixteen focused
O-Proj and configuration tests passed, including the GLM-5 architecture gate
and the feature-specific query-ownership assertion.

The following end-to-end results cover the unquantized O-Proj path. The unit
tests additionally cover ModelOpt NVFP4 packed-weight gathering, logical scale
gathering and full CUTLASS-layout reassembly, preservation of replicated global
scales, and rejection of non-CUTLASS NVFP4 post-load layouts. A quantized
end-to-end model result is still required before promoting the NVFP4 adapter
beyond example status.

Online accuracy was evaluated on August 23, 2026, using four NVIDIA H20 GPUs,
GLM-4.7-Flash BF16, and the 1,319-question GSM8K test set. TP4, PCP4TP1, and
PCP2TP2 used eager execution, DCP1, 5-shot prompting, `temperature=0`, seed 42,
concurrency 32, and `max_tokens=4096`. Each topology first passed a 32-question
output gate, then completed the full dataset.

| Parallel strategy | 32-question gate | Full-set correct | Full-set accuracy | Failed requests | Invalid answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 baseline | 25/32 | 1129/1319 | 85.5951% | 0 | 0 |
| PCP4TP1 | 25/32 | 1135/1319 | 86.0500% | 0 | 0 |
| PCP2TP2 | 24/32 | 1131/1319 | 85.7468% | 0 | 0 |

PCP4TP1 and PCP2TP2 differ from TP4 by +0.4549 and +0.1516 percentage points,
respectively. The three reduction topologies can follow different
autoregressive generation paths because their floating-point reduction orders
differ.

Based on checkpoint safetensors metadata, the O-Proj weights across 48 layers
total `1,006,632,960` bytes, or `3.2242%` of all model parameter bytes. With the
feature enabled:

- PCP4TP1 reduces the resident O-Proj weight on each rank from 100% of the full
  O-Proj weight to 25%, a 75% reduction; and
- PCP2TP2 reduces the resident O-Proj weight on each rank from 100% of the
  original TP2 shard to 50%, a 50% reduction.

These percentages describe parameter weights only. They exclude the CUDA
context, KV cache, communication buffers, the shared single-layer prefetch
buffer, and allocator fragmentation, and therefore do not represent the total
HBM reduction.

The same four-H20 host was used to measure process-level memory with a fixed
32 GiB KV cache. `Model loading took` is the model runner's allocated-memory
measurement after loading and post-load processing. Ready HBM is the
`nvidia-smi` value after engine initialization.

| Parallel strategy | Feature | Model-load memory/rank | Ready HBM/rank |
| --- | --- | ---: | ---: |
| PCP4TP1 | disabled | 17.02 GiB | 59,461 MiB |
| PCP4TP1 | enabled | 16.58 GiB | 60,021 MiB |
| PCP2TP2 | disabled | 15.15 GiB | 56,541 MiB |
| PCP2TP2 | enabled | 15.17 GiB | 57,140 MiB |

PCP4TP1 reduces model-load allocation by 0.44 GiB per rank. For PCP2TP2, the
smaller local parameter shard and the current persistent gather staging have
similar sizes, yielding a 0.02 GiB measured increase. Ready HBM includes the
communication and runtime workspaces initialized by the feature and is 560 MiB
and 599 MiB higher for PCP4TP1 and PCP2TP2, respectively. These measurements
distinguish the parameter-layout reduction from total serving-process HBM.

Serving performance used random 32,768-token inputs and 1,024 generated tokens,
`ignore_eos`, `temperature=0`, seed 12345, prefix caching disabled, and a fixed
32 GiB KV cache. Each row is one completed controlled run; all requests
succeeded. Throughput is generated-token throughput.

| Topology | Feature | Concurrency | Mean TTFT | Mean TPOT | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| PCP4TP1 | disabled | 1 | 2,027.49 ms | 82.33 ms | 11.87 tok/s |
| PCP4TP1 | enabled | 1 | 2,030.59 ms | 86.39 ms | 11.33 tok/s |
| PCP4TP1 | disabled | 8 | 9,368.98 ms | 89.98 ms | 80.51 tok/s |
| PCP4TP1 | enabled | 8 | 9,402.45 ms | 94.69 ms | 76.84 tok/s |
| PCP4TP1 | disabled | 16 | 17,376.00 ms | 97.19 ms | 139.50 tok/s |
| PCP4TP1 | enabled | 16 | 17,438.39 ms | 104.49 ms | 131.04 tok/s |
| PCP2TP2 | disabled | 1 | 2,091.03 ms | 89.73 ms | 10.91 tok/s |
| PCP2TP2 | enabled | 1 | 2,151.26 ms | 95.16 ms | 10.29 tok/s |
| PCP2TP2 | disabled | 8 | 9,008.10 ms | 95.98 ms | 76.17 tok/s |
| PCP2TP2 | enabled | 8 | 8,968.01 ms | 103.23 ms | 71.27 tok/s |
| PCP2TP2 | disabled | 16 | 16,404.51 ms | 103.03 ms | 133.75 tok/s |
| PCP2TP2 | enabled | 16 | 16,417.18 ms | 108.64 ms | 127.70 tok/s |

Across concurrency 1/8/16, enabling the feature changes mean TTFT by
`+0.15%/+0.36%/+0.36%` for PCP4TP1 and `+2.88%/-0.45%/+0.08%` for PCP2TP2.
Mean TPOT changes by `+4.93%/+5.23%/+7.51%` and
`+6.05%/+7.55%/+5.45%`, respectively; generated-token throughput changes by
`-4.55%/-4.56%/-6.06%` and `-5.68%/-6.43%/-4.52%`.

### GLM-5.2 SM100 smoke validation (previous implementation)

A follow-up smoke test used four NVIDIA SM100 GPUs and a GLM-5.2 ModelOpt
NVFP4 checkpoint whose configuration excludes self-attention from
quantization. The O-Proj therefore exercised the unquantized path. The test
used PCP4TP1, DCP1, EP4, eager execution, FP8 KV cache, a 4,096-token model
length, prefix caching disabled, `gpu_memory_utilization=0.90`, and the existing
NVIDIA DeepSeekV3.2 model-class route. Both arms used the same locally prebuilt
SM100 extensions.

| Feature | Model-load memory/rank | Available KV cache/rank | KV cache tokens |
| --- | ---: | ---: | ---: |
| disabled | 129.88 GiB | 31.39 GiB | 706,624 |
| enabled | 122.74 GiB | 38.60 GiB | 868,800 |

The feature reduced model-load allocation by 7.14 GiB per rank and increased
KV capacity by 162,176 tokens. Ready HBM did not decrease because the fixed
GPU-memory-utilization target assigned the freed model memory to KV cache.

Both variants completed all requests in two eight-request serving smokes. One
of two deterministic continuation checks matched exactly; the short-context
continuation diverged. Different collective reduction order can change
autoregressive paths, so this smoke does not replace a model accuracy
evaluation.

The following single cold A/B pair is directional only and is not treated as
stable performance evidence:

| Workload | Metric | Disabled | Enabled | Change |
| --- | --- | ---: | ---: | ---: |
| 1,024 input / 32 output / c8 | Mean TTFT | 574.71 ms | 630.86 ms | +9.77% |
| 1,024 input / 32 output / c8 | Mean TPOT | 199.40 ms | 216.18 ms | +8.41% |
| 1,024 input / 32 output / c8 | Output throughput | 37.19 tok/s | 34.28 tok/s | -7.84% |
| 128 input / 128 output / c8 | Mean TTFT | 457.97 ms | 502.81 ms | +9.79% |
| 128 input / 128 output / c8 | Mean TPOT | 202.81 ms | 211.60 ms | +4.34% |
| 128 input / 128 output / c8 | Output throughput | 39.02 tok/s | 37.37 tok/s | -4.25% |

## PCP8 parameter projections

The following projections use Hugging Face checkpoint tensor metadata and
assume TP1, PCP8, DCP1, and an implementation compatible with each model's
O-Proj representation. They include the current single-layer full-weight
prefetch buffer and exclude communication, staging, and allocator overhead.

| Model/checkpoint | Total parameter bytes | Shardable O-Proj | Local O-Proj at PCP8 | Per-rank persistent saving | Resulting persistent weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.2 BF16 | 1,403.186 GiB | 14.812 GiB | 1.852 GiB | 12.773 GiB | 1,390.413 GiB |
| GLM-5.2 FP8 | 703.723 GiB | 7.408 GiB | 0.926 GiB | 6.388 GiB | 697.335 GiB |
| DeepSeek-V4 Flash-Base | 274.436 GiB | 1.375 GiB | 0.172 GiB | 1.172 GiB | 273.264 GiB |
| DeepSeek-V4 Flash | 148.648 GiB | 1.375 GiB | 0.172 GiB | 1.172 GiB | 147.476 GiB |
| DeepSeek-V4 Pro-Base | 1,495.734 GiB | 6.782 GiB | 0.848 GiB | 5.825 GiB | 1,489.910 GiB |
| DeepSeek-V4 Pro | 805.319 GiB | 6.782 GiB | 0.848 GiB | 5.825 GiB | 799.495 GiB |

For a combined TP size `T`, the local O-Proj, prefetch buffer, and per-rank
saving in this table are divided by `T`.

## Future work

1. Add multi-run stability tests and traces of weight-gather/attention overlap.
2. Measure prefill O-Proj cost: the peer-shard accumulation runs `P` GEMMs of
   `K/(T*P)` plus `P-1` output additions instead of one GEMM of `K/T`.
3. Integrate the shared receive buffer with a concurrency-safe model-level
   workspace manager and remove the DBO restriction.
4. Add CUDA Graph and `torch.compile`-capturable paths; the decode modes
   already use static shapes.
5. Wire DeepSeek-V4 once its attention supports PCP.
6. Evaluate a dedicated PCP communicator or stream to avoid serialization with
   attention PCP collectives.
