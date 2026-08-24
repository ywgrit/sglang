# Five Performance PR Portfolio Design

Date: 2026-08-24

## Objective

Produce five independent, technically honest performance pull requests for SGLang and SGLang-Omni. Each PR must remove a measured bottleneck, preserve correctness, include focused tests, and carry reproducible evidence from the exact hardware used. A PR is not counted as a delivered performance result until the optimized path is observed and an end-to-end benchmark clears the measured noise band.

The first implementation target is SGLang issue #31504, Step 3. The remaining targets are ordered follow-ups, not five branches developed simultaneously.

## Why the original candidate list changed

Several earlier candidates are no longer valid on current upstream:

- SGLang prefill graph `input_embeds` storage and replay are already implemented, including registry and unit-test coverage.
- SGLang-Omni Qwen3-TTS Predictor SDPA GQA landed in PR #1164.
- Fun-ASR encoder `torch.compile` support and tests are already present on `main`.
- SGLang-Omni prefill graph work is active under issue #1357 with named owners and multiple open PRs.
- The Qwen3-TTS suppress-token path has already moved to sparse per-request token handling; the previously considered dense-bias removal is no longer a current gap.

These items must not be reimplemented merely to reach a PR count.

## Approaches considered

### A. Implement five small changes immediately

This maximizes visible branch count but risks artificial PR splitting, weak benchmarks, and collisions with fast-moving upstream work. Rejected.

### B. Complete one deep distributed-kernel integration before choosing anything else

This provides the strongest single artifact but creates schedule risk if FlashInfer API or multi-GPU validation exposes a blocker. Useful for the first PR, but too brittle as the whole portfolio strategy.

### C. Staged portfolio with explicit fallbacks

Implement one high-value task at a time. Before starting each task, refresh the issue, open-PR search, and current code. Keep a ranked fallback for every slot. This is the selected approach.

## Portfolio

### PR 1 — FlashInfer AllReduce + RMSNorm + static FP8 quantization

Repository: `sgl-project/sglang`

Upstream reference: issue #31504, Step 3

Goal: for eligible tensor-parallel layers, use FlashInfer's `kARResidualRMSNormFP8Quant` pattern so the all-reduce epilogue emits a pre-quantized FP8 activation consumed directly by the following static per-tensor FP8 linear. This removes the standalone activation-quantization launch and its HBM round trip.

Current upstream state:

- Norm-level static FP8 fusion landed in PR #33471.
- The FP8 linear path already accepts `(fp8_input, input_scale, original_dtype)`.
- The FlashInfer all-reduce integration still selects `kARResidualRMSNorm` and returns BF16/FP16 norm output.
- No open or merged SGLang PR currently implements the quantizing all-reduce pattern.

Scope:

- Add a FlashInfer wrapper for all-reduce + residual + RMSNorm + per-tensor FP8 quantization.
- Reuse the existing static-FP8 eligibility check; do not duplicate quantization-scheme detection.
- Pass only the necessary downstream-linear/scale information through the layer-normalization and communicator boundary.
- Return the same pre-quantized tuple contract used by the norm-level fusion.
- Wire a single-consumer QKV/GDN input-projection site first. Do not quantize the post-attention norm whose output also feeds BF16 router/routed-expert consumers.
- Preserve eager and non-quantizing all-reduce fallbacks.

Non-goals:

- Dynamic or block-wise FP8 activation quantization.
- MXFP8, Marlin, ROCm/AITER, multi-node MNNVL qualification, SiluAndMul fusion, or dual-output post-attention norm fusion.
- Enabling the feature for a model without proving the producer has exactly the intended FP8 consumer contract.

Expected code surfaces:

- `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/communicator.py`
- the minimum model wiring needed for Qwen3.5/NemotronH single-consumer input projections
- `test/registered/unit/layers/test_flashinfer_comm_fusion.py`
- focused layernorm/communicator/model unit tests
- a small two-rank benchmark or extension of the existing communication benchmark

Correctness contract:

- Residual output matches the non-quantizing fused path within the existing norm tolerance.
- Dequantized FP8 norm output matches the BF16/FP16 reference within FP8 tolerance.
- Downstream linear output dtype remains the original activation dtype.
- Unsupported scale shape, quantization method, topology, world size, shape, dtype, or FlashInfer version follows the existing path without changing output type.
- CUDA Graph capture/replay produces stable outputs and does not allocate dynamic buffers in replay.

Performance contract:

- Nsight Systems or CUDA-profiler evidence shows that the standalone static-FP8 quant kernel disappears at the wired sites.
- Isolated two-rank latency uses interleaved A/B rounds after warm-up and reports median and tail distribution.
- End-to-end H20 TP=2 decode compares identical model, workload, process, clocks, and software SHA. A gain is claimed only if it clears the measured A/A noise band.

### PR 2 — Whisper compiled/fused encoder path

Repository: `sgl-project/sglang-omni`

Upstream reference: ASR roadmap #1396, W-PR6

Goal: qualify and implement a compiled/fused Whisper encoder path on top of the landed bucketed encoder CUDA Graph work.

Scope is gated by profiling. First establish encoder-only and end-to-end baselines for the graph-enabled path; then compile stable fixed/bucketed encoder shapes or fuse a proven hot projection/pointwise chain. The PR must retain eager fallback for unqualified shapes. It must not assume that `torch.compile` is beneficial merely because it reduces Python code.

Tests cover compile selection, bucket hits/misses, eager fallback, output parity, and graph/compile mutual-exclusion rules. Benchmark gates include WER/CER, encoder latency, requests/s, mean/p95 latency, compile warm-up cost, and peak memory.

### PR 3 — Qwen3-TTS streaming-vocoder event-based completion and pinned staging

Repository: `sgl-project/sglang-omni`

Upstream reference: issue #1418, section 6

Goal: remove the per-decode `stream.synchronize()` and pageable host-to-device upload from `streaming_vocoder.py` by introducing pinned staging plus an explicit launch/resolve event lifecycle.

The design must preserve bad-code validation, request ordering, cancellation, state ownership, and final flush. Launch may enqueue work, but waveform consumption and error reporting may occur only after the corresponding event resolves. Batches cannot reuse staging storage while work is in flight.

Tests cover pending/completed states, staging-slot reuse, cancellation, bad rows, final flush, and deterministic mode. Benchmarks report H2D latency, host blocked time, TTFA, audio seconds/s, and latency at c1/c8/c16.

### PR 4 — Qwen3-TTS follow-up-window CUDA Graphs

Repository: `sgl-project/sglang-omni`

Upstream reference: issue #1418, section 6

Goal: graph stable follow-up vocoder window shapes instead of graphing only the initial window.

Use a bounded key consisting of the semantically required dimensions, such as batch size and valid frame count. Record hit, miss, capture, and eager-fallback counters. Do not right-pad across the convolution receptive field unless a valid-region argument and bit-parity test prove it safe. Limit graph count and memory explicitly.

Tests cover keying, capture reuse, bounded eviction/disable policy, eager fallback, exact emitted waveform slices, and interaction with the PR 3 launch/resolve lifecycle. Benchmarks report hit rate, graph memory, capture time, steady-state vocoder latency, and end-to-end streaming metrics.

### PR 5 — Qwen3-TTS uploaded-reference single-flight and service batching

Repository: `sgl-project/sglang-omni`

Upstream reference: issue #1418, section 6

Goal: ensure concurrent cold requests for the same uploaded voice perform one reference encode, and add service-level batching only where the codec/reference hook can prove equivalent batched semantics.

The existing ad-hoc reference path already uses `ReferenceEncodeService`; the uploaded-voice cache-miss branch currently performs direct encoding. First unify cache misses behind keyed single-flight. Batch encoding is a second phase of the same PR only if profiling shows additional value and input padding/masking is correct; otherwise the PR remains a focused single-flight performance fix.

Tests cover N concurrent identical misses, different-key independence, leader failure propagation, retry after failure, cache insertion, cancellation, and batch parity. Benchmarks report encoder invocation count, cold-start wall latency, throughput under repeated/shared and unique voices, and cache memory.

## Execution order and dependency rules

1. Implement PR 1 locally and run CPU/mock tests before renting a GPU.
2. Rent one two-card H20 NVLink instance only when PR 1 reaches the GPU-validation gate.
3. In parallel with benchmark collection only at the workflow level, prepare PR 2's profile harness; do not begin a second implementation branch until PR 1 has a reviewable commit.
4. Implement PR 3 before PR 4 because the follow-up graph path must use the final launch/resolve ownership model.
5. PR 5 is independent of PR 3/4 and is the fallback if Whisper compile provides no measurable benefit.

Each feature PR branches directly from the then-current upstream `main`. Planning commits are never included in a feature PR.

## Local and remote workflow

Local laptop:

- source edits, unit tests that do not require CUDA, static checks, focused mocked dispatch tests, and benchmark-driver development
- one isolated git worktree per feature branch
- raw benchmark schema and command lines prepared before GPU rental

Remote H20 instance:

- exact source commit transferred through git
- environment and model cache retained on the instance data disk
- GPU-only correctness, CUDA Graph, profiler, NCCL, and end-to-end runs
- raw JSON/CSV/traces copied back before shutdown

No benchmark result is copied from H100/H200 literature and relabeled as H20.

## H20 validation gate

Before performance work, record:

- `nvidia-smi` and driver/CUDA versions
- `nvidia-smi topo -m`
- NVLink status
- peer-to-peer bandwidth/latency
- `nccl-tests` two-rank all-reduce bandwidth
- PyTorch, SGLang, SGLang-Omni, FlashInfer, Triton, and model revisions
- Nsight Systems availability and Nsight Compute counter permission

If peer access or NVLink is absent, PR 1 may still receive correctness testing, but no representative communication-performance claim is made.

## Benchmark discipline

- Declare correctness and performance gates before looking at final numbers.
- Use the same host, model revision, dataset revision, environment, and client for both arms.
- Separate warm-up/capture/compile time from steady-state measurement.
- Run at least one A/A noise measurement and interleave or cross the A/B order.
- Use at least five measured repeats for small effects; use more when the noise band is comparable to the expected gain.
- Save raw per-request or per-iteration results, command lines, environment, git SHA, and profiler evidence.
- Report negative and inconclusive results. An isolated kernel win is not an end-to-end win.

## Resume and PR wording

Before merge, describe work as an open-source contribution or submitted PR, never as an upstream SGLang feature already shipped. Performance numbers must name H20, topology, model, precision, workload, baseline, and measurement scope. If a change is not merged, the resume must say “submitted” or “implemented and validated in PR #…”.

## Stop and fallback conditions

- If a conflicting upstream PR appears before coding starts, stop and choose the next ranked task.
- If a correctness contract cannot be met, do not publish a performance claim.
- If an optimization remains inside the A/A noise band after a properly powered rerun, retain the profile/negative result but replace it as a resume performance PR.
- If a task requires hardware not available on the chosen H20 instance, keep its local design/tests and advance an H20-compatible fallback instead.
