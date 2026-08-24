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

The five slots do not have equal readiness. PR 1 is implementation-ready. PR 2 has a defined producer and consumer but still requires an isolated kernel spike. PR 3 and PR 4 begin as profiling spikes and become PRs only after a concrete hot path clears their promotion gate. PR 5 is a concrete model-adoption task, but still requires a baseline showing that prefill is material for the selected workload.

Ranked fallbacks, rechecked before each branch starts:

1. another unclaimed #31504 producer site with measured H20 headroom;
2. Qwen3-TTS mixed sampled/argmax batch graph retention from #1418;
3. `dots.tts` or AuDar-TTS prefill graph adoption from #1357;
4. a newly qualified H20-compatible target from the Omni KDA roadmap #1650.

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
- Generalize the existing fused-AR-quant configuration and tuple handoff instead of creating a second communicator mechanism.
- Reuse `_fp8_static_input_scale` for consumer eligibility and static scale extraction.
- Use `kARResidualRMSNormFP8Quant` for a full-attention QKV projection whose normalized activation has only the FP8 consumer.
- For Qwen3.5 GDN, which consumes the same normalized activation through FP8 `in_proj_qkvz` and BF16 `in_proj_ba`, use and validate the dual-output `kARResidualRMSNormOutFP8Quant` pattern. Preserve the BF16 norm output and pass the FP8 tuple only to the eligible projection.
- Do not quantize the post-attention norm whose output also feeds router/routed-expert consumers; that broader dual-output site remains out of scope.
- Preserve eager and non-quantizing all-reduce fallbacks.

Non-goals:

- Dynamic or block-wise FP8 activation quantization.
- MXFP8, Marlin, ROCm/AITER, multi-node MNNVL qualification, SiluAndMul fusion, or dual-output post-attention/router norm fusion.
- Enabling the feature for a model without proving the producer has exactly the intended FP8 consumer contract.

Expected code surfaces:

- `python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `python/sglang/srt/layers/layernorm.py`
- `python/sglang/srt/layers/communicator.py`
- Qwen3.5 full-attention QKV quant-only wiring and GDN explicit dual-output wiring; NemotronH is included only after each projection's consumer contract is verified
- `test/registered/unit/layers/test_flashinfer_comm_fusion.py`
- focused layernorm/communicator/model unit tests
- a small two-rank benchmark or extension of the existing communication benchmark

Required call chain:

1. Model construction supplies the exact eligible projection(s), not an architecture name or a Boolean guessed from checkpoint metadata.
2. `LayerCommunicator` reuses its existing `enable_fused_ar_quant` and `fused_ar_quant_keep_bf16` policy, extended with static per-tensor CUDA mode.
3. The layernorm helper resolves eligibility through `_fp8_static_input_scale` and selects quant-only or norm-plus-quant output.
4. The FlashInfer wrapper preallocates residual, optional norm, and FP8 outputs, then calls the exact pattern with a device-resident `scale_factor`.
5. The downstream FP8 linear consumes `(fp8, scale, original_dtype)`; a GDN BF16 consumer receives the separately materialized norm output.

Correctness contract:

- Residual output matches the non-quantizing fused path within the existing norm tolerance.
- Fused quant output matches `static_quant_fp8(nonquant_norm, input_scale)` under the same scale convention, not merely a dequantized cosine check.
- Standard RMSNorm and Gemma/Qwen3.5 semantics are tested separately. The implementation must choose exactly one weight representation: raw `weight` with FlashInfer `weight_bias=1.0`, or the existing pre-added `gemma_weight` with `weight_bias=0.0`. Passing `gemma_weight` with a second bias is forbidden.
- Downstream linear output dtype remains the original activation dtype.
- FP16/BF16, scale dtype/device/contiguity, quant-only tuples, and BF16-plus-FP8 dual output are covered.
- Downstream GEMM and a fixed model-level output/logit gate agree with the baseline within the declared FP8 tolerance.
- CUDA Graph capture followed by repeated replay with changing inputs produces fresh, stable outputs and does not allocate dynamic buffers in replay.

Collective-safe eligibility and fallback:

- Exact FlashInfer enum and call-signature support for `quant_out`, `scale_factor`, `weight_bias`, and the selected pattern is probed before any collective.
- Backend, workspace capacity, process-group identity, dtype, shape, scale, and tuple-mode decisions are derived from rank-invariant metadata. Availability failures discovered during initialization are synchronized across the participating ranks.
- A rank may use the existing non-quantizing fused path only when every rank makes the same pre-collective decision.
- Once the quantizing collective is launched, errors fail fast. There is no catch-and-fallback path after one rank may have entered the kernel.
- Two-rank tests cover both the normal path and a deliberately unavailable capability path without collective mismatch.

Performance contract:

- Nsight Systems or CUDA-profiler evidence shows that the standalone static-FP8 quant kernel disappears at the wired sites.
- Isolated two-rank latency uses interleaved A/B rounds after warm-up and reports median and tail distribution.
- End-to-end H20 TP=2 decode compares identical model, workload, process, clocks, and software SHA. A gain is claimed only if it clears the measured A/A noise band.

Before renting a GPU, the benchmark checkpoint and revision must be fixed and checked from metadata or a dry-run memory estimate. It must fit `2 x H20 96GB`, use an allowlisted architecture, contain a wired static per-tensor FP8 projection, enable FlashInfer all-reduce under TP=2 with DP attention off and `moe_a2a_backend=none`, and show the target standalone quant kernel in a baseline trace. If no such checkpoint is available, PR 1 remains a correctness/integration contribution and a different H20-compatible target takes the performance-claim slot.

### PR 2 — Gated RMSNorm + static FP8 quantization

Repository: `sgl-project/sglang`

Upstream reference: issue #31504, Step 2

Goal: fuse static per-tensor FP8 quantization into the block-internal gated RMSNorm that feeds the GDN output projection. Unlike PR 1, this is a local producer-kernel optimization and applies even when no tensor-parallel all-reduce fusion is active.

This is a separate PR because it changes a different kernel and producer contract, has different topology coverage, and can be evaluated independently. It must reuse the pre-quantized linear tuple contract from the landed norm fusion. The initial spike must prove that the gated norm is a material H20 hot path and that an extra FP8 output beats the current gated norm plus standalone quant sequence. If it does not clear isolated A/A noise, the slot falls back rather than publishing a neutral custom kernel.

Tests cover gate-before/after-norm semantics, head/group shapes, FP16/BF16, residual absence, FP8 saturation, downstream output parity, and CUDA Graph replay. Benchmarks report isolated kernel latency, removed launches, memory traffic where counters are available, and end-to-end decode.

### PR 3 — Qwen3-TTS mixed sampled/argmax Predictor graph retention

Repository: `sgl-project/sglang-omni`

Upstream reference: issue #1418, section 6

Goal: keep mixed batches containing sampled and argmax rows on the Predictor CUDA Graph path while preserving exact per-row sampling semantics.

Current `_predictor_graph_signature` deliberately rejects mixed batches because sampled-row tensors have count-dependent shapes. The candidate first profiles the frequency and cost of this fallback. If material, replace the shape-changing sub-batch path with a fixed full-batch masked sampling contract: sampled rows execute the seeded categorical path, argmax rows retain the exact eager tie-breaking result, and inactive padded rows cannot alter RNG position or output. Graph keys remain bounded by batch bucket and sampling-shape signature rather than the exact row mask.

Tests cover all-argmax, all-sampled, alternating and sparse sampled masks, bucket padding, per-row top-k/top-p/temperature, semantic positions, deterministic seeds, graph reuse after row-mask changes, eager fallback, and exact token/embedding identity. Benchmarks report mixed-batch graph hit rate, Predictor kernel/launch count, step latency, end-to-end requests/s, TTFA, and generated-audio quality. If mixed batches are rare or the full-batch sampling cost cancels graph savings, this slot falls back.

### PR 4 — Qwen3-TTS reference-encode service batching

Repository: `sgl-project/sglang-omni`

Upstream reference: issue #1418, section 6

Goal: add service-level batching for Qwen3-TTS reference encoding only if a cold unique-voice workload shows material encoder headroom.

Open PR #1625 already implements uploaded-voice single-flight and must not be duplicated. This candidate branches from then-current main and targets `_Qwen3TTSAdhocReferenceHook.can_encode_batch` / `encode_batch`, configures `ReferenceEncodeService(max_batch_size, max_batch_wait_ms)`, and reuses `_Qwen3TTSRefCodeBatcher` to batch the speaker-embedding work that remains per request. It proceeds only when profiling shows material unique-key cold-request headroom and when padding, masking, output splitting, cache insertion, cancellation, and failure isolation are equivalent to independent encodes.

Tests cover variable reference lengths, batch formation, output-to-request mapping, per-item failure, cancellation, cache insertion, and batched-vs-independent parity. Benchmarks report encoder invocation count, achieved batch size, cold unique-voice throughput, mean/p95 latency, and memory. If profiling shows batching cannot amortize the work, this slot falls back.

### PR 5 — MOSS-TTS-Local prefill CUDA Graph adoption

Repository: `sgl-project/sglang-omni`

Upstream reference: issue #1357 backlog

Goal: adopt the shared SGLang-Omni prefill graph policy for MOSS-TTS-Local without duplicating the upstream graph runner.

Before implementation, verify that no open PR has claimed this backlog item and establish that prefill contributes enough wall time to justify graph capture. Reuse `OmniPrefillInputs` and the centralized bucket/cap policy available on then-current main. Add only model-specific static-input preparation, output slicing, capability declaration, and explicit eager fallback required by MOSS-TTS-Local.

Tests follow #1357's adapter acceptance contract: eligible real request, A-B-A stale-state protection, replay plus eager fallback, deterministic output/quality parity, and capture-memory accounting. Benchmarks report graph hit rate, capture/startup cost, VRAM, prefill latency, TTFA, and end-to-end throughput at c1 and loaded concurrency. If prefill is not material, replace this slot rather than defaulting on a neutral graph.

## Execution order and dependency rules

1. Coordinate ownership for #31504 Step 3 before substantive coding, then implement PR 1 locally and run CPU/mock tests before renting a GPU. Open a draft PR early once the local contract is credible.
2. Rent one two-card H20 NVLink instance only when PR 1 reaches the GPU-validation gate and the checkpoint gate is satisfied.
3. Do not begin PR 2 until PR 1 has a reviewable commit; reuse shared tuple infrastructure but keep the gated-norm kernel and evidence independent.
4. Run PR 3 and PR 4 qualification profiles before treating them as PR slots. A failed promotion gate selects a fallback.
5. PR 5 may proceed independently after refreshing #1357 ownership and dependencies.

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
