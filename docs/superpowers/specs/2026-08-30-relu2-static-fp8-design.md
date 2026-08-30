# ReLU2 + Static FP8 Activation Quantization Design

## Status and scope

This design targets the dense MLP path in `NemotronHMLP` on CUDA when the
`down_proj` is loaded with serialized ModelOpt per-tensor static FP8 activation
scales.  It fuses the existing standalone ReLU2 and static activation
quantization passes into one producer kernel, then lets the existing FP8 linear
consumer use the produced FP8 tensor without quantizing it again.

The initial implementation is intentionally narrow:

- CUDA, BF16 activation input, contiguous storage;
- `NemotronHMLP` dense MLP only;
- serialized ModelOpt FP8 with exactly one positive finite `input_scale`;
- TP size 1 only until a real multi-rank model test establishes the sharded
  contract;
- the established SGLang prequantized tuple contract, consumed by the general
  FP8 linear or FlashInfer per-tensor FP8 BMM path;
- original BF16 output from `down_proj`.

The change does not add a public activation wrapper, a serving flag, a second
FP8 linear abstraction, dynamic/per-token scaling, MoE support, Marlin support,
SM120 GEMV support, ROCm support, or a new GEMM implementation.

The ModelOpt tuple consumer is implemented in the local rebased prerequisite
branch `codex/pr1-rebase-20260830` at commit
`71b5af4cc000c043a98476560dbed8b967251c9c`, which is also the current remote
head of PR #36501.  Development is stacked on that exact consumer contract
rather than copying it into this patch.  Until #36501 lands, the ReLU2 branch
is tested as a stack and must not be submitted as a standalone diff against
`main`.  If #36501 changes its tuple contract during review, this design and
the pinned prerequisite are revised before production code proceeds.

## Motivation and measured promotion bar

The baseline dense block performs:

1. BF16 `up_proj` output;
2. a full read and BF16 write for `relu2`;
3. another full BF16 read and FP8 write for `static_quant_fp8`;
4. the existing FP8 `down_proj`.

The proposed producer performs the ReLU2 computation, the required BF16
rounding, scale division, FP8 clamp, and FP8 store in one pass.  It removes one
intermediate BF16 tensor traversal and one kernel launch.  This is expected to
matter most at large token counts; it must not be promoted solely from a
microbenchmark.

For reference only, vLLM PR #53793 reports an 18.79% shared-MLP speedup and a
1.34% request-throughput improvement on 4x GB200 for Nemotron Ultra NVFP4.  The
SGLang implementation must be independently measured on its actual H20
ModelOpt FP8 dispatch and must not reuse those values as SGLang results.

Promotion requires all of the following on a real Nemotron ModelOpt FP8
checkpoint:

- the fused route is observed in the actual model call path;
- no standalone ReLU2 or static-quant kernel remains between `up_proj` and
  `down_proj` in an Nsight trace;
- numerical parity passes at the model boundary with identical greedy token IDs
  and per-layer outputs within `atol=rtol=1e-2` for BF16;
- the shared-MLP benchmark improves by at least 5% geometric mean over token
  counts `{1, 8, 32, 128, 512, 2048, 8192, 32768}` at width 15680, or the
  end-to-end result demonstrates another
  clear deployment benefit;
- six counterordered end-to-end server lifetimes show request throughput no
  worse than `-1%`, with the paired 95% confidence interval also crossing no
  worse than `-2%`; TTFT and ITL medians must each remain within `+2%`.

If the actual H20 backend cannot consume the prequantized tensor without a
larger framework contract, or the promotion bar is missed, the experiment is
recorded as NO-GO rather than expanded into a large patch.

## Eligibility and fail-closed dispatch

Eligibility has two stages.  `NemotronHMLP` records only immutable intent and
configuration facts during construction.  `ModelOptFp8LinearMethod` finalizes a
layer capability flag in `process_weights_after_loading`, after the checkpoint
scale and selected backend exist.  `forward` then performs invocation-specific
shape, dtype, device, and contiguity checks after `up_proj`.  The model queries
these two narrow predicates and does not duplicate backend internals.

Eligibility requires:

1. CUDA execution and BF16 model activations;
2. TP size exactly 1;
3. `down_proj.quant_method` is `ModelOptFp8LinearMethod` backed by a serialized
   checkpoint;
4. weight processing retained a scalar `down_proj.input_scale`;
5. neither Marlin nor SM120 GEMV is enabled for the layer; both disable the
   initial optimization globally rather than changing ownership by invocation;
6. runtime input is CUDA BF16, rank at least 2, contiguous, non-empty, and
   `x.shape[-1] == down_proj.input_size_per_partition`;
7. the scale is positive and finite when weights finish loading.

The concrete forward predicate is therefore the conjunction of the finalized
layer capability and
`x.is_cuda`, `x.dtype == torch.bfloat16`, `x.dim() >= 2`, `x.is_contiguous()`,
`x.numel() > 0`, and
`x.shape[-1] == down_proj.input_size_per_partition`.  The capability is false
until `process_weights_after_loading` has reduced the checkpoint scale to the
single `Parameter` that the consumer owns.  That `Parameter` is not rebound
after capability finalization; the producer and tuple handoff return the exact
same object.

Any unmet or unknown condition uses the unchanged baseline
`ReLU2 -> down_proj(BF16)` path.  The optimized branch is not selected by model
name alone.

## Producer semantics

Add one internal Triton kernel in
`python/sglang/kernels/ops/activation/relu2_fp8_quant.py`.  Its Python API is:

```python
relu2_and_static_quant_fp8(x_bf16, input_scale) -> x_fp8
```

The accepted input is a contiguous, non-empty CUDA BF16 tensor of rank at least
2 whose last dimension exactly equals the consuming down projection's
`input_size_per_partition`; leading dimensions are flattened.  No pointer
alignment or hidden-width divisibility is required: a one-dimensional element
grid and masked tail handle every eligible element count.  Rank, dtype, device,
contiguity, zero-numel input, or hidden-width mismatch stays on the model's
unchanged baseline path and never calls the producer.  Once called, the
producer assumes these preconditions and does not contain a second fallback
policy.  Its output has the same shape and device and dtype
`torch.float8_e4m3fn`.

For each element the kernel must preserve the baseline's observable ordering:

```text
relu_f32   = max(float32(x_bf16), 0)
relu2_bf16 = bf16(relu_f32 * relu_f32)
scale_inv  = float32(1.0) / float32(input_scale)
q_f32      = float32(relu2_bf16) * scale_inv
q_fp8      = fp8(clamp(q_f32, fp8_min, fp8_max))
```

The explicit BF16 rounding and reciprocal-then-multiply ordering are required.
Replacing the latter with direct division may change FP32 rounding; omitting
the former computes a different pipeline (`FP32 ReLU2 -> FP8`).  Either change
may fail bitwise comparison around rounding and saturation boundaries.

The current CUDA ReLU2 treats NaN like a failed `x > 0` comparison and emits
zero; negative infinity also emits zero, while positive infinity is clipped to
the FP8 maximum after squaring.  The fused kernel must preserve those semantics.
It must support CUDA Graph replay without host synchronization or per-replay
metadata allocation.

## Consumer contract

The optimization uses SGLang's existing internal tuple contract
`(fp8_tensor, input_scale, original_dtype)` rather than FP8 dtype alone:

- the general `apply_fp8_linear` path already recognizes FP8 input, skips
  quantization, reuses `input_scale`, and defaults the output to BF16;
- PR #36501's `ModelOptFp8LinearMethod.apply` validates tuple arity, requires an
  E4M3FN tensor, requires scalar/device validity, requires object identity with
  the finalized `layer.input_scale`, requires the explicit output dtype to
  equal `layer.orig_dtype`, rejects Marlin and bare FP8, then routes to the
  existing general or FlashInfer BMM consumer without requantization;
- `apply_fp8_linear_bmm_flashinfer` from that prerequisite skips
  `static_quant_fp8` only for the tuple's FP8 tensor, passes the tuple's scalar
  scale to FlashInfer, and requests BF16 output;
- BF16 input must retain the current BMM behavior byte-for-byte;
- the ReLU2 producer returns exactly the checkpoint-owned
  `down_proj.input_scale`; a different tensor or value is rejected by the
  ReLU2 eligibility/handoff helper before `down_proj`.

No new tuple/dataclass or `LinearBase` change is introduced.  Tests must prove
that bare FP8, malformed tuples, a tuple carrying another layer's scale, and an
ineligible model branch do not enter the optimized ModelOpt path.

## Model integration

The baseline remains:

```python
x, _ = self.up_proj(x)
x = self.act_fn(x)
x, _ = self.down_proj(x)
```

After the runtime predicate above passes, the eligible route is:

```python
x, _ = self.up_proj(x)
x_fp8 = relu2_and_static_quant_fp8(x, self.down_proj.input_scale)
x = (x_fp8, self.down_proj.input_scale, torch.bfloat16)
x, _ = self.down_proj(x)
```

The model owns only the branch, producer call, and established tuple assembly.
Scale validation and backend capability remain inside the ModelOpt method.  The
static flag is absent/false until `process_weights_after_loading` finalizes it;
runtime checks run after `up_proj` and can fall back independently.

## Test-first implementation order

The first production edit is forbidden until the following failing tests are
added and their failures are observed:

1. **Kernel reference tests** across eligible token counts
   `{1, 7, 32, 128, 1024}`.  For production-legal existing ReLU2 widths
   `{16, 128, 4480, 15680}`, compare fused FP8 bits against the actual
   `static_quant_fp8(relu2(x), scale)` chain.  For odd tail widths
   `{1, 15, 127}`, where the existing vectorized ReLU2 kernel is not a legal
   baseline invocation, compare against a scalar tensor reference that exactly
   implements BF16 rounding followed by reciprocal-then-multiply static
   quantization.  Cover negative, zero, BF16 rounding boundaries,
   NaN/infinities, FP8 saturation, and masked tails.  Separately prove that
   zero-token, non-contiguous, wrong model width, rank, dtype, and device cases
   execute the baseline model branch and never invoke the producer.
2. **Consumer tests** for both general and FlashInfer BMM ModelOpt dispatch,
   proving BF16 input quantizes once, a valid tuple quantizes zero times, exact
   scale object/value is preserved, and output remains BF16 with bias/no-bias
   coverage.  Bare FP8, malformed tuples, and a mismatched scale must fail
   closed.
3. **Eligibility tests** covering every fail-closed condition and proving
   `NemotronHMLP.forward` selects exactly one of baseline/fused paths.
4. **CUDA Graph tests** capturing and replaying producer plus each actual
   selected consumer with stable pointers and changed input values.

After implementation, run the existing ReLU2 and ModelOpt FP8 test suites to
guard unrelated callers.

## Benchmark and profiling protocol

Before measurement, record the resolved `origin/main` baseline SHA, #36501
prerequisite SHA, and candidate SHA in the raw result manifest.  Use the
unmodified baseline stack and candidate stack with identical
software, clocks, model weights, warmup, and workload.  Run interleaved ABBA
trials and retain raw JSON/CSV.

1. Producer microbenchmark: baseline ReLU2 + static quant versus fused producer
   at width 15680 and token counts `{1, 8, 32, 128, 512, 2048, 8192, 32768}`;
   200 warmups and 1000 timed iterations per point.
2. Shared-MLP benchmark: `up_proj -> activation/quant -> down_proj`, reporting
   both absolute microseconds and speedup.
3. Nsight Systems/Compute: kernel launches, HBM bytes, achieved bandwidth,
   occupancy, and whether the removed BF16 intermediate traffic is visible.
4. Real Nemotron 9B ModelOpt FP8 service A/B: request throughput, TTFT, ITL,
   output-token throughput, peak memory, and correctness on identical prompts.

The report must state hotspot share and use Amdahl's law to explain why an
operator gain may yield a much smaller end-to-end gain.  Synthetic model data
may validate mechanics but cannot be presented as end-to-end evidence.

## Change budget and stop conditions

Expected production surface:

- one focused producer kernel and Python wrapper;
- one small ModelOpt capability query; tuple consumption remains in #36501;
- one narrow branch in `NemotronHMLP`;
- focused tests and benchmark wiring.

Production means changes under `python/sglang/`, excluding tests, benchmarks,
and docs.  The fixed Triton design makes the 110-200 net-line target auditable;
it must remain below 250 net lines unless a maintainer explicitly requests a
broader abstraction.  Stop and
redesign if the patch needs changes to generic `LinearBase`, introduces a new
public container type, duplicates an FP8 GEMM backend, or needs model-wide
configuration plumbing.

## Rollback and compatibility

The baseline path remains present and is selected automatically when capability
checks fail.  The implementation introduces no checkpoint-format or public API
change.  Removing the fused branch and kernel restores prior behavior; the
prequantized ModelOpt consumer remains owned and tested by prerequisite #36501.
