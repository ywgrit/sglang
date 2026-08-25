# Gated RMSNorm Static FP8 Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse Qwen3.5 GDN's gated RMSNorm and static per-tensor FP8 activation quantization so `out_proj` consumes a pre-quantized activation without launching `static_quant_fp8`.

**Architecture:** Extend the existing FLA Triton gated-norm kernel with an optional FP8 store epilogue using the consumer's device-resident reciprocal scale. Keep the kernel API tensor-only, and let the Qwen3.5 model wrap the reshaped FP8 result as `(fp8, scale, original_dtype)` immediately before `out_proj`. Unsupported consumers and non-CUDA backends retain the exact existing tensor path.

**Tech Stack:** Python 3.12, PyTorch 2.13, Triton, SGLang ModelOpt FP8 linear dispatch, CUDA Graph, pytest.

---

## File map

- Modify `python/sglang/kernels/ops/attention/fla/layernorm_gated.py`: optional static-FP8 epilogue, scale validation, output allocation, and `RMSNorm.forward` API.
- Modify `python/sglang/srt/models/qwen3_5.py`: select the exact static scalar `out_proj` scale, preserve backend fallbacks, reshape FP8 output, and form the linear tuple.
- Temporarily reuse PR1 changes in `python/sglang/srt/layers/layernorm.py` and `python/sglang/srt/layers/quantization/modelopt_quant.py`: ModelOpt eligibility and tuple consumption. These commits are a stacked dependency and must be dropped after PR1 lands.
- Create `test/registered/kernels/ops/attention/test_layernorm_gated_fp8_quant.py`: real CUDA correctness, dtype, shape, saturation, and CUDA Graph replay.
- Create `test/registered/unit/models/test_qwen3_5_gated_rmsnorm_fp8_quant.py`: import-light/model dispatch contracts, unsupported-consumer fallback, and tuple reshaping.
- Create `benchmark/kernels/bench_gated_rmsnorm_fp8_quant.py`: interleaved split-versus-fused latency and JSON/CSV evidence.

### Task 1: Establish the stacked ModelOpt tuple prerequisite

- [x] Cherry-pick PR1's test-first ModelOpt tuple commits `b9afd777dc`, `15a4fce220`, `74c23c6d50`, and `c1f6a19196` onto this branch.
- [x] Resolve only upstream drift; do not copy PR1 all-reduce code.
- [x] Run the import-light ModelOpt tuple test and `git diff --check`.
- [x] Record in the branch notes that these commits disappear when rebased after PR1 merges.

### Task 2: Define the gated-kernel FP8 contract (RED)

- [ ] Add CUDA tests comparing `rms_norm_gated(..., quant_scale=scale)` against `static_quant_fp8(rms_norm_gated(...), scale)` for FP16 and BF16.
- [ ] Cover `norm_before_gate=True/False`, swish/sigmoid where supported, grouped and ungrouped shapes, non-multiple token counts, and 3D input reshape.
- [ ] Assert FP8 dtype, exact shape, scalar FP32 same-device contiguous scale requirements, and saturation at E4M3FN finite limits.
- [ ] Add CUDA Graph capture followed by changing-input replay; verify fresh output and stable addresses.
- [ ] Run the focused test remotely and verify it fails because `quant_scale` is not accepted.

### Task 3: Implement the Triton FP8 epilogue (GREEN)

- [ ] Add `QuantScale` and a compile-time `QUANTIZE_STATIC_FP8` flag to `_layer_norm_fwd_1pass_kernel`.
- [ ] After normalization and gating, first round `y` to the original activation dtype (matching the split kernel's intermediate store), then compute `clamp(y / scale, -448, 448)` in FP32 and store directly through an FP8 output pointer.
- [ ] Allocate FP8 output only when `quant_scale` is present; keep the existing output dtype and launch unchanged otherwise.
- [ ] Validate scale `numel`, dtype, device, contiguity, and FP8 dtype before launch.
- [ ] Thread `quant_scale` through `rms_norm_gated`, `layernorm_fn`, and `RMSNorm.forward` without changing existing callers.
- [ ] Run the focused CUDA tests until green, then run existing gated RMSNorm tests.

### Task 4: Define and implement Qwen3.5 wiring

- [ ] Write failing unit/model tests showing static scalar ModelOpt `out_proj` selects fusion while unquantized, Marlin, dynamic, vector-scale, CPU/XPU/NPU, and incompatible paths do not.
- [ ] Cover `SGLANG_DISABLE_GATED_RMSNORM_FP8_QUANT=1` as an explicit eager/split fallback kill switch.
- [ ] Import and use the shared `_fp8_static_input_scale` consumer probe; do not inspect checkpoint names or architecture strings.
- [ ] Add one helper that runs gated norm, restores `z_shape_og`, flattens heads, and returns either a tensor or `(fp8, scale, original_dtype)`.
- [ ] Use the helper in both the standard forward and XPU-shaped forward without enabling quantization outside CUDA.
- [ ] Prove `out_proj` receives the tuple only after all shape operations, and output dtype remains the original activation dtype.
- [ ] Run focused model tests and existing Qwen3.5 tests.

### Task 5: Benchmark and GPU validation driver

- [x] Write an interleaved benchmark comparing gated RMSNorm + standalone `static_quant_fp8` with the fused epilogue at representative Qwen3.5 GDN shapes.
- [x] Save metadata, warmup, A/A noise, p10/p50/p90, speedup, correctness, and CUDA Graph replay to JSON and CSV.
- [ ] Confirm profiler evidence contains one fused gated-norm launch and no standalone quant launch in the fused arm.
- [ ] Run correctness and benchmark on the rented CUDA host; do not claim a speedup until it clears A/A noise.

### Task 6: Full verification and PR preparation

- [ ] Run focused Python tests, `py_compile`, Ruff/pre-commit for touched files, and `git diff --check`.
- [ ] Rebase onto current `origin/main`; retain or drop the stacked ModelOpt commits according to PR1 merge state.
- [ ] Save raw remote reports and exact hardware/software revisions.
- [ ] Prepare a PR description separating measured kernel improvement from any unmeasured end-to-end estimate.
