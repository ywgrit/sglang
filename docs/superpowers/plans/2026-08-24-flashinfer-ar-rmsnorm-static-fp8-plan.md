# FlashInfer AllReduce + RMSNorm + Static FP8 Quantization Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement SGLang roadmap issue #31504 Step 3 for Qwen3.5 on NVIDIA Hopper/Blackwell: fuse tensor-parallel AllReduce, residual addition, Gemma RMSNorm, and static per-tensor FP8 activation quantization, while preserving an optional BF16 side output for GDN layers.

**Architecture:** Add a fail-closed FlashInfer quant-fusion capability probe beside the existing unified AllReduce probe. Add one custom-op wrapper that selects FlashInfer pattern 2 (quant-only) or pattern 4 (BF16 + quant), returning tensor-only outputs. LayerNorm translates those tensors into the tuple contracts already accepted by SGLang FP8 linears. Qwen3.5 registers a consumer during construction based on quant-method capability (before ModelOpt scales are finalized); `LayerCommunicator` and LayerNorm validate the finalized scalar scale immediately before dispatch. Ineligible or not-yet-ready consumers retain the existing plain AllReduce + RMSNorm path. AMD per-group and CUDA per-tensor tuple contracts remain separate.

**Tech Stack:** Python 3.10+, PyTorch custom ops/FakeTensor, FlashInfer `comm.allreduce_fusion`, SGLang `LayerCommunicator`, unittest/pytest, CUDA Graph capture, NCCL TP=2 kernel validation plus checkpoint-sized TP=N, NVIDIA H20/H100/H200.

## Non-negotiable contracts

- Scope the first PR to Qwen3.5. Do not advertise NemotronH support until its consumer tuple contract is independently verified.
- The roadmap's Qwen3.5-V2 target uses `ModelOptFp8LinearMethod`. Supporting only native `Fp8LinearMethod` or compressed-tensors is not completion: ModelOpt must both expose its scalar scale to the producer and consume pre-quantized tuple input without launching `static_quant_fp8` again.
- Use `kARResidualRMSNormFP8Quant` for full-attention QKV, with no BF16 norm output.
- Use `kARResidualRMSNormOutFP8Quant` for GDN, because `in_proj_qkvz` consumes FP8 while `in_proj_ba` consumes BF16.
- Preserve Gemma semantics by passing `GemmaRMSNorm.gemma_weight` and leaving FlashInfer `weight_bias=0.0`. Never add one twice.
- Pass the checkpoint's device-resident scalar `input_scale` directly as `scale_factor`; never call `.item()` and never create a host scalar in the hot path.
- Capability and eligibility decisions must be rank-invariant before entering a collective. Missing quant capability disables only quant fusion, not ordinary FlashInfer AllReduce fusion.
- Do not catch a FlashInfer exception after any rank launches the collective and continue through a local fallback. A post-launch failure must fail the request/process consistently.
- No performance or CUDA Graph claim is valid until it is measured on the exact remote checkpoint and topology.

## Worktree and branch setup

### Task 1: Create an isolated feature worktree from current upstream

**Files:**
- Create: `/home/wx/.config/superpowers/worktrees/sglang/flashinfer-ar-rmsnorm-static-fp8/`

**Step 1: Refresh upstream without touching the user's dirty main worktree**

Run:

```bash
cd /home/wx/Documents/github/llm/sglang
git fetch origin main
git status --short
```

Expected: fetch succeeds; the existing untracked user notes remain untouched.

**Step 2: Create the feature worktree**

Run:

```bash
git worktree add /home/wx/.config/superpowers/worktrees/sglang/flashinfer-ar-rmsnorm-static-fp8 -b codex/flashinfer-ar-rmsnorm-static-fp8 origin/main
```

Expected: a new branch and worktree at the fetched `origin/main`. The three portfolio-planning commits must not appear in this feature branch.

**Step 3: Record the base**

Run:

```bash
cd /home/wx/.config/superpowers/worktrees/sglang/flashinfer-ar-rmsnorm-static-fp8
git log -1 --oneline
git status --short
```

Expected: the current upstream commit and an empty status.

## FlashInfer capability and kernel wrapper

### Task 2: Add failing tests for independent quant capability detection

**Files:**
- Modify: `test/registered/unit/layers/test_flashinfer_comm_fusion.py`
- Modify later: `python/sglang/srt/layers/flashinfer_comm_fusion.py`

**Step 1: Extend the fake FlashInfer enum and signature**

Add two enum members and explicit quant arguments to `_FakeFlashInferComm`:

```python
class AllReduceFusionPattern:
    kAllReduce = object()
    kARResidualRMSNorm = object()
    kARResidualRMSNormFP8Quant = object()
    kARResidualRMSNormOutFP8Quant = object()

def allreduce_fusion(
    self,
    *,
    input,
    workspace,
    pattern,
    output=None,
    residual_out=None,
    norm_out=None,
    residual_in=None,
    rms_gamma=None,
    rms_eps=None,
    quant_out=None,
    scale_out=None,
    scale_factor=None,
    weight_bias=0.0,
    **_kwargs,
):
    ...
```

Do not implement quant math yet; these tests cover only capability inspection.

**Step 2: Add three capability tests**

Add tests that call a pure helper named `_supports_allreduce_rmsnorm_static_fp8_quant`:

```python
def test_static_fp8_quant_capability_requires_both_patterns_and_arguments(self):
    self.assertTrue(
        fusion._supports_allreduce_rmsnorm_static_fp8_quant(_FakeFlashInferComm())
    )

    missing_pattern = _FakeFlashInferComm()
    delattr(
        missing_pattern.AllReduceFusionPattern,
        "kARResidualRMSNormOutFP8Quant",
    )
    self.assertFalse(
        fusion._supports_allreduce_rmsnorm_static_fp8_quant(missing_pattern)
    )

def test_missing_quant_signature_does_not_disable_plain_allreduce(self):
    class LegacyComm(_FakeFlashInferComm):
        def allreduce_fusion(self, *, input, workspace, pattern, output=None):
            return output

    self.assertFalse(
        fusion._supports_allreduce_rmsnorm_static_fp8_quant(LegacyComm())
    )
    self.assertFalse(fusion._flashinfer_allreduce_unavailable)
```

Use separate fake enum classes rather than permanently deleting a shared class attribute; restore every patched global in `finally` or use `patch.object`.

The required signature set is:

```python
{"quant_out", "scale_factor", "weight_bias"}
```

`scale_out` is not required for static per-tensor quantization.

Add a mocked two-rank/exact-group test for a future startup helper
`_synchronize_allreduce_quant_capability`: local rank 0 reports the new API and
local rank 1 does not. The mocked CPU-group `all_reduce(MAX)` must cache false
on both ranks, and subsequent preflight must choose ordinary fusion without a
hot-path CPU collective. Also assert that attention TP, MoE TP, and EP paths
pass their own coordinator's `cpu_group`, never unconditional
`get_tp_group().cpu_group`.

**Step 3: Run the focused tests and observe RED**

Run:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py \
  -k 'static_fp8_quant_capability or missing_quant_signature'
```

Expected: failure because `_supports_allreduce_rmsnorm_static_fp8_quant` does not exist. If collection fails for a missing package such as `orjson`, install the repository's development dependencies in an isolated virtual environment first; do not reinterpret a collection error as a test failure.

**Step 4: Commit the RED tests**

```bash
git add test/registered/unit/layers/test_flashinfer_comm_fusion.py
git commit -m "test: define FlashInfer FP8 allreduce capability contract"
```

### Task 3: Implement the quant capability probe

**Files:**
- Modify: `python/sglang/srt/layers/flashinfer_comm_fusion.py:24-100`

**Step 1: Add a pure capability helper**

Add:

```python
_flashinfer_allreduce_quant_available = False
_flashinfer_allreduce_quant_capability_by_group = {}


def _supports_allreduce_rmsnorm_static_fp8_quant(comm) -> bool:
    patterns = getattr(comm, "AllReduceFusionPattern", None)
    if patterns is None:
        return False
    required_patterns = (
        "kARResidualRMSNormFP8Quant",
        "kARResidualRMSNormOutFP8Quant",
    )
    if not all(hasattr(patterns, name) for name in required_patterns):
        return False
    try:
        params = inspect.signature(comm.allreduce_fusion).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return {"quant_out", "scale_factor", "weight_bias"}.issubset(params)


def is_flashinfer_allreduce_quant_available() -> bool:
    return _flashinfer_allreduce_quant_available
```

Add a group-aware startup synchronization helper. Factor exact group
selection into `_get_allreduce_group(use_attn_tp_group)` so it returns
`(world_size, rank, coordinator)` for attention TP, EP, or MoE TP using the
same rules as `ensure_workspace_initialized`:

```python
def _synchronize_allreduce_quant_capability(
    use_attn_tp_group: bool,
) -> bool:
    world_size, _, coordinator = _get_allreduce_group(use_attn_tp_group)
    if world_size <= 1:
        return False
    key = coordinator.cpu_group
    unavailable = torch.tensor(
        [0 if _flashinfer_allreduce_quant_available else 1],
        dtype=torch.int32,
    )
    dist.all_reduce(
        unavailable,
        op=dist.ReduceOp.MAX,
        group=coordinator.cpu_group,
    )
    available = unavailable.item() == 0
    _flashinfer_allreduce_quant_capability_by_group[key] = available
    return available


def _is_allreduce_quant_capability_available_for_group(
    use_attn_tp_group: bool,
) -> bool:
    world_size, _, coordinator = _get_allreduce_group(use_attn_tp_group)
    if world_size <= 1:
        return False
    # Fail closed if startup synchronization has not happened.
    return _flashinfer_allreduce_quant_capability_by_group.get(
        coordinator.cpu_group, False
    )
```

The cache is per exact CPU process group. Do not mutate the process-global local
probe based on one subgroup, because a rank may participate in multiple groups.
Every rank registers the consumer without consulting this flag.

At the very start of `pre_initialize_workspaces`, before its existing
`_flashinfer_allreduce_unavailable or _flashinfer_comm is None` early return,
call startup synchronization for the MoE and attention groups:

```python
_synchronize_allreduce_quant_capability(use_attn_tp_group=False)
_synchronize_allreduce_quant_capability(use_attn_tp_group=True)
```

`BaseRunner._pre_initialize_flashinfer_allreduce_workspace` already invokes
this function during warmup on all ranks before CUDA Graph capture. Keep that
call site unchanged, but add a test proving capability sync occurs before the
early return even when the local `_flashinfer_comm` is absent. This avoids any
CPU collective in layer forward or graph replay.

Keep `_sync_allreduce_unavailable_across_tp` on the enclosing TP CPU group.
The flag it synchronizes is process-global and disables both MoE and attention
fusion, so an exact EP-subgroup vote can leave crossing attention-TP ranks in
different paths and deadlock. The vote must encode the complete local unusable
condition (`_flashinfer_allreduce_unavailable or _flashinfer_comm is None`) and
store the MAX-reduced result back into the global flag. Workspace rendezvous
and quant capability caching remain scoped to the exact attention-TP, EP, or
MoE-TP group.

At startup the collective order on every TP rank is:

1. exact MoE-group quant capability vote;
2. exact attention-TP quant capability vote;
3. enclosing TP unavailable vote;
4. only then the unavailable/missing-comm early return and workspace setup.

Add hybrid TP+EP and missing-local-comm regressions for this ordering. A rank
with no FlashInfer comm must not return while healthy peers enter workspace
collectives.

**Step 2: Set the flag during the existing import probe**

Immediately after obtaining `allreduce_params`, set:

```python
_flashinfer_allreduce_quant_available = (
    _supports_allreduce_rmsnorm_static_fp8_quant(comm)
)
```

If this is false, log at debug level. Do not set `_flashinfer_allreduce_unavailable` and do not warn as though ordinary fusion is broken.

**Step 3: Run GREEN**

Run the command from Task 2 Step 3.

Expected: all selected tests pass.

**Step 4: Commit**

```bash
git add python/sglang/srt/layers/flashinfer_comm_fusion.py
git commit -m "feat: detect FlashInfer static FP8 allreduce support"
```

### Task 4: Add failing numerical tests for both FlashInfer quant patterns

**Files:**
- Modify: `test/registered/unit/layers/test_flashinfer_comm_fusion.py`

**Step 1: Teach the fake backend the exact quant semantics**

Refactor its RMSNorm baseline into a local helper. For quant patterns:

```python
quantized = torch.clamp(
    norm_fp32 / scale_factor.to(torch.float32),
    min=torch.finfo(torch.float8_e4m3fn).min,
    max=torch.finfo(torch.float8_e4m3fn).max,
).to(torch.float8_e4m3fn)
residual_out.copy_(expected_residual)
quant_out.copy_(quantized)
if pattern is self.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant:
    norm_out.copy_(expected_norm)
return quant_out
```

Assert in the fake that `weight_bias == 0.0` and save the exact `scale_factor` object in `self.calls` so the test can verify no host conversion/copy occurred.

**Step 2: Add quant-only and dual-output tests**

Add one parameterized/subtest loop over `keep_bf16=False, True`. Patch:

- `_flashinfer_comm` to the fake.
- `_flashinfer_allreduce_quant_available=True`.
- `_flashinfer_allreduce_quant_capability_by_group` so the exact selected group is cached as available (capability synchronization itself is tested separately).
- workspace manager to an initialized fake workspace.
- `ensure_workspace_initialized=True` if needed to isolate the wrapper.
- `get_parallel()` world sizes so the selected MoE TP group has `world_size=4`.

Call the future public preflight wrapper:

```python
quant_out, residual_out, norm_out = (
    fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
        input_tensor,
        residual,
        gemma_weight,
        input_scale,
        eps,
        use_attn_tp_group=False,
        keep_bf16=keep_bf16,
    )
)
```

Assert:

- `quant_out.dtype == torch.float8_e4m3fn`.
- `residual_out` equals `input * world_size + residual`.
- dequantized/quantized output equals the baseline static quantization exactly.
- `norm_out.numel() == 0` for quant-only.
- `norm_out` equals Gemma RMSNorm baseline for dual-output.
- the fake received `input_scale` by object identity.
- the fake received `weight_bias == 0.0`.
- pattern 2 was used for quant-only and pattern 4 for dual-output.

**Step 3: Add pre-collective fallback tests**

Test that the wrapper returns `(None, None, None)` without invoking the fake collective for each of:

- one rank's quant capability false (all peers fall back via exact CPU group);
- scalar scale absent or `numel() != 1`;
- non-contiguous input/residual/weight;
- world size one;
- workspace unavailable.

Also test that an exception raised by `allreduce_fusion` propagates instead of falling back.

**Step 4: Run RED**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py \
  -k 'static_fp8_quant and not capability'
```

Expected: failure because the wrapper is absent.

**Step 5: Commit RED tests**

```bash
git add test/registered/unit/layers/test_flashinfer_comm_fusion.py
git commit -m "test: cover FlashInfer fused allreduce FP8 patterns"
```

### Task 5: Implement Python preflight plus a tensor-only custom op

**Files:**
- Modify: `python/sglang/srt/layers/flashinfer_comm_fusion.py:760-870`

**Step 1: Add a FakeTensor implementation**

```python
def fake_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant_op(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale_factor: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
    keep_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quant_out = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    residual_out = torch.empty_like(residual)
    norm_out = (
        torch.empty_like(input_tensor)
        if keep_bf16
        else input_tensor.new_empty((0,))
    )
    return quant_out, residual_out, norm_out
```

**Step 2: Add the private real custom op**

Register `_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant_op` as the
custom op. Its annotated and actual return is always exactly three tensors.
It assumes public preflight has already established capability, tensor
eligibility, exact-group workspace readiness, and scalar scale readiness. It
asserts the workspace is present, allocates outputs, launches FlashInfer once,
and returns the three tensors. It must never return `None`.

**Step 3: Add the public Python preflight wrapper**

Add an unregistered Python function
`try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant`. Its pre-launch
checks mirror `flashinfer_allreduce_residual_rmsnorm`, with these additions:

```python
if not _is_allreduce_quant_capability_available_for_group(use_attn_tp_group):
    return None, None, None
if scale_factor is None or scale_factor.numel() != 1:
    return None, None, None
if scale_factor.device != input_tensor.device:
    return None, None, None
```

It performs all rank-invariant shape/dtype/contiguity checks, calls
`ensure_workspace_initialized`, then invokes the private custom op. Fallback is
legal only here, before the private op/collective is entered. This split is
required because the registered op schema cannot declare Tensor outputs and
then return `(None, None, None)`.

Inside the private op, allocate outputs as in the fake. Select:

```python
pattern = (
    _flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant
    if keep_bf16
    else _flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant
)
```

Call the unified API once inside the private op:

```python
kwargs = dict(
    input=input_tensor,
    workspace=workspace_manager.workspace,
    pattern=pattern,
    launch_with_pdl=True,
    residual_out=residual_out,
    norm_out=norm_out if keep_bf16 else None,
    quant_out=quant_out,
    scale_factor=scale_factor,
    residual_in=residual,
    rms_gamma=weight,
    rms_eps=eps,
    weight_bias=0.0,
    use_oneshot=use_oneshot,
    fp32_acc=fp32_acc,
)
if _flashinfer_allreduce_supports_trigger_completion:
    kwargs["trigger_completion_at_end"] = trigger_completion_at_end
_flashinfer_comm.allreduce_fusion(**kwargs)
return quant_out, residual_out, norm_out
```

Match the existing wrapper's exact argument names for `use_oneshot`/`fp32_acc`; inspect upstream FlashInfer once more before implementation and adjust the plan snippet if its primary-source signature differs.

Do not wrap this call in `try/except`.

**Step 4: Run GREEN plus the existing suite**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py
```

Expected: all tests in the file pass on the configured one-GPU test environment; no old backend/workspace test regresses.

**Step 5: Commit**

```bash
git add python/sglang/srt/layers/flashinfer_comm_fusion.py
git commit -m "feat: fuse FlashInfer allreduce RMSNorm and static FP8 quant"
```

### Task 5A: Add ModelOpt FP8 pre-quantized input support

**Files:**
- Modify: `test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py`
- Modify: `python/sglang/srt/layers/quantization/modelopt_quant.py:610-658`
- Modify: `python/sglang/srt/layers/layernorm.py:370-425`

This is required by #31504's primary Qwen3.5-V2/ModelOpt target. The norm-level fusion merged in #33471 taught native and compressed-tensors linears to consume tuples, but current `ModelOptFp8LinearMethod.apply` still accepts only a tensor and launches its own static quantization.

**Step 1: Add a failing ModelOpt dispatch test**

Extend `TestModeloptFp8PerTensorLinear` with a CUDA test that builds one processed ModelOpt layer, quantizes an input once, then calls:

```python
actual, _ = layer((qinput, layer.input_scale, torch.bfloat16))
```

Patch or instrument `static_quant_fp8` and assert it is not called from the linear method. Compare the output against the ordinary BF16-input path and require the same shape/dtype with the existing FP8 tolerance.

Add a separate, undecorated
`TestModeloptFp8PrequantizedDispatch(CustomTestCase)` for the mock-based
CPU-discoverable unit. Do not put it inside
`TestModeloptFp8PerTensorLinear`, because that class has an SM90+ `skipIf`
decorator and would silently skip the contract on a no-GPU runner:

```python
with patch.object(modelopt_quant, "apply_fp8_linear") as apply:
    method.apply(layer, (qinput, scale, torch.bfloat16))
    apply.assert_called_once_with(
        input=qinput,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        input_scale=scale,
        bias=None,
        cutlass_fp8_supported=method.cutlass_fp8_supported,
        pre_quant_output_dtype=torch.bfloat16,
    )
```

Test that Marlin rejects/is ineligible for tuple production, because Marlin deletes `input_scale` and consumes unquantized activations.

**Step 2: Run RED**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py \
  -k 'Modelopt and prequant'
```

Expected: failure because `ModelOptFp8LinearMethod.apply` treats the tuple as a tensor.

**Step 3: Implement tuple consumption before all ModelOpt special paths**

At the top of `ModelOptFp8LinearMethod.apply`, handle tuple input with an
explicit Marlin rejection before SM120 GEMV and FlashInfer BMM dispatch:

```python
if isinstance(x, tuple):
    if self.use_marlin:
        raise TypeError("ModelOpt FP8 Marlin cannot consume pre-quantized input")
    qx, x_scale = x[0], x[1]
    out_dtype = x[2] if len(x) > 2 else None
    return apply_fp8_linear(
        input=qx,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        input_scale=x_scale,
        bias=bias,
        cutlass_fp8_supported=self.cutlass_fp8_supported,
        pre_quant_output_dtype=out_dtype,
    )
```

This deliberately bypasses `apply_fp8_linear_bmm_flashinfer`, whose current implementation begins by calling `static_quant_fp8` on a BF16 tensor. It also bypasses the SM120 GEMV branch, which likewise quantizes internally. Marlin must raise before this branch uses its rearranged weight, and is never registered as an eligible producer consumer. Benchmark the non-Marlin dispatch choice later; correctness and removal of the standalone quant kernel come first.

Validate that `apply_fp8_linear` accepts `pre_quant_output_dtype` on the pinned SGLang base before committing.

**Step 4: Recognize ModelOpt as a static per-tensor consumer**

Tighten native FP8 and extend `_is_static_per_tensor_fp8_linear` for ModelOpt.
Construction-time capability requires that an `input_scale` parameter exists,
but deliberately does not require `numel() == 1`:

```python
input_scale = getattr(linear, "input_scale", None)
if input_scale is None:
    return False

# Keep the existing imports/class checks.
if isinstance(quant_method, Fp8LinearMethod):
    return not (
        getattr(quant_method, "block_quant", False)
        or getattr(quant_method, "use_mxfp8", False)
        or getattr(quant_method, "use_marlin", False)
    )

try:
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp8LinearMethod,
    )
except ImportError:
    ModelOptFp8LinearMethod = ()
if isinstance(quant_method, ModelOptFp8LinearMethod):
    return not getattr(quant_method, "use_marlin", False)
```

This excludes native dynamic-activation FP8 (`input_scale is None`) while still
allowing ModelOpt's pre-load vector parameter. The existing
`_fp8_static_input_scale` `numel() == 1` check remains authoritative after this
class check at forward time.

**Step 5: Run GREEN and the existing ModelOpt backend test**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py \
  -k 'Modelopt'
```

Expected: ordinary and pre-quantized ModelOpt cases pass on SM90+.

**Step 6: Commit**

```bash
git add \
  python/sglang/srt/layers/quantization/modelopt_quant.py \
  python/sglang/srt/layers/layernorm.py \
  test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py
git commit -m "feat: let ModelOpt FP8 linears consume fused quant input"
```

## LayerNorm and communicator integration

### Task 6: Add failing LayerNorm tuple-contract tests

**Files:**
- Create: `test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py`
- Modify later: `python/sglang/srt/layers/layernorm.py`

**Step 1: Test static consumer discovery**

Test two deliberately separate decisions:

1. Construction-time consumer capability via `_is_static_per_tensor_fp8_linear`.
2. Forward-time readiness via `_fp8_static_input_scale`.

Build minimal fake linears with:

- `ModelOptFp8LinearMethod`, pre-load vector `input_scale` -> consumer-capable but not forward-ready;
- the same ModelOpt linear after scale finalization to one element -> consumer-capable and forward-ready;
- native `Fp8LinearMethod`, static scalar `input_scale` -> eligible;
- compressed-tensors static W8A8 scalar scale -> eligible when importable;
- block/MXFP8/Marlin/dynamic/no scale -> consumer-ineligible;
- otherwise-compatible vector scale -> consumer-capable but not forward-ready.

This distinction is mandatory because ModelOpt creates one scale per packed
partition and collapses it to a scalar only in
`process_weights_after_loading`. A constructor-time `numel() == 1` gate would
permanently disable the target path.

**Step 2: Test the quant-only contract**

Patch `try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant` to return known tensors. Call the future method:

```python
result = norm.forward_with_allreduce_fusion_static_fp8_quant(
    x,
    residual,
    quant_linear=static_fp8_linear,
    use_attn_tp_group=False,
    keep_bf16=False,
)
```

Expected:

```python
((quant_out, input_scale, x.dtype), residual_out)
```

Assert the same `input_scale` tensor object is forwarded.

**Step 3: Test the GDN dual-output contract**

With `keep_bf16=True`, expected:

```python
((norm_out, quant_out, input_scale, x.dtype), residual_out)
```

**Step 4: Test Gemma weight semantics and fallback**

For `GemmaRMSNorm`, assert the wrapper receives `norm.gemma_weight`, not raw `norm.weight`. Test that missing static scale, `residual=None`, or wrapper precondition failure returns `None`, allowing the caller to use ordinary fusion.

**Step 5: Run RED and commit**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py
```

Expected: failure because the method is absent.

```bash
git add test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py
git commit -m "test: define allreduce static FP8 layernorm tuples"
```

### Task 7: Implement the shared LayerNorm helper

**Files:**
- Modify: `python/sglang/srt/layers/layernorm.py:370-410,870-920,1220-1265`

**Step 1: Add the shared helper**

```python
def _forward_with_allreduce_fusion_static_fp8_quant(
    norm_module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    quant_linear: Optional[nn.Module],
    use_attn_tp_group: bool = True,
    keep_bf16: bool = False,
):
    if residual is None:
        return None
    scale = _fp8_static_input_scale(quant_linear)
    if scale is None:
        return None
    from sglang.srt.layers.flashinfer_comm_fusion import (
        try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant,
    )
    quant_out, residual_out, norm_out = (
        try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
            input_tensor=x,
            residual=residual,
            weight=weight,
            scale_factor=scale,
            eps=norm_module.variance_epsilon,
            max_token_num=max(x.shape[0], 2048),
            use_attn_tp_group=use_attn_tp_group,
            keep_bf16=keep_bf16,
        )
    )
    if quant_out is None:
        return None
    if keep_bf16:
        return (norm_out, quant_out, scale, x.dtype), residual_out
    return (quant_out, scale, x.dtype), residual_out
```

Static FP8 here is NVIDIA-only. Do not add a staged separate-quant fallback: when the single kernel is unavailable, return `None` and let the existing ordinary AR+RMSNorm path run; the downstream FP8 linear then performs its established quantization.

**Step 2: Add methods to both norm classes**

For `RMSNorm`, pass `self.weight`. For `GemmaRMSNorm`, pass `self.gemma_weight`. Both methods accept `quant_linear`, `use_attn_tp_group`, and `keep_bf16`.

**Step 3: Run GREEN**

Run the Task 6 command and the existing layernorm fusion suite:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py \
  test/registered/layers/test_layernorm_fusion.py
```

Expected: all selected tests pass on their registered environments.

**Step 4: Commit**

```bash
git add python/sglang/srt/layers/layernorm.py
git commit -m "feat: expose allreduce static FP8 norm fusion"
```

### Task 8: Add failing communicator dispatch tests

**Files:**
- Modify: `test/registered/unit/layers/test_layer_communicator_fusion_gate.py`
- Modify later: `python/sglang/srt/layers/communicator.py`

**Step 1: Build a minimal `prepare_attn` fixture**

Create fake hidden/residual objects and a fake norm recording calls. Patch the communication predicates so the flow reaches the fused branch without a real collective.

The communicator constructor gains:

```python
fused_ar_quant_linear: Optional[torch.nn.Module] = None
```

**Step 2: Cover CUDA static-FP8 dispatch**

With `enable_fused_ar_quant=True`, `_use_aiter=False`, and a consumer linear present, assert:

1. `forward_with_allreduce_fusion_static_fp8_quant` is called first with the exact consumer object.
2. `keep_bf16` is forwarded.
3. A non-`None` result is returned unchanged.
4. The ordinary fusion method is not called.

**Step 3: Cover safe fallback and AMD preservation**

Assert ordinary `forward_with_allreduce_fusion` is called when:

- CUDA static helper returns `None`;
- no consumer is registered;
- quant fusion is disabled.

Assert `_use_aiter=True` still calls only the existing per-group helper and preserves its old tuple shape.

**Step 4: Run RED and commit**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_layer_communicator_fusion_gate.py
```

Expected: new dispatch tests fail because the constructor and CUDA branch are absent.

```bash
git add test/registered/unit/layers/test_layer_communicator_fusion_gate.py
git commit -m "test: cover communicator static FP8 fusion dispatch"
```

### Task 9: Implement communicator dispatch

**Files:**
- Modify: `python/sglang/srt/layers/communicator.py:450-475,585-612`

**Step 1: Store the exact consumer**

Add the optional constructor parameter and:

```python
self.fused_ar_quant_linear = fused_ar_quant_linear
```

Do not infer the consumer from the norm or global model state.

**Step 2: Split AMD and CUDA branches**

Preserve the existing AMD branch exactly. Add the CUDA branch:

```python
elif (
    self.enable_fused_ar_quant
    and self.fused_ar_quant_linear is not None
    and hasattr(
        self.input_layernorm,
        "forward_with_allreduce_fusion_static_fp8_quant",
    )
):
    quant_result = (
        self.input_layernorm.forward_with_allreduce_fusion_static_fp8_quant(
            hidden_states,
            residual,
            quant_linear=self.fused_ar_quant_linear,
            use_attn_tp_group=False,
            keep_bf16=self.fused_ar_quant_keep_bf16,
        )
    )
```

If `quant_result is None`, retain the existing ordinary fusion call.

**Step 3: Run GREEN and commit**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_layer_communicator_fusion_gate.py
```

Expected: all communicator gate tests pass.

```bash
git add python/sglang/srt/layers/communicator.py
git commit -m "feat: dispatch CUDA static FP8 allreduce fusion"
```

## Qwen3.5 model integration

### Task 10: Add failing Qwen3.5 tuple-routing tests

**Files:**
- Create: `test/registered/unit/models/test_qwen3_5_fused_ar_quant.py`
- Modify later: `python/sglang/srt/models/qwen3_5.py`

**Step 1: Define separate eligibility helpers**

The tests require two predicates:

```python
_linear_accepts_group_fp8_tuple(linear)  # existing block/MXFP8 behavior
_linear_accepts_static_fp8_tuple(linear) # quant-method capability only
```

The static predicate delegates to `_is_static_per_tensor_fp8_linear` and must
be true for ModelOpt FP8 even while its pre-load `input_scale` is a vector. It
must also recognize native per-tensor FP8 and compatible compressed-tensors
static W8A8 FP8, and be false for Marlin, block/MXFP8, and dynamic consumers.
It must not call `_fp8_static_input_scale`, because scale cardinality is a
forward-time readiness property.

Do not broaden the old AMD predicate and accidentally pass a 4-tuple to block-FP8 code.

**Step 2: Test full-attention routing**

For an eligible static-FP8 `qkv_proj`, a `(fp8, scale, orig_dtype)` tuple is passed directly to the projection on CUDA. For an ordinary/non-static projection, tuple input raises a clear `TypeError`; it must never silently quantize/dequantize the wrong representation.

**Step 3: Test GDN routing**

For `(bf16, fp8, scale, orig_dtype)`:

- `in_proj_qkvz` receives `(fp8, scale, orig_dtype)`.
- `in_proj_ba` receives `bf16`.
- dual-stream and single-stream paths make the same selection.

Retain the AMD `(bf16, fp8, group_scale)` behavior.

**Step 4: Test model registration**

Patch environment/runtime predicates and instantiate the smallest viable Qwen3.5 layer (or call a factored pure helper if full construction is too heavy). Assert:

- full attention registers `qkv_proj`, `keep_bf16=False`;
- GDN registers `in_proj_qkvz`, `keep_bf16=True`;
- a ModelOpt consumer with a pre-load vector scale is still registered;
- CUDA static FP8 is not gated by `enable_aiter_allreduce_fusion`;
- `SGLANG_DISABLE_FUSED_AR_QUANT=1` disables both backends as the existing user opt-out promises.

**Step 5: Run RED and commit**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/models/test_qwen3_5_fused_ar_quant.py
```

Expected: failures for missing static helpers and CUDA routing.

```bash
git add test/registered/unit/models/test_qwen3_5_fused_ar_quant.py
git commit -m "test: define Qwen3.5 static FP8 allreduce routing"
```

### Task 11: Implement Qwen3.5 backend-specific tuple contracts

**Files:**
- Modify: `python/sglang/srt/models/qwen3_5.py:210-265,590-675,850-875,1075-1095,1155-1195`

**Step 1: Split the gate from AMD-only backend eligibility**

Refactor the opt-out into a backend-neutral function:

```python
def _fused_ar_quant_disabled() -> bool:
    return get_bool_env_var("SGLANG_DISABLE_FUSED_AR_QUANT", default="false")
```

Keep `_enable_qwen35_fused_ar_quant()` as the AMD decision if renaming it would cause unnecessary churn. Add a CUDA decision that requires:

- `_is_cuda` and not `_use_aiter`;
- static-FP8 tuple consumer capability on the exact quant method, without checking scale cardinality;
- ordinary FlashInfer allreduce fusion configuration remains enabled by its existing runtime gate.

Do not consult the process-local FlashInfer import/API capability in the model
constructor: asymmetric local decisions would make ranks choose different
collectives. Startup synchronizes capability per exact group; public preflight
reads that cache. Do not initialize a workspace or enter a collective in the
model constructor. The LayerNorm helper calls `_fp8_static_input_scale` at
forward time, after `process_weights_after_loading`, and returns `None` if the
scale is absent/non-scalar so the communicator takes ordinary AR+RMSNorm.

**Step 2: Split tuple selectors**

Implement one explicit selector whose interpretation is anchored by the exact
consumer quant method, not tuple length alone:

```python
def _select_fused_ar_input_for_linear(hidden_states, linear):
    if not isinstance(hidden_states, tuple):
        return hidden_states

    accepts_static = _linear_accepts_static_fp8_tuple(linear)
    accepts_group = _linear_accepts_group_fp8_tuple(linear)

    # CUDA quant-only contract: (fp8, scalar_scale, orig_dtype).
    if accepts_static and len(hidden_states) == 3 and isinstance(
        hidden_states[2], torch.dtype
    ):
        return hidden_states

    # AMD quant-only contract: (fp8, group_scale).
    if accepts_group and len(hidden_states) == 2:
        return hidden_states

    # CUDA GDN dual-output contract.
    if len(hidden_states) == 4:
        hs_bf16, hs_fp8, hs_scale, orig_dtype = hidden_states
        if accepts_static:
            return hs_fp8, hs_scale, orig_dtype
        return hs_bf16

    # AMD GDN dual-output contract.
    if len(hidden_states) == 3:
        hs_bf16, hs_fp8, hs_scale = hidden_states
        if accepts_group:
            return hs_fp8, hs_scale
        return hs_bf16

    raise TypeError(
        f"{linear.__class__.__name__} cannot consume fused AR quant tuple input"
    )
```

This resolves the only length ambiguity: both AMD dual-output and CUDA
quant-only have length 3, but the CUDA tuple's last entry is a `torch.dtype`
and its consumer is a static per-tensor method, while the AMD tuple's last entry
is a scale tensor and its consumer is block/MXFP8.

Unit-test all four contracts and the TypeError branch. Do not introduce a
dataclass/NamedTuple into the graph boundary in this PR.

**Step 3: Add CUDA GDN handler**

Rename `_forward_input_proj_fused_quant_amd` to
`_forward_input_proj_fused_quant`. It first selects `hs_bf16 =
hidden_states[0]`, then obtains `hs_qkvz` through the consumer-aware selector.
Preserve BF16 for `in_proj_ba` and retain current alternate-stream behavior.
Change `_forward_input_proj` to call this routine for tuple input on either AMD
or CUDA, rather than guarding it with `_use_aiter`.

**Step 4: Register consumers in both decoder layer types**

Set:

```python
enable_fused_ar_quant = amd_eligible or cuda_static_eligible
fused_ar_quant_linear = exact_projection if enable_fused_ar_quant else None
```

Pass the exact projection into `LayerCommunicator`.

**Step 5: Remove CUDA tuple guards tied to `_use_aiter`**

At every full-attention projection preparation path, select a tuple whenever
`hidden_states` is a tuple, using the backend-aware selector. In particular,
`forward_prepare_cuda_fused` must begin with:

```python
projection_input = _select_fused_ar_input_for_linear(
    hidden_states, self.qkv_proj
)
qkv, _ = self.qkv_proj(projection_input)
```

After projection, derive token count from a tensor output (`qkv.shape[0]` or
`q_out.shape[0]`), never `hidden_states.shape[0]`, because the fused input may
be a tuple. Make the same selection in `native` and `fused_gate` paths.

Add an `attn_output_gate=True` CUDA-fused-path regression test that passes the
static three-tuple, verifies `qkv_proj` receives it, and proves no tuple
`.shape` access occurs. Verify every other QKV preparation variant either
consumes the tuple or deliberately falls back before it is produced.

**Step 6: Run GREEN plus relevant existing tests**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/models/test_qwen3_5_fused_ar_quant.py \
  test/registered/unit/models/test_qwen3_5_packed_weight_loader.py \
  test/registered/unit/layers/test_layer_communicator_fusion_gate.py
```

Expected: all selected tests pass.

**Step 7: Commit**

```bash
git add python/sglang/srt/models/qwen3_5.py
git commit -m "feat: enable Qwen3.5 FlashInfer allreduce FP8 fusion"
```

## Local integration, benchmark, and remote GPU gate

### Task 12: Add a focused benchmark and CUDA Graph smoke test

**Files:**
- Create: `benchmark/kernels/bench_flashinfer_allreduce_rmsnorm_fp8_quant.py`
- Create or modify: `test/registered/unit/layers/test_flashinfer_comm_fusion.py`

**Step 1: Add CLI benchmark cases**

The benchmark must compare, at TP=2 for representative token counts `1, 8, 32, 128, 512`:

1. unfused AllReduce + residual RMSNorm + static FP8 quant;
2. existing FlashInfer AllReduce + RMSNorm, then static FP8 quant;
3. new quant-only pattern;
4. new dual-output pattern.

Record median, p10/p90, warmup count, iterations, hidden size, dtype, backend, GPU name, FlashInfer/SGLang commits, and CUDA Graph eager/replay modes. Synchronize outside the timed region consistently.

**Step 2: Add graph capture/replay smoke coverage**

Capture with stable input, residual, weight, and device-resident scale addresses. Replay after changing input values in place. Assert output changes and matches the eager baseline. Do not allocate a new scale tensor between capture and replay.

**Step 3: Run syntax and CPU-discoverable checks locally**

```bash
python -m compileall -q \
  python/sglang/srt/layers/flashinfer_comm_fusion.py \
  python/sglang/srt/layers/layernorm.py \
  python/sglang/srt/layers/communicator.py \
  python/sglang/srt/models/qwen3_5.py \
  python/sglang/srt/layers/quantization/modelopt_quant.py \
  benchmark/kernels/bench_flashinfer_allreduce_rmsnorm_fp8_quant.py
ruff check \
  python/sglang/srt/layers/flashinfer_comm_fusion.py \
  python/sglang/srt/layers/layernorm.py \
  python/sglang/srt/layers/communicator.py \
  python/sglang/srt/models/qwen3_5.py \
  python/sglang/srt/layers/quantization/modelopt_quant.py \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py \
  test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py \
  test/registered/unit/layers/test_layer_communicator_fusion_gate.py \
  test/registered/unit/models/test_qwen3_5_fused_ar_quant.py \
  test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py \
  benchmark/kernels/bench_flashinfer_allreduce_rmsnorm_fp8_quant.py
```

Expected: both commands exit 0.

**Step 4: Commit**

```bash
git add benchmark/kernels/bench_flashinfer_allreduce_rmsnorm_fp8_quant.py \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py
git commit -m "bench: measure FlashInfer allreduce FP8 fusion"
```

### Task 13: Identify the exact checkpoint before renting GPUs

**Files:**
- Modify: `docs/superpowers/specs/2026-08-24-five-performance-pr-portfolio-design.md` only in the planning branch, not the feature PR; or record in the final validation report.

**Step 1: Find a public Qwen3.5 static per-tensor FP8 checkpoint**

Use primary model configuration/weight metadata to identify candidates, then
perform a real load-time inspection. The on-disk checkpoint may contain
separate q/k/v or packed-partition scale entries rather than a literal
`qkv_proj.input_scale` key. Completion requires that, after SGLang weight
loading and `process_weights_after_loading`, the merged `qkv_proj.input_scale`
and GDN `in_proj_qkvz.input_scale` both exist, are device tensors, and have
`numel() == 1`. Do not infer this from the repository name or raw key names
alone.

**Step 2: Verify rental image prerequisites before asking the user to start it**

Required:

- two H20/H100/H200 GPUs in one machine;
- peer access/NVLink or the topology intended for the benchmark;
- enough aggregate and per-rank VRAM for the exact checkpoint at the computed TP=N;
- current NVIDIA driver compatible with the chosen PyTorch/FlashInfer image;
- persistent data volume for source, wheels, and checkpoint cache.

**Step 3: Issue the explicit rental signal**

Only after Tasks 1-12 pass locally and the exact checkpoint memory calculation
is complete, tell the user exactly: “现在租 `{N}×{GPU型号}`（同一台机器，按量）”，
plus estimated validation duration and shutdown criteria. `{N}` and the GPU
type must come from measured/authoritative checkpoint weight size plus runtime,
workspace, CUDA Graph, and KV-cache headroom. Do not hard-code 2×H20: the
#31504 example Qwen3.5-397B-A17B-NVFP4-V2 may exceed 2×96GB after runtime
overhead. A separate 2-GPU kernel benchmark does not prove the exact model fits
two GPUs. Before this computed signal, keep saying no GPU is needed.

### Task 14: Run TP=2 kernel validation and checkpoint-sized end-to-end validation

**Files:**
- Create locally after the run: `artifacts/flashinfer-ar-rmsnorm-static-fp8/validation.md` (do not include large raw logs in the PR unless maintainers request them)

**Step 1: Reproduce the environment**

Record exact commands and versions:

```bash
nvidia-smi
python - <<'PY'
import torch, flashinfer
print(torch.__version__)
print(torch.version.cuda)
print(flashinfer.__version__)
PY
git rev-parse HEAD
```

**Step 2: Run unit and graph tests at TP=2**

Use `torchrun --nproc-per-node=2` for the collective/graph smoke entry point. Expected: quant-only and dual-output match baseline within the quantization-defined tolerance, with no hang and successful graph replay.

**Step 3: Run the benchmark**

Run eager and graph modes for both FlashInfer backends supported by the hardware (Hopper single-node normally `trtllm`; do not claim `mnnvl` results unless actually run). Save raw JSON/CSV and the summarized table.

**Step 4: Run the exact Qwen3.5 checkpoint**

Start SGLang at the Task 13 computed `TP=N` with FlashInfer AllReduce fusion
enabled. Compare:

- fusion disabled baseline;
- existing AR+RMSNorm only;
- new AR+RMSNorm+static-FP8.

Validate deterministic prompts/output equivalence within expected FP8 variation, CUDA Graph capture/replay, warmup, decode stability, and memory. Report TTFT, ITL, throughput, and kernel-level latency; do not cherry-pick only one favorable metric.

If `N > 2`, make the rental sequence explicit. Either rent N GPUs once and use
two ranks/GPUs for the kernel microbenchmark before the TP=N model run, or do
two short stages (2 GPUs for kernel validation, shut them down, then N GPUs for
the model). Never keep an idle first instance running while the second is used.

**Step 5: Stop rental promptly**

After logs and artifacts are copied to persistent storage, stop the paid GPU instance. Tell the user validation is complete and whether the instance can be released.

### Task 15: Final regression, rebase, and review package

**Files:**
- All modified feature files

**Step 1: Rebase on current upstream**

```bash
git fetch origin main
git rebase origin/main
```

Expected: clean rebase or conflicts resolved without dropping upstream changes.

**Step 2: Run the complete focused test matrix**

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py \
  test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py \
  test/registered/unit/layers/test_layer_communicator_fusion_gate.py \
  test/registered/unit/models/test_qwen3_5_fused_ar_quant.py \
  test/registered/unit/models/test_qwen3_5_packed_weight_loader.py \
  test/registered/layers/test_layernorm_fusion.py

PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py \
  -k 'Modelopt'
```

Expected: all selected tests pass. The separate CPU mock dispatch class must
run even without SM90; document skips and their hardware reason for the real
ModelOpt numerical cases.

**Step 3: Inspect diff and commit history**

```bash
git diff --check origin/main...HEAD
git status --short
git log --oneline origin/main..HEAD
```

Expected: no whitespace errors, clean worktree, and small reviewable commits.

**Step 4: Prepare (but do not publish) ownership and PR text**

Prepare a concise issue comment describing exact Step 3 scope and a PR body containing:

- why pattern 2 vs pattern 4 is required;
- static scale and tuple contracts;
- collective-safety/fallback behavior;
- TP=2 kernel/unit/graph and TP=N exact-checkpoint validation commands/results;
- measured performance table and limitations.

Do not comment, push, or open a PR without the user's explicit approval, because each mutates remote state.

**Step 5: Request independent code review**

Use `superpowers:requesting-code-review`. Resolve Blocker/Important findings with RED/GREEN regression tests, rerun Task 15 Steps 2-3, and only then present the branch for user approval.

## Resume-safe completion criteria

The item may be described as “implemented” only when local code and focused
tests pass. After Task 14, distinguish “kernel/collective path validated at
TP=2 on 2×{GPU}” from “exact checkpoint validated at TP=N on N×{GPU}”; include
only configurations actually run. It may be described as an “SGLang PR” only
after a real PR URL exists. Until merged, the resume must say “submitted” or
“open-source contribution under review,” never imply upstream adoption.
