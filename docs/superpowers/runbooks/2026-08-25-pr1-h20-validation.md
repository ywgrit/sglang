# PR 1 — H20 validation runbook

This runbook is prepared before rental. Do not add measured values until the
commands have actually run. Keep the feature branch and planning branch SHAs in
the final report so that validation tooling is not confused with PR content.

## Fixed inputs and stop conditions

- Feature: `codex/flashinfer-ar-rmsnorm-static-fp8`
- Planning/validation tooling: `codex/five-performance-pr-portfolio`
- Checkpoint: `nvidia/Qwen3.5-397B-A17B-NVFP4-V2`
- Checkpoint revision: `8f590eae8f10bf55d9a46f79ea0280bde435c9f8`
- Weight files: 243,575,846,403 bytes (226.848 GiB), 26 safetensor shards
- Full-model topology: TP=4 on one `4 x H20 96GB` host
- Kernel topology: ranks/GPU 0-1 of the same host, TP=2
- Persistent free storage before download: at least 400 GB

Stop without benchmarking if any of these is false:

1. all four GPUs are H20-class SM90 devices on the same host;
2. ranks 0-3 have peer access and the topology is recorded;
3. the installed FlashInfer exposes all-reduce patterns 2 and 4 plus
   `quant_out`, `scale_factor`, and `weight_bias` arguments;
4. all processed Qwen3.5 `qkv_proj.input_scale` and
   `in_proj_qkvz.input_scale` parameters are contiguous, positive, finite,
   CUDA FP32 scalars;
5. the model loads without CPU or GPU OOM and leaves defensible graph/KV-cache
   headroom;
6. the baseline trace contains a standalone static-FP8 activation-quantization
   kernel at a wired site. Without that baseline kernel, there is no launch to
   remove and no performance claim to make.

## 1. Capture the machine and container

Run on the host and save every output under a persistent result directory:

```bash
mkdir -p /workspace/results/pr1/{env,scale,kernel,e2e,profiles}
nvidia-smi -q > /workspace/results/pr1/env/nvidia-smi-q.txt
nvidia-smi topo -m > /workspace/results/pr1/env/topology.txt
nvidia-smi nvlink --status > /workspace/results/pr1/env/nvlink-status.txt
df -h /workspace > /workspace/results/pr1/env/disk.txt
free -h > /workspace/results/pr1/env/host-memory.txt
docker pull lmsysorg/sglang:nightly-dev-cu13
docker image inspect lmsysorg/sglang:nightly-dev-cu13 \
  > /workspace/results/pr1/env/container-image.json
```

Use the pulled digest, never the mutable tag alone, in the validation report.
Start one container and keep the Hugging Face cache and results on persistent
storage:

```bash
docker run --rm -it --gpus all --ipc=host --network=host --privileged \
  --shm-size 64g \
  -v /workspace:/workspace \
  -v /workspace/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:nightly-dev-cu13 bash
```

Inside the container, use the feature source without replacing its pinned
runtime dependencies:

```bash
cd /workspace/sglang
git status --short
git rev-parse codex/flashinfer-ar-rmsnorm-static-fp8
git rev-parse codex/five-performance-pr-portfolio
git worktree add /workspace/sglang-pr1 codex/flashinfer-ar-rmsnorm-static-fp8
git worktree add /workspace/sglang-pr1-tools codex/five-performance-pr-portfolio
cd /workspace/sglang-pr1
export PYTHONPATH=/workspace/sglang-pr1/python
```

If either worktree already exists, verify its branch and SHA instead of adding
another one.

## 2. Prove runtime compatibility before downloading/loading the model

```bash
nvidia-smi
nvidia-smi topo -m
python - <<'PY'
import inspect
import torch
import flashinfer
from flashinfer import comm

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("flashinfer", flashinfer.__version__)
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
print("patterns", comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
      comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant)
parameters = inspect.signature(comm.allreduce_fusion).parameters
required = {"quant_out", "scale_factor", "weight_bias"}
print("allreduce_fusion_parameters", tuple(parameters))
assert required.issubset(parameters)
assert torch.cuda.device_count() == 4
assert all(torch.cuda.get_device_capability(i)[0] == 9 for i in range(4))
PY
```

Run the CUDA sample `p2pBandwidthLatencyTest` and `nccl-tests` all-reduce test if
they are present in the image. Record absence explicitly; do not invent a
topology claim from `nvidia-smi topo -m` alone.

## 3. Run local-contract tests in the real image

```bash
set -o pipefail
cd /workspace/sglang-pr1
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/test_flashinfer_comm_fusion.py \
  test/registered/unit/layers/test_layer_communicator_fusion_gate.py \
  test/registered/unit/layers/test_layernorm_allreduce_fp8_quant.py \
  test/registered/unit/layers/quantization/test_fp8_blockwise_linear_backends.py \
  test/registered/unit/models/test_qwen3_5_fused_ar_quant.py \
  2>&1 | tee /workspace/results/pr1/kernel/focused-pytest.txt
test "${PIPESTATUS[0]}" -eq 0
```

Any failure is investigated before the collective benchmark. Do not relabel a
collection failure as a skipped GPU test.

## 4. Validate TP=2 correctness, eager latency, and CUDA Graph replay

Use the Hopper `trtllm` backend. Do not report `mnnvl` unless it is separately
run and its topology requirements are satisfied.

```bash
set -o pipefail
cd /workspace/sglang-pr1
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  benchmark/kernels/bench_flashinfer_allreduce_rmsnorm_fp8_quant.py \
  --backend trtllm --dtype bf16 --tokens 1 8 32 128 512 \
  --hidden-size 4096 --mode both --warmup 100 --iterations 1000 --repeats 9 \
  --json-out /workspace/results/pr1/kernel/h20-tp2.json \
  --csv-out /workspace/results/pr1/kernel/h20-tp2.csv \
  2>&1 | tee /workspace/results/pr1/kernel/h20-tp2.log
test "${PIPESTATUS[0]}" -eq 0
```

The benchmark must report all four cases: the production split baseline,
existing FlashInfer all-reduce+RMSNorm followed by separate quantization,
pattern 2 quant-only, and pattern 4 BF16+FP8 dual output. Correctness includes
changed-input graph replay and an E4M3 lattice-distance limit of two buckets.

## 5. Download the pinned checkpoint and inspect processed scales

Download on persistent storage. If the provider supports a CPU-only mode for
the same disk, do the download there before paid GPU time; otherwise start it
immediately after the runtime gate.

```bash
huggingface-cli download nvidia/Qwen3.5-397B-A17B-NVFP4-V2 \
  --revision 8f590eae8f10bf55d9a46f79ea0280bde435c9f8
```

Load once without CUDA Graph capture and fail closed on the processed scale
contract:

```bash
set -o pipefail
cd /workspace/sglang-pr1
scale_run=/workspace/results/pr1/scale/inspection-$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=/workspace/sglang-pr1/python \
python /workspace/sglang-pr1-tools/scripts/inspect_qwen35_static_fp8_scales.py \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4-V2 \
  --revision 8f590eae8f10bf55d9a46f79ea0280bde435c9f8 \
  --tp-size 4 --quantization modelopt_mixed --disable-radix-cache \
  --disable-cuda-graph --trust-remote-code --mem-fraction-static 0.80 \
  --inspection-timeout-seconds 3600 \
  --inspection-output-dir "$scale_run" \
  2>&1 | tee "${scale_run}.log"
test "${PIPESTATUS[0]}" -eq 0
```

The output directory must not already exist; the inspector creates it and
refuses to reuse older rank reports. Confirm its `summary.json` says both
`"workers_succeeded": true` and `"valid": true` on all four ranks. The report
is load-time evidence; raw checkpoint key names are not a substitute.

## 6. Three full-model arms

Use the same pinned image, model revision, static-memory fraction, CUDA Graph
settings, prompts, request schedule, and clocks for all arms. Restart the server
between arms and preserve logs. The only intended differences are:

| Arm | FlashInfer AR backend | `SGLANG_DISABLE_FUSED_AR_QUANT` |
| --- | --- | --- |
| split baseline | forced off | `1` |
| existing AR+RMSNorm | `trtllm` | `1` |
| new AR+RMSNorm+FP8 | `trtllm` | `0` |

Common launch suffix:

```bash
python -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4-V2 \
  --revision 8f590eae8f10bf55d9a46f79ea0280bde435c9f8 \
  --tp-size 4 --quantization modelopt_mixed --disable-radix-cache \
  --trust-remote-code --mem-fraction-static 0.80
```

For the existing arm add `--flashinfer-allreduce-fusion-backend trtllm` and
export `SGLANG_DISABLE_FUSED_AR_QUANT=1`. For the new arm add the same backend
and export `SGLANG_DISABLE_FUSED_AR_QUANT=0`. For the split baseline, export
`SGLANG_DISABLE_FUSED_AR_QUANT=1` and add
`--enforce-disable-flashinfer-allreduce-fusion`; merely omitting the backend is
not sufficient because Qwen3.5/SM90/TP=4 may auto-enable FlashInfer fusion.

First run deterministic output/logit and CUDA Graph replay checks. Then run A/A
noise measurement and interleaved/crossed A/B orders for TTFT, ITL, throughput,
and memory. Collect an Nsight Systems trace of the same decode window for each
arm. The new-arm claim requires the standalone static-FP8 quant kernel to be
present in the existing arm and absent at the wired sites in the new arm.

The exact serving benchmark command, prompt file hash, concurrency matrix, and
profile window are written to `artifacts/flashinfer-ar-rmsnorm-static-fp8/validation.md`
before looking at final numbers; no favorable-only metric selection is allowed.

## 7. Copy results and shut down

```bash
cd /workspace/results
find pr1 -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum > pr1/SHA256SUMS.txt
du -sh pr1
```

After checking that JSON, CSV, logs, environment files, and profiler traces are
on persistent storage (and copied locally if the provider deletes the disk),
stop the paid GPU instance. Do not leave it running while analyzing results.
