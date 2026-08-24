#!/usr/bin/env python3
"""Benchmark FlashInfer TP=2 allreduce + RMSNorm + static FP8 fusion.

Run this on one two-GPU node with, for example::

    torchrun --standalone --nproc-per-node=2 \
      benchmark/kernels/bench_flashinfer_allreduce_rmsnorm_fp8_quant.py

The module is deliberately import-light: importing it for its pure contract
helpers does not initialize Torch, CUDA, SGLang, or a process group.  No sample
or placeholder performance numbers are emitted; every timing in the outputs is
measured by the current invocation.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


DEFAULT_TOKEN_COUNTS = (1, 8, 32, 128, 512)
CASE_NAMES = (
    "split_ar_rmsnorm_static_fp8",
    "flashinfer_ar_rmsnorm_then_static_fp8",
    "flashinfer_ar_rmsnorm_static_fp8",
    "flashinfer_ar_rmsnorm_static_fp8_bf16",
)
MAX_BASELINE_SATURATION_RATE = 0.01
MAX_FP8_ULPS = 2
GPU_FABRIC_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,fabric.state,fabric.status",
    "--format=csv,noheader",
)

METADATA_FIELDS = (
    "timestamp_utc",
    "command",
    "hostname",
    "gpu",
    "gpu_count",
    "gpu_names",
    "gpu_topology",
    "nvidia_driver_version",
    "nccl_version",
    "nvlink_status",
    "gpu_fabric",
    "torch_version",
    "torch_git_version",
    "cuda_version",
    "flashinfer_version",
    "flashinfer_path",
    "flashinfer_commit",
    "sglang_commit",
    "world_size",
    "hidden_size",
    "dtype",
    "backend",
    "requested_backend",
    "mode",
    "warmup",
    "iterations",
    "repeats",
    "token_counts",
    "seed",
    "eps",
    "scale",
    "correctness_only",
)

RESULT_FIELDS = (
    "mode",
    "world_size",
    "token_count",
    "case",
    "available",
    "correctness",
    "error",
    "sample_count",
    "p10_us",
    "p50_us",
    "p90_us",
    "speedup_vs_split",
    "speedup_vs_existing_2kernel",
)


def _linear_percentile(sorted_samples, quantile):
    if not sorted_samples:
        raise ValueError("percentile samples must not be empty")
    position = (len(sorted_samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_samples[lower])
    fraction = position - lower
    return float(
        sorted_samples[lower]
        + (sorted_samples[upper] - sorted_samples[lower]) * fraction
    )


def percentile_summary(samples):
    """Return linearly interpolated p10/p50/p90 CUDA latency in microseconds."""
    ordered = sorted(float(value) for value in samples)
    return {
        "p10_us": _linear_percentile(ordered, 0.10),
        "p50_us": _linear_percentile(ordered, 0.50),
        "p90_us": _linear_percentile(ordered, 0.90),
    }


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def validate_torchrun_environment(environ):
    """Validate and return rank placement from a standard local torchrun."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing = [name for name in required if name not in environ]
    if missing:
        raise RuntimeError(
            "A standard torchrun environment is required; missing "
            + ", ".join(missing)
        )
    try:
        rank = int(environ["RANK"])
        world_size = int(environ["WORLD_SIZE"])
        local_rank = int(environ["LOCAL_RANK"])
        local_world_size = int(environ["LOCAL_WORLD_SIZE"])
    except ValueError as error:
        raise RuntimeError(
            "torchrun rank environment values must be integers"
        ) from error
    if world_size != 2 or local_world_size != 2:
        raise RuntimeError(
            "This benchmark requires WORLD_SIZE=2 and LOCAL_WORLD_SIZE=2; "
            f"got WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}."
        )
    if rank not in (0, 1) or local_rank not in (0, 1):
        raise RuntimeError(
            "TP=2 torchrun ranks must be 0 or 1; "
            f"got RANK={rank}, LOCAL_RANK={local_rank}."
        )
    return rank, world_size, local_rank


def validate_rank_placement(placements):
    """Require two ranks on one hostname, bound to distinct local GPUs."""
    if len(placements) != 2 or len({hostname for hostname, _ in placements}) != 1:
        raise RuntimeError(
            "All two benchmark ranks must run on exactly one host; "
            f"observed placements {placements}."
        )
    local_ranks = {local_rank for _, local_rank in placements}
    if local_ranks != {0, 1}:
        raise RuntimeError(
            "The two benchmark local ranks must be exactly {0, 1}; "
            f"observed placements {placements}."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure eager and CUDA-graph latency for FlashInfer allreduce + "
            "RMSNorm + static FP8 quantization on one TP=2 NVIDIA node."
        )
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_TOKEN_COUNTS),
        help="token counts (default: 1 8 32 128 512)",
    )
    parser.add_argument("--hidden-size", type=positive_int, default=4096)
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument(
        "--mode", choices=("eager", "graph", "both"), default="both"
    )
    parser.add_argument(
        "--backend", choices=("auto", "trtllm", "mnnvl"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.02,
        help=(
            "positive static FP8 activation scale <= 0.1; baseline "
            "saturation is measured from the quantized output"
        ),
    )
    parser.add_argument(
        "--json-out",
        default="flashinfer_allreduce_rmsnorm_fp8_quant.json",
    )
    parser.add_argument(
        "--csv-out",
        default="flashinfer_allreduce_rmsnorm_fp8_quant.csv",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="run correctness (including graph replay validation) without timing",
    )
    return parser


def _load_runtime():
    import flashinfer
    import torch
    import torch.distributed as dist

    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.layers.flashinfer_comm_fusion import (
        cleanup_flashinfer_workspace,
        flashinfer_allreduce_residual_rmsnorm,
        pre_initialize_workspaces,
        resolve_flashinfer_allreduce_fusion_backend,
        try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant,
    )
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.runtime_context import get_context, get_server_args

    return SimpleNamespace(**locals())


def _barrier(runtime, device):
    try:
        runtime.dist.barrier(device_ids=[device.index])
    except TypeError:
        runtime.dist.barrier()


def _all_ranks_true(runtime, value, device):
    flag = runtime.torch.tensor(
        1 if value else 0, dtype=runtime.torch.int32, device=device
    )
    runtime.dist.all_reduce(flag, op=runtime.dist.ReduceOp.MIN)
    return bool(flag.item())


def _rank_max(runtime, value, device):
    latency = runtime.torch.tensor(value, dtype=runtime.torch.float64, device=device)
    runtime.dist.all_reduce(latency, op=runtime.dist.ReduceOp.MAX)
    return float(latency.item())


def _dtype_from_name(torch, name):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _make_inputs(runtime, token_count, hidden_size, dtype, seed, rank, device):
    input_generator = runtime.torch.Generator(device=device)
    input_generator.manual_seed(seed + rank * 104729 + token_count)
    weight_generator = runtime.torch.Generator(device=device)
    weight_generator.manual_seed(seed + hidden_size)
    input_tensor = runtime.torch.randn(
        token_count,
        hidden_size,
        device=device,
        dtype=runtime.torch.float32,
        generator=input_generator,
    ).to(dtype)
    residual = runtime.torch.randn(
        token_count,
        hidden_size,
        device=device,
        dtype=runtime.torch.float32,
        generator=input_generator,
    ).to(dtype)
    raw_weight = (
        runtime.torch.randn(
            hidden_size,
            device=device,
            dtype=runtime.torch.float32,
            generator=weight_generator,
        )
        * 0.1
    ).to(dtype)
    return (
        input_tensor.contiguous(),
        residual.contiguous(),
        raw_weight.contiguous(),
    )


def _build_gemma_norm(runtime, hidden_size, eps, dtype, device, raw_weight):
    """Load checkpoint-style raw weight and retain the adjusted graph buffer."""
    gemma_norm = runtime.GemmaRMSNorm(hidden_size, eps=eps).to(
        device=device, dtype=dtype
    )
    adjusted_pointer = gemma_norm.gemma_weight.data_ptr()
    gemma_norm._weight_loader(gemma_norm.weight, raw_weight)
    if gemma_norm.gemma_weight.data_ptr() != adjusted_pointer:
        raise RuntimeError("Gemma adjusted weight storage changed during loading")
    return gemma_norm, gemma_norm.gemma_weight


def _make_case_state(
    runtime,
    case_name,
    base_input,
    base_residual,
    gemma_norm,
    gemma_weight,
    scale,
    args,
):
    input_tensor = base_input.clone()
    residual = base_residual.clone()

    def split_ar_rmsnorm_static_fp8():
        allreduced = runtime.tensor_model_parallel_all_reduce(input_tensor)
        norm_out, residual_out = gemma_norm.forward_cuda(
            allreduced, residual
        )
        quant_out, _ = runtime.static_quant_fp8(
            norm_out.contiguous(), scale, repeat_scale=False
        )
        return quant_out, residual_out, norm_out

    def flashinfer_ar_rmsnorm_then_static_fp8():
        norm_out, residual_out = runtime.flashinfer_allreduce_residual_rmsnorm(
            input_tensor=input_tensor,
            residual=residual,
            weight=gemma_weight,
            eps=args.eps,
            max_token_num=max(args.tokens),
            use_attn_tp_group=True,
        )
        if norm_out is None:
            return None
        quant_out, _ = runtime.static_quant_fp8(
            norm_out.contiguous(), scale, repeat_scale=False
        )
        return quant_out, residual_out, norm_out

    def flashinfer_ar_rmsnorm_static_fp8(keep_bf16=False):
        quant_out, residual_out, norm_out = (
            runtime.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                input_tensor=input_tensor,
                residual=residual,
                weight=gemma_weight,
                scale_factor=scale,
                eps=args.eps,
                max_token_num=max(args.tokens),
                use_attn_tp_group=True,
                keep_bf16=keep_bf16,
            )
        )
        if quant_out is None:
            return None
        return quant_out, residual_out, norm_out

    calls = {
        CASE_NAMES[0]: split_ar_rmsnorm_static_fp8,
        CASE_NAMES[1]: flashinfer_ar_rmsnorm_then_static_fp8,
        CASE_NAMES[2]: lambda: flashinfer_ar_rmsnorm_static_fp8(False),
        CASE_NAMES[3]: lambda: flashinfer_ar_rmsnorm_static_fp8(True),
    }
    return SimpleNamespace(
        name=case_name,
        input=input_tensor,
        residual=residual,
        weight=gemma_weight,
        gemma_norm=gemma_norm,
        scale=scale,
        call=calls[case_name],
        graph=None,
        output=None,
    )


def _prepare_state(state, base_input, base_residual):
    state.input.copy_(base_input)
    state.residual.copy_(base_residual)


def _tolerances(dtype):
    if str(dtype).endswith("float32"):
        residual = {"rtol": 2e-4, "atol": 2e-4}
        norm = {"rtol": 3e-4, "atol": 3e-4}
    elif str(dtype).endswith("bfloat16"):
        residual = {"rtol": 1e-2, "atol": 1e-2}
        norm = {"rtol": 2e-2, "atol": 2e-2}
    else:
        residual = {"rtol": 5e-3, "atol": 5e-3}
        norm = {"rtol": 1e-2, "atol": 1e-2}
    return residual, norm


def _finite_e4m3_values(torch, device):
    """Return the sorted unique finite E4M3 lattice on ``device``."""
    bit_patterns = torch.arange(256, dtype=torch.uint8, device=device)
    values = bit_patterns.view(torch.float8_e4m3fn).float()
    return torch.unique(values[torch.isfinite(values)], sorted=True)


def _fp8_lattice_indices(torch, values, lattice, label):
    float_values = values.float()
    if not bool(torch.isfinite(float_values).all()):
        raise AssertionError(f"{label} contains non-finite FP8 values")
    indices = torch.searchsorted(lattice, float_values)
    in_range = indices < lattice.numel()
    safe_indices = indices.clamp(max=lattice.numel() - 1)
    exact = in_range & lattice[safe_indices].eq(float_values)
    if not bool(exact.all()):
        raise AssertionError(f"{label} contains values outside the E4M3 lattice")
    return safe_indices


def _unravel_flat_index(flat_index, shape):
    position = []
    for size in reversed(shape):
        position.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(position))


def _local_fp8_saturation_error(
    torch,
    baseline_quant,
    max_saturation_rate=MAX_BASELINE_SATURATION_RATE,
):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    saturation_rate = (
        baseline_quant.float().abs().eq(fp8_max).float().mean()
    )
    if bool(saturation_rate > max_saturation_rate):
        return (
            f"baseline FP8 saturation rate {saturation_rate.item():.2%} "
            f"exceeds {max_saturation_rate:.2%}; choose a larger --scale"
        )
    return None


def _validate_baseline_quantization(runtime, baseline_quant, device):
    local_error = _local_fp8_saturation_error(runtime.torch, baseline_quant)
    all_representative = _all_ranks_true(runtime, local_error is None, device)
    if not all_representative:
        errors = _gather_error(runtime, local_error)
        raise RuntimeError(
            "Benchmark scale is not representative on every rank: " + errors
        )


def _assert_case_matches_baseline(runtime, actual, baseline, dtype, scale, case_name):
    actual_quant, actual_residual, actual_norm = actual
    baseline_quant, baseline_residual, baseline_norm = baseline
    residual_tol, norm_tol = _tolerances(dtype)
    if actual_quant.shape != baseline_quant.shape:
        raise AssertionError(
            f"{case_name} quant shape mismatch: {actual_quant.shape} != "
            f"{baseline_quant.shape}"
        )
    if actual_quant.device != baseline_quant.device:
        raise AssertionError(
            f"{case_name} quant device mismatch: {actual_quant.device} != "
            f"{baseline_quant.device}"
        )
    if actual_quant.dtype != baseline_quant.dtype:
        raise AssertionError(
            f"{case_name} quant dtype mismatch: {actual_quant.dtype} != "
            f"{baseline_quant.dtype}"
        )
    runtime.torch.testing.assert_close(
        actual_residual,
        baseline_residual,
        msg=lambda message: f"{case_name} residual mismatch: {message}",
        **residual_tol,
    )
    lattice = _finite_e4m3_values(runtime.torch, baseline_quant.device)
    actual_indices = _fp8_lattice_indices(
        runtime.torch, actual_quant, lattice, f"{case_name} actual quant"
    )
    baseline_indices = _fp8_lattice_indices(
        runtime.torch, baseline_quant, lattice, "baseline quant"
    )
    bucket_distance = (actual_indices - baseline_indices).abs()
    violations = bucket_distance > MAX_FP8_ULPS
    if bool(violations.any()):
        flat_distance = bucket_distance.reshape(-1)
        flat_position = flat_distance.argmax().item()
        max_bucket_distance = flat_distance[flat_position].item()
        position = _unravel_flat_index(flat_position, actual_quant.shape)
        raise AssertionError(
            f"{case_name} FP8 bucket distance exceeds {MAX_FP8_ULPS}: "
            f"max_bucket_distance={max_bucket_distance}, position={position}"
        )
    if case_name in (CASE_NAMES[1], CASE_NAMES[3]):
        runtime.torch.testing.assert_close(
            actual_norm,
            baseline_norm,
            msg=lambda message: f"{case_name} BF16/FP output mismatch: {message}",
            **norm_tol,
        )


def _gather_error(runtime, local_error):
    gathered = [None] * runtime.dist.get_world_size()
    runtime.dist.all_gather_object(gathered, local_error)
    return "; ".join(
        f"rank {rank}: {message}"
        for rank, message in enumerate(gathered)
        if message is not None
    )


def _check_eager_correctness(runtime, states, base_input, base_residual, dtype, device):
    outcomes = {}
    baseline_state = states[CASE_NAMES[0]]
    _prepare_state(baseline_state, base_input, base_residual)
    baseline = baseline_state.call()
    runtime.torch.cuda.synchronize(device)
    _validate_baseline_quantization(runtime, baseline[0], device)
    outcomes[CASE_NAMES[0]] = {"available": True, "correctness": True, "error": ""}

    for case_name in CASE_NAMES[1:]:
        state = states[case_name]
        _prepare_state(state, base_input, base_residual)
        actual = state.call()
        runtime.torch.cuda.synchronize(device)
        available = _all_ranks_true(runtime, actual is not None, device)
        if not available:
            outcomes[case_name] = {
                "available": False,
                "correctness": False,
                "error": "FlashInfer path returned unavailable on at least one rank",
            }
            continue

        local_error = None
        try:
            _assert_case_matches_baseline(
                runtime,
                actual,
                baseline,
                dtype,
                state.scale,
                case_name,
            )
        except AssertionError as error:
            local_error = str(error)
        correct = _all_ranks_true(runtime, local_error is None, device)
        outcomes[case_name] = {
            "available": True,
            "correctness": correct,
            "error": "" if correct else _gather_error(runtime, local_error),
        }
    return outcomes


def _capture_and_validate_graph(
    runtime,
    state,
    baseline_template,
    base_input,
    base_residual,
    dtype,
    rank,
    device,
):
    _prepare_state(state, base_input, base_residual)
    scale_pointer = state.scale.data_ptr()
    with runtime.graph_capture() as capture_context:
        graph = runtime.torch.cuda.CUDAGraph()
        with runtime.torch.cuda.graph(graph, stream=capture_context.stream):
            captured_output = state.call()
    captured_everywhere = _all_ranks_true(
        runtime, captured_output is not None, device
    )
    if not captured_everywhere:
        return False, "FlashInfer path became unavailable during graph capture"
    state.graph = graph
    state.output = captured_output

    graph.replay()
    runtime.torch.cuda.synchronize(device)
    old_residual = state.output[1].clone()
    changed_input = (base_input.float() * 0.5 + (rank + 1) * 0.125).to(dtype)
    changed_residual = (base_residual.float() * -0.25 + 0.375).to(dtype)
    state.input.copy_(changed_input)
    state.residual.copy_(changed_residual)
    graph.replay()
    runtime.torch.cuda.synchronize(device)
    replay_output = tuple(value.clone() for value in state.output)

    local_error = None
    try:
        if state.scale.data_ptr() != scale_pointer:
            raise AssertionError("scale data_ptr changed across capture/replay")
        if runtime.torch.equal(old_residual, replay_output[1]):
            raise AssertionError(
                "graph replay output did not change after input mutation"
            )

        _prepare_state(baseline_template, changed_input, changed_residual)
        eager_baseline = baseline_template.call()
        runtime.torch.cuda.synchronize(device)
        _assert_case_matches_baseline(
            runtime,
            replay_output,
            eager_baseline,
            dtype,
            state.scale,
            state.name,
        )
    except AssertionError as error:
        local_error = str(error)
    correct = _all_ranks_true(runtime, local_error is None, device)
    error = "" if correct else _gather_error(runtime, local_error)
    _prepare_state(state, base_input, base_residual)
    return correct, error


def _measure(runtime, state, prepare, warmup, iterations, repeats, device):
    for _ in range(warmup):
        prepare()
        runtime.torch.cuda.synchronize(device)
        _barrier(runtime, device)
        if state.graph is not None:
            state.graph.replay()
        else:
            state.output = state.call()
        runtime.torch.cuda.synchronize(device)

    start = runtime.torch.cuda.Event(enable_timing=True)
    end = runtime.torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        for _ in range(iterations):
            prepare()
            runtime.torch.cuda.synchronize(device)
            _barrier(runtime, device)
            start.record()
            if state.graph is not None:
                state.graph.replay()
            else:
                state.output = state.call()
            end.record()
            end.synchronize()
            samples.append(
                _rank_max(runtime, start.elapsed_time(end) * 1000.0, device)
            )
    return samples


def _base_result(mode, world_size, token_count, case_name, outcome):
    return {
        "mode": mode,
        "world_size": world_size,
        "token_count": token_count,
        "case": case_name,
        "available": outcome["available"],
        "correctness": outcome["correctness"],
        "error": outcome["error"],
        "sample_count": 0,
        "p10_us": None,
        "p50_us": None,
        "p90_us": None,
        "speedup_vs_split": None,
        "speedup_vs_existing_2kernel": None,
    }


def _run_shape(runtime, args, mode, token_count, dtype, rank, world_size, device):
    base_input, base_residual, raw_weight = _make_inputs(
        runtime,
        token_count,
        args.hidden_size,
        dtype,
        args.seed,
        rank,
        device,
    )
    gemma_norm, gemma_weight = _build_gemma_norm(
        runtime,
        hidden_size=args.hidden_size,
        eps=args.eps,
        dtype=dtype,
        device=device,
        raw_weight=raw_weight,
    )
    scale = runtime.torch.tensor(
        args.scale, dtype=runtime.torch.float32, device=device
    )
    states = {
        case_name: _make_case_state(
            runtime,
            case_name,
            base_input,
            base_residual,
            gemma_norm,
            gemma_weight,
            scale,
            args,
        )
        for case_name in CASE_NAMES
    }
    outcomes = _check_eager_correctness(
        runtime, states, base_input, base_residual, dtype, device
    )

    if mode == "graph":
        for case_name in CASE_NAMES:
            if not outcomes[case_name]["correctness"]:
                continue
            baseline_template = _make_case_state(
                runtime,
                CASE_NAMES[0],
                base_input,
                base_residual,
                gemma_norm,
                gemma_weight,
                scale,
                args,
            )
            correct, error = _capture_and_validate_graph(
                runtime,
                states[case_name],
                baseline_template,
                base_input,
                base_residual,
                dtype,
                rank,
                device,
            )
            outcomes[case_name]["correctness"] = correct
            outcomes[case_name]["error"] = error

    rows = []
    for case_name in CASE_NAMES:
        outcome = outcomes[case_name]
        row = _base_result(mode, world_size, token_count, case_name, outcome)
        if (
            outcome["available"]
            and outcome["correctness"]
            and not args.correctness_only
        ):
            state = states[case_name]
            samples = _measure(
                runtime,
                state,
                lambda current=state: _prepare_state(
                    current, base_input, base_residual
                ),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            )
            row.update(percentile_summary(samples))
            row["sample_count"] = len(samples)
        rows.append(row)

    split_p50 = rows[0]["p50_us"]
    existing_p50 = rows[1]["p50_us"]
    for row in rows:
        if row["p50_us"] is not None:
            if split_p50 is not None:
                row["speedup_vs_split"] = split_p50 / row["p50_us"]
            if existing_p50 is not None:
                row["speedup_vs_existing_2kernel"] = (
                    existing_p50 / row["p50_us"]
                )
    return rows


def _run_readonly(command, cwd=None):
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return completed.stdout.strip() or None


def _package_commit(package_path):
    if not package_path:
        return None
    path = Path(package_path).resolve()
    start = path if path.is_dir() else path.parent
    return _run_readonly(("git", "rev-parse", "HEAD"), cwd=start)


def _nccl_version(torch):
    try:
        version = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        return None
    if isinstance(version, (tuple, list)):
        return ".".join(str(component) for component in version)
    return str(version)


def _metadata(runtime, args, resolved_backend, world_size, device, repo_root):
    flashinfer_path = str(Path(runtime.flashinfer.__file__).resolve())
    try:
        flashinfer_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        flashinfer_version = getattr(runtime.flashinfer, "__version__", "unknown")
    gpu_names = [
        runtime.torch.cuda.get_device_name(index)
        for index in range(runtime.torch.cuda.device_count())
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "hostname": socket.gethostname(),
        "gpu": runtime.torch.cuda.get_device_name(device),
        "gpu_count": runtime.torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "gpu_topology": _run_readonly(("nvidia-smi", "topo", "-m")),
        "nvidia_driver_version": _run_readonly(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        ),
        "nccl_version": _nccl_version(runtime.torch),
        "nvlink_status": _run_readonly(("nvidia-smi", "nvlink", "--status")),
        "gpu_fabric": _run_readonly(GPU_FABRIC_QUERY),
        "torch_version": runtime.torch.__version__,
        "torch_git_version": getattr(runtime.torch.version, "git_version", None),
        "cuda_version": runtime.torch.version.cuda,
        "flashinfer_version": flashinfer_version,
        "flashinfer_path": flashinfer_path,
        "flashinfer_commit": _package_commit(flashinfer_path),
        "sglang_commit": _run_readonly(("git", "rev-parse", "HEAD"), cwd=repo_root),
        "world_size": world_size,
        "hidden_size": args.hidden_size,
        "dtype": args.dtype,
        "backend": resolved_backend,
        "requested_backend": args.backend,
        "mode": args.mode,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "token_counts": list(args.tokens),
        "seed": args.seed,
        "eps": args.eps,
        "scale": args.scale,
        "correctness_only": args.correctness_only,
    }


def _write_outputs(metadata, results, json_path, csv_path):
    if json_path:
        output = Path(json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
        )
    if csv_path:
        output = Path(csv_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(METADATA_FIELDS) + list(RESULT_FIELDS)
        with output.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for result in results:
                row = dict(metadata)
                row["gpu_names"] = json.dumps(row["gpu_names"])
                row["token_counts"] = json.dumps(row["token_counts"])
                row.update(result)
                writer.writerow(row)


def _format_measurement(value):
    return "-" if value is None else f"{value:.2f}"


def _print_results(metadata, results):
    print(json.dumps(metadata, indent=2))
    print()
    print(
        "mode   tokens case                                          "
        "correct p10(us) p50(us) p90(us) split-x existing-x"
    )
    for row in results:
        correctness = "PASS" if row["correctness"] else "FAIL"
        print(
            f"{row['mode']:<7}{row['token_count']:<7}"
            f"{row['case']:<46}{correctness:<8}"
            f"{_format_measurement(row['p10_us']):>8} "
            f"{_format_measurement(row['p50_us']):>8} "
            f"{_format_measurement(row['p90_us']):>8} "
            f"{_format_measurement(row['speedup_vs_split']):>7} "
            f"{_format_measurement(row['speedup_vs_existing_2kernel']):>10}"
        )
        if row["error"]:
            print(f"        error: {row['error']}")


def _validate_launch(torch, args, world_size, local_rank):
    if world_size != 2:
        raise RuntimeError(
            "This benchmark requires exactly one local TP=2 torchrun job "
            f"(WORLD_SIZE=2); got WORLD_SIZE={world_size}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "Two visible local CUDA devices are required; "
            f"found {torch.cuda.device_count()}."
        )
    if local_rank not in (0, 1):
        raise RuntimeError(f"LOCAL_RANK must be 0 or 1; got {local_rank}")
    if not math.isfinite(args.scale) or not 0 < args.scale <= 0.1:
        raise ValueError(
            "--scale must be positive and <= 0.1; the upper bound avoids "
            "degenerate near-zero FP8 outputs, while baseline saturation is "
            "measured separately"
        )
    if args.eps <= 0 or not math.isfinite(args.eps):
        raise ValueError("--eps must be finite and positive")


def cleanup_runtime(runtime, model_parallel_started):
    """Tear down partial model-parallel state before distributed state."""
    try:
        if model_parallel_started:
            try:
                runtime.cleanup_flashinfer_workspace()
            finally:
                runtime.destroy_model_parallel()
    finally:
        if runtime.dist.is_initialized():
            runtime.destroy_distributed_environment()


def main(argv=None):
    args = build_parser().parse_args(argv)
    rank, world_size, local_rank = validate_torchrun_environment(os.environ)
    import torch

    _validate_launch(torch, args, world_size, local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    runtime = _load_runtime()
    dtype = _dtype_from_name(runtime.torch, args.dtype)
    repo_root = Path(__file__).resolve().parents[2]

    model_parallel_started = False
    results = []
    metadata = None
    runtime.set_custom_all_reduce(True)
    with runtime.get_context().override_server_args(
        flashinfer_allreduce_fusion_backend=args.backend,
        nnodes=1,
    ):
        try:
            runtime.init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                distributed_init_method="env://",
                backend="nccl",
            )
            model_parallel_started = True
            runtime.initialize_model_parallel(tensor_model_parallel_size=world_size)

            placements = [None] * world_size
            runtime.dist.all_gather_object(
                placements, (socket.gethostname(), local_rank)
            )
            validate_rank_placement(placements)

            runtime.pre_initialize_workspaces(
                max_token_num=max(args.tokens),
                hidden_dim=args.hidden_size,
                dtype=dtype,
            )
            resolved_backend = runtime.resolve_flashinfer_allreduce_fusion_backend(
                runtime.get_server_args()
            )
            modes = ("eager", "graph") if args.mode == "both" else (args.mode,)
            for mode in modes:
                for token_count in args.tokens:
                    results.extend(
                        _run_shape(
                            runtime,
                            args,
                            mode,
                            token_count,
                            dtype,
                            rank,
                            world_size,
                            device,
                        )
                    )

            metadata = _metadata(
                runtime, args, resolved_backend, world_size, device, repo_root
            )
            output_error = None
            if rank == 0:
                try:
                    _print_results(metadata, results)
                    _write_outputs(metadata, results, args.json_out, args.csv_out)
                except Exception as error:
                    # Every peer must observe rank-0 output failure before any
                    # rank leaves the common collective sequence.  Re-raise
                    # below on all ranks; never continue with partial output.
                    output_error = f"{type(error).__name__}: {error}"
            output_status = [output_error]
            runtime.dist.broadcast_object_list(output_status, src=0)
            if output_status[0] is not None:
                raise RuntimeError(f"rank-0 output failed: {output_status[0]}")
            _barrier(runtime, device)
            if not all(row["correctness"] for row in results):
                raise RuntimeError(
                    "At least one benchmark path was unavailable or failed "
                    "correctness; no successful timing was recorded for that path."
                )
        finally:
            cleanup_runtime(runtime, model_parallel_started)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
