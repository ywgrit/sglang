#!/usr/bin/env python3
"""Load Qwen3.5 through SGLang and validate processed static-FP8 scales.

This is validation tooling for the FlashInfer all-reduce/RMSNorm/FP8 work.  It
is intentionally kept on the planning branch, outside the feature PR.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import sys
import time
import traceback
from pathlib import Path


TARGET_SUFFIXES = ("qkv_proj", "in_proj_qkvz")
EXPECTED_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4-V2"
EXPECTED_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
EXPECTED_TP_SIZE = 4
EXPECTED_QUANTIZATION = "modelopt_mixed"
EXPECTED_PROJECTION_COUNTS = {"qkv_proj": 15, "in_proj_qkvz": 45}


def inspect_processed_scales(model, torch, eligibility=None):
    """Return a JSON-serializable report for the two fused projection types."""
    if eligibility is None:
        from sglang.srt.layers.layernorm import _is_static_per_tensor_fp8_linear

        eligibility = _is_static_per_tensor_fp8_linear
    targets = {suffix: [] for suffix in TARGET_SUFFIXES}
    for name, module in model.named_modules():
        suffix = next((value for value in TARGET_SUFFIXES if name.endswith(value)), None)
        if suffix is None:
            continue
        scale = getattr(module, "input_scale", None)
        if scale is None:
            record = {
                "module": name,
                "valid": False,
                "reason": "input_scale is absent",
            }
        else:
            detached = scale.detach()
            quant_method = getattr(module, "quant_method", None)
            eligible_consumer = bool(eligibility(quant_method, module))
            finite = bool(torch.isfinite(detached.float()).all().item())
            positive = bool((detached.float() > 0).all().item())
            record = {
                "module": name,
                "quant_method": type(quant_method).__name__,
                "eligible_static_fp8_consumer": eligible_consumer,
                "shape": list(detached.shape),
                "numel": detached.numel(),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
                "is_cuda": detached.is_cuda,
                "contiguous": detached.is_contiguous(),
                "finite": finite,
                "positive": positive,
            }
            record["valid"] = all(
                (
                    record["numel"] == 1,
                    detached.dtype == torch.float32,
                    record["is_cuda"],
                    record["contiguous"],
                    finite,
                    positive,
                    eligible_consumer,
                )
            )
            if not record["valid"]:
                record["reason"] = (
                    "projection is not an eligible static-FP8 tuple consumer"
                    if not eligible_consumer
                    else "processed scale violates the scalar CUDA FP32 contract"
                )
        targets[suffix].append(record)

    missing = [suffix for suffix, records in targets.items() if not records]
    count_mismatches = {
        suffix: {"expected": EXPECTED_PROJECTION_COUNTS[suffix], "actual": len(records)}
        for suffix, records in targets.items()
        if len(records) != EXPECTED_PROJECTION_COUNTS[suffix]
    }
    invalid = [
        record
        for records in targets.values()
        for record in records
        if not record["valid"]
    ]
    return {
        "valid": not missing and not count_mismatches and not invalid,
        "missing_projection_types": missing,
        "count_mismatches": count_mismatches,
        "counts": {suffix: len(records) for suffix, records in targets.items()},
        "invalid": invalid,
        "targets": targets,
    }


def validate_exact_target(server_args):
    """Refuse a green report for anything except the pinned TP=4 checkpoint."""
    expected = {
        "model_path": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "tp_size": EXPECTED_TP_SIZE,
        "quantization": EXPECTED_QUANTIZATION,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(server_args, name, None)}
        for name, value in expected.items()
        if getattr(server_args, name, None) != value
    }
    if mismatches:
        raise ValueError(
            "scale inspection is pinned to one exact checkpoint: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    terminate_deadline = time.monotonic() + 10
    for process in processes:
        process.join(timeout=max(0, terminate_deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    kill_deadline = time.monotonic() + 10
    for process in processes:
        process.join(timeout=max(0, kill_deadline - time.monotonic()))


def wait_for_processes(processes, timeout_seconds, poll_interval=0.5):
    """Bound model loading and stop every rank after the first failed rank."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        for process in processes:
            process.join(timeout=0)
        exitcodes = [process.exitcode for process in processes]
        if any(code is not None and code != 0 for code in exitcodes):
            _terminate_processes(processes)
            return False
        if all(code is not None for code in exitcodes):
            return all(code == 0 for code in exitcodes)
        if time.monotonic() >= deadline:
            _terminate_processes(processes)
            return False
        time.sleep(poll_interval)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_output_dir(output_dir):
    """Create a new run directory; never reuse reports from an older run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _worker(server_args, port_args, gpu_id, tp_rank, output_dir):
    report_path = Path(output_dir) / f"rank-{tp_rank}.json"
    try:
        import torch

        from sglang.benchmark.one_batch import load_model
        from sglang.srt.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        runner, _tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
        torch_runner = getattr(runner, "torch_runner", None)
        if torch_runner is None:
            raise RuntimeError("scale inspection requires SGLang's PyTorch ModelRunner")
        report = inspect_processed_scales(torch_runner.model, torch)
        report.update(
            {
                "rank": tp_rank,
                "gpu_id": gpu_id,
                "hostname": socket.gethostname(),
                "model_path": server_args.model_path,
                "revision": server_args.revision,
                "tp_size": server_args.tp_size,
                "quantization": server_args.quantization,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(gpu_id),
                "compute_capability": list(torch.cuda.get_device_capability(gpu_id)),
            }
        )
        _write_json(report_path, report)
        if not report["valid"]:
            raise RuntimeError(
                "processed Qwen3.5 FP8 scale contract failed; see " + str(report_path)
            )
    except BaseException as error:
        if not report_path.exists():
            _write_json(
                report_path,
                {
                    "valid": False,
                    "rank": tp_rank,
                    "gpu_id": gpu_id,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        try:
            if "destroy_model_parallel" in locals():
                destroy_model_parallel()
            if "destroy_distributed_environment" in locals():
                destroy_distributed_environment()
        except Exception:
            pass


def _aggregate(output_dir, tp_size, workers_succeeded):
    reports = []
    for rank in range(tp_size):
        path = Path(output_dir) / f"rank-{rank}.json"
        if not path.exists():
            reports.append({"rank": rank, "valid": False, "error": "report missing"})
        else:
            reports.append(json.loads(path.read_text()))
    aggregate = {
        "valid": (
            workers_succeeded
            and len(reports) == tp_size
            and all(item.get("valid") for item in reports)
        ),
        "workers_succeeded": workers_succeeded,
        "model_path": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "quantization": EXPECTED_QUANTIZATION,
        "tp_size": tp_size,
        "ranks": reports,
    }
    _write_json(Path(output_dir) / "summary.json", aggregate)
    return aggregate


def main(argv=None):
    from sglang.benchmark.one_batch import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--inspection-output-dir", required=True)
    parser.add_argument("--inspection-timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(args)
    try:
        validate_exact_target(server_args)
    except ValueError as error:
        parser.error(str(error))
    if args.inspection_timeout_seconds <= 0:
        parser.error("--inspection-timeout-seconds must be positive")
    server_args.dist_timeout = min(
        server_args.dist_timeout or 1800, args.inspection_timeout_seconds
    )
    try:
        prepare_output_dir(args.inspection_output_dir)
    except FileExistsError:
        parser.error(
            "--inspection-output-dir must name a fresh, non-existent run directory"
        )
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    processes = []
    for tp_rank in range(server_args.tp_size):
        process = multiprocessing.Process(
            target=_worker,
            args=(
                server_args,
                port_args,
                tp_rank,
                tp_rank,
                args.inspection_output_dir,
            ),
        )
        process.start()
        processes.append(process)
    workers_succeeded = wait_for_processes(
        processes, timeout_seconds=args.inspection_timeout_seconds
    )

    aggregate = _aggregate(
        args.inspection_output_dir,
        server_args.tp_size,
        workers_succeeded=workers_succeeded,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if not workers_succeeded or not aggregate["valid"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
