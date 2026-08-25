#!/usr/bin/env python3
"""Benchmark gated RMSNorm + static FP8 quantization.

The split arm stores the norm result in the model dtype and launches
``static_quant_fp8``.  The fused arm asks the gated-norm Triton kernel to store
the same rounded values directly as FP8.  GPU imports are intentionally lazy
so report helpers remain testable on a CPU-only development machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


def interleaved_order(rounds: int) -> list[tuple[str, str]]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    return [
        ("split", "fused") if index % 2 == 0 else ("fused", "split")
        for index in range(rounds)
    ]


def _nearest_rank(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize empty samples")
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def summarize(samples: Iterable[float]) -> dict[str, float | int]:
    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("cannot summarize empty samples")
    return {
        "count": len(values),
        "mean_us": statistics.fmean(values),
        "p10_us": _nearest_rank(values, 0.10),
        "p50_us": _nearest_rank(values, 0.50),
        "p90_us": _nearest_rank(values, 0.90),
    }


def build_report(
    *,
    metadata: dict[str, Any],
    config: dict[str, Any],
    split_samples: Iterable[float],
    fused_samples: Iterable[float],
    correctness: dict[str, Any],
    aa_samples: Iterable[float] | None = None,
) -> dict[str, Any]:
    split_values = [float(value) for value in split_samples]
    fused_values = [float(value) for value in fused_samples]
    arms = {
        "split": {"samples_us": split_values, "summary": summarize(split_values)},
        "fused": {"samples_us": fused_values, "summary": summarize(fused_values)},
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "metadata": metadata,
        "config": config,
        "correctness": correctness,
        "arms": arms,
        "speedup": arms["split"]["summary"]["p50_us"]
        / arms["fused"]["summary"]["p50_us"],
    }
    if aa_samples is not None:
        aa_values = [float(value) for value in aa_samples]
        report["arms"]["split_aa"] = {
            "samples_us": aa_values,
            "summary": summarize(aa_values),
        }
        split_p50 = arms["split"]["summary"]["p50_us"]
        aa_p50 = report["arms"]["split_aa"]["summary"]["p50_us"]
        report["aa_noise_fraction"] = abs(aa_p50 - split_p50) / split_p50
    return report


def write_reports(report: dict[str, Any], prefix: Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm", "sample", "latency_us"))
        writer.writeheader()
        for arm, values in report["arms"].items():
            for index, latency in enumerate(values["samples_us"]):
                writer.writerow(
                    {"arm": arm, "sample": index, "latency_us": latency}
                )
    return json_path, csv_path


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _time_cuda_calls(torch: Any, function: Callable[[], Any], count: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--scale", type=float, default=0.03125)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if min(args.tokens, args.heads, args.head_dim, args.warmup, args.rounds, args.iterations) < 1:
        raise ValueError("shape and iteration arguments must be positive")

    import torch

    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    rows = args.tokens * args.heads
    x = torch.randn((rows, args.head_dim), device="cuda", dtype=dtype)
    gate = torch.randn_like(x)
    weight = torch.randn((args.head_dim,), device="cuda", dtype=dtype)
    scale = torch.tensor([args.scale], device="cuda", dtype=torch.float32)

    def split() -> Any:
        normed = rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=gate,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="swish",
        )
        return static_quant_fp8(normed, scale, repeat_scale=False)[0]

    def fused() -> Any:
        return rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=gate,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="swish",
            quant_scale=scale,
        )

    for _ in range(args.warmup):
        split()
        fused()
    torch.cuda.synchronize()
    split_expected = split()
    fused_actual = fused()
    exact = bool(torch.equal(split_expected, fused_actual))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fused()
    x.copy_(torch.randn_like(x) + 1)
    gate.copy_(torch.randn_like(gate) - 1)
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = bool(torch.equal(graph_output, split()))

    functions = {"split": split, "fused": fused}
    samples = {"split": [], "fused": []}
    for first, second in interleaved_order(args.rounds):
        for arm in (first, second):
            samples[arm].extend(
                _time_cuda_calls(torch, functions[arm], args.iterations)
            )
    aa_samples = _time_cuda_calls(torch, split, args.iterations * 2)

    device = torch.cuda.current_device()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[0],
        "git_sha": _command_output(["git", "rev-parse", "HEAD"]),
        "pid": os.getpid(),
    }
    config = {
        "tokens": args.tokens,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "rows": rows,
        "dtype": args.dtype,
        "scale": args.scale,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "iterations": args.iterations,
        "seed": args.seed,
    }
    report = build_report(
        metadata=metadata,
        config=config,
        split_samples=samples["split"],
        fused_samples=samples["fused"],
        aa_samples=aa_samples,
        correctness={"exact": exact, "cuda_graph_exact": graph_exact},
    )
    json_path, csv_path = write_reports(report, args.output_prefix)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    if not exact or not graph_exact:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
