import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


BENCHMARK = (
    Path(__file__).parents[4]
    / "benchmark/kernels/bench_gated_rmsnorm_fp8_quant.py"
)


def load_benchmark():
    spec = importlib.util.spec_from_file_location("bench_gated_rmsnorm_fp8", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGatedRMSNormFP8BenchmarkContract(unittest.TestCase):
    def test_interleaved_order_balances_first_arm(self):
        benchmark = load_benchmark()
        self.assertEqual(
            benchmark.interleaved_order(4),
            [
                ("split", "fused"),
                ("fused", "split"),
                ("split", "fused"),
                ("fused", "split"),
            ],
        )

    def test_summary_uses_sorted_nearest_rank_percentiles(self):
        benchmark = load_benchmark()
        summary = benchmark.summarize([9.0, 1.0, 7.0, 3.0, 5.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["p10_us"], 1.0)
        self.assertEqual(summary["p50_us"], 5.0)
        self.assertEqual(summary["p90_us"], 9.0)
        self.assertEqual(summary["mean_us"], 5.0)

    def test_reports_preserve_raw_samples_and_metadata(self):
        benchmark = load_benchmark()
        report = benchmark.build_report(
            metadata={"gpu": "test-gpu", "git_sha": "abc123"},
            config={"tokens": 8, "heads": 4, "head_dim": 64},
            split_samples=[4.0, 6.0],
            fused_samples=[2.0, 3.0],
            correctness={"exact": True, "cuda_graph_exact": True},
        )
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["metadata"]["gpu"], "test-gpu")
        self.assertEqual(report["arms"]["split"]["samples_us"], [4.0, 6.0])
        self.assertEqual(report["arms"]["fused"]["samples_us"], [2.0, 3.0])
        self.assertEqual(report["speedup"], 2.0)

        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "gated-rmsnorm-fp8"
            json_path, csv_path = benchmark.write_reports(report, prefix)
            loaded = json.loads(json_path.read_text())
            self.assertEqual(loaded, report)
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["arm"] for row in rows}, {"split", "fused"})


if __name__ == "__main__":
    unittest.main()
