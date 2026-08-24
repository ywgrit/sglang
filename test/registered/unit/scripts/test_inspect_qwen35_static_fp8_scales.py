import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inspect_qwen35_static_fp8_scales.py"
SPEC = importlib.util.spec_from_file_location("inspect_qwen35_static_fp8_scales", SCRIPT_PATH)
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


class _BoolResult:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _Scale:
    def __init__(self):
        self.dtype = "fp32"
        self.is_cuda = True
        self.shape = ()
        self.device = "cuda:0"

    def detach(self):
        return self

    def float(self):
        return self

    def numel(self):
        return 1

    def is_contiguous(self):
        return True

    def __gt__(self, _value):
        return _BoolResult(True)


class _Torch:
    float32 = "fp32"

    @staticmethod
    def isfinite(_scale):
        return _BoolResult(True)


class _Projection:
    def __init__(self, quant_method):
        self.input_scale = _Scale()
        self.quant_method = quant_method


class _Model:
    def __init__(self, qkv_method, qkvz_method):
        self._modules = []
        for index in range(INSPECTOR.EXPECTED_PROJECTION_COUNTS["qkv_proj"]):
            self._modules.append(
                (f"model.layers.{index}.self_attn.qkv_proj", _Projection(qkv_method))
            )
        for index in range(INSPECTOR.EXPECTED_PROJECTION_COUNTS["in_proj_qkvz"]):
            self._modules.append(
                (
                    f"model.layers.{index}.linear_attn.in_proj_qkvz",
                    _Projection(qkvz_method),
                )
            )

    def named_modules(self):
        return iter(self._modules)


class _Process:
    def __init__(self, exitcode=None):
        self.exitcode = exitcode
        self.terminated = False

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.terminated = True

    def is_alive(self):
        return self.exitcode is None and not self.terminated


class TestQwen35ScaleInspector(unittest.TestCase):
    def test_rejects_scalar_scale_on_ineligible_quant_method(self):
        eligible = object()
        ineligible = object()
        model = _Model(eligible, ineligible)

        report = INSPECTOR.inspect_processed_scales(
            model,
            _Torch,
            eligibility=lambda method, _module: method is eligible,
        )

        self.assertFalse(report["valid"])
        self.assertEqual(len(report["invalid"]), 45)
        self.assertTrue(
            all("consumer" in item["reason"] for item in report["invalid"])
        )

    def test_requires_exact_projection_counts(self):
        eligible = object()
        model = _Model(eligible, eligible)
        model._modules.pop()

        report = INSPECTOR.inspect_processed_scales(
            model,
            _Torch,
            eligibility=lambda _method, _module: True,
        )

        self.assertFalse(report["valid"])
        self.assertEqual(
            report["count_mismatches"],
            {"in_proj_qkvz": {"expected": 45, "actual": 44}},
        )

    def test_rejects_any_target_other_than_pinned_tp4_checkpoint(self):
        valid = SimpleNamespace(
            model_path=INSPECTOR.EXPECTED_MODEL,
            revision=INSPECTOR.EXPECTED_REVISION,
            tp_size=4,
            quantization="modelopt_mixed",
        )
        INSPECTOR.validate_exact_target(valid)

        for field, value in (
            ("model_path", "other/model"),
            ("revision", "main"),
            ("tp_size", 2),
            ("quantization", "modelopt_fp8"),
        ):
            bad = SimpleNamespace(**vars(valid))
            setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                INSPECTOR.validate_exact_target(bad)

    def test_first_failed_rank_terminates_other_workers(self):
        failed = _Process(exitcode=1)
        blocked = _Process(exitcode=None)

        result = INSPECTOR.wait_for_processes(
            [blocked, failed], timeout_seconds=60, poll_interval=0
        )

        self.assertFalse(result)
        self.assertTrue(blocked.terminated)


if __name__ == "__main__":
    unittest.main()
