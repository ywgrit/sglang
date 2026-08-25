import os
import unittest
from unittest.mock import Mock, patch

import torch

import sglang.srt.models.qwen3_5 as qwen3_5


class TestQwen35GatedRMSNormStaticFP8Quant(unittest.TestCase):
    def test_helper_is_exposed(self):
        self.assertTrue(
            hasattr(qwen3_5, "_prepare_gated_rmsnorm_out_proj_input")
        )

    @staticmethod
    def _inputs(dtype=torch.bfloat16):
        core = torch.randn(24, 64, dtype=dtype)
        gate = torch.randn(24, 64, dtype=dtype)
        shape_with_heads = (6, 4, 64)
        norm_fp8 = torch.empty((24, 64), dtype=torch.float8_e4m3fn)
        norm_bf16 = torch.empty((24, 64), dtype=dtype)
        return core, gate, shape_with_heads, norm_fp8, norm_bf16

    def test_eligible_cuda_consumer_receives_reshaped_tuple(self):
        core, gate, shape, norm_fp8, _ = self._inputs(torch.float16)
        scale = torch.tensor([0.03125], dtype=torch.float32)
        norm = Mock(return_value=norm_fp8)
        out_proj = Mock()

        with (
            patch.object(qwen3_5, "_is_cuda", True),
            patch.object(
                qwen3_5, "_fp8_static_input_scale", return_value=scale
            ) as scale_probe,
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("SGLANG_DISABLE_GATED_RMSNORM_FP8_QUANT", None)
            actual = qwen3_5._prepare_gated_rmsnorm_out_proj_input(
                norm, out_proj, core, gate, shape
            )

        scale_probe.assert_called_once_with(out_proj)
        norm.assert_called_once_with(core, gate, quant_scale=scale)
        self.assertIs(actual[1], scale)
        self.assertEqual(actual[2], torch.float16)
        self.assertEqual(actual[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(actual[0].shape, (shape[0], shape[1] * shape[2]))
        self.assertEqual(actual[0].data_ptr(), norm_fp8.data_ptr())

    def test_ineligible_consumer_preserves_existing_tensor_path(self):
        core, gate, shape, _, norm_bf16 = self._inputs()
        norm = Mock(return_value=norm_bf16)
        out_proj = Mock()

        with (
            patch.object(qwen3_5, "_is_cuda", True),
            patch.object(qwen3_5, "_fp8_static_input_scale", return_value=None),
        ):
            actual = qwen3_5._prepare_gated_rmsnorm_out_proj_input(
                norm, out_proj, core, gate, shape
            )

        norm.assert_called_once_with(core, gate)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.dtype, core.dtype)
        self.assertEqual(actual.shape, (shape[0], shape[1] * shape[2]))

    def test_non_cuda_backend_does_not_probe_or_quantize(self):
        core, gate, shape, _, norm_bf16 = self._inputs()
        norm = Mock(return_value=norm_bf16)
        out_proj = Mock()

        with (
            patch.object(qwen3_5, "_is_cuda", False),
            patch.object(qwen3_5, "_fp8_static_input_scale") as scale_probe,
        ):
            actual = qwen3_5._prepare_gated_rmsnorm_out_proj_input(
                norm, out_proj, core, gate, shape
            )

        scale_probe.assert_not_called()
        norm.assert_called_once_with(core, gate)
        self.assertIsInstance(actual, torch.Tensor)

    def test_kill_switch_preserves_existing_tensor_path(self):
        core, gate, shape, _, norm_bf16 = self._inputs()
        norm = Mock(return_value=norm_bf16)
        out_proj = Mock()

        with (
            patch.object(qwen3_5, "_is_cuda", True),
            patch.object(qwen3_5, "_fp8_static_input_scale") as scale_probe,
            patch.dict(
                os.environ,
                {"SGLANG_DISABLE_GATED_RMSNORM_FP8_QUANT": "1"},
            ),
        ):
            actual = qwen3_5._prepare_gated_rmsnorm_out_proj_input(
                norm, out_proj, core, gate, shape
            )

        scale_probe.assert_not_called()
        norm.assert_called_once_with(core, gate)
        self.assertIsInstance(actual, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
