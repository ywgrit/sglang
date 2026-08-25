import inspect
import unittest

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestRMSNormGatedStaticFP8API(unittest.TestCase):
    def test_quant_scale_is_an_explicit_api_parameter(self):
        self.assertIn("quant_scale", inspect.signature(rms_norm_gated).parameters)


class TestRMSNormGatedStaticFP8Quant(CustomTestCase):
    FP8_DTYPE = torch.float8_e4m3fn

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    @staticmethod
    def _inputs(shape, dtype, *, weight_scale=1.0):
        torch.manual_seed(7)
        hidden = shape[-1]
        x = torch.randn(shape, dtype=dtype)
        gate = torch.randn(shape, dtype=dtype)
        weight = torch.randn(hidden, dtype=dtype) * weight_scale
        scale = torch.tensor([0.03125], dtype=torch.float32)
        return x, gate, weight, scale

    @staticmethod
    def _split_reference(
        x,
        gate,
        weight,
        scale,
        *,
        group_size,
        norm_before_gate,
        activation,
    ):
        normed = rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=gate,
            eps=1e-6,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=True,
            activation=activation,
        )
        quantized, _ = static_quant_fp8(
            normed.reshape(-1, normed.shape[-1]),
            scale,
            repeat_scale=False,
        )
        return quantized.reshape(normed.shape)

    def test_fused_quant_matches_split_kernel_exactly(self):
        cases = (
            ((13, 256), torch.bfloat16, 64, True, "swish"),
            ((7, 4, 128), torch.float16, None, True, "sigmoid"),
            ((9, 192), torch.bfloat16, 64, False, "swish"),
        )
        for shape, dtype, group_size, norm_before_gate, activation in cases:
            with self.subTest(
                shape=shape,
                dtype=dtype,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                activation=activation,
            ):
                x, gate, weight, scale = self._inputs(shape, dtype)
                expected = self._split_reference(
                    x,
                    gate,
                    weight,
                    scale,
                    group_size=group_size,
                    norm_before_gate=norm_before_gate,
                    activation=activation,
                )
                actual = rms_norm_gated(
                    x=x,
                    weight=weight,
                    bias=None,
                    z=gate,
                    eps=1e-6,
                    group_size=group_size,
                    norm_before_gate=norm_before_gate,
                    is_rms_norm=True,
                    activation=activation,
                    quant_scale=scale,
                )
                self.assertEqual(actual.dtype, self.FP8_DTYPE)
                self.assertEqual(actual.shape, x.shape)
                self.assertTrue(torch.equal(actual, expected))

    def test_fused_quant_matches_split_saturation(self):
        x, gate, weight, scale = self._inputs(
            (11, 128), torch.bfloat16, weight_scale=1000.0
        )
        expected = self._split_reference(
            x,
            gate,
            weight,
            scale,
            group_size=64,
            norm_before_gate=True,
            activation="swish",
        )
        actual = rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=gate,
            eps=1e-6,
            group_size=64,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="swish",
            quant_scale=scale,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.isfinite(actual.float()).all())
        self.assertLessEqual(actual.float().abs().max().item(), 448.0)

    def test_quant_scale_contract_is_rejected_before_launch(self):
        x, gate, weight, scale = self._inputs((5, 128), torch.bfloat16)
        invalid_scales = (
            torch.ones(2, dtype=torch.float32),
            scale.to(torch.bfloat16),
            torch.ones(2, dtype=torch.float32)[::2],
            torch.tensor([0.03125], dtype=torch.float32, device="cpu"),
        )
        for invalid in invalid_scales:
            with self.subTest(scale=invalid), self.assertRaises(
                (AssertionError, ValueError)
            ):
                rms_norm_gated(
                    x=x,
                    weight=weight,
                    bias=None,
                    z=gate,
                    is_rms_norm=True,
                    quant_scale=invalid,
                )

    def test_cuda_graph_replay_uses_fresh_inputs_and_stable_output(self):
        x, gate, weight, scale = self._inputs((16, 128), torch.bfloat16)
        # Warm Triton compilation before capture.
        rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=gate,
            is_rms_norm=True,
            quant_scale=scale,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = rms_norm_gated(
                x=x,
                weight=weight,
                bias=None,
                z=gate,
                is_rms_norm=True,
                quant_scale=scale,
            )
        output_ptr = output.data_ptr()
        first = output.clone()

        x.copy_(torch.randn_like(x) + 2)
        gate.copy_(torch.randn_like(gate) - 1)
        graph.replay()
        torch.cuda.synchronize()
        second = output.clone()
        expected = self._split_reference(
            x,
            gate,
            weight,
            scale,
            group_size=None,
            norm_before_gate=True,
            activation="swish",
        )

        self.assertEqual(output.data_ptr(), output_ptr)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(second, expected))


if __name__ == "__main__":
    unittest.main()
