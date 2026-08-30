import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRelu2StaticFp8Quant(unittest.TestCase):
    @staticmethod
    def _scalar_reference(x, scale):
        relu = torch.where(x.float() > 0, x.float(), 0.0)
        activated = (relu * relu).to(torch.bfloat16)
        scale_inv = torch.ones((), device=x.device) / scale.float()
        fp8 = torch.finfo(torch.float8_e4m3fn)
        return (
            (activated.float() * scale_inv)
            .clamp(fp8.min, fp8.max)
            .to(torch.float8_e4m3fn)
        )

    def test_matches_bf16_rounding_and_static_quantization(self):
        from sglang.kernels.ops.activation.relu2_fp8_quant import (
            relu2_and_static_quant_fp8,
        )

        values = torch.tensor(
            [
                float("nan"),
                float("-inf"),
                -1.0,
                -0.0,
                0.0,
                0.25,
                1.00390625,
                float("inf"),
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        x = values.repeat(7, 17).contiguous()
        scale = torch.tensor([0.03125], device="cuda", dtype=torch.float32)

        actual = relu2_and_static_quant_fp8(x, scale)
        expected = self._scalar_reference(x, scale)

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertEqual(actual.shape, x.shape)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
