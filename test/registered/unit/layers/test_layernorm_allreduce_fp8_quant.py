import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers import flashinfer_comm_fusion, layernorm
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestLayerNormAllReduceStaticFp8Quant(unittest.TestCase):
    @staticmethod
    def _make_modelopt_linear(input_scale, *, use_marlin=False):
        # Keep the readiness check honest: _fp8_static_input_scale must see the
        # real ModelOpt method type, while no hardware-dependent constructor or
        # GEMM is needed for this hand-off contract.
        quant_method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
        quant_method.use_marlin = use_marlin
        return types.SimpleNamespace(
            quant_method=quant_method,
            input_scale=input_scale,
        )

    @staticmethod
    def _norm_cases():
        rms = layernorm.RMSNorm(4, eps=1e-5)
        gemma = layernorm.GemmaRMSNorm(4, eps=2e-5)
        return (
            ("rms", rms, rms.weight),
            ("gemma", gemma, gemma.gemma_weight),
        )

    def _assert_wrapper_call(
        self,
        wrapper,
        *,
        norm,
        x,
        residual,
        expected_weight,
        scale,
        use_attn_tp_group,
        keep_bf16,
    ):
        wrapper.assert_called_once()
        kwargs = wrapper.call_args.kwargs
        self.assertLessEqual(
            {
                "input_tensor",
                "residual",
                "weight",
                "scale_factor",
                "eps",
                "max_token_num",
                "use_attn_tp_group",
                "keep_bf16",
            },
            set(kwargs),
        )
        self.assertIs(kwargs["input_tensor"], x)
        self.assertIs(kwargs["residual"], residual)
        self.assertIs(kwargs["weight"], expected_weight)
        self.assertIs(kwargs["scale_factor"], scale)
        self.assertEqual(kwargs["eps"], norm.variance_epsilon)
        self.assertEqual(kwargs["max_token_num"], max(x.shape[0], 2048))
        self.assertIs(kwargs["use_attn_tp_group"], use_attn_tp_group)
        self.assertIs(kwargs["keep_bf16"], keep_bf16)

    def test_quant_only_returns_modelopt_tuple_and_exact_norm_weight(self):
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        quant_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        residual_out = torch.empty_like(x)
        empty_norm = torch.empty(0, dtype=x.dtype)
        input_scale = torch.tensor(0.125, dtype=torch.float32)
        quant_linear = self._make_modelopt_linear(input_scale)

        for name, norm, expected_weight in self._norm_cases():
            with self.subTest(norm=name), mock.patch.object(
                flashinfer_comm_fusion,
                "try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant",
                return_value=(quant_out, residual_out, empty_norm),
            ) as wrapper:
                result = norm.forward_with_allreduce_fusion_static_fp8_quant(
                    x,
                    residual,
                    quant_linear,
                    use_attn_tp_group=False,
                    keep_bf16=False,
                )

            quant_tuple, actual_residual = result
            self.assertIs(quant_tuple[0], quant_out)
            self.assertIs(quant_tuple[1], input_scale)
            self.assertIs(quant_tuple[2], x.dtype)
            self.assertIs(actual_residual, residual_out)
            self._assert_wrapper_call(
                wrapper,
                norm=norm,
                x=x,
                residual=residual,
                expected_weight=expected_weight,
                scale=input_scale,
                use_attn_tp_group=False,
                keep_bf16=False,
            )

    def test_dual_output_returns_norm_and_quant_tuple_with_exact_scale(self):
        # Exercise the non-default max_token_num branch without allocating a
        # meaningful amount of memory.
        x = torch.randn(2050, 4, dtype=torch.float16)
        residual = torch.randn_like(x)
        quant_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        residual_out = torch.empty_like(x)
        norm_out = torch.empty_like(x)
        input_scale = torch.tensor(0.25, dtype=torch.float32)
        quant_linear = self._make_modelopt_linear(input_scale)

        for name, norm, expected_weight in self._norm_cases():
            with self.subTest(norm=name), mock.patch.object(
                flashinfer_comm_fusion,
                "try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant",
                return_value=(quant_out, residual_out, norm_out),
            ) as wrapper:
                result = norm.forward_with_allreduce_fusion_static_fp8_quant(
                    x,
                    residual,
                    quant_linear,
                    use_attn_tp_group=True,
                    keep_bf16=True,
                )

            quant_tuple, actual_residual = result
            self.assertIs(quant_tuple[0], norm_out)
            self.assertIs(quant_tuple[1], quant_out)
            self.assertIs(quant_tuple[2], input_scale)
            self.assertIs(quant_tuple[3], x.dtype)
            self.assertIs(actual_residual, residual_out)
            self._assert_wrapper_call(
                wrapper,
                norm=norm,
                x=x,
                residual=residual,
                expected_weight=expected_weight,
                scale=input_scale,
                use_attn_tp_group=True,
                keep_bf16=True,
            )

    def test_local_preconditions_fall_back_without_calling_wrapper(self):
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        bad_linears = (
            ("missing_scale", self._make_modelopt_linear(None)),
            (
                "non_scalar_scale",
                self._make_modelopt_linear(torch.tensor([0.125, 0.25])),
            ),
            (
                "modelopt_marlin",
                self._make_modelopt_linear(
                    torch.tensor(0.125), use_marlin=True
                ),
            ),
            (
                "unsupported_quant_method",
                types.SimpleNamespace(
                    quant_method=object(), input_scale=torch.tensor(0.125)
                ),
            ),
        )

        for norm_name, norm, _ in self._norm_cases():
            with self.subTest(
                norm=norm_name, condition="missing_residual"
            ), mock.patch.object(
                flashinfer_comm_fusion,
                "try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant",
            ) as wrapper:
                result = norm.forward_with_allreduce_fusion_static_fp8_quant(
                    x, None, self._make_modelopt_linear(torch.tensor(0.125))
                )
            self.assertIsNone(result)
            wrapper.assert_not_called()

            for condition, quant_linear in bad_linears:
                with self.subTest(
                    norm=norm_name, condition=condition
                ), mock.patch.object(
                    flashinfer_comm_fusion,
                    "try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant",
                ) as wrapper:
                    result = norm.forward_with_allreduce_fusion_static_fp8_quant(
                        x, residual, quant_linear
                    )
                self.assertIsNone(result)
                wrapper.assert_not_called()

    def test_wrapper_fallback_is_propagated_as_none(self):
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        input_scale = torch.tensor(0.125, dtype=torch.float32)
        quant_linear = self._make_modelopt_linear(input_scale)

        for name, norm, _ in self._norm_cases():
            with self.subTest(norm=name), mock.patch.object(
                flashinfer_comm_fusion,
                "try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant",
                return_value=(None, None, None),
            ) as wrapper:
                result = norm.forward_with_allreduce_fusion_static_fp8_quant(
                    x, residual, quant_linear
                )
            self.assertIsNone(result)
            wrapper.assert_called_once()

    def test_gemma_plain_allreduce_forwards_exact_group_selection(self):
        norm = layernorm.GemmaRMSNorm(4, eps=3e-5)
        x = torch.randn(2, 4)
        residual = torch.randn_like(x)
        expected = mock.sentinel.result

        with mock.patch.object(
            layernorm, "_forward_with_allreduce_fusion", return_value=expected
        ) as shared:
            actual = norm.forward_with_allreduce_fusion(
                x, residual, use_attn_tp_group=False
            )

        self.assertIs(actual, expected)
        shared.assert_called_once_with(
            norm,
            x,
            residual,
            None,
            norm.gemma_weight,
            use_attn_tp_group=False,
        )


if __name__ == "__main__":
    unittest.main()
