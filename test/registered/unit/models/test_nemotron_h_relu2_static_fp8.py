import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.srt.models.nemotron_h import NemotronHMLP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Projection(nn.Module):
    def __init__(self, output, quant_method=None, scale=None):
        super().__init__()
        self.output = output
        self.quant_method = quant_method
        self.input_scale = scale
        self.tp_size = 1
        self.seen = None

    def forward(self, x):
        self.seen = x
        return self.output, None


class _Activation(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.seen = None

    def forward(self, x):
        self.seen = x
        return self.output


class _Capability:
    def __init__(self, eligible):
        self.eligible = eligible

    def can_consume_relu2_static_fp8_quant(self, layer, x):
        return self.eligible


class TestNemotronHRelu2StaticFp8Dispatch(unittest.TestCase):
    @staticmethod
    def _make_mlp(eligible):
        x = torch.ones((2, 16), dtype=torch.bfloat16)
        activated = torch.full_like(x, 2)
        output = torch.full((2, 8), 3, dtype=torch.bfloat16)
        scale = nn.Parameter(torch.tensor([0.125]), requires_grad=False)

        mlp = object.__new__(NemotronHMLP)
        nn.Module.__init__(mlp)
        mlp.up_proj = _Projection(x)
        mlp.act_fn = _Activation(activated)
        mlp.down_proj = _Projection(
            output, quant_method=_Capability(eligible), scale=scale
        )
        return mlp, x, activated, output, scale

    def test_eligible_path_hands_exact_scale_and_bf16_dtype_to_down_proj(self):
        mlp, x, _, output, scale = self._make_mlp(eligible=True)
        qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)

        with mock.patch(
            "sglang.srt.models.nemotron_h._relu2_and_static_quant_fp8",
            return_value=qx,
        ) as producer:
            actual = mlp(torch.zeros_like(x))

        self.assertIs(actual, output)
        producer.assert_called_once_with(x, scale)
        self.assertIsNone(mlp.act_fn.seen)
        self.assertIs(mlp.down_proj.seen[0], qx)
        self.assertIs(mlp.down_proj.seen[1], scale)
        self.assertIs(mlp.down_proj.seen[2], torch.bfloat16)

    def test_ineligible_path_keeps_existing_relu2_input(self):
        mlp, _, activated, output, _ = self._make_mlp(eligible=False)

        actual = mlp(torch.zeros((2, 16), dtype=torch.bfloat16))

        self.assertIs(actual, output)
        self.assertIs(mlp.down_proj.seen, activated)


class TestModelOptRelu2StaticFp8Capability(unittest.TestCase):
    @staticmethod
    def _make_method():
        method = object.__new__(ModelOptFp8LinearMethod)
        method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
        method.cutlass_fp8_supported = False
        method.enable_flashinfer_bmm = False
        method.use_marlin = False
        method.use_sm120_gemv = False
        return method

    @staticmethod
    def _make_layer(scale=0.125):
        return SimpleNamespace(
            weight=nn.Parameter(
                torch.zeros((16, 16), dtype=torch.float32), requires_grad=False
            ),
            weight_scale=nn.Parameter(torch.ones(1), requires_grad=False),
            input_scale=nn.Parameter(torch.tensor([scale]), requires_grad=False),
            logical_widths=[16],
            tp_size=1,
            input_size_per_partition=16,
            orig_dtype=torch.bfloat16,
        )

    def _finish_loading(self, method, layer):
        with mock.patch(
            "sglang.srt.layers.quantization.modelopt_quant.requantize_with_max_scale",
            return_value=(torch.ones(1), torch.zeros((16, 16))),
        ):
            method.process_weights_after_loading(layer)

    def test_positive_finite_loaded_scale_enables_exact_runtime_contract(self):
        method = self._make_method()
        layer = self._make_layer()
        self._finish_loading(method, layer)
        x = mock.Mock(
            is_cuda=True,
            dtype=torch.bfloat16,
            is_contiguous=lambda: True,
            numel=lambda: 32,
            dim=lambda: 2,
            shape=(2, 16),
            device=layer.input_scale.device,
        )

        self.assertTrue(method.can_consume_relu2_static_fp8_quant(layer, x))

    def test_capability_fails_closed_before_loading_and_for_bad_runtime_shape(self):
        method = self._make_method()
        layer = self._make_layer()
        x = mock.Mock(
            is_cuda=True,
            dtype=torch.bfloat16,
            is_contiguous=lambda: True,
            numel=lambda: 30,
            dim=lambda: 2,
            shape=(2, 15),
            device=layer.input_scale.device,
        )

        self.assertFalse(method.can_consume_relu2_static_fp8_quant(layer, x))

    def test_nonfinite_loaded_scale_never_enables_capability(self):
        method = self._make_method()
        layer = self._make_layer(float("nan"))
        self._finish_loading(method, layer)
        x = mock.Mock(
            is_cuda=True,
            dtype=torch.bfloat16,
            is_contiguous=lambda: True,
            numel=lambda: 32,
            dim=lambda: 2,
            shape=(2, 16),
            device=layer.input_scale.device,
        )

        self.assertFalse(method.can_consume_relu2_static_fp8_quant(layer, x))


if __name__ == "__main__":
    unittest.main()
