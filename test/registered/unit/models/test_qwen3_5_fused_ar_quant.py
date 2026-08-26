from unittest.mock import patch

import torch

from sglang.srt.models import qwen3_5
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class _Linear:
    quant_method = object()


class _ModelOptFp8LinearMethod:
    use_marlin = False


class _RecordingLinear:
    def __init__(self, quant_method):
        self.quant_method = quant_method
        self.input = None

    def __call__(self, hidden_states):
        self.input = hidden_states
        return hidden_states, None


def test_cuda_modelopt_fp8_tuple_reuses_existing_handoff():
    bf16 = torch.randn(2, 8)
    fp8 = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    linear = _Linear()
    linear.quant_method = _ModelOptFp8LinearMethod()

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch(
            "sglang.srt.layers.quantization.modelopt_quant.ModelOptFp8LinearMethod",
            _ModelOptFp8LinearMethod,
        ),
    ):
        assert qwen3_5._linear_accepts_fp8_tuple(linear)
        dual_result = qwen3_5._select_fused_ar_input_for_linear(
            (bf16, fp8, scale), linear
        )
        quant_result = qwen3_5._select_fused_ar_input_for_linear((fp8, scale), linear)
        assert dual_result[0] is fp8 and dual_result[1] is scale
        assert quant_result[0] is fp8 and quant_result[1] is scale


def test_gdn_routes_dual_output_to_quantized_and_bf16_projections():
    bf16 = torch.randn(2, 8)
    fp8 = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    gdn = type("GDN", (), {})()
    gdn.in_proj_qkvz = _RecordingLinear(_ModelOptFp8LinearMethod())
    gdn.in_proj_ba = _RecordingLinear(object())
    gdn.alt_stream = None

    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch.object(qwen3_5, "check_cuda_graph_backend", return_value=False),
        patch(
            "sglang.srt.layers.quantization.modelopt_quant.ModelOptFp8LinearMethod",
            _ModelOptFp8LinearMethod,
        ),
    ):
        qkvz, ba = qwen3_5.Qwen3_5GatedDeltaNet._forward_input_proj_fused_quant(
            gdn, (bf16, fp8, scale)
        )

    assert gdn.in_proj_qkvz.input[0] is fp8
    assert gdn.in_proj_qkvz.input[1] is scale
    assert gdn.in_proj_ba.input is bf16
    assert qkvz is gdn.in_proj_qkvz.input
    assert ba is bf16


def test_cuda_non_modelopt_fp8_method_does_not_accept_tuple():
    linear = _Linear()
    linear.quant_method = type("Fp8LinearMethod", (), {"block_quant": False})()
    with (
        patch.object(qwen3_5, "_use_aiter", False),
        patch(
            "sglang.srt.layers.quantization.modelopt_quant.ModelOptFp8LinearMethod",
            _ModelOptFp8LinearMethod,
        ),
    ):
        assert not qwen3_5._linear_accepts_fp8_tuple(linear)


def test_amd_tuple_predicate_remains_per_group_only():
    linear = _Linear()
    with patch.object(qwen3_5, "_use_aiter", True):
        assert not qwen3_5._linear_accepts_fp8_tuple(linear)

    linear.quant_method = type(
        "Fp8LinearMethod", (), {"block_quant": True, "use_mxfp8": False}
    )()
    with patch.object(qwen3_5, "_use_aiter", True):
        assert qwen3_5._linear_accepts_fp8_tuple(linear)
