"""Numerics for the FP8 dense-linear GEMM backends (--fp8-gemm-backend).

Real layer path vs a dequantized-reference matmul, in three formats: FP8
blockwise, MXFP8, and per-tensor FP8 (auto dispatch). Backend sets adapt to
the device SM, so one file covers SM90 / SM100 / SM120.
"""

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.layernorm import (
    _fp8_static_input_scale,
    _is_static_per_tensor_fp8_linear,
)
from sglang.srt.layers.quantization import fp8_utils, modelopt_quant
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

FP8_MAX = 448.0

# (M, N, K), N and K multiples of the (128, 128) weight block.
FP8_BLOCK_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
    (128, 1024, 1024),
]

# (M, N, K); K must be a multiple of 256 (flashinfer trtllm mxfp8 requirement).
MXFP8_SHAPES = [
    (64, 512, 512),
    (5, 384, 768),
]

# (M, N, K); per-tensor has no block-alignment constraints.
PER_TENSOR_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
]


def _fp8_block_backends():
    sm = get_device_sm()
    if 100 <= sm < 110:
        return ["triton", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass"]
    if sm >= 120:
        # cutlass is the SM120-only explicit backend; the trtllm / deepgemm
        # kernels do not support consumer Blackwell.
        return ["triton", "cutlass"]
    if sm == 90:
        # flashinfer_deepgemm (swapAB) is SM90-only.
        return ["triton", "deep_gemm", "flashinfer_deepgemm"]
    return []


def _mxfp8_backends():
    # MXFP8 linear is validated on SM100/103 only.
    if get_device_sm() in (100, 103):
        return [
            "auto",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "flashinfer_cutedsl",
        ]
    return []


def _quantize_fp8_blockwise(w: torch.Tensor, block: int = 128):
    """Per (block, block) tile fp8 quantization; returns checkpoint-format
    (w_fp8 [N, K], scale_inv fp32 [N/block, K/block]) and the dequant reference."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[:, None, :, None]).reshape(n, k)
    return w_fp8.reshape(n, k), scale, w_dequant


def _quantize_mxfp8(w: torch.Tensor, block: int = 32):
    """Per (1, block) group e8m0 quantization; returns checkpoint-format
    (w_fp8 [N, K], scale uint8 [N, K/block]) and the dequant reference."""
    n, k = w.shape
    groups = w.float().reshape(n, k // block, block)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(min=-127, max=127)
    scale = torch.pow(2.0, exp)
    w_fp8 = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[..., None]).reshape(n, k)
    scale_e8m0 = (exp + 127).to(torch.uint8)
    return w_fp8.reshape(n, k), scale_e8m0, w_dequant


def _make_linear(quant_config, n: int, k: int):
    return make_tp1_column_parallel_linear(
        quant_config, n, k, skip_block_quant_check=True
    )


class TestModeloptFp8PrequantizedDispatch(CustomTestCase):
    """CPU/mock contracts for ModelOpt's fused-RMSNorm FP8 hand-off."""

    @staticmethod
    def _make_method(*, use_marlin: bool = False):
        # ModelOptFp8LinearMethod.__init__ probes the active accelerator.  These
        # dispatch tests exercise no kernel, so construct the real method type
        # while making every hardware-derived choice explicit.
        method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
        method.quant_config = mock.sentinel.quant_config
        method.cutlass_fp8_supported = True
        method.enable_flashinfer_bmm = True
        method.use_marlin = use_marlin
        method.use_sm120_gemv = True
        return method

    @staticmethod
    def _make_sm120_module():
        module = types.ModuleType("sglang.kernels.ops.gemm.sm120_fp8_gemv")
        module.use_sm120_fp8_gemv = mock.Mock(return_value=True)
        module.sm120_fp8_gemv = mock.Mock()
        return module

    def test_tuple_dispatches_directly_to_fp8_linear(self):
        method = self._make_method()
        qinput = torch.empty((2, 16), dtype=torch.float8_e4m3fn)
        input_scale = torch.tensor(0.125)
        bias = torch.randn(8, dtype=torch.bfloat16)
        layer = types.SimpleNamespace(
            weight=torch.empty((16, 8), dtype=torch.float8_e4m3fn),
            weight_scale=torch.tensor(0.25),
            input_scale=torch.tensor(0.5),
            use_flashinfer_bmm=True,
            sm120_gemv_alpha=torch.ones(1),
        )
        expected = torch.empty((2, 8), dtype=torch.bfloat16)
        sm120_module = self._make_sm120_module()

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang.kernels.ops.gemm.sm120_fp8_gemv": sm120_module},
            ),
            mock.patch.object(
                modelopt_quant, "apply_fp8_linear", return_value=expected
            ) as fp8_linear,
            mock.patch.object(
                modelopt_quant, "apply_fp8_linear_bmm_flashinfer"
            ) as flashinfer_bmm,
            mock.patch.object(fp8_utils, "static_quant_fp8") as requantize,
        ):
            actual = method.apply(
                layer, (qinput, input_scale, torch.bfloat16), bias=bias
            )

        self.assertIs(actual, expected)
        fp8_linear.assert_called_once_with(
            input=qinput,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=input_scale,
            bias=bias,
            cutlass_fp8_supported=True,
            pre_quant_output_dtype=torch.bfloat16,
        )
        flashinfer_bmm.assert_not_called()
        requantize.assert_not_called()
        sm120_module.use_sm120_fp8_gemv.assert_not_called()
        sm120_module.sm120_fp8_gemv.assert_not_called()

    def test_marlin_rejects_tuple_before_accessing_rearranged_weights(self):
        method = self._make_method(use_marlin=True)
        qinput = torch.empty((2, 16), dtype=torch.float8_e4m3fn)
        input_scale = torch.tensor(0.125)

        class PoisonedMarlinLayer:
            @property
            def weight(self):
                raise AssertionError("Marlin weight must not be read")

        marlin_op = mock.Mock()
        fake_sglang_ops = types.SimpleNamespace(apply_fp8_marlin_linear=marlin_op)
        with (
            mock.patch.object(torch.ops, "sglang", fake_sglang_ops),
            self.assertRaises(ValueError),
        ):
            method.apply(
                PoisonedMarlinLayer(),
                (qinput, input_scale, torch.bfloat16),
            )
        marlin_op.assert_not_called()

    def test_modelopt_static_scale_construction_and_forward_eligibility(self):
        method = self._make_method()
        pre_load_scale = torch.ones(3, dtype=torch.float32)
        layer = types.SimpleNamespace(
            quant_method=method,
            input_scale=pre_load_scale,
        )

        self.assertTrue(_is_static_per_tensor_fp8_linear(method, layer))
        self.assertIsNone(_fp8_static_input_scale(layer))

        finalized_scale = torch.tensor(0.125, dtype=torch.float32)
        layer.input_scale = finalized_scale
        self.assertTrue(_is_static_per_tensor_fp8_linear(method, layer))
        self.assertIs(_fp8_static_input_scale(layer), finalized_scale)

    def test_modelopt_marlin_or_missing_scale_is_ineligible(self):
        marlin_method = self._make_method(use_marlin=True)
        marlin_layer = types.SimpleNamespace(
            quant_method=marlin_method,
            input_scale=torch.tensor(0.125),
        )
        self.assertFalse(
            _is_static_per_tensor_fp8_linear(marlin_method, marlin_layer)
        )
        self.assertIsNone(_fp8_static_input_scale(marlin_layer))

        method = self._make_method()
        missing_scale_layer = types.SimpleNamespace(
            quant_method=method,
            input_scale=None,
        )
        self.assertFalse(
            _is_static_per_tensor_fp8_linear(method, missing_scale_layer)
        )
        self.assertIsNone(_fp8_static_input_scale(missing_scale_layer))


class _LinearBackendCheck(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _check_backend(self, backend: str, allowed, shapes, build_layer):
        if backend not in allowed:
            self.skipTest(f"{backend} not in SM{get_device_sm()} backend set")
        torch.manual_seed(7)
        for m, n, k in shapes:
            with self.subTest(backend=backend, shape=(m, n, k)):
                with mock.patch.object(
                    fp8_utils,
                    "FP8_GEMM_RUNNER_BACKEND",
                    Fp8GemmRunnerBackend(backend),
                ):
                    layer, w_dequant = build_layer(n, k)
                    layer.quant_method.process_weights_after_loading(layer)

                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                    out, _ = layer(x)

                    ref = x.float() @ w_dequant.T
                    # atol covers single-element UE8M0 scale-rounding outliers
                    # (deep_gemm); a wrong kernel/layout fails by orders more.
                    assert_output_close(self, out, ref, rtol=5e-2, atol=1e-1)


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestFp8BlockwiseLinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_inv, w_dequant = _quantize_fp8_blockwise(w)
        load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_inv)
        return layer, w_dequant

    def _run(self, backend: str):
        self._check_backend(
            backend, _fp8_block_backends(), FP8_BLOCK_SHAPES, self._build_layer
        )

    def test_triton(self):
        self._run("triton")

    def test_deep_gemm(self):
        self._run("deep_gemm")

    def test_flashinfer_trtllm(self):
        self._run("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_deepgemm(self):
        self._run("flashinfer_deepgemm")

    def test_cutlass(self):
        self._run("cutlass")


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestMxfp8LinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            use_mxfp8=True,
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_e8m0, w_dequant = _quantize_mxfp8(w)
        load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_e8m0)
        return layer, w_dequant

    def _run(self, backend: str):
        self._check_backend(backend, _mxfp8_backends(), MXFP8_SHAPES, self._build_layer)

    def test_flashinfer_trtllm(self):
        self._run("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_cutedsl(self):
        self._run("flashinfer_cutedsl")

    def test_auto(self):
        if "auto" not in _mxfp8_backends():
            self.skipTest(f"auto not in SM{get_device_sm()} MXFP8 backend set")
        with mock.patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.AUTO,
        ):
            self.assertEqual(
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
            )
        self._run("auto")

    @unittest.skipUnless(get_device_sm() >= 100, "Requires Blackwell FlashInfer")
    def test_auto_falls_back_when_cutedsl_is_unsupported(self):
        with (
            mock.patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                Fp8GemmRunnerBackend.AUTO,
            ),
            mock.patch.object(fp8_utils, "get_device_sm", return_value=107),
            mock.patch.object(
                fp8_utils._raw_flashinfer_mm_mxfp8,
                "is_backend_supported",
                return_value=False,
            ) as is_backend_supported,
        ):
            self.assertEqual(
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS,
            )
            is_backend_supported.assert_called_once_with("cute-dsl", 107)


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestModeloptFp8PerTensorLinear(_LinearBackendCheck):
    """Per-tensor FP8 (ModelOptFp8LinearMethod, static scales) on the auto
    dispatch path -- the checkpoint style of nvidia/*-FP8 models."""

    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True, packed_modules_mapping={}
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        scale = (w.float().abs().max() / FP8_MAX).clamp(min=1e-12)
        w_fp8 = (w.float() / scale).to(torch.float8_e4m3fn)
        # 0-dim scales exercise weight_loader_v2's scalar reshape branch.
        load_linear_weights(
            layer,
            weight=w_fp8,
            weight_scale=scale,
            input_scale=torch.tensor(1.0 / FP8_MAX, device="cuda"),
        )
        w_dequant = w_fp8.float() * scale
        return layer, w_dequant

    def test_auto(self):
        self._check_backend("auto", ["auto"], PER_TENSOR_SHAPES, self._build_layer)

    def test_prequantized_tuple_matches_bf16_and_skips_requantization(self):
        torch.manual_seed(11)
        layer, _ = self._build_layer(n=256, k=512)
        layer.quant_method.process_weights_after_loading(layer)
        method = layer.quant_method

        # Establish the ordinary static-FP8 result without either optional
        # specialized path, then hand the exact same quantized activation to
        # ModelOpt as a fused producer would.
        method.use_sm120_gemv = False
        layer.use_flashinfer_bmm = False
        x = torch.randn((5, 512), device="cuda", dtype=torch.bfloat16) / 10
        reference = method.apply(layer, x)
        qinput, _ = fp8_utils.static_quant_fp8(
            x, layer.input_scale, repeat_scale=False
        )

        method.use_sm120_gemv = True
        layer.use_flashinfer_bmm = True
        layer.sm120_gemv_alpha = torch.ones(1, device="cuda")
        sm120_module = TestModeloptFp8PrequantizedDispatch._make_sm120_module()
        with (
            mock.patch.dict(
                sys.modules,
                {"sglang.kernels.ops.gemm.sm120_fp8_gemv": sm120_module},
            ),
            mock.patch.object(
                fp8_utils,
                "static_quant_fp8",
                wraps=fp8_utils.static_quant_fp8,
            ) as requantize,
            mock.patch.object(
                modelopt_quant, "apply_fp8_linear_bmm_flashinfer"
            ) as flashinfer_bmm,
        ):
            output = method.apply(
                layer,
                (qinput, layer.input_scale, torch.bfloat16),
            )

        requantize.assert_not_called()
        flashinfer_bmm.assert_not_called()
        sm120_module.use_sm120_fp8_gemv.assert_not_called()
        sm120_module.sm120_fp8_gemv.assert_not_called()
        self.assertEqual(output.shape, reference.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        assert_output_close(self, output, reference, rtol=5e-2, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
