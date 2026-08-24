"""CPU contracts for Qwen3.5 fused all-reduce quantized hand-off.

The fused kernels are intentionally not exercised here.  These tests cover the
construction-time consumer decision and the tuple routing immediately around
the Qwen3.5 projections, where confusing AMD per-group tuples with CUDA static
per-tensor tuples would otherwise silently feed the wrong representation to a
linear layer.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import flashinfer_comm_fusion as flashinfer_ar
from sglang.srt.layers import layernorm as layernorm_mod
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.srt.models import qwen3_5 as qwen35
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
    )
except ImportError:
    CompressedTensorsLinearMethod = None
    CompressedTensorsW8A8Fp8 = None


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Linear:
    def __init__(self, quant_method=None, input_scale=None, output=None):
        self.quant_method = quant_method
        self.input_scale = input_scale
        self.output = output
        self.inputs = []

    def __call__(self, hidden_states):
        self.inputs.append(hidden_states)
        return self.output, None


def _native_fp8_method(*, block=False, mxfp8=False, marlin=False):
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.block_quant = block
    method.use_mxfp8 = mxfp8
    method.use_marlin = marlin
    return method


def _modelopt_fp8_method(*, marlin=False):
    method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
    method.use_marlin = marlin
    return method


def _static_linear(*, modelopt=False, scale=None, output=None):
    if scale is None:
        scale = torch.ones(1, dtype=torch.float32)
    method = _modelopt_fp8_method() if modelopt else _native_fp8_method()
    return _Linear(method, scale, output)


def _group_linear(*, block=False, mxfp8=False, output=None):
    return _Linear(
        _native_fp8_method(block=block, mxfp8=mxfp8),
        torch.ones(2, dtype=torch.float32),
        output,
    )


class TestQwen35FusedArConsumerPredicates(unittest.TestCase):
    def test_group_predicate_preserves_block_and_mxfp8_contract(self):
        block = _group_linear(block=True)
        mxfp8 = _group_linear(mxfp8=True)
        static = _static_linear()

        self.assertTrue(qwen35._linear_accepts_group_fp8_tuple(block))
        self.assertTrue(qwen35._linear_accepts_group_fp8_tuple(mxfp8))
        self.assertFalse(qwen35._linear_accepts_group_fp8_tuple(static))

        # Keep the compatibility predicate narrow: widening it to static FP8
        # makes a CUDA 3-tuple indistinguishable from an AMD GDN 3-tuple.
        self.assertTrue(qwen35._linear_accepts_fp8_tuple(block))
        self.assertTrue(qwen35._linear_accepts_fp8_tuple(mxfp8))
        self.assertFalse(qwen35._linear_accepts_fp8_tuple(static))

    def test_static_predicate_delegates_to_construction_time_capability(self):
        linear = _static_linear(modelopt=True, scale=torch.ones(8))
        with (
            mock.patch.object(
                qwen35,
                "_is_static_per_tensor_fp8_linear",
                return_value=True,
            ) as capability,
            mock.patch.object(
                layernorm_mod,
                "_fp8_static_input_scale",
                side_effect=AssertionError("forward-time scale readiness queried"),
            ),
        ):
            self.assertTrue(qwen35._linear_accepts_static_fp8_tuple(linear))

        capability.assert_called_once_with(linear.quant_method, linear)

    def test_static_predicate_classifies_supported_quant_methods(self):
        cases = [
            (
                "modelopt vector pre-load scale",
                _static_linear(modelopt=True, scale=torch.ones(8)),
                True,
            ),
            ("modelopt scalar scale", _static_linear(modelopt=True), True),
            (
                "modelopt marlin",
                _Linear(_modelopt_fp8_method(marlin=True), torch.ones(1)),
                False,
            ),
            ("native static", _static_linear(), True),
            (
                "native dynamic",
                _Linear(_native_fp8_method(), input_scale=None),
                False,
            ),
            (
                "native marlin",
                _Linear(_native_fp8_method(marlin=True), torch.ones(1)),
                False,
            ),
            ("native block", _group_linear(block=True), False),
            ("native mxfp8", _group_linear(mxfp8=True), False),
        ]
        for name, linear, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    qwen35._linear_accepts_static_fp8_tuple(linear), expected
                )

    @unittest.skipUnless(
        CompressedTensorsLinearMethod is not None,
        "compressed-tensors is not importable",
    )
    def test_static_predicate_accepts_compressed_tensors_static_fp8(self):
        method = CompressedTensorsLinearMethod.__new__(CompressedTensorsLinearMethod)
        scheme = CompressedTensorsW8A8Fp8.__new__(CompressedTensorsW8A8Fp8)
        scheme.is_static_input_scheme = True
        linear = _Linear(method, torch.ones(1))
        linear.scheme = scheme

        self.assertTrue(qwen35._linear_accepts_static_fp8_tuple(linear))

        scheme.is_static_input_scheme = False
        self.assertFalse(qwen35._linear_accepts_static_fp8_tuple(linear))


class TestQwen35FusedArTupleSelector(unittest.TestCase):
    def setUp(self):
        self.bf16 = torch.randn(2, 4, dtype=torch.bfloat16)
        self.fp8 = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        self.scalar_scale = torch.tensor([0.125], dtype=torch.float32)
        self.group_scale = torch.ones(2, 1, dtype=torch.float32)
        self.static = _static_linear()
        self.group = _group_linear(block=True)
        self.ordinary = _Linear()

    def test_cuda_quant_only_tuple_is_preserved_for_static_consumer(self):
        fused = (self.fp8, self.scalar_scale, torch.bfloat16)

        selected = qwen35._select_fused_ar_input_for_linear(fused, self.static)

        self.assertIs(selected, fused)

    def test_amd_quant_only_tuple_is_preserved_for_group_consumer(self):
        fused = (self.fp8, self.group_scale)

        selected = qwen35._select_fused_ar_input_for_linear(fused, self.group)

        self.assertIs(selected, fused)

    def test_cuda_gdn_tuple_selects_static_qkv_and_bf16_ba(self):
        fused = (self.bf16, self.fp8, self.scalar_scale, torch.bfloat16)

        selected_qkv = qwen35._select_fused_ar_input_for_linear(fused, self.static)
        selected_ba = qwen35._select_fused_ar_input_for_linear(fused, self.ordinary)

        self.assertEqual(len(selected_qkv), 3)
        self.assertIs(selected_qkv[0], self.fp8)
        self.assertIs(selected_qkv[1], self.scalar_scale)
        self.assertIs(selected_qkv[2], torch.bfloat16)
        self.assertIs(selected_ba, self.bf16)

    def test_amd_gdn_tuple_selects_group_qkv_and_bf16_ba(self):
        fused = (self.bf16, self.fp8, self.group_scale)

        selected_qkv = qwen35._select_fused_ar_input_for_linear(fused, self.group)
        selected_ba = qwen35._select_fused_ar_input_for_linear(fused, self.ordinary)

        self.assertEqual(len(selected_qkv), 2)
        self.assertIs(selected_qkv[0], self.fp8)
        self.assertIs(selected_qkv[1], self.group_scale)
        self.assertIs(selected_ba, self.bf16)

    def test_mismatched_or_unsupported_consumers_raise_instead_of_guessing(self):
        cases = [
            (
                "CUDA quant-only to group",
                (self.fp8, self.scalar_scale, torch.bfloat16),
                self.group,
            ),
            (
                "CUDA quant-only to ordinary",
                (self.fp8, self.scalar_scale, torch.bfloat16),
                self.ordinary,
            ),
            ("AMD quant-only to static", (self.fp8, self.group_scale), self.static),
            ("AMD quant-only to ordinary", (self.fp8, self.group_scale), self.ordinary),
            (
                "CUDA GDN to group",
                (self.bf16, self.fp8, self.scalar_scale, torch.bfloat16),
                self.group,
            ),
            ("AMD GDN to static", (self.bf16, self.fp8, self.group_scale), self.static),
        ]
        for name, fused, linear in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    TypeError, "cannot consume fused AR quant tuple"
                ):
                    qwen35._select_fused_ar_input_for_linear(fused, linear)


class TestQwen35GdnFusedArRouting(unittest.TestCase):
    def _run_input_projection(self, fused, qkv_linear):
        qkv_linear.output = torch.randn(2, 6)
        ba = _Linear(output=torch.randn(2, 2))
        # Bypass the heavyweight constructor while retaining class method
        # lookup for the common tuple projection helper.
        gdn = qwen35.Qwen3_5GatedDeltaNet.__new__(qwen35.Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(gdn)
        gdn.in_proj_qkvz = qkv_linear
        gdn.in_proj_ba = ba
        gdn.alt_stream = None
        with mock.patch.multiple(
            qwen35,
            _use_aiter=False,
            _is_cpu=False,
            _is_npu=False,
            _gdn_use_alt_stream=False,
        ), mock.patch.object(qwen35, "check_cuda_graph_backend", return_value=False):
            projected = qwen35.Qwen3_5GatedDeltaNet._forward_input_proj(gdn, fused)
        return projected, qkv_linear, ba

    def test_cuda_static_gdn_routes_exact_tuple_to_qkv_and_bf16_to_ba(self):
        bf16 = torch.randn(2, 4, dtype=torch.bfloat16)
        fp8 = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.125])
        fused = (bf16, fp8, scale, torch.bfloat16)

        projected, qkv, ba = self._run_input_projection(fused, _static_linear())

        self.assertIs(qkv.inputs[0][0], fp8)
        self.assertIs(qkv.inputs[0][1], scale)
        self.assertIs(qkv.inputs[0][2], torch.bfloat16)
        self.assertIs(ba.inputs[0], bf16)
        self.assertIs(projected[0], qkv.output)
        self.assertIs(projected[1], ba.output)

    def test_amd_group_gdn_keeps_existing_projection_routing(self):
        bf16 = torch.randn(2, 4, dtype=torch.bfloat16)
        fp8 = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        scale = torch.ones(2, 1)
        fused = (bf16, fp8, scale)

        _, qkv, ba = self._run_input_projection(fused, _group_linear(block=True))

        self.assertIs(qkv.inputs[0][0], fp8)
        self.assertIs(qkv.inputs[0][1], scale)
        self.assertIs(ba.inputs[0], bf16)


class TestQwen35FullAttentionFusedArRouting(unittest.TestCase):
    def setUp(self):
        self.positions = torch.arange(2)
        self.fp8 = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        self.scale = torch.tensor([0.125])
        self.fused = (self.fp8, self.scale, torch.bfloat16)
        self.qkv_out = torch.randn(2, 12)
        self.qkv = _static_linear(output=self.qkv_out)
        self.attn = SimpleNamespace(
            qkv_proj=self.qkv,
            attn_output_gate=True,
            q_size=4,
            kv_size=2,
            num_heads=1,
            num_kv_heads=1,
            head_dim=4,
            q_norm=SimpleNamespace(
                weight=SimpleNamespace(data=torch.ones(4)), variance_epsilon=1e-6
            ),
            k_norm=SimpleNamespace(weight=SimpleNamespace(data=torch.ones(2))),
            rotary_emb=SimpleNamespace(
                cos_sin_cache=torch.empty(0),
                rotary_dim=4,
                __call__=lambda positions, q, k: (q, k),
            ),
            _apply_qk_norm=lambda q, k: (q, k),
        )

    def test_native_prepare_selects_static_tuple_before_qkv(self):
        self.attn.rotary_emb = lambda positions, q, k: (q, k)
        with mock.patch.object(qwen35, "_use_aiter", False):
            qwen35.Qwen3_5AttentionDecoderLayer.forward_prepare_native(
                self.attn, self.positions, self.fused
            )

        self.assertIs(self.qkv.inputs[0], self.fused)

    def test_fused_gate_prepare_selects_static_tuple_before_qkv(self):
        self.attn.rotary_emb = lambda positions, q, k: (q, k)
        q = torch.randn(2, 4)
        k = torch.randn(2, 2)
        gate = torch.randn(2, 4)
        with (
            mock.patch.object(qwen35, "_use_aiter", False),
            mock.patch.object(
                qwen35,
                "fused_qk_gemma_rmsnorm_with_gate",
                return_value=(q, k, gate),
            ),
        ):
            qwen35.Qwen3_5AttentionDecoderLayer.forward_prepare_fused_gate(
                self.attn, self.positions, self.fused
            )

        self.assertIs(self.qkv.inputs[0], self.fused)

    def test_cuda_fused_prepare_uses_qkv_tensor_for_token_count(self):
        q = torch.randn(2, 4)
        k = torch.randn(2, 2)
        gate = torch.randn(2, 4)
        with mock.patch.object(
            qwen35,
            "fused_qk_gemma_rmsnorm_rope_gate",
            return_value=(q, k, gate),
            create=True,
        ):
            prepared = qwen35.Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused(
                self.attn, self.positions, self.fused
            )

        self.assertIs(self.qkv.inputs[0], self.fused)
        self.assertEqual(prepared[0].shape[0], q.shape[0])
        self.assertEqual(prepared[1].shape[0], q.shape[0])
        self.assertEqual(prepared[3].shape[0], q.shape[0])


class TestQwen35FusedArRegistration(unittest.TestCase):
    EXPECTED_KEYS = {
        "enable_fused_ar_quant",
        "fused_ar_quant_keep_bf16",
        "fused_ar_quant_linear",
    }

    def _decision(
        self,
        linear,
        *,
        keep_bf16,
        cuda,
        aiter,
        enable_aiter_allreduce_fusion=True,
    ):
        clear_gate_cache = getattr(
            qwen35._enable_qwen35_fused_ar_quant, "cache_clear", lambda: None
        )
        clear_gate_cache()
        if cuda:
            get_exec_patch = mock.patch.object(
                qwen35,
                "get_exec",
                side_effect=AssertionError("CUDA registration queried runtime gate"),
            )
        else:
            get_exec_patch = mock.patch.object(
                qwen35,
                "get_exec",
                return_value=SimpleNamespace(
                    comm=SimpleNamespace(
                        enable_aiter_allreduce_fusion=(
                            enable_aiter_allreduce_fusion
                        )
                    )
                ),
            )
        try:
            with (
                mock.patch.multiple(
                    qwen35,
                    _is_cuda=cuda,
                    _is_hip=aiter,
                    _use_aiter=aiter,
                ),
                get_exec_patch,
                mock.patch.object(
                    flashinfer_ar,
                    "_is_allreduce_quant_capability_available_for_group",
                    side_effect=AssertionError("FlashInfer capability queried"),
                ),
                mock.patch.object(
                    flashinfer_ar,
                    "ensure_workspace_initialized",
                    side_effect=AssertionError("FlashInfer workspace queried"),
                ),
                mock.patch.object(
                    flashinfer_ar,
                    "_synchronize_allreduce_quant_capability",
                    side_effect=AssertionError("FlashInfer collective queried"),
                ),
            ):
                return qwen35._fused_ar_quant_communicator_kwargs(
                    linear, keep_bf16=keep_bf16
                )
        finally:
            clear_gate_cache()

    def test_full_attention_registers_exact_cuda_static_consumer(self):
        qkv_proj = _static_linear()

        decision = self._decision(
            qkv_proj, keep_bf16=False, cuda=True, aiter=False
        )

        self.assertEqual(set(decision), self.EXPECTED_KEYS)
        self.assertTrue(decision["enable_fused_ar_quant"])
        self.assertFalse(decision["fused_ar_quant_keep_bf16"])
        self.assertIs(decision["fused_ar_quant_linear"], qkv_proj)

    def test_gdn_registers_exact_amd_group_consumer_and_keeps_bf16(self):
        in_proj_qkvz = _group_linear(block=True)

        decision = self._decision(
            in_proj_qkvz, keep_bf16=True, cuda=False, aiter=True
        )

        self.assertEqual(set(decision), self.EXPECTED_KEYS)
        self.assertTrue(decision["enable_fused_ar_quant"])
        self.assertTrue(decision["fused_ar_quant_keep_bf16"])
        self.assertIs(decision["fused_ar_quant_linear"], in_proj_qkvz)

    def test_amd_group_registration_preserves_aiter_runtime_gate(self):
        in_proj_qkvz = _group_linear(block=True)

        decision = self._decision(
            in_proj_qkvz,
            keep_bf16=True,
            cuda=False,
            aiter=True,
            enable_aiter_allreduce_fusion=False,
        )

        self.assertEqual(set(decision), self.EXPECTED_KEYS)
        self.assertFalse(decision["enable_fused_ar_quant"])
        self.assertTrue(decision["fused_ar_quant_keep_bf16"])
        self.assertIsNone(decision["fused_ar_quant_linear"])

    def test_modelopt_vector_scale_is_eligible_before_weights_are_loaded(self):
        qkv_proj = _static_linear(modelopt=True, scale=torch.ones(8))

        decision = self._decision(
            qkv_proj, keep_bf16=False, cuda=True, aiter=False
        )

        self.assertTrue(decision["enable_fused_ar_quant"])
        self.assertIs(decision["fused_ar_quant_linear"], qkv_proj)

    def test_backend_neutral_opt_out_disables_cuda_and_amd(self):
        cases = [
            ("CUDA static", _static_linear(), True, False),
            ("AMD group", _group_linear(block=True), False, True),
        ]
        with mock.patch.dict(
            os.environ, {"SGLANG_DISABLE_FUSED_AR_QUANT": "1"}, clear=False
        ):
            for name, linear, cuda, aiter in cases:
                with self.subTest(name=name):
                    decision = self._decision(
                        linear, keep_bf16=(not cuda), cuda=cuda, aiter=aiter
                    )
                    self.assertFalse(decision["enable_fused_ar_quant"])
                    self.assertIsNone(decision["fused_ar_quant_linear"])

    def test_unsupported_consumer_is_disabled_without_registration(self):
        decision = self._decision(
            _Linear(), keep_bf16=False, cuda=True, aiter=False
        )

        self.assertEqual(set(decision), self.EXPECTED_KEYS)
        self.assertFalse(decision["enable_fused_ar_quant"])
        self.assertFalse(decision["fused_ar_quant_keep_bf16"])
        self.assertIsNone(decision["fused_ar_quant_linear"])


if __name__ == "__main__":
    unittest.main()
