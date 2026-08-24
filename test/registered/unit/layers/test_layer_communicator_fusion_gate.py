import inspect
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class _RecordingNorm(torch.nn.Module):
    def __init__(
        self,
        *,
        static_result=None,
        ordinary_result=None,
        per_group_result=None,
    ):
        super().__init__()
        self.static_result = static_result
        self.ordinary_result = ordinary_result
        self.per_group_result = per_group_result
        self.static_calls = []
        self.ordinary_calls = []
        self.per_group_calls = []
        self.events = []

    def forward_with_allreduce_fusion_static_fp8_quant(
        self,
        x,
        residual=None,
        quant_linear=None,
        use_attn_tp_group=True,
        keep_bf16=False,
    ):
        self.events.append("static")
        self.static_calls.append(
            {
                "x": x,
                "residual": residual,
                "quant_linear": quant_linear,
                "use_attn_tp_group": use_attn_tp_group,
                "keep_bf16": keep_bf16,
            }
        )
        return self.static_result

    def forward_with_allreduce_fusion(
        self,
        x,
        residual=None,
        post_residual_addition=None,
        use_attn_tp_group=True,
    ):
        self.events.append("ordinary")
        self.ordinary_calls.append(
            {
                "x": x,
                "residual": residual,
                "post_residual_addition": post_residual_addition,
                "use_attn_tp_group": use_attn_tp_group,
            }
        )
        return self.ordinary_result

    def forward_with_allreduce_fusion_quant_per_group(
        self,
        x,
        residual=None,
        group_size=128,
        use_attn_tp_group=True,
        keep_bf16=False,
    ):
        self.events.append("per_group")
        self.per_group_calls.append(
            {
                "x": x,
                "residual": residual,
                "group_size": group_size,
                "use_attn_tp_group": use_attn_tp_group,
                "keep_bf16": keep_bf16,
            }
        )
        return self.per_group_result


def _identity_communicate(*, hidden_states, **_kwargs):
    return hidden_states


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Hybrid EP+TP must not fuse the post-experts all-reduce away.

    The fused residual+LN reduces over a single group, but with moe_ep_size > 1
    and moe_tp_size > 1 the post-experts reduction spans two disjoint groups
    (_MOE_EP then _MOE_TP) and should_skip_post_experts_all_reduce() drops both
    once fusion is published. The result is activations reduced over only half
    the peers -- wrong output, no crash. Observed as garbage completions on
    Qwen3-30B-A3B with --tp-size 4 --ep-size 2.
    """

    def _should_fuse(self, *, moe_ep_size, moe_tp_size):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(), forward_batch
            )

    def test_hybrid_ep_tp_does_not_fuse(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))


class TestPrepareAttnStaticFp8Fusion(CustomTestCase):
    def _make_communicator(
        self,
        norm,
        *,
        consumer=None,
        enable_fused_ar_quant=True,
        keep_bf16=False,
    ):
        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator.input_layernorm = norm
        communicator.enable_fused_ar_quant = enable_fused_ar_quant
        communicator.fused_ar_quant_keep_bf16 = keep_bf16
        communicator.fused_ar_quant_linear = consumer
        communicator._communicate_simple_fn = _identity_communicate
        communicator._context = types.SimpleNamespace()
        communicator.qkv_latent_func = None
        return communicator

    def _prepare_attn(self, communicator, hidden_states, residual, *, use_aiter):
        with (
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(comm, "apply_aiter_all_reduce_fusion", return_value=False),
            patch.object(comm, "_use_aiter", use_aiter),
        ):
            return LayerCommunicator.prepare_attn(
                communicator,
                hidden_states,
                residual,
                types.SimpleNamespace(),
            )

    def _inputs(self):
        hidden_states = torch.tensor([[1.0, 2.0]])
        hidden_states._sglang_needs_allreduce_fusion = True
        residual = torch.tensor([[3.0, 4.0]])
        return hidden_states, residual

    def _assert_call(self, calls, **expected_values):
        self.assertEqual(len(calls), 1)
        actual_values = calls[0]
        for name, expected_value in expected_values.items():
            self.assertIn(name, actual_values)
            if name in ("x", "residual", "quant_linear"):
                self.assertIs(actual_values[name], expected_value)
            else:
                self.assertEqual(actual_values[name], expected_value)

    def test_constructor_preserves_exact_fused_ar_quant_consumer(self):
        consumer = torch.nn.Identity()
        context = types.SimpleNamespace()
        parameters = inspect.signature(LayerCommunicator).parameters
        self.assertIn(
            "fused_ar_quant_linear",
            parameters,
        )
        self.assertIs(parameters["fused_ar_quant_linear"].default, None)

        with (
            patch.object(comm.CommunicateContext, "init_new", return_value=context),
            patch.object(LayerCommunicator, "_post_init_communicate"),
            patch.object(
                comm,
                "get_spec",
                return_value=types.SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(
                comm.SpeculativeAlgorithm,
                "from_string",
                return_value=types.SimpleNamespace(),
            ),
        ):
            communicator = LayerCommunicator(
                layer_scatter_modes=types.SimpleNamespace(),
                input_layernorm=torch.nn.Identity(),
                post_attention_layernorm=torch.nn.Identity(),
                fused_ar_quant_linear=consumer,
            )

        self.assertIs(communicator.fused_ar_quant_linear, consumer)

    def test_cuda_static_fp8_fusion_receives_consumer_and_returns_its_tuple(self):
        hidden_states, residual = self._inputs()
        consumer = torch.nn.Identity()
        static_hidden = (
            torch.tensor([[5.0, 6.0]]),
            torch.tensor([[7]], dtype=torch.uint8),
            torch.tensor(0.25),
            hidden_states.dtype,
        )
        static_residual = torch.tensor([[6.0, 7.0]])
        ordinary_result = (
            torch.tensor([[8.0, 9.0]]),
            torch.tensor([[10.0, 11.0]]),
        )
        norm = _RecordingNorm(
            static_result=(static_hidden, static_residual),
            ordinary_result=ordinary_result,
        )
        communicator = self._make_communicator(
            norm,
            consumer=consumer,
            keep_bf16=True,
        )

        result = self._prepare_attn(
            communicator,
            hidden_states,
            residual,
            use_aiter=False,
        )

        self._assert_call(
            norm.static_calls,
            x=hidden_states,
            residual=residual,
            quant_linear=consumer,
            use_attn_tp_group=False,
            keep_bf16=True,
        )
        self.assertEqual(norm.events, ["static"])
        self.assertIs(result[0], static_hidden)
        self.assertIs(result[1], static_residual)
        self.assertEqual(norm.ordinary_calls, [])
        self.assertEqual(norm.per_group_calls, [])

    def test_cuda_static_fp8_none_falls_back_to_ordinary_fusion_once(self):
        hidden_states, residual = self._inputs()
        consumer = torch.nn.Identity()
        ordinary_hidden = torch.tensor([[8.0, 9.0]])
        ordinary_residual = torch.tensor([[10.0, 11.0]])
        norm = _RecordingNorm(
            static_result=None,
            ordinary_result=(ordinary_hidden, ordinary_residual),
        )
        communicator = self._make_communicator(norm, consumer=consumer)

        result = self._prepare_attn(
            communicator,
            hidden_states,
            residual,
            use_aiter=False,
        )

        self.assertIs(result[0], ordinary_hidden)
        self.assertIs(result[1], ordinary_residual)
        self._assert_call(
            norm.static_calls,
            x=hidden_states,
            residual=residual,
            quant_linear=consumer,
            use_attn_tp_group=False,
            keep_bf16=False,
        )
        self._assert_call(
            norm.ordinary_calls,
            x=hidden_states,
            residual=residual,
            use_attn_tp_group=False,
        )
        self.assertEqual(norm.events, ["static", "ordinary"])
        self.assertEqual(norm.per_group_calls, [])

    def test_cuda_fusion_without_consumer_skips_static_fp8_helper(self):
        hidden_states, residual = self._inputs()
        ordinary_result = (
            torch.tensor([[12.0, 13.0]]),
            torch.tensor([[14.0, 15.0]]),
        )
        norm = _RecordingNorm(ordinary_result=ordinary_result)
        communicator = self._make_communicator(norm, consumer=None)

        result = self._prepare_attn(
            communicator,
            hidden_states,
            residual,
            use_aiter=False,
        )

        self.assertIs(result[0], ordinary_result[0])
        self.assertIs(result[1], ordinary_result[1])
        self.assertEqual(norm.static_calls, [])
        self._assert_call(
            norm.ordinary_calls,
            x=hidden_states,
            residual=residual,
            use_attn_tp_group=False,
        )
        self.assertEqual(norm.events, ["ordinary"])
        self.assertEqual(norm.per_group_calls, [])

    def test_cuda_fusion_disabled_skips_static_fp8_helper(self):
        hidden_states, residual = self._inputs()
        consumer = torch.nn.Identity()
        ordinary_result = (
            torch.tensor([[16.0, 17.0]]),
            torch.tensor([[18.0, 19.0]]),
        )
        norm = _RecordingNorm(ordinary_result=ordinary_result)
        communicator = self._make_communicator(
            norm,
            consumer=consumer,
            enable_fused_ar_quant=False,
        )

        result = self._prepare_attn(
            communicator,
            hidden_states,
            residual,
            use_aiter=False,
        )

        self.assertIs(result[0], ordinary_result[0])
        self.assertIs(result[1], ordinary_result[1])
        self.assertEqual(norm.static_calls, [])
        self._assert_call(
            norm.ordinary_calls,
            x=hidden_states,
            residual=residual,
            use_attn_tp_group=False,
        )
        self.assertEqual(norm.events, ["ordinary"])
        self.assertEqual(norm.per_group_calls, [])

    def test_aiter_preserves_per_group_quant_fusion_and_tuple_output(self):
        hidden_states, residual = self._inputs()
        consumer = torch.nn.Identity()
        bf16_output = torch.tensor([[20.0, 21.0]])
        fp8_output = torch.tensor([[22]], dtype=torch.uint8)
        scale_output = torch.tensor([[0.5]])
        per_group_hidden = (bf16_output, fp8_output, scale_output)
        per_group_residual = torch.tensor([[23.0, 24.0]])
        norm = _RecordingNorm(
            per_group_result=(per_group_hidden, per_group_residual)
        )
        communicator = self._make_communicator(
            norm,
            consumer=consumer,
            keep_bf16=True,
        )

        result = self._prepare_attn(
            communicator,
            hidden_states,
            residual,
            use_aiter=True,
        )

        self.assertIs(result[0], per_group_hidden)
        self.assertIs(result[1], per_group_residual)
        self._assert_call(
            norm.per_group_calls,
            x=hidden_states,
            residual=residual,
            use_attn_tp_group=False,
            keep_bf16=True,
        )
        self.assertEqual(norm.events, ["per_group"])
        self.assertEqual(norm.static_calls, [])
        self.assertEqual(norm.ordinary_calls, [])


if __name__ == "__main__":
    unittest.main()
