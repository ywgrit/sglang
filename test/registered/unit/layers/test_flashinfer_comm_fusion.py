import contextlib
import inspect
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Collectives are mocked and world_size is a plain int, so the world_size=4
# cases need one real CUDA device.
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class _FakeWorkspace:
    def __init__(self, backend, world_size, dtype=torch.bfloat16):
        self.backend = backend
        self.world_size = world_size
        self.metadata = {"use_fp32_lamport": dtype == torch.float32}

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kAllReduce = object()
        kARResidualRMSNorm = object()
        kARResidualRMSNormFP8Quant = object()
        kARResidualRMSNormOutFP8Quant = object()

    def __init__(self):
        self.calls = []

    def create_allreduce_fusion_workspace(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWorkspace(
            kwargs["backend"], kwargs["world_size"], dtype=kwargs["dtype"]
        )

    @staticmethod
    def _residual_rmsnorm(input, workspace, residual_in, rms_gamma, rms_eps):
        allreduced = input * workspace.world_size
        residual_out = allreduced + residual_in
        variance = residual_out.to(torch.float32).pow(2).mean(
            dim=-1, keepdim=True
        )
        norm_fp32 = (
            residual_out.to(torch.float32)
            * torch.rsqrt(variance + rms_eps)
            * rms_gamma.to(torch.float32)
        )
        return residual_out, norm_fp32

    def allreduce_fusion(
        self,
        *,
        input,
        workspace,
        pattern,
        output=None,
        residual_out=None,
        norm_out=None,
        residual_in=None,
        rms_gamma=None,
        rms_eps=None,
        quant_out=None,
        scale_out=None,
        scale_factor=None,
        weight_bias=0.0,
        **_kwargs,
    ):
        if pattern is self.AllReduceFusionPattern.kAllReduce:
            allreduced = input * workspace.world_size
            if output is None:
                return allreduced
            output.copy_(allreduced)
            return output

        rmsnorm_patterns = (
            self.AllReduceFusionPattern.kARResidualRMSNorm,
            self.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
            self.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
        )
        if pattern not in rmsnorm_patterns:
            raise ValueError(f"Unexpected pattern: {pattern}")

        expected_residual, norm_fp32 = self._residual_rmsnorm(
            input,
            workspace,
            residual_in,
            rms_gamma,
            rms_eps,
        )
        expected_norm = norm_fp32.to(input.dtype)
        residual_out.copy_(expected_residual)

        if pattern is self.AllReduceFusionPattern.kARResidualRMSNorm:
            norm_out.copy_(expected_norm)
            return norm_out

        assert weight_bias == 0.0
        quantized = torch.clamp(
            norm_fp32 / scale_factor.to(torch.float32),
            min=torch.finfo(torch.float8_e4m3fn).min,
            max=torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
        quant_out.copy_(quantized)
        if (
            pattern
            is self.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant
        ):
            norm_out.copy_(expected_norm)
        self.calls.append(
            {
                "pattern": pattern,
                "scale_factor": scale_factor,
                "weight_bias": weight_bias,
                "rms_gamma": rms_gamma,
                "norm_out": norm_out,
            }
        )
        return quant_out


class TestFlashInferAllReduceQuantCapability(CustomTestCase):
    def test_static_fp8_quant_capability_requires_both_patterns_and_arguments(self):
        comm = _FakeFlashInferComm()
        self.assertTrue(
            fusion._supports_allreduce_rmsnorm_static_fp8_quant(comm)
        )

        for missing_pattern in (
            "kARResidualRMSNormFP8Quant",
            "kARResidualRMSNormOutFP8Quant",
        ):
            patterns = types.SimpleNamespace(
                **{
                    name: getattr(comm.AllReduceFusionPattern, name)
                    for name in (
                        "kARResidualRMSNormFP8Quant",
                        "kARResidualRMSNormOutFP8Quant",
                    )
                    if name != missing_pattern
                }
            )
            with self.subTest(missing_pattern=missing_pattern), patch.object(
                comm, "AllReduceFusionPattern", patterns
            ):
                self.assertFalse(
                    fusion._supports_allreduce_rmsnorm_static_fp8_quant(comm)
                )

        for missing_argument in ("quant_out", "scale_factor", "weight_bias"):
            parameters = {
                name: parameter
                for name, parameter in inspect.signature(
                    _FakeFlashInferComm.allreduce_fusion
                ).parameters.items()
                if name != missing_argument
            }
            legacy_signature = inspect.Signature(parameters.values())
            with self.subTest(missing_argument=missing_argument), patch.object(
                _FakeFlashInferComm.allreduce_fusion,
                "__signature__",
                legacy_signature,
                create=True,
            ):
                self.assertFalse(
                    fusion._supports_allreduce_rmsnorm_static_fp8_quant(comm)
                )

    def test_missing_quant_signature_does_not_disable_plain_allreduce(self):
        class _LegacyFlashInferComm(_FakeFlashInferComm):
            def allreduce_fusion(self, *, input, workspace, pattern, output=None):
                return output

        with patch.object(fusion, "_flashinfer_allreduce_unavailable", False):
            self.assertFalse(
                fusion._supports_allreduce_rmsnorm_static_fp8_quant(
                    _LegacyFlashInferComm()
                )
            )
            self.assertFalse(fusion._flashinfer_allreduce_unavailable)

    def test_startup_synchronizes_capability_on_exact_group(self):
        attn_cpu_group = object()
        moe_tp_cpu_group = object()
        ep_cpu_group = object()

        cases = (
            (
                "attention_tp",
                True,
                1,
                attn_cpu_group,
                {
                    "get_attn_tp_group": MagicMock(
                        return_value=types.SimpleNamespace(
                            world_size=2, cpu_group=attn_cpu_group
                        )
                    )
                },
            ),
            (
                "moe_tp",
                False,
                1,
                moe_tp_cpu_group,
                {
                    "get_moe_tp_group": MagicMock(
                        return_value=types.SimpleNamespace(
                            world_size=2, cpu_group=moe_tp_cpu_group
                        )
                    )
                },
            ),
            (
                "moe_ep",
                False,
                2,
                ep_cpu_group,
                {
                    "get_moe_ep_group": MagicMock(
                        return_value=types.SimpleNamespace(
                            world_size=2, cpu_group=ep_cpu_group
                        )
                    )
                },
            ),
        )

        capability_cases = (
            ("all_available", True, False, 0, True),
            ("remote_unavailable", True, True, 0, False),
            ("local_unavailable", False, False, 1, False),
        )
        for name, use_attn_tp_group, moe_ep_size, cpu_group, group_patches in cases:
            for (
                capability_case,
                local_available,
                remote_unavailable,
                expected_incoming_flag,
                expected_available,
            ) in capability_cases:
                def emulate_collective(flag, *, op, group):
                    self.assertEqual(flag.item(), expected_incoming_flag)
                    self.assertIs(op, fusion.dist.ReduceOp.MAX)
                    self.assertIs(group, cpu_group)
                    if remote_unavailable:
                        flag.fill_(1)

                with (
                    self.subTest(
                        name=name, capability_case=capability_case
                    ),
                    patch.object(
                        fusion,
                        "_flashinfer_allreduce_quant_available",
                        local_available,
                        create=True,
                    ),
                    patch.object(
                        fusion,
                        "_flashinfer_allreduce_quant_capability_by_group",
                        {},
                        create=True,
                    ),
                    patch.object(
                        fusion,
                        "get_parallel",
                        return_value=types.SimpleNamespace(moe_ep_size=moe_ep_size),
                    ),
                    patch.object(
                        fusion,
                        "get_tp_group",
                        side_effect=AssertionError(
                            "must use the exact coordinator"
                        ),
                    ),
                    patch.multiple(fusion, **group_patches),
                    patch.object(
                        fusion.dist,
                        "all_reduce",
                        side_effect=emulate_collective,
                    ) as all_reduce,
                ):
                    fusion._synchronize_allreduce_quant_capability(
                        use_attn_tp_group
                    )

                    self.assertEqual(
                        fusion._flashinfer_allreduce_quant_capability_by_group,
                        {cpu_group: expected_available},
                    )
                    all_reduce.assert_called_once()

                    all_reduce.reset_mock()
                    self.assertEqual(
                        fusion._is_allreduce_quant_capability_available_for_group(
                            use_attn_tp_group
                        ),
                        expected_available,
                    )
                    all_reduce.assert_not_called()

    def test_pre_initialize_syncs_both_capabilities_before_early_return(self):
        cases = (
            (None, False, False),
            (_FakeFlashInferComm(), True, False),
            (_FakeFlashInferComm(), False, True),
        )
        for comm, unavailable, expect_initialization in cases:
            events = []
            synchronize = MagicMock(
                side_effect=lambda use_attn_tp_group: events.append(
                    ("quant", use_attn_tp_group)
                )
            )
            synchronize_unavailable = MagicMock(
                side_effect=lambda: events.append(("unavailable",))
            )
            initialize = MagicMock(
                side_effect=lambda **kwargs: events.append(
                    ("initialize", kwargs["use_attn_tp_group"])
                )
            )
            with (
                self.subTest(comm=comm, unavailable=unavailable),
                patch.object(fusion, "_flashinfer_comm", comm),
                patch.object(
                    fusion, "_flashinfer_allreduce_unavailable", unavailable
                ),
                patch.object(
                    fusion,
                    "_synchronize_allreduce_quant_capability",
                    synchronize,
                    create=True,
                ),
                patch.object(
                    fusion,
                    "_sync_allreduce_unavailable_across_tp",
                    synchronize_unavailable,
                ),
                patch.object(
                    fusion, "ensure_workspace_initialized", initialize
                ),
            ):
                fusion.pre_initialize_workspaces(
                    max_token_num=8,
                    hidden_dim=16,
                    dtype=torch.bfloat16,
                )

            self.assertEqual(
                events,
                [("quant", False), ("quant", True), ("unavailable",)]
                + (
                    [("initialize", False), ("initialize", True)]
                    if expect_initialization
                    else []
                ),
            )
            synchronize_unavailable.assert_called_once_with()
            if not expect_initialization:
                initialize.assert_not_called()

    def test_missing_local_comm_votes_unavailable(self):
        tp_cpu_group = object()

        def emulate_vote(flag, *, op, group):
            self.assertEqual(flag.item(), 1)
            self.assertIs(op, fusion.dist.ReduceOp.MAX)
            self.assertIs(group, tp_cpu_group)

        with (
            patch.object(fusion, "_flashinfer_allreduce_unavailable", False),
            patch.object(fusion, "_flashinfer_comm", None),
            patch.object(
                fusion,
                "get_tp_group",
                return_value=types.SimpleNamespace(
                    world_size=4, cpu_group=tp_cpu_group
                ),
            ),
            patch.object(fusion.dist, "all_reduce", side_effect=emulate_vote),
        ):
            fusion._sync_allreduce_unavailable_across_tp()

            self.assertTrue(fusion._flashinfer_allreduce_unavailable)

    def test_remote_unavailable_vote_updates_global_flag(self):
        tp_cpu_group = object()

        def emulate_vote(flag, *, op, group):
            self.assertEqual(flag.item(), 0)
            self.assertIs(op, fusion.dist.ReduceOp.MAX)
            self.assertIs(group, tp_cpu_group)
            flag.fill_(1)

        with (
            patch.object(fusion, "_flashinfer_allreduce_unavailable", False),
            patch.object(fusion, "_flashinfer_comm", _FakeFlashInferComm()),
            patch.object(
                fusion,
                "get_tp_group",
                return_value=types.SimpleNamespace(
                    world_size=4, cpu_group=tp_cpu_group
                ),
            ),
            patch.object(fusion.dist, "all_reduce", side_effect=emulate_vote),
        ):
            fusion._sync_allreduce_unavailable_across_tp()

            self.assertTrue(fusion._flashinfer_allreduce_unavailable)

    def test_single_rank_missing_comm_sets_unavailable_without_collective(self):
        with (
            patch.object(fusion, "_flashinfer_allreduce_unavailable", False),
            patch.object(fusion, "_flashinfer_comm", None),
            patch.object(
                fusion,
                "get_tp_group",
                return_value=types.SimpleNamespace(world_size=1),
            ),
            patch.object(fusion.dist, "all_reduce") as all_reduce,
        ):
            fusion._sync_allreduce_unavailable_across_tp()

            self.assertTrue(fusion._flashinfer_allreduce_unavailable)
            all_reduce.assert_not_called()

    def test_hybrid_ep_failure_is_voted_across_enclosing_tp(self):
        tp_cpu_group = object()

        for local_unavailable in (False, True):
            def emulate_tp_vote(flag, *, op, group):
                self.assertEqual(flag.item(), int(local_unavailable))
                self.assertIs(op, fusion.dist.ReduceOp.MAX)
                self.assertIs(group, tp_cpu_group)
                # Model a failure from one of the partitioned EP subgroups.
                flag.fill_(1)

            with (
                self.subTest(local_unavailable=local_unavailable),
                patch.object(
                    fusion,
                    "_flashinfer_allreduce_unavailable",
                    local_unavailable,
                ),
                patch.object(
                    fusion, "_flashinfer_comm", _FakeFlashInferComm()
                ),
                patch.object(
                    fusion,
                    "get_tp_group",
                    return_value=types.SimpleNamespace(
                        world_size=4, cpu_group=tp_cpu_group
                    ),
                ),
                patch.object(
                    fusion,
                    "_get_allreduce_group",
                    side_effect=AssertionError(
                        "the global unavailable vote must not use an EP subgroup"
                    ),
                ),
                patch.object(
                    fusion.dist,
                    "all_reduce",
                    side_effect=emulate_tp_vote,
                ) as all_reduce,
            ):
                fusion._sync_allreduce_unavailable_across_tp()

                self.assertTrue(fusion._flashinfer_allreduce_unavailable)
                all_reduce.assert_called_once()
                # Every TP rank now takes the same later attention fallback
                # before consulting its exact attention workspace group.
                self.assertFalse(
                    fusion.ensure_workspace_initialized(use_attn_tp_group=True)
                )


def _torch_rmsnorm_fp32(residual_out, weight, eps):
    variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (
        residual_out.to(torch.float32)
        * torch.rsqrt(variance + eps)
        * weight.to(torch.float32)
    )


def _torch_allreduce_residual_rmsnorm_baseline(
    input_tensor, residual, weight, world_size, eps
):
    allreduced = input_tensor * world_size
    residual_out = allreduced + residual
    norm_out = _torch_rmsnorm_fp32(residual_out, weight, eps).to(
        input_tensor.dtype
    )
    return norm_out, residual_out


def _torch_static_fp8_quant(tensor_fp32, scale_factor):
    return torch.clamp(
        tensor_fp32 / scale_factor.to(torch.float32),
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)


class TestFlashInferWorkspaceIdentity(CustomTestCase):
    def test_initialized_workspace_is_recreated_for_identity_mismatch(self):
        old_device_group = object()
        old_cpu_group = object()
        new_device_group = object()
        new_cpu_group = object()
        cases = (
            (
                "group_and_rank",
                {
                    "rank": 0,
                    "group": (old_device_group, old_cpu_group),
                    "backend": "trtllm",
                    "dtype": torch.bfloat16,
                },
                {
                    "rank": 1,
                    "backend": "trtllm",
                    "dtype": torch.bfloat16,
                },
            ),
            (
                "backend",
                {
                    "rank": 1,
                    "group": (new_device_group, new_cpu_group),
                    "backend": "trtllm",
                    "dtype": torch.bfloat16,
                },
                {
                    "rank": 1,
                    "backend": "mnnvl",
                    "dtype": torch.bfloat16,
                },
            ),
            (
                "dtype",
                {
                    "rank": 1,
                    "group": (new_device_group, new_cpu_group),
                    "backend": "trtllm",
                    "dtype": torch.bfloat16,
                },
                {
                    "rank": 1,
                    "backend": "trtllm",
                    "dtype": torch.float16,
                },
            ),
        )
        for case_name, old_identity, requested_identity in cases:
            with self.subTest(case_name=case_name):
                manager = fusion.FlashInferWorkspaceManager()
                old_workspace = _FakeWorkspace(
                    old_identity["backend"], 2, dtype=old_identity["dtype"]
                )
                old_workspace.is_buffer_size_sufficient = MagicMock(
                    return_value=True
                )
                manager.workspace = old_workspace
                manager.initialized = True
                manager.world_size = 2
                manager.rank = old_identity["rank"]
                manager.group = old_identity["group"]
                manager.max_token_num = 16
                manager.hidden_dim = 8
                manager.dtype = old_identity["dtype"]
                manager.backend = old_identity["backend"]
                manager.use_fp32_lamport = False

                new_workspace = _FakeWorkspace(
                    requested_identity["backend"],
                    2,
                    dtype=requested_identity["dtype"],
                )
                create_workspace = MagicMock(return_value=new_workspace)
                preflight = MagicMock(return_value=True)
                with (
                    patch.object(fusion, "_flashinfer_comm", _FakeFlashInferComm()),
                    patch.object(
                        fusion,
                        "_create_allreduce_fusion_workspace",
                        create_workspace,
                    ),
                    patch.object(
                        fusion,
                        "_flashinfer_create_workspace_supports_group",
                        True,
                    ),
                    patch.object(
                        fusion,
                        "_flashinfer_create_workspace_supports_comm_backend",
                        False,
                    ),
                    patch.object(fusion, "_TorchDistBackend", None),
                    patch.object(fusion, "_mnnvl_comm_backend", None),
                    patch.object(
                        fusion,
                        "_preflight_check_workspace_memory",
                        preflight,
                    ),
                    patch.object(
                        fusion,
                        "in_the_same_node_as",
                        return_value=[True, True],
                    ),
                    patch.object(
                        fusion, "_flashinfer_allreduce_unavailable", False
                    ),
                ):
                    manager.initialize(
                        world_size=2,
                        rank=requested_identity["rank"],
                        max_token_num=16,
                        hidden_dim=8,
                        backend=requested_identity["backend"],
                        group=new_cpu_group,
                        dtype=requested_identity["dtype"],
                        device_group=new_device_group,
                        cpu_group=new_cpu_group,
                    )

                old_workspace.is_buffer_size_sufficient.assert_not_called()
                create_workspace.assert_called_once()
                preflight.assert_called_once_with(
                    world_size=2,
                    max_token_num=16,
                    hidden_dim=8,
                    dtype=requested_identity["dtype"],
                    cpu_group=new_cpu_group,
                )
                self.assertIs(manager.workspace, new_workspace)
                self.assertEqual(manager.rank, requested_identity["rank"])
                self.assertEqual(
                    manager.group, (new_device_group, new_cpu_group)
                )
                self.assertEqual(manager.backend, requested_identity["backend"])
                self.assertEqual(manager.dtype, requested_identity["dtype"])
                self.assertIs(
                    create_workspace.call_args.kwargs["group"], new_device_group
                )
                self.assertEqual(
                    create_workspace.call_args.kwargs["backend"],
                    requested_identity["backend"],
                )
                self.assertEqual(
                    create_workspace.call_args.kwargs["rank"],
                    requested_identity["rank"],
                )
                self.assertEqual(
                    create_workspace.call_args.kwargs["dtype"],
                    requested_identity["dtype"],
                )


class TestFlashInferAllReduceStaticFp8Quant(CustomTestCase):
    @contextlib.contextmanager
    def _patched_quant_path(
        self,
        fake_comm,
        *,
        world_size=4,
        capability_available=True,
        workspace_available=True,
        manager_overrides=None,
    ):
        cpu_group = object()
        coordinator = types.SimpleNamespace(
            world_size=world_size,
            rank_in_group=0,
            cpu_group=cpu_group,
            device_group=object(),
        )
        manager_overrides = manager_overrides or {}
        manager_dtype = manager_overrides.get("dtype", torch.bfloat16)
        workspace = _FakeWorkspace("trtllm", world_size, dtype=manager_dtype)
        manager = fusion.FlashInferWorkspaceManager()
        manager.workspace = workspace
        manager.initialized = True
        manager.world_size = world_size
        manager.rank = 0
        manager.group = (coordinator.device_group, coordinator.cpu_group)
        manager.max_token_num = 2048
        manager.hidden_dim = 8
        manager.dtype = manager_dtype
        manager.backend = workspace.backend
        manager.use_fp32_lamport = manager_dtype == torch.float32
        for name, value in manager_overrides.items():
            setattr(manager, name, value)
        parallel = types.SimpleNamespace(
            attn_tp_size=world_size,
            attn_tp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=world_size,
            moe_tp_rank=0,
        )
        capability_cache = (
            {cpu_group: capability_available} if world_size > 1 else {}
        )
        ensure_workspace = MagicMock(return_value=workspace_available)
        with (
            patch.object(fusion, "_flashinfer_comm", fake_comm),
            patch.object(fusion, "_flashinfer_allreduce_unavailable", False),
            patch.object(
                fusion, "_flashinfer_allreduce_quant_available", True
            ),
            patch.object(
                fusion,
                "_flashinfer_allreduce_quant_capability_by_group",
                capability_cache,
            ),
            patch.object(fusion, "is_flashinfer_available", return_value=True),
            patch.object(fusion, "get_parallel", return_value=parallel),
            patch.object(fusion, "get_moe_tp_group", return_value=coordinator),
            patch.object(
                fusion,
                "ensure_workspace_initialized",
                ensure_workspace,
            ),
            patch.object(fusion, "_get_workspace_manager", return_value=manager),
        ):
            yield coordinator, manager, ensure_workspace

    def _inputs(self, dtype=torch.bfloat16):
        device = torch.device("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(7)
        return (
            torch.randn(
                3,
                8,
                dtype=dtype,
                device=device,
                generator=generator,
            ),
            torch.randn(
                3,
                8,
                dtype=dtype,
                device=device,
                generator=generator,
            ),
            # This represents Gemma's already-adjusted weight. The fused call
            # must use it directly with weight_bias=0, never add one again.
            torch.randn(
                8,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            + 1,
            torch.tensor(0.03125, dtype=torch.float32, device=device),
        )

    def test_static_fp8_quant_patterns_match_rmsnorm_baseline(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        world_size = 4
        eps = 1e-6
        for keep_bf16 in (False, True):
            with self.subTest(keep_bf16=keep_bf16):
                input_tensor, residual, gemma_weight, input_scale = self._inputs()
                fake_comm = _FakeFlashInferComm()
                with self._patched_quant_path(fake_comm, world_size=world_size):
                    quant_out, residual_out, norm_out = (
                        fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                            input_tensor=input_tensor,
                            residual=residual,
                            weight=gemma_weight,
                            scale_factor=input_scale,
                            eps=eps,
                            use_attn_tp_group=False,
                            keep_bf16=keep_bf16,
                        )
                    )

                expected_norm, expected_residual = (
                    _torch_allreduce_residual_rmsnorm_baseline(
                        input_tensor,
                        residual,
                        gemma_weight,
                        world_size,
                        eps,
                    )
                )
                expected_quant = _torch_static_fp8_quant(
                    _torch_rmsnorm_fp32(
                        expected_residual, gemma_weight, eps
                    ),
                    input_scale,
                )

                self.assertEqual(quant_out.dtype, torch.float8_e4m3fn)
                torch.testing.assert_close(residual_out, expected_residual)
                torch.testing.assert_close(
                    quant_out.to(torch.float32),
                    expected_quant.to(torch.float32),
                    rtol=0,
                    atol=0,
                )
                if keep_bf16:
                    self.assertEqual(norm_out.dtype, torch.bfloat16)
                    torch.testing.assert_close(norm_out, expected_norm)
                else:
                    self.assertEqual(norm_out.numel(), 0)

                self.assertEqual(len(fake_comm.calls), 1)
                call = fake_comm.calls[0]
                expected_pattern = (
                    fake_comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant
                    if keep_bf16
                    else fake_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant
                )
                self.assertIs(call["pattern"], expected_pattern)
                self.assertIs(call["scale_factor"], input_scale)
                self.assertIs(call["rms_gamma"], gemma_weight)
                self.assertEqual(call["weight_bias"], 0.0)
                if keep_bf16:
                    self.assertIs(call["norm_out"], norm_out)
                else:
                    self.assertIsNone(call["norm_out"])

    def test_static_fp8_quant_fallbacks_happen_before_collective(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        input_tensor, residual, weight, scale = self._inputs()
        generator = torch.Generator(device=input_tensor.device)
        generator.manual_seed(11)
        noncontiguous_input = torch.randn(
            8,
            3,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
            generator=generator,
        ).t()
        noncontiguous_residual = torch.randn(
            8,
            3,
            dtype=residual.dtype,
            device=residual.device,
            generator=generator,
        ).t()
        noncontiguous_weight = torch.randn(
            16,
            dtype=weight.dtype,
            device=weight.device,
            generator=generator,
        )[::2]
        float64_input, float64_residual, float64_weight, _ = self._inputs(
            dtype=torch.float64
        )
        cases = (
            (
                "group_capability_unavailable",
                {},
                {"capability_available": False},
                False,
            ),
            ("scale_absent", {"scale_factor": None}, {}, False),
            (
                "scale_not_scalar",
                {"scale_factor": torch.ones(2, device=scale.device)},
                {},
                False,
            ),
            (
                "scale_wrong_device",
                {"scale_factor": torch.tensor(scale.item(), device="cpu")},
                {},
                False,
            ),
            (
                "scale_wrong_dtype",
                {
                    "scale_factor": torch.tensor(
                        scale.item(), dtype=torch.bfloat16, device=scale.device
                    )
                },
                {},
                False,
            ),
            (
                "input_noncontiguous",
                {"input_tensor": noncontiguous_input},
                {},
                False,
            ),
            (
                "residual_noncontiguous",
                {"residual": noncontiguous_residual},
                {},
                False,
            ),
            (
                "weight_noncontiguous",
                {"weight": noncontiguous_weight},
                {},
                False,
            ),
            (
                "unsupported_matching_dtype",
                {
                    "input_tensor": float64_input,
                    "residual": float64_residual,
                    "weight": float64_weight,
                },
                {},
                False,
            ),
            ("single_rank", {}, {"world_size": 1}, False),
            (
                "workspace_unavailable",
                {},
                {"workspace_available": False},
                True,
            ),
        )
        for (
            name,
            argument_overrides,
            path_overrides,
            expect_workspace_check,
        ) in cases:
            fake_comm = _FakeFlashInferComm()
            arguments = {
                "input_tensor": input_tensor,
                "residual": residual,
                "weight": weight,
                "scale_factor": scale,
                "eps": 1e-6,
                "use_attn_tp_group": False,
                "keep_bf16": False,
            }
            arguments.update(argument_overrides)
            with self.subTest(name=name), self._patched_quant_path(
                fake_comm, **path_overrides
            ) as (_, _, ensure_workspace):
                result = (
                    fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                        **arguments
                    )
                )

            self.assertEqual(result, (None, None, None))
            self.assertEqual(fake_comm.calls, [])
            if expect_workspace_check:
                ensure_workspace.assert_called_once()
            else:
                ensure_workspace.assert_not_called()

    def test_static_fp8_quant_rejects_invalid_manager_after_ensure(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        input_tensor, residual, weight, scale = self._inputs()
        cases = (
            ("missing_workspace", {"workspace": None}),
            ("wrong_group", {"group": (object(), object())}),
        )
        for case_name, manager_overrides in cases:
            fake_comm = _FakeFlashInferComm()
            with self.subTest(case_name=case_name), self._patched_quant_path(
                fake_comm, manager_overrides=manager_overrides
            ) as (_, _, ensure_workspace):
                with self.assertRaises(RuntimeError):
                    fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                        input_tensor=input_tensor,
                        residual=residual,
                        weight=weight,
                        scale_factor=scale,
                        use_attn_tp_group=False,
                    )

            ensure_workspace.assert_called_once()
            self.assertEqual(fake_comm.calls, [])

    def test_static_fp8_quant_compile_uses_preinitialized_metadata_only(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        input_tensor, residual, weight, scale = self._inputs()
        fake_comm = _FakeFlashInferComm()
        expected = (
            torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn),
            torch.empty_like(residual),
            input_tensor.new_empty((0,)),
        )
        private_op = MagicMock(return_value=expected)
        with self._patched_quant_path(fake_comm) as (
            _,
            manager,
            ensure_workspace,
        ):
            manager.is_buffer_size_sufficient = MagicMock(
                side_effect=AssertionError(
                    "compile preflight must not call the manager validator"
                )
            )
            manager.workspace.is_buffer_size_sufficient = MagicMock(
                side_effect=AssertionError(
                    "compile preflight must not call the opaque workspace"
                )
            )
            with (
                patch.object(torch.compiler, "is_compiling", return_value=True),
                patch.object(
                    fusion,
                    "_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant_op",
                    private_op,
                ),
            ):
                result = (
                    fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                        input_tensor=input_tensor,
                        residual=residual,
                        weight=weight,
                        scale_factor=scale,
                        use_attn_tp_group=False,
                    )
                )

        self.assertIs(result, expected)
        ensure_workspace.assert_not_called()
        manager.is_buffer_size_sufficient.assert_not_called()
        manager.workspace.is_buffer_size_sufficient.assert_not_called()
        private_op.assert_called_once()
        self.assertEqual(fake_comm.calls, [])

    def test_static_fp8_quant_compile_rejects_invalid_static_workspace(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        input_tensor, residual, weight, scale = self._inputs()
        cases = (
            ("token_capacity", {"max_token_num": input_tensor.shape[0] - 1}),
            ("hidden_capacity", {"hidden_dim": input_tensor.shape[-1] - 1}),
            ("dtype", {"dtype": torch.float16}),
            ("rank", {"rank": 1}),
            ("group", {"group": (object(), object())}),
        )
        for case_name, manager_overrides in cases:
            fake_comm = _FakeFlashInferComm()
            private_op = MagicMock(
                side_effect=AssertionError("invalid metadata reached kernel")
            )
            with self.subTest(case_name=case_name), self._patched_quant_path(
                fake_comm, manager_overrides=manager_overrides
            ) as (_, manager, ensure_workspace):
                manager.is_buffer_size_sufficient = MagicMock(
                    side_effect=AssertionError(
                        "compile preflight must use static capacity metadata"
                    )
                )
                manager.workspace.is_buffer_size_sufficient = MagicMock(
                    side_effect=AssertionError(
                        "compile preflight must not call the opaque workspace"
                    )
                )
                with (
                    patch.object(
                        torch.compiler, "is_compiling", return_value=True
                    ),
                    patch.object(
                        fusion,
                        "_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant_op",
                        private_op,
                    ),
                ):
                    result = fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                        input_tensor=input_tensor,
                        residual=residual,
                        weight=weight,
                        scale_factor=scale,
                        use_attn_tp_group=False,
                    )

            self.assertEqual(result, (None, None, None))
            ensure_workspace.assert_not_called()
            manager.is_buffer_size_sufficient.assert_not_called()
            manager.workspace.is_buffer_size_sufficient.assert_not_called()
            private_op.assert_not_called()
            self.assertEqual(fake_comm.calls, [])

    def test_static_fp8_quant_accepts_supported_cuda_dtypes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                input_tensor, residual, weight, scale = self._inputs(dtype=dtype)
                fake_comm = _FakeFlashInferComm()
                expected = (
                    torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn),
                    torch.empty_like(residual),
                    input_tensor.new_empty((0,)),
                )
                private_op = MagicMock(return_value=expected)
                with self._patched_quant_path(
                    fake_comm, manager_overrides={"dtype": dtype}
                ) as (_, _, ensure_workspace), patch.object(
                    fusion,
                    "_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant_op",
                    private_op,
                ):
                    result = fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                        input_tensor=input_tensor,
                        residual=residual,
                        weight=weight,
                        scale_factor=scale,
                        use_attn_tp_group=False,
                    )

                self.assertIs(result, expected)
                ensure_workspace.assert_called_once()
                private_op.assert_called_once()
                self.assertEqual(fake_comm.calls, [])

    def test_static_fp8_quant_collective_exception_propagates(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for FlashInfer FP8 fusion contract")

        input_tensor, residual, weight, scale = self._inputs()
        fake_comm = _FakeFlashInferComm()
        fake_comm.allreduce_fusion = MagicMock(
            side_effect=RuntimeError("collective launch failed")
        )
        with self._patched_quant_path(fake_comm):
            with self.assertRaisesRegex(RuntimeError, "collective launch failed"):
                fusion.try_flashinfer_allreduce_residual_rmsnorm_static_fp8_quant(
                    input_tensor=input_tensor,
                    residual=residual,
                    weight=weight,
                    scale_factor=scale,
                    eps=1e-6,
                    use_attn_tp_group=False,
                    keep_bf16=False,
                )

        fake_comm.allreduce_fusion.assert_called_once()


class TestFlashInferCommFusion(CustomTestCase):
    def test_auto_backend_resolves_by_arch(self):
        single_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=1
        )
        multi_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=2
        )

        # Blackwell: mnnvl on both single-node and multi-node.
        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node), "mnnvl"
            )

        # SM90: auto uses trtllm on single-node, multi-node is unsupported.
        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

        # Architectures outside SM90/SM10X are unsupported. Both pre-SM90
        # and post-SM10X devices (e.g. SM120) must fail closed.
        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(single_node)
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

    def test_explicit_backend_validation(self):
        single_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=1
        )
        multi_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=2
        )
        single_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=1
        )
        multi_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=2
        )

        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_mnnvl),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_trtllm),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl)
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl),
                "mnnvl",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                for args in (
                    single_node_mnnvl,
                    multi_node_mnnvl,
                    single_node_trtllm,
                    multi_node_trtllm,
                ):
                    with self.subTest(backend=args.flashinfer_allreduce_fusion_backend):
                        with self.assertRaises(ValueError):
                            fusion.resolve_flashinfer_allreduce_fusion_backend(args)

    def test_allreduce_fusion_backends_match_torch_baseline(self):
        fake_comm = _FakeFlashInferComm()
        original_comm = fusion._flashinfer_comm
        original_create = fusion._create_allreduce_fusion_workspace
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._create_allreduce_fusion_workspace = (
                fake_comm.create_allreduce_fusion_workspace
            )
            fusion._flashinfer_allreduce_unavailable = False

            for backend in ("trtllm", "mnnvl"):
                with self.subTest(backend=backend):
                    world_size = 4
                    manager = fusion.FlashInferWorkspaceManager()
                    manager.workspace = _FakeWorkspace(backend, world_size)
                    manager.initialized = True
                    buffers[manager_key] = manager
                    if not torch.cuda.is_available():
                        self.skipTest("FlashInfer allreduce custom op is CUDA-only")
                    device = torch.device("cuda")
                    torch.manual_seed(0)
                    input_tensor = torch.randn(4, 8, dtype=torch.float32, device=device)
                    residual = torch.randn(4, 8, dtype=torch.float32, device=device)
                    weight = torch.randn(8, dtype=torch.float32, device=device)
                    eps = 1e-6

                    expected_norm, expected_residual = (
                        _torch_allreduce_residual_rmsnorm_baseline(
                            input_tensor, residual, weight, world_size, eps
                        )
                    )

                    with (
                        patch.object(
                            fusion, "is_flashinfer_available", return_value=True
                        ),
                        get_parallel().override(attn_tp_size=world_size),
                        patch.object(
                            fusion, "ensure_workspace_initialized", return_value=True
                        ),
                    ):
                        norm_out, residual_out = (
                            fusion.flashinfer_allreduce_residual_rmsnorm(
                                input_tensor=input_tensor,
                                residual=residual,
                                weight=weight,
                                eps=eps,
                                max_token_num=8,
                            )
                        )

                    torch.testing.assert_close(norm_out, expected_norm)
                    torch.testing.assert_close(residual_out, expected_residual)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._create_allreduce_fusion_workspace = original_create
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable


_GROUP_KEY = ("device_group", "cpu_group")
_OTHER_GROUP_KEY = ("other_device_group", "other_cpu_group")


class TestFlashInferAllReduceOnly(CustomTestCase):
    def _make_manager(self, world_size, group_key=_GROUP_KEY, backend="trtllm"):
        manager = fusion.FlashInferWorkspaceManager()
        manager.workspace = _FakeWorkspace(backend, world_size)
        manager.initialized = True
        manager.world_size = world_size
        manager.group = group_key
        manager.max_token_num = 2048
        manager.hidden_dim = 4096
        manager.dtype = torch.float32
        manager.backend = backend
        manager.use_fp32_lamport = True
        return manager

    @contextlib.contextmanager
    def _patched_attn_workspace(self, manager):
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        original_comm = fusion._flashinfer_comm
        original_unavailable = fusion._flashinfer_allreduce_unavailable

        buffers[manager_key] = manager
        fusion._flashinfer_comm = _FakeFlashInferComm()
        fusion._flashinfer_allreduce_unavailable = False
        try:
            yield
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._flashinfer_allreduce_unavailable = original_unavailable
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager

    def _can_use(self, input_, world_size=4, group_key=_GROUP_KEY):
        return fusion.can_use_flashinfer_allreduce(
            input_,
            use_attn_tp_group=True,
            expected_world_size=world_size,
            expected_group_key=group_key,
        )

    def test_allreduce_output_equals_input_times_world_size(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for flashinfer custom op")
        world_size = 4
        manager = self._make_manager(world_size)
        manager.dtype = torch.bfloat16
        manager.use_fp32_lamport = False
        with self._patched_attn_workspace(manager):
            input_ = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
            expected = input_ * world_size

            with get_parallel().override(attn_tp_size=world_size):
                self.assertTrue(self._can_use(input_, world_size=world_size))
                result = fusion.flashinfer_allreduce(input_, use_attn_tp_group=True)

            torch.testing.assert_close(result, expected)

    def test_shape_guard_rejects_non_2d(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            self.assertFalse(self._can_use(torch.randn(16)))
            self.assertFalse(self._can_use(torch.randn(2, 8, 16)))

    def test_shape_guard_rejects_non_contiguous(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            non_contiguous = torch.randn(16, 8).t()
            self.assertFalse(non_contiguous.is_contiguous())
            self.assertFalse(self._can_use(non_contiguous))

    def test_rejects_when_unavailable(self):
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_allreduce_unavailable = True
            self.assertFalse(self._can_use(torch.randn(8, 16)))
        finally:
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_rejects_when_workspace_uninitialized(self):
        with self._patched_attn_workspace(fusion.FlashInferWorkspaceManager()):
            with get_parallel().override(attn_tp_size=4):
                self.assertFalse(self._can_use(torch.randn(8, 16)))

    def test_rejects_when_workspace_group_differs(self):
        """A workspace rendezvoused on other peers must not be reused.

        Under hybrid EP+TP (e.g. tp=4, ep=2) the MoE-TP and MoE-EP groups have
        the same world size but pair different ranks, so a workspace built for
        one silently reduces across the wrong peers when used by the other --
        wrong output rather than a crash.
        """
        with self._patched_attn_workspace(self._make_manager(2)):
            self.assertFalse(
                self._can_use(
                    torch.randn(8, 16), world_size=2, group_key=_OTHER_GROUP_KEY
                )
            )

    def test_rejects_when_workspace_world_size_differs(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            self.assertFalse(self._can_use(torch.randn(8, 16), world_size=2))

    def test_fp32_initialization_caches_allocated_lamport_mode(self):
        """FP32 startup allocation must remain eligible for FP32 all-reduce.

        The initialization API's legacy use_fp32_lamport argument defaults to
        False, while FlashInfer derives the allocated TRT-LLM mode from dtype.
        Eligibility must follow the workspace metadata rather than that input.
        """
        fake_comm = _FakeFlashInferComm()
        manager = fusion.FlashInferWorkspaceManager()
        with (
            patch.object(fusion, "_flashinfer_comm", fake_comm),
            patch.object(
                fusion,
                "_create_allreduce_fusion_workspace",
                fake_comm.create_allreduce_fusion_workspace,
            ),
            patch.object(
                fusion, "_preflight_check_workspace_memory", return_value=True
            ),
        ):
            manager.initialize(
                world_size=4,
                rank=0,
                max_token_num=8,
                hidden_dim=4096,
                backend="trtllm",
                dtype=torch.float32,
            )

        self.assertTrue(manager.use_fp32_lamport)
        with self._patched_attn_workspace(manager):
            self.assertTrue(
                self._can_use(
                    torch.randn(8, 16, dtype=torch.float32), group_key=(None, None)
                )
            )

    def test_rejects_when_token_num_exceeds_workspace_capacity(self):
        """Oversized all-reduces fall back without triggering a warning.

        TRT-LLM warns whenever its size validator rejects an operation. The
        total element count already proves that this operation cannot use the
        workspace, so the validator must not be invoked.
        """
        manager = self._make_manager(4)
        manager.max_token_num = 8
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=True)
        with self._patched_attn_workspace(manager):
            self.assertFalse(self._can_use(torch.randn(9, 4096)))

        manager.workspace.is_buffer_size_sufficient.assert_not_called()

    def test_reshaped_input_within_total_capacity_reaches_validator(self):
        manager = self._make_manager(4)
        manager.max_token_num = 8
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=True)
        input_ = torch.randn(9, 16)
        with self._patched_attn_workspace(manager):
            self.assertTrue(self._can_use(input_))

        manager.workspace.is_buffer_size_sufficient.assert_called_once_with(
            tp_size=4,
            num_tokens=9,
            hidden_dim=16,
            dtype=input_.dtype,
        )

    def test_non_fp32_dtype_change_reaches_validator(self):
        manager = self._make_manager(4)
        manager.dtype = torch.bfloat16
        manager.use_fp32_lamport = False
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=True)
        input_ = torch.randn(8, 16, dtype=torch.float16)
        with self._patched_attn_workspace(manager):
            self.assertTrue(self._can_use(input_))

        manager.workspace.is_buffer_size_sufficient.assert_called_once()

    def test_mnnvl_capacity_decision_reaches_validator(self):
        manager = self._make_manager(4, backend="mnnvl")
        manager.max_token_num = 8
        manager.workspace.is_buffer_size_sufficient = MagicMock(return_value=False)
        with self._patched_attn_workspace(manager):
            self.assertFalse(self._can_use(torch.randn(9, 4096)))

        manager.workspace.is_buffer_size_sufficient.assert_called_once()

    def test_compiling_rejects_when_token_num_exceeds_workspace_capacity(self):
        manager = self._make_manager(4)
        manager.max_token_num = 8
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16)))
                self.assertFalse(self._can_use(torch.randn(9, 16)))

    def test_rejects_when_hidden_dim_exceeds_workspace_capacity(self):
        manager = self._make_manager(4)
        manager.hidden_dim = 16
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16)))
                self.assertFalse(self._can_use(torch.randn(8, 17)))

    def test_rejects_when_dtype_mismatches_workspace(self):
        manager = self._make_manager(4)
        manager.dtype = torch.bfloat16
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16, dtype=torch.bfloat16)))
                self.assertFalse(self._can_use(torch.randn(8, 16, dtype=torch.float32)))


class _FakeGroupCoordinator:
    def __init__(self, world_size):
        self.world_size = world_size
        self._fi_workspace_hint = None


class TestTagGroupsForFlashInferAllReduceOnly(CustomTestCase):
    """The MoE workspace rendezvouses on the EP group when moe_ep_size > 1 and
    on the MoE-TP group otherwise, so only that one group may be tagged."""

    def _tag(self, *, attn_tp, moe_ep, moe_tp):
        from sglang.srt.distributed import parallel_state as ps

        with patch.object(ps, "_ENABLE_FLASHINFER_ALLREDUCE_ONLY", True), patch.object(
            ps, "_ATTN_TP", attn_tp
        ), patch.object(ps, "_MOE_EP", moe_ep), patch.object(ps, "_MOE_TP", moe_tp):
            ps._tag_groups_for_flashinfer_allreduce_only()

    def test_hybrid_ep_tp_tags_only_the_ep_group(self):
        attn_tp = _FakeGroupCoordinator(4)
        moe_ep = _FakeGroupCoordinator(2)
        moe_tp = _FakeGroupCoordinator(2)

        self._tag(attn_tp=attn_tp, moe_ep=moe_ep, moe_tp=moe_tp)

        self.assertEqual(attn_tp._fi_workspace_hint, "attn_tp")
        self.assertEqual(moe_ep._fi_workspace_hint, "moe")
        self.assertIsNone(moe_tp._fi_workspace_hint)

    def test_pure_moe_tp_tags_only_the_moe_tp_group(self):
        attn_tp = _FakeGroupCoordinator(4)
        moe_ep = _FakeGroupCoordinator(1)
        moe_tp = _FakeGroupCoordinator(4)

        self._tag(attn_tp=attn_tp, moe_ep=moe_ep, moe_tp=moe_tp)

        self.assertEqual(moe_tp._fi_workspace_hint, "moe")
        self.assertIsNone(moe_ep._fi_workspace_hint)

    def test_shared_coordinator_prefers_attn_tp(self):
        # tp=4, ep=4: _ATTN_TP is _MOE_EP is _TP. Either workspace spans the
        # same peers, but the choice must be deterministic.
        shared = _FakeGroupCoordinator(4)
        moe_tp = _FakeGroupCoordinator(1)

        self._tag(attn_tp=shared, moe_ep=shared, moe_tp=moe_tp)

        self.assertEqual(shared._fi_workspace_hint, "attn_tp")
        self.assertIsNone(moe_tp._fi_workspace_hint)


if __name__ == "__main__":
    unittest.main()
