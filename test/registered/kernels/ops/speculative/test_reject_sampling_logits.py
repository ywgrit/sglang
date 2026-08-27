import os
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_from_logits_triton,
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize(
    ("batch_size", "num_slots", "expected"),
    [(1, 8, False), (2, 8, True), (4, 2, False), (4, 4, True), (8, 2, True)],
)
def test_logits_rejection_profitability_gate(batch_size, num_slots, expected):
    assert (
        eagle_utils._is_logits_rejection_shape_profitable(batch_size, num_slots)
        is expected
    )


def _kernel_runtime_available():
    return torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"


def _kernel_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _outputs(batch_size: int, slots: int, device: str):
    return (
        torch.full((batch_size * slots,), -1, dtype=torch.int32, device=device),
        torch.full((batch_size, slots), -1, dtype=torch.int32, device=device),
        torch.zeros((batch_size,), dtype=torch.int32, device=device),
    )


def _run_sampler(
    sampler,
    *,
    logits,
    temperatures,
    draft_probs,
    candidates,
    coins,
    final_coins,
):
    batch_size, slots = candidates.shape
    predicts, accept_index, accept_token_num = _outputs(
        batch_size, slots, logits.device
    )
    retrieve_index = torch.arange(
        batch_size * slots, dtype=torch.int32, device=logits.device
    ).view(batch_size, slots)

    common = dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_index,
        retrive_next_sibling=retrieve_index,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    if sampler is chain_speculative_sampling_triton:
        target_probs = torch.softmax(
            logits.view(batch_size, slots, -1) / temperatures[:, None, :], dim=-1
        )
        sampler(target_probs=target_probs, **common)
    else:
        sampler(target_logits=logits, temperatures=temperatures, **common)
    if logits.is_cuda:
        torch.cuda.synchronize()
    return predicts, accept_index, accept_token_num


def _run_eagle_dispatch(
    renorm,
    *,
    batch_size=8,
    is_cuda=True,
    is_musa=False,
    logits_sampler_return=None,
    enable_async_assert=False,
    target_logits=None,
    logit_bias=None,
    use_real_logits_sampler=False,
):
    slots, vocab_size = 3, 37
    logits = (
        torch.randn(batch_size * slots, vocab_size)
        if target_logits is None
        else target_logits
    )
    device = logits.device
    temperatures = torch.full((batch_size, 1), 0.7, device=device)
    draft_probs = torch.full(
        (batch_size, slots, vocab_size),
        1.0 / vocab_size,
        dtype=torch.float32,
        device=device,
    )
    verify_input = SimpleNamespace(
        draft_token_num=slots,
        draft_token=torch.tensor(
            [0, 3, 5] * batch_size, dtype=torch.int32, device=device
        ),
        max_tree_depth=slots,
        tree_topk=1,
        retrieve_index=torch.arange(
            batch_size * slots, dtype=torch.int32, device=device
        ).view(batch_size, slots),
        retrieve_next_token=torch.zeros(
            (batch_size, slots), dtype=torch.int32, device=device
        ),
        retrieve_next_sibling=torch.zeros(
            (batch_size, slots), dtype=torch.int32, device=device
        ),
        draft_probs=draft_probs,
    )
    sampling_info = SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=logit_bias,
        is_all_greedy=False,
        temperatures=temperatures,
        need_top_k_sampling=renorm == "top-k",
        need_top_p_sampling=renorm == "top-p",
        top_ks=torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device),
        top_ps=torch.full((batch_size,), 0.9, dtype=torch.float32, device=device),
        sampling_seed=None,
    )
    batch = SimpleNamespace(
        device=device,
        seq_lens=torch.full((batch_size,), 4, dtype=torch.int32, device=device),
        sampling_info=sampling_info,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    logits_output = SimpleNamespace(next_token_logits=logits)
    coins = torch.zeros((batch_size, slots), dtype=torch.float32, device=device)
    final_coins = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
    logits_call = {}
    probability_call = {}

    def fake_logits_sampler(**kwargs):
        logits_call.update(kwargs)
        kwargs["predicts"].zero_()
        kwargs["accept_index"].zero_()
        kwargs["accept_token_num"].zero_()
        if logits_sampler_return is not None:
            return logits_sampler_return
        return torch.ones((2, batch_size * slots, 1), dtype=torch.float32)

    def fake_probability_sampler(**kwargs):
        probability_call.update(kwargs)
        kwargs["predicts"].zero_()
        kwargs["accept_index"].zero_()
        kwargs["accept_token_num"].zero_()

    spec = SimpleNamespace(
        speculative_use_rejection_sampling=True,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    tp_group = SimpleNamespace(world_size=1)
    real_softmax = torch.nn.functional.softmax
    needs_renorm = renorm != "none"
    needs_probability_fallback = needs_renorm or not (
        is_cuda and eagle_utils._is_logits_rejection_shape_profitable(batch_size, slots)
    )
    softmax_side_effect = (
        real_softmax
        if needs_probability_fallback
        else AssertionError("unexpected probability fallback")
    )
    logits_sampler_context = (
        nullcontext()
        if use_real_logits_sampler
        else patch(
            "sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_from_logits_triton",
            side_effect=fake_logits_sampler,
        )
    )
    with (
        patch.object(eagle_utils, "_is_cuda", is_cuda),
        patch.object(eagle_utils, "_is_cpu", False),
        patch.object(eagle_utils, "_is_npu", False),
        patch.object(eagle_utils, "_is_hip", False),
        patch.object(eagle_utils, "_is_musa", is_musa),
        patch.object(eagle_utils, "_is_xpu", False),
        patch.object(eagle_utils, "get_spec", return_value=spec),
        patch.object(eagle_utils, "_verify_coins", return_value=(coins, final_coins)),
        patch("torch.nn.functional.softmax", side_effect=softmax_side_effect),
        patch("sgl_kernel.top_k_renorm_prob", side_effect=lambda probs, _: probs),
        patch("sgl_kernel.top_p_renorm_prob", side_effect=lambda probs, _: probs),
        logits_sampler_context,
        patch(
            "sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton",
            side_effect=fake_probability_sampler,
        ),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ),
        patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_ASYNC_ASSERT.get",
            return_value=enable_async_assert,
        ),
    ):
        eagle_utils.eagle_sample(verify_input, batch, logits_output)

    return (
        probability_call,
        logits_call,
        logits,
        temperatures,
        draft_probs,
        coins,
        final_coins,
    )


@pytest.mark.parametrize("renorm", ["none", "top-k", "top-p"])
def test_eagle_rejection_dispatch_preserves_probability_fallback(renorm):
    (
        probability_call,
        logits_call,
        logits,
        temperatures,
        draft_probs,
        coins,
        final_coins,
    ) = _run_eagle_dispatch(renorm)
    needs_renorm = renorm != "none"
    captured = probability_call if needs_renorm else logits_call
    assert bool(probability_call) is needs_renorm
    assert bool(logits_call) is not needs_renorm
    if needs_renorm:
        assert captured["target_probs"].shape == (
            draft_probs.shape[0],
            draft_probs.shape[1],
            draft_probs.shape[2],
        )
    else:
        assert captured["target_logits"] is logits
        assert captured["temperatures"] is temperatures
    assert captured["draft_probs"] is draft_probs
    assert captured["uniform_samples"] is coins
    assert captured["uniform_samples_for_final_sampling"] is final_coins


@pytest.mark.parametrize("batch_size", [1, 4])
def test_eagle_small_verify_grid_uses_probability_fallback(batch_size):
    probability_call, logits_call, *_ = _run_eagle_dispatch(
        "none", batch_size=batch_size
    )

    assert probability_call
    assert not logits_call


def test_eagle_musa_uses_probability_fallback():
    probability_call, logits_call, *_ = _run_eagle_dispatch(
        "none", is_cuda=False, is_musa=True
    )

    assert probability_call
    assert not logits_call


@pytest.mark.parametrize("invalid_block_sumexp", [0.0, float("nan")])
def test_eagle_logits_rejection_async_asserts_invalid_normalization(
    invalid_block_sumexp,
):
    target_block_stats = torch.ones((2, 24, 1), dtype=torch.float32)
    target_block_stats[1].fill_(invalid_block_sumexp)

    with pytest.raises(RuntimeError, match="normalization"):
        _run_eagle_dispatch(
            "none",
            logits_sampler_return=target_block_stats,
            enable_async_assert=True,
        )


@pytest.mark.skipif(
    not _kernel_runtime_available(), reason="requires CUDA or Triton interpreter"
)
@pytest.mark.parametrize("invalid_source", ["all-masked", "postprocessor-nan"])
def test_eagle_logits_rejection_detects_actual_invalid_stats(invalid_source):
    child_case = os.environ.get("SGLANG_REJECTION_INVALID_STATS_CHILD")
    if torch.cuda.is_available() and child_case != invalid_source:
        env = os.environ.copy()
        env["SGLANG_REJECTION_INVALID_STATS_CHILD"] = invalid_source
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_eagle_logits_rejection_detects_actual_invalid_stats[{invalid_source}]",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        return

    device = _kernel_device()
    logits = torch.randn(24, 37, dtype=torch.float32, device=device)
    logit_bias = None
    if invalid_source == "all-masked":
        logits.fill_(float("-inf"))
    else:
        logit_bias = torch.zeros((8, 37), dtype=torch.float32, device=device)
        logit_bias[0, 0] = float("nan")

    expected_error = (
        "device-side assert triggered" if logits.is_cuda else "normalization"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        _run_eagle_dispatch(
            "none",
            enable_async_assert=True,
            target_logits=logits,
            logit_bias=logit_bias,
            use_real_logits_sampler=True,
        )
        if logits.is_cuda:
            torch.cuda.synchronize()


@pytest.mark.skipif(
    not _kernel_runtime_available(), reason="requires CUDA or Triton interpreter"
)
@pytest.mark.parametrize(
    "scenario", ["all-accepted", "first-reject", "middle-reject", "nan-q"]
)
def test_logits_sampler_matches_probability_sampler(scenario):
    torch.manual_seed(42)
    device = _kernel_device()
    batch_size, slots, vocab_size = 2, 4, 37
    temperatures = torch.tensor([[0.7], [1.0]], device=device)
    logits = torch.randn(
        batch_size * slots, vocab_size, dtype=torch.float32, device=device
    )
    candidates = torch.tensor(
        [[0, 3, 5, 7], [0, 2, 4, 6]], dtype=torch.int32, device=device
    )
    draft_probs = torch.full(
        (batch_size, slots, vocab_size),
        1.0 / vocab_size,
        dtype=torch.float32,
        device=device,
    )
    coins = torch.zeros((batch_size, slots), dtype=torch.float32, device=device)
    final_coins = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)

    if scenario == "first-reject":
        coins[:, 0] = 0.99
        draft_probs[:, 0].zero_()
        draft_probs[0, 0, 3] = 1.0
        draft_probs[1, 0, 2] = 1.0
    elif scenario == "middle-reject":
        coins[:, 1] = 0.99
        draft_probs[:, 1].zero_()
        draft_probs[0, 1, 5] = 1.0
        draft_probs[1, 1, 4] = 1.0
    elif scenario == "nan-q":
        coins[:, 0] = 0.5
        draft_probs[:, 0].fill_(float("nan"))

    expected = _run_sampler(
        chain_speculative_sampling_triton,
        logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        candidates=candidates,
        coins=coins,
        final_coins=final_coins,
    )
    actual = _run_sampler(
        chain_speculative_sampling_from_logits_triton,
        logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        candidates=candidates,
        coins=coins,
        final_coins=final_coins,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.skipif(
    not _kernel_runtime_available(), reason="requires CUDA or Triton interpreter"
)
def test_logits_sampler_preserves_degenerate_residual_fallback():
    device = _kernel_device()
    batch_size, slots, vocab_size = 1, 3, 37
    logits = torch.zeros(
        (batch_size * slots, vocab_size), dtype=torch.float32, device=device
    )
    temperatures = torch.tensor([[0.7]], device=device)
    draft_probs = torch.full(
        (batch_size, slots, vocab_size),
        1.0 / vocab_size,
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.tensor([[0, 3, 5]], dtype=torch.int32, device=device)
    coins = torch.ones((batch_size, slots), dtype=torch.float32, device=device)
    final_coins = torch.tensor([0.5], dtype=torch.float32, device=device)

    predicts, _, accept_token_num = _run_sampler(
        chain_speculative_sampling_from_logits_triton,
        logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        candidates=candidates,
        coins=coins,
        final_coins=final_coins,
    )
    assert accept_token_num.tolist() == [0]
    assert predicts[0].item() == vocab_size - 1


@pytest.mark.skipif(
    not _kernel_runtime_available(), reason="requires CUDA or Triton interpreter"
)
def test_logits_sampler_rejects_when_coin_times_q_equals_p():
    device = _kernel_device()
    batch_size, slots, vocab_size = 1, 2, 2
    logits = torch.zeros(
        (batch_size * slots, vocab_size), dtype=torch.float32, device=device
    )
    temperatures = torch.ones((batch_size, 1), device=device)
    draft_probs = torch.zeros(
        (batch_size, slots, vocab_size), dtype=torch.float32, device=device
    )
    draft_probs[:, :, 0] = 1.0
    candidates = torch.zeros((batch_size, slots), dtype=torch.int32, device=device)
    # softmax([0, 0])[0] == 0.5 and q(0) == 1.0, so the reachable half-open
    # coin 0.5 exercises the exact equality boundary: coin * q == p.
    coins = torch.full((batch_size, slots), 0.5, dtype=torch.float32, device=device)
    final_coins = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    for sampler in (
        chain_speculative_sampling_triton,
        chain_speculative_sampling_from_logits_triton,
    ):
        predicts, _, accept_token_num = _run_sampler(
            sampler,
            logits=logits,
            temperatures=temperatures,
            draft_probs=draft_probs,
            candidates=candidates,
            coins=coins,
            final_coins=final_coins,
        )
        assert accept_token_num.tolist() == [0]
        assert predicts.tolist() == [1, -1]


@pytest.mark.skipif(
    not _kernel_runtime_available(), reason="requires CUDA or Triton interpreter"
)
def test_logits_sampler_handles_fully_masked_vocabulary_blocks():
    device = _kernel_device()
    batch_size, slots, vocab_size = 1, 3, 8193
    logits = torch.full(
        (batch_size * slots, vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    logits[:, [1, 8192]] = torch.tensor([0.0, 2.0], device=device)
    temperatures = torch.tensor([[0.7]], device=device)
    draft_probs = torch.zeros(
        (batch_size, slots, vocab_size), dtype=torch.float32, device=device
    )
    candidates = torch.tensor([[0, 1, 8192]], dtype=torch.int32, device=device)
    coins = torch.zeros((batch_size, slots), dtype=torch.float32, device=device)
    final_coins = torch.tensor([0.5], dtype=torch.float32, device=device)

    expected = _run_sampler(
        chain_speculative_sampling_triton,
        logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        candidates=candidates,
        coins=coins,
        final_coins=final_coins,
    )
    actual = _run_sampler(
        chain_speculative_sampling_from_logits_triton,
        logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        candidates=candidates,
        coins=coins,
        final_coins=final_coins,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Graph")
def test_logits_sampler_cuda_graph_replay_reuses_capture_allocations():
    torch.manual_seed(7)
    device = "cuda"
    batch_size, slots, vocab_size = 2, 4, 37
    logits = torch.randn(
        batch_size * slots, vocab_size, dtype=torch.float32, device=device
    )
    temperatures = torch.tensor([[0.7], [1.0]], device=device)
    draft_probs = torch.full(
        (batch_size, slots, vocab_size),
        1.0 / vocab_size,
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.tensor(
        [[0, 3, 5, 7], [0, 2, 4, 6]], dtype=torch.int32, device=device
    )
    coins = torch.zeros((batch_size, slots), dtype=torch.float32, device=device)
    final_coins = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    expected = _run_sampler(
        chain_speculative_sampling_triton,
        logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        candidates=candidates,
        coins=coins,
        final_coins=final_coins,
    )

    predicts, accept_index, accept_token_num = _outputs(batch_size, slots, device)
    retrieve_index = torch.arange(
        batch_size * slots, dtype=torch.int32, device=device
    ).view(batch_size, slots)
    kwargs = dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_index,
        retrive_next_sibling=retrieve_index,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_logits=logits,
        temperatures=temperatures,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    chain_speculative_sampling_from_logits_triton(**kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target_block_stats = chain_speculative_sampling_from_logits_triton(**kwargs)
    torch.cuda.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == allocated_after_capture
    assert target_block_stats.shape[0] == 2
    for actual_tensor, expected_tensor in zip(
        (predicts, accept_index, accept_token_num), expected
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
