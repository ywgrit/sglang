import torch
import triton
import triton.language as tl

_LOGITS_STATS_BLOCK_SIZE = 4096


@triton.jit
def _temperature_logits_block_stats_kernel(
    TargetLogits,
    Temperatures,
    BlockMax,
    BlockSumexp,
    stride_logits_row,
    stride_logits_v,
    stride_temp_b,
    stride_stats_row,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    v_offsets = block_idx * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE

    temperature = tl.load(Temperatures + (row_idx // NUM_SLOTS) * stride_temp_b)
    logits = tl.load(
        TargetLogits + row_idx * stride_logits_row + v_offsets * stride_logits_v,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    logits = logits / temperature
    block_max = tl.max(logits, axis=0)
    block_sumexp = tl.sum(tl.exp(logits - block_max), axis=0)
    # Grammar masks can make a complete vocabulary block -inf. Avoid letting
    # the resulting (-inf) - (-inf) contaminate otherwise valid row statistics.
    block_sumexp = tl.where(block_max == float("-inf"), 0.0, block_sumexp)
    stats_offset = row_idx * stride_stats_row + block_idx
    tl.store(BlockMax + stats_offset, block_max)
    tl.store(BlockSumexp + stats_offset, block_sumexp)


@triton.jit
def _global_lse_from_block_stats(
    BlockMax,
    BlockSumexp,
    row_idx,
    stride_stats_row,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_STATS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_STATS)
    mask = offsets < NUM_BLOCKS
    stats_offsets = row_idx * stride_stats_row + offsets
    block_max = tl.load(BlockMax + stats_offsets, mask=mask, other=float("-inf"))
    global_max = tl.max(block_max, axis=0)
    block_sumexp = tl.load(BlockSumexp + stats_offsets, mask=mask, other=0.0)
    global_sumexp = tl.sum(block_sumexp * tl.exp(block_max - global_max), axis=0)
    return global_max + tl.log(global_sumexp)


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetValues,
    DraftProbs,
    Temperatures,
    TargetBlockMax,
    TargetBlockSumexp,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    stride_temp_b,
    stride_stats_row,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_STATS_BLOCKS: tl.constexpr,
    BLOCK_STATS: tl.constexpr,
    TARGET_IS_LOGITS: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        if TARGET_IS_LOGITS:
            stats_row = pid * NUM_SLOTS + cur_prob_row
            target_lse = _global_lse_from_block_stats(
                TargetBlockMax,
                TargetBlockSumexp,
                stats_row,
                stride_stats_row,
                NUM_STATS_BLOCKS,
                BLOCK_STATS,
            )
            temperature = tl.load(Temperatures + pid * stride_temp_b)
            target_logit = tl.load(TargetValues + offset_prob).to(tl.float32)
            p = tl.exp(target_logit / temperature - target_lse)
        else:
            p = tl.load(TargetValues + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetValues + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    if TARGET_IS_LOGITS:
        stats_row = pid * NUM_SLOTS + cur_prob_row
        target_lse = _global_lse_from_block_stats(
            TargetBlockMax,
            TargetBlockSumexp,
            stats_row,
            stride_stats_row,
            NUM_STATS_BLOCKS,
            BLOCK_STATS,
        )
        temperature = tl.load(Temperatures + pid * stride_temp_b)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        if TARGET_IS_LOGITS:
            target_logit = tl.load(p_ptr, mask=mask, other=float("-inf")).to(tl.float32)
            p_val = tl.exp(target_logit / temperature - target_lse)
        else:
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
            # Treat NaN q (degenerate draft rows) as 0: residual falls back to p.
            q_val = tl.where(q_val == q_val, q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            if TARGET_IS_LOGITS:
                target_logit = tl.load(p_ptr, mask=mask, other=float("-inf")).to(
                    tl.float32
                )
                p_val = tl.exp(target_logit / temperature - target_lse)
            else:
                p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same NaN-q guard as pass 1.
                q_val = tl.where(q_val == q_val, q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def _launch_chain_speculative_sampling(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_values,
    draft_probs,
    temperatures=None,
    target_block_stats=None,
    target_is_logits=False,
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_values.shape[-1]

    if target_is_logits:
        stride_tp_b = target_values.stride(0) * num_slots
        stride_tp_s = target_values.stride(0)
        stride_tp_v = target_values.stride(1)
        block_max = target_block_stats[0]
        block_sumexp = target_block_stats[1]
        num_stats_blocks = block_max.shape[1]
        stride_stats_row = block_max.stride(0)
        stride_temp_b = temperatures.stride(0)
    else:
        stride_tp_b, stride_tp_s, stride_tp_v = target_values.stride()
        temperatures = target_values
        block_max = target_values
        block_sumexp = target_values
        num_stats_blocks = 1
        stride_stats_row = 0
        stride_temp_b = 0

    speculative_sampling_classic_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_values,
        draft_probs,
        temperatures,
        block_max,
        block_sumexp,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        stride_temp_b,
        stride_stats_row,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        NUM_STATS_BLOCKS=num_stats_blocks,
        BLOCK_STATS=triton.next_power_of_2(num_stats_blocks),
        TARGET_IS_LOGITS=target_is_logits,
    )


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    _launch_chain_speculative_sampling(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_values=target_probs,
        draft_probs=draft_probs,
    )


def chain_speculative_sampling_from_logits_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_logits,
    temperatures,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    batch_size, num_slots = candidates.shape
    num_rows, vocab_size = target_logits.shape
    assert num_rows == batch_size * num_slots
    assert target_logits.stride(1) == 1
    assert target_logits.dtype == torch.float32

    num_stats_blocks = triton.cdiv(vocab_size, _LOGITS_STATS_BLOCK_SIZE)
    target_block_stats = target_logits.new_empty((2, num_rows, num_stats_blocks))
    block_max = target_block_stats[0]
    block_sumexp = target_block_stats[1]
    _temperature_logits_block_stats_kernel[(num_rows, num_stats_blocks)](
        target_logits,
        temperatures,
        block_max,
        block_sumexp,
        target_logits.stride(0),
        target_logits.stride(1),
        temperatures.stride(0),
        block_max.stride(0),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=_LOGITS_STATS_BLOCK_SIZE,
        num_warps=8,
    )
    _launch_chain_speculative_sampling(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_values=target_logits,
        draft_probs=draft_probs,
        temperatures=temperatures,
        target_block_stats=target_block_stats,
        target_is_logits=True,
    )
    return target_block_stats
