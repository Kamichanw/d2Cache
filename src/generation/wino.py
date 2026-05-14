import torch
import torch.nn.functional as F

from src.cache import BlockdCache, dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import (
    check_can_generate,
    get_block_mask,
    get_initial_new_tokens,
    prepare_logits_for_generation,
    register,
    sample_tokens,
)
from src.generation.vanilla import confidence_unmasking, maybe_extend_new_block
from src.utils import certainty_density, is_block_diffusion


@torch.no_grad()
def wino_generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    num_transfer_tokens: int = 1,
    max_new_tokens: int | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int = None,  # type: ignore
    eos_token_id: int | None = None,
    sigma: float | None = None,
    stop_until_eos: bool = False,
    # PC sampler
    debias: bool = False,
    clip_alpha: float | None = None,
    # wino
    wide_in_thres: float = 0.6,
    narrow_out_thres: float = 0.9,
    num_last_wide_in: torch.Tensor = None,  # type: ignore
    generation_block_length: int | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    """
    Run one Wino decoding step with a temporary shadow block.

    The shadow block is used only for leave-one-out confidence estimation, so it
    is forwarded without KV caching and never becomes part of the committed frame.
    """
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    device = frame.prompts.device
    block_indices = torch.nonzero(block_mask[0], as_tuple=False).squeeze(-1)
    block_start = block_indices[0].item()
    block_end = block_indices[-1].item() + 1
    block_width = int(block_end - block_start)
    generation_block_length = generation_block_length or block_width

    can_generate = check_can_generate(
        frame,
        eligible_mask=block_mask,
        num_transfer_tokens=num_transfer_tokens,
        stop_until_eos=stop_until_eos,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )

    if not torch.any(can_generate):
        return None

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()

    # filtered inputs
    prompts_active = frame.prompts[can_generate]
    generated_active = frame.generated_tokens[can_generate]

    # append a block of mask_token_id to input ids
    x = F.pad(
        torch.cat([prompts_active, generated_active], dim=-1),
        (0, block_width),
        value=mask_token_id,
    )

    active_batch_size, total_len = x.shape
    active_attn_mask = attention_mask[can_generate]

    # prepare attention mask & position ids
    # see figure 2(b) of the original paper for details
    active_attn_mask_ext = F.pad(active_attn_mask, (0, block_width), value=1)

    prefix_pos_ids = (torch.cumsum(active_attn_mask, dim=1) - 1) * active_attn_mask
    position_ids = torch.zeros(
        (active_batch_size, total_len), device=device, dtype=torch.long
    )
    position_ids[:, :-block_width] = prefix_pos_ids
    position_ids[:, -block_width:] = prefix_pos_ids[
        :, prompt_length + block_start : prompt_length + block_end
    ]

    if is_block_diffusion(model):
        main_positions = torch.arange(total_len - block_width, device=device)
        shadow_positions = prompt_length + torch.arange(
            block_start, block_end, device=device
        )
        block_ids = torch.cat([main_positions, shadow_positions]).div(
            generation_block_length, rounding_mode="floor"
        )
        final_mask = (
            (block_ids[:, None] >= block_ids[None, :])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(active_batch_size, -1, -1, -1)
            .clone()
        )
        final_mask &= active_attn_mask_ext[:, None, None, :].bool()
    else:
        final_mask = (
            active_attn_mask_ext.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, 1, total_len, total_len)
            .bool()
            .clone()
        )

    # ----- apply wino mask constraints -----
    # nothing attends to shadow block (except shadow block itself)
    final_mask[:, :, :-block_width, -block_width:] = False

    # shadow block attends to current block with ~eye (leave-one-out)
    r_start = total_len - block_width
    c_start = prompt_length + block_start
    final_mask[:, :, r_start:, c_start : c_start + block_width] &= ~torch.eye(
        block_width, device=device, dtype=torch.bool
    )

    # ----- forward with the shadow block -----
    outputs = model(
        x,
        attention_mask=final_mask,
        position_ids=position_ids,
        output_hidden_states=output_hidden_states,
        use_cache=False,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)

    # combine logits (B, block_length, vocab_size): main block + shadow block
    block_mask_curr = transfer_index_mask[can_generate][:, block_start:block_end]
    combined_logits = torch.where(
        block_mask_curr.unsqueeze(-1),
        logits[:, prompt_length + block_start : prompt_length + block_end],
        logits[:, -block_width:],
    ).to(torch.float64)

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    combined_conf, x0, p = sample_tokens(
        combined_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        clip_alpha=clip_alpha,
        alg=alg,
    )

    # for masked positions, sample_tokens' confidence already matches x0; for unmasked, gather current tokens.
    current_tokens = generated_active[:, block_start:block_end]
    current_conf = torch.gather(p, dim=-1, index=current_tokens.unsqueeze(-1)).squeeze(
        -1
    )
    confidence = torch.where(block_mask_curr, combined_conf, current_conf)

    # ----- unmasking (wide in) -----
    scores = torch.where(block_mask_curr, confidence, -torch.inf)
    if sigma is not None and sigma > 0:
        scores = (
            confidence
            * certainty_density(~remaining_mask[can_generate], sigma=sigma)[
                :, block_start:block_end
            ]
        )
    selected_indices = confidence_unmasking(
        scores=scores,
        transfer_index_mask=block_mask_curr,
        min_transfer_tokens=num_transfer_tokens,
        max_transfer_tokens=torch.maximum(
            torch.clamp(
                (block_mask_curr.sum(dim=1) * 0.7).int(),
                min=5,
                max=20,  # adopted from the official implementation
            ),
            torch.full(
                (active_batch_size,),
                num_transfer_tokens,
                device=scores.device,
                dtype=torch.long,
            ),
        ),
        threshold=wide_in_thres,
    )

    unmask_mask_block = torch.zeros_like(block_mask_curr, dtype=torch.bool)
    for i, idx in enumerate(selected_indices):
        if idx.numel() > 0:
            unmask_mask_block[i, idx.long()] = True

    unmask_mask = torch.zeros_like(transfer_index_mask[can_generate], dtype=torch.bool)
    unmask_mask[:, block_start:block_end] = unmask_mask_block

    # ----- remasking (narrow out) -----
    remask_mask = torch.zeros_like(unmask_mask)
    current_wide_in = unmask_mask_block.sum(dim=1)
    num_last_wide_in[can_generate] = current_wide_in

    # only consider samples that have unmasked at least one token
    can_remask = current_wide_in > 0

    # use shadow block probs for already-unmasked tokens
    shadow_conf = torch.where(~block_mask_curr, confidence, torch.inf)

    for i in range(active_batch_size):
        if not can_remask[i]:
            continue

        row_conf = shadow_conf[i]
        row_remask = row_conf < narrow_out_thres

        target_k = max(int(current_wide_in[i].item()) - num_transfer_tokens, 0)

        if row_remask.sum() > target_k:
            if target_k <= 0:
                row_remask[:] = False
            else:
                # select bottom-k to remask, only when k is valid
                _, idx = torch.topk(row_conf.view(-1), k=target_k, largest=False)
                row_remask = (
                    torch.zeros_like(row_remask)
                    .view(-1)
                    .scatter_(0, idx, True)
                    .view_as(row_remask)
                )

        remask_mask[i, block_start:block_end] = row_remask

    # construct delta
    decoded_tokens = torch.full_like(generated_active, INVALID_TOKEN_ID)
    decoded_tokens[:, block_start:block_end].masked_scatter_(
        block_mask_curr, x0[block_mask_curr]
    )
    decoded_tokens[remask_mask] = mask_token_id

    total_mask = (
        unmask_mask | remask_mask
    )  # we need to transfer remasking tokens as well
    active_transfer_index = tuple(
        torch.nonzero(total_mask[i], as_tuple=False).squeeze(-1)
        for i in range(active_batch_size)
    )

    transfer_index_iter = iter(active_transfer_index)
    transfer_index = tuple(
        (
            next(transfer_index_iter)
            if is_active
            else torch.tensor([], dtype=torch.long, device=device)
        )
        for is_active in can_generate
    )

    confidence_ext = torch.full_like(
        generated_active, -torch.inf, dtype=confidence.dtype
    )
    confidence_ext[:, block_start:block_end].masked_scatter_(
        block_mask_curr, confidence[block_mask_curr]
    )
    confidence_ext[remask_mask] = 1.0

    probs_ext = None
    if output_probs:
        probs_ext = torch.full(
            (*generated_active.shape, p.size(-1)),
            -torch.inf,
            device=device,
            dtype=p.dtype,
        )
        probs_ext[:, block_start:block_end] = torch.where(
            block_mask_curr.unsqueeze(-1), p, -torch.inf
        )
        dummy_probs = torch.zeros((p.size(-1),), device=device, dtype=p.dtype)
        dummy_probs[mask_token_id] = 1.0
        probs_ext[remask_mask] = dummy_probs.unsqueeze(0)

    delta = FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=decoded_tokens,
        confidence=confidence_ext,
        probs=probs_ext,
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
        extra=dict(num_last_wide_in=num_last_wide_in),
    ).to(model.dtype)
    if is_block_diffusion(model):
        active_seq_idx = torch.nonzero(can_generate, as_tuple=True)[0]
        return maybe_extend_new_block(
            frame,
            delta,
            generation_block_length,
            active_seq_idx=active_seq_idx,
            block_mask=block_mask[can_generate],
            num_transfer_tokens=num_transfer_tokens,
            stop_until_eos=stop_until_eos,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
    return delta


@register("wino")
def wino_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int | None,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    block_length: int = 32,
    num_transfer_tokens: int = 1,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    sigma: float | None = None,
    mask_token_id: int = None,  # type: ignore
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
    # wino
    wide_in_thres: float = 0.6,
    narrow_out_thres: float = 0.9,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache: dCache | None = None,
) -> DecodeRecord:
    """
    Wino decoding strategy.
    """
    assert isinstance(mask_token_id, int)
    if stop_until_eos:
        assert isinstance(eos_token_id, int)
    assert attention_mask is not None
    if max_new_tokens is None:
        assert stop_until_eos and isinstance(eos_token_id, int)

    block_aligned = is_block_diffusion(model)
    initial_frame = Frame.create_initial_frame(
        input_ids,
        num_new_tokens=get_initial_new_tokens(
            input_ids.size(-1),
            block_length,
            max_new_tokens,
            block_aligned=block_aligned,
        ),
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)
    initial_frame = initial_frame.as_batch()
    batch_size = initial_frame.generated_tokens.size(0)
    attention_mask = attention_mask.to(model.device)

    frame = initial_frame
    deltas = []

    block_idx = 0
    while True:
        block_mask = get_block_mask(
            frame, block_idx, block_length, block_aligned=block_aligned
        )
        if not torch.any(block_mask):
            break
        total_length = frame.prompts.size(-1) + frame.generated_tokens.size(-1)
        if attention_mask.size(-1) < total_length:
            attention_mask = F.pad(
                attention_mask,
                (0, total_length - attention_mask.size(-1)),
                value=1,
            )

        num_last_wide_in = torch.full(
            (batch_size,), 30, device=model.device, dtype=torch.long
        )

        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        block_deltas = []
        while True:
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            delta = wino_generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                attention_mask=attention_mask,
                num_transfer_tokens=num_transfer_tokens,
                max_new_tokens=max_new_tokens,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sigma=sigma,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                mask_token_id=mask_token_id,
                wide_in_thres=wide_in_thres,
                narrow_out_thres=narrow_out_thres,
                num_last_wide_in=num_last_wide_in,
                generation_block_length=block_length,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )

            if delta is None:
                break

            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            # update num_last_wide_in based on Wide In count
            num_last_wide_in = delta.extra.pop("num_last_wide_in")

            prev_length = frame.generated_tokens.size(-1)
            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta, mask_token_id=mask_token_id)
            if frame.generated_tokens.size(-1) > prev_length:
                break

        if cache is not None:
            cache.on_block_end(block_mask, start_frame, block_deltas)

        deltas.extend(block_deltas)
        block_idx += 1

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
