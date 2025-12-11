import os
import torch
import torch.nn.functional as F

from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import prepare_logits_for_generation, sample_tokens
from src.generation.vanilla import confidence_unmasking
from src.utils import certainty_density, register


@torch.no_grad()
def wino_generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int = None,  # type: ignore
    sigma: float | None = None,
    # PC sampler
    debias: bool = False,
    clip_alpha: float | None = None,
    # wino
    wide_in_thres: float = 0.6,
    narrow_out_thres: float = 0.9,
    last_accept: torch.Tensor | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    device = frame.prompts.device
    block_indices = torch.nonzero(block_mask[0], as_tuple=False).squeeze(-1)
    block_start = block_indices[0].item()
    block_end = block_indices[-1].item() + 1
    block_length = int(block_end - block_start)

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()
    can_generate = (block_mask & transfer_index_mask).any(dim=-1)

    if not torch.any(can_generate):
        return None

    # filtered inputs
    prompts_active = frame.prompts[can_generate]
    generated_active = frame.generated_tokens[can_generate]

    # append a block of mask_token_id to input ids
    x = F.pad(
        torch.cat([prompts_active, generated_active], dim=-1),
        (0, block_length),
        value=mask_token_id,
    )

    active_batch_size, total_len = x.shape
    active_attn_mask = attention_mask[can_generate]

    # prepare attention mask & position ids
    # see figure 2(b) of the original paper for details
    active_attn_mask_ext = F.pad(active_attn_mask, (0, block_length), value=1)

    prefix_pos_ids = (torch.cumsum(active_attn_mask, dim=1) - 1) * active_attn_mask
    position_ids = torch.zeros(
        (active_batch_size, total_len), device=device, dtype=torch.long
    )
    position_ids[:, :-block_length] = prefix_pos_ids
    position_ids[:, -block_length:] = prefix_pos_ids[
        :, prompt_length + block_start : prompt_length + block_end
    ]

    final_mask = (
        active_attn_mask_ext.unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, 1, total_len, total_len)
        .bool()
        .clone()
    )

    # Apply Wino mask constraints
    # nothing attends to shadow block (except shadow block itself)
    final_mask[:, :, :-block_length, -block_length:] = False

    # shadow block attends to current block with ~eye (leave-one-out)
    r_start = total_len - block_length
    c_start = prompt_length + block_start
    final_mask[:, :, r_start:, c_start : c_start + block_length] &= ~torch.eye(
        block_length, device=device, dtype=torch.bool
    )

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
        logits[:, -block_length:],
    )

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

    current_tokens = generated_active[:, block_start:block_end]
    # for masked positions, sample_tokens' confidence already matches x0; for unmasked, gather current tokens.
    current_conf = torch.gather(p, dim=-1, index=current_tokens.unsqueeze(-1)).squeeze(
        -1
    )
    confidence = torch.where(block_mask_curr, combined_conf, current_conf)

    # Unmasking (Wide In)
    scores = torch.where(block_mask_curr, confidence, -torch.inf)
    if sigma is not None and sigma > 0:
        scores = (
            confidence
            * certainty_density(~remaining_mask, sigma=sigma)[:, block_start:block_end]
        )
    selected_indices = confidence_unmasking(
        scores=scores,
        transfer_index_mask=block_mask_curr,
        min_transfer_tokens=torch.ones(
            active_batch_size, device=scores.device, dtype=torch.long
        ),
        max_transfer_tokens=torch.clamp(
            (block_mask_curr.sum(dim=1) * 0.7).int(), min=5, max=20
        ),
        threshold=wide_in_thres,
    )

    unmask_mask_block = torch.zeros_like(block_mask_curr, dtype=torch.bool)
    for b, idx in enumerate(selected_indices):
        if idx.numel() > 0:
            unmask_mask_block[b, idx.long()] = True

    unmask_mask = torch.zeros_like(transfer_index_mask[can_generate], dtype=torch.bool)
    unmask_mask[:, block_start:block_end] = unmask_mask_block

    # Remasking (Narrow Out)
    remask_mask = torch.zeros_like(unmask_mask)
    if last_accept is not None:
        active_last_accept = last_accept[can_generate]
        # only consider samples that have unmasked at least one token
        can_remask = unmask_mask.sum(dim=1) > 1

        # use shadow block probs for already-unmasked tokens
        shadow_conf = torch.where(
            ~block_mask_curr,
            confidence,
            torch.tensor(float("inf"), device=device),
        )

        for b in range(active_batch_size):
            if not can_remask[b]:
                continue

            row_conf = shadow_conf[b]
            row_remask = row_conf < narrow_out_thres

            target_k = int(active_last_accept[b].item()) - 1
            if target_k > 0 and row_remask.sum() >= active_last_accept[b]:
                # select bottom-k to remask, only when k is valid
                _, idx = torch.topk(
                    row_conf.view(-1),
                    k=target_k,
                    largest=False,
                )
                row_remask = (
                    torch.zeros_like(row_remask)
                    .view(-1)
                    .scatter_(0, idx, True)
                    .view_as(row_remask)
                )

            remask_mask[b, block_start:block_end] = row_remask

    # construct delta
    decoded_tokens = torch.full_like(generated_active, INVALID_TOKEN_ID)
    decoded_tokens[:, block_start:block_end].masked_scatter_(
        block_mask_curr, x0[block_mask_curr]
    )

    total_mask = unmask_mask & ~remask_mask
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

    return FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=decoded_tokens,
        confidence=confidence_ext,
        probs=probs_ext,
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
    )


@register.gen_strategy("wino")
def wino_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    sigma: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    # wino
    wide_in_thres: float = 0.6,
    narrow_out_thres: float = 0.9,
    # other params
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> DecodeRecord:
    """
    Wino decoding strategy.
    """
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID", -1))
    pad_token_id = pad_token_id or int(os.environ.get("PAD_TOKEN_ID", -1))

    if -1 in [mask_token_id, pad_token_id]:
        raise ValueError(
            "mask_token_id and pad_token_id must be provided either as arguments or environment variables."
        )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).long()

    attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(model.device)

    frame = initial_frame
    deltas = []

    last_accept = torch.full(
        (input_ids.size(0),), 30, device=model.device, dtype=torch.long
    )

    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        while True:
            delta = wino_generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                attention_mask=attention_mask,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sigma=sigma,
                mask_token_id=mask_token_id,
                wide_in_thres=wide_in_thres,
                narrow_out_thres=narrow_out_thres,
                last_accept=last_accept,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )

            if delta is None:
                break

            # update last_accept based on Wide In count
            last_accept.fill_(0)
            for b in range(len(delta.transfer_index)):
                idxs = delta.transfer_index[b]
                if idxs.numel() > 0:
                    # count tokens that were mask_id in current frame (Wide In)
                    tokens_at_idxs = frame.generated_tokens[b, idxs]
                    wide_in_count = (tokens_at_idxs == mask_token_id).sum().item()
                    last_accept[b] = wide_in_count

            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
