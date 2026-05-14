import torch
import torch.nn.functional as F
import torch.distributions as dists
from typing import Any, Callable

from src.cache import dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import (
    check_can_generate,
    get_block_causal_attention_mask,
    get_block_mask,
    get_initial_new_tokens,
    prepare_logits_for_generation,
    sample_tokens,
    register,
)
from src.utils import certainty_density, is_block_diffusion


def maybe_extend_new_block(
    frame: Frame,
    delta: FrameDelta,
    block_length: int,
    active_seq_idx: torch.Tensor,
    block_mask: torch.Tensor,
    num_transfer_tokens: int,
    stop_until_eos: bool,
    mask_token_id: int,
    eos_token_id: int | None,
    max_new_tokens: int | None,
):
    """
    Append a masked block when block diffusion generation exhausts the current block.

    The returned delta first applies the sampled token updates, then inserts a
    new masked suffix if the active sequences still need room to continue.
    """
    batch_size, gen_length = frame.generated_tokens.shape
    device = frame.generated_tokens.device

    next_frame = frame.apply_delta(delta, mask_token_id=mask_token_id)
    full_block_mask = torch.zeros(
        (batch_size, gen_length), dtype=torch.bool, device=device
    )
    full_block_mask[active_seq_idx] = block_mask
    can_continue_block = check_can_generate(
        next_frame,
        eligible_mask=full_block_mask,
        num_transfer_tokens=num_transfer_tokens,
        stop_until_eos=stop_until_eos,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )
    eos_seen = (
        (next_frame.generated_tokens == eos_token_id).any(dim=-1)
        if eos_token_id is not None
        else torch.zeros(batch_size, dtype=torch.bool, device=device)
    )
    if torch.any(can_continue_block) or (stop_until_eos and torch.all(eos_seen)):
        return delta

    append_length = block_length
    if max_new_tokens is not None:
        append_length = min(block_length, max_new_tokens - gen_length)
        if append_length <= 0:
            return delta

    full_decoded_tokens = torch.full(
        (batch_size, gen_length),
        INVALID_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    full_decoded_tokens[active_seq_idx] = delta.decoded_tokens
    full_decoded_tokens = F.pad(
        full_decoded_tokens, (0, append_length), value=mask_token_id
    )

    full_confidence = full_probs = None
    if delta.confidence is not None:
        full_confidence = torch.full(
            (batch_size, gen_length),
            -torch.inf,
            dtype=delta.confidence.dtype,
            device=device,
        )
        full_confidence[active_seq_idx] = delta.confidence
        full_confidence = F.pad(full_confidence, (0, append_length), value=-torch.inf)

    if delta.probs is not None:
        full_probs = torch.full(
            (batch_size, gen_length, delta.probs.size(-1)),
            -torch.inf,
            dtype=delta.probs.dtype,
            device=device,
        )
        full_probs[active_seq_idx] = delta.probs
        full_probs = F.pad(full_probs, (0, 0, 0, append_length), value=-torch.inf)

    insert_index = torch.full(
        (batch_size, append_length),
        gen_length,
        dtype=torch.long,
        device=device,
    )
    insert_src_index = (
        gen_length + torch.arange(append_length, dtype=torch.long, device=device)
    ).expand(batch_size, -1)

    return FrameDelta(
        transfer_index=delta.transfer_index,
        decoded_tokens=full_decoded_tokens,
        confidence=full_confidence,
        probs=full_probs,
        intermediate=delta.intermediate,
        insert_index=insert_index,
        insert_src_index=insert_src_index,
        extra=delta.extra,
    )


@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: int,
    unmasking_fn: Callable,
    block_length: int,
    max_new_tokens: int | None = None,
    attention_mask: torch.Tensor | None = None,
    cache: dCache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int = None,  # type: ignore
    eos_token_id: int | None = None,
    sigma: float | None = None,
    stop_until_eos: bool = False,
    debias: bool = False,
    clip_alpha: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    device = block_mask.device
    input_ids = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)

    input_padding_mask = (
        attention_mask
        if attention_mask is not None and attention_mask.dim() == 2
        else None
    )

    if is_block_diffusion(model):
        attention_mask = get_block_causal_attention_mask(
            seq_length=input_ids.size(-1),
            block_length=block_length,
            batch_size=batch_size,
            device=device,
            attention_mask=input_padding_mask,
        )

    if input_padding_mask is not None:
        position_ids = input_padding_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(input_padding_mask == 0, 0)
    else:
        position_ids = (
            torch.arange(
                input_ids.size(-1),
                dtype=torch.long,
                device=device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

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
    active_seq_idx = torch.nonzero(can_generate, as_tuple=True)[0]

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()

    if cache is not None:
        cache.active_seq_mask = can_generate

    if attention_mask is not None and attention_mask.size(0) == batch_size:
        attention_mask = attention_mask[active_seq_idx]
    block_mask = block_mask[active_seq_idx]
    outputs = model(
        input_ids[active_seq_idx],
        attention_mask=attention_mask,
        position_ids=position_ids[active_seq_idx],
        output_hidden_states=output_hidden_states,
        past_key_values=cache,
        use_cache=cache is not None,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)
    if cache is not None and cache.active_q_mask is not None:
        active_q_mask = cache.active_q_mask
        if active_q_mask.size(0) == batch_size:
            active_q_mask = active_q_mask[active_seq_idx]
        if model.config.model_type.lower() == "dream":
            valid_mask = active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = active_q_mask[:, prompt_length:]
        transfer_index_mask = transfer_index_mask[active_seq_idx] & valid_mask
    else:
        transfer_index_mask = transfer_index_mask[active_seq_idx]
    logits = logits[:, prompt_length:]

    # sampling tokens for all generated positions
    confidence, x0, p = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        clip_alpha=clip_alpha,
        alg=alg,
    )
    scores = confidence = torch.where(transfer_index_mask, confidence, -torch.inf)
    if sigma is not None and sigma > 0:
        scores = confidence * certainty_density(
            ~remaining_mask[active_seq_idx], sigma=sigma
        )

    # delegate token selection to strategy-specific unmasking logic
    transfer_index, extra = unmasking_fn(
        active_seq_idx=active_seq_idx,
        scores=scores,
        probs=p,
        transfer_index_mask=transfer_index_mask,
        block_mask=block_mask,
        num_transfer_tokens=num_transfer_tokens,
    )
    if len(transfer_index) != int(active_seq_idx.numel()):
        raise ValueError(
            "Transfer selector must return one index tensor per active sequence."
        )

    full_transfer_index = [
        torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    for seq_idx, index in zip(active_seq_idx.tolist(), transfer_index):
        full_transfer_index[seq_idx] = index

    delta = FrameDelta(
        transfer_index=tuple(full_transfer_index),
        decoded_tokens=torch.where(transfer_index_mask, x0, INVALID_TOKEN_ID),
        confidence=confidence,
        probs=(
            torch.where(transfer_index_mask.unsqueeze(-1), p, -torch.inf)
            if output_probs
            else None
        ),
        intermediate=Intermediate(
            hidden_states=(
                tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
                if output_hidden_states
                else tuple()
            )
        ),
        extra=extra,
    )
    if is_block_diffusion(model):
        return maybe_extend_new_block(
            frame,
            delta,
            block_length,
            active_seq_idx=active_seq_idx,
            block_mask=block_mask,
            num_transfer_tokens=num_transfer_tokens,
            stop_until_eos=stop_until_eos,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )
    return delta


def confidence_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    min_transfer_tokens: torch.Tensor | int,
    max_transfer_tokens: torch.Tensor | None = None,
    # parallel decoding
    threshold: float | torch.Tensor | None = None,
    factor: float | None = None,
    # EB sampler
    gamma: float | None = None,
    p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Select tokens to be fixed based on token probs, i.e., confidence.
    It consists of parallel decoding and low-confidence remasking.
    Args:
        token_probs: A tensor of shape [B, gen_length] containing the probabilities of each token.
        transfer_index_mask: A boolean tensor of shape [B, gen_length] indicating which tokens can be transferred.
        min_transfer_tokens: A tensor of shape [B,] indicating the minimum number of tokens to be transferred at each step.
        max_transfer_tokens: Maximum cap of shape [B,] for tokens transferred when threshold/factor select too many.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        gamma: threshold of upper bound of joint dependence error, see https://arxiv.org/pdf/2505.24857.
        p: A tensor of shape [B, gen_length, vocab_size] containing the probabilities of each token. Must be provided if using EB sampler.
    """

    if (threshold is not None) + (factor is not None) + (gamma is not None) > 1:
        raise ValueError(
            "Only one of `threshold`, `factor`, or `gamma` should be provided."
        )

    batch_size, _ = scores.shape

    if isinstance(min_transfer_tokens, int):
        min_transfer_tokens = torch.full(
            (batch_size,),
            min_transfer_tokens,
            device=scores.device,
            dtype=torch.long,
        )

    if min_transfer_tokens.numel() != batch_size:
        raise ValueError(
            "`min_transfer_tokens` must have shape (batch_size,) to match scores."
        )

    if max_transfer_tokens is not None:
        if max_transfer_tokens.numel() != batch_size:
            raise ValueError(
                "`max_transfer_tokens` must have shape (batch_size,) to match scores."
            )
    else:
        max_transfer_tokens = torch.sum(transfer_index_mask, dim=-1)
    num_transfer_tokens = torch.minimum(min_transfer_tokens, max_transfer_tokens)

    confidence = torch.where(transfer_index_mask, scores, -torch.inf)
    transfer_index = [torch.tensor([]) for _ in range(batch_size)]
    if threshold is not None or factor is not None:
        if threshold is not None:
            # 1.a select all tokens whose confidence is above the threshold
            col_indices = torch.nonzero(confidence >= threshold, as_tuple=False)[:, 1]
            counts = torch.sum(confidence >= threshold, dim=-1).cpu().tolist()
            transfer_index = list(torch.split(col_indices, counts))
            # check if there are too many tokens to be decoded in any sequence
            # in this case, we only keep the top-k tokens with highest confidence
            # to do so, we simply clear the transfer_index for those sequences and faill back to top-k selection later
            for i, t in enumerate(transfer_index):
                if t.numel() > max_transfer_tokens[i]:
                    transfer_index[i] = torch.tensor([])
                    num_transfer_tokens[i] = max_transfer_tokens[i]
        elif factor is not None:
            # 1.b unmask top-n* tokens, where n* = argmax_{n} (n + 1)(1 - nth_largest_conf) < factor
            num_unmasked_tokens = torch.sum(transfer_index_mask, dim=-1, keepdim=True)
            for i in range(batch_size):
                sorted_conf, _ = torch.sort(
                    confidence[i][transfer_index_mask[i]],
                    dim=-1,
                    descending=True,
                )
                for n in range(1, num_unmasked_tokens[i] + 1):
                    if (n + 1) * (1 - sorted_conf[n - 1]) >= factor:
                        break
                transfer_index[i] = torch.topk(confidence[i], min(n - 1, int(max_transfer_tokens[i].item())), dim=-1).indices  # type: ignore
    elif gamma is not None:
        # 1.c EB sampler: select tokens based on entropy bound
        if p is None:
            raise ValueError(
                "Probabilities of all tokens `p` must be provided for EB sampler."
            )
        _, ids = torch.sort(confidence, dim=-1, descending=True)
        entropy = torch.gather(
            dists.Categorical(probs=p.float()).entropy(), dim=-1, index=ids
        )
        acc_entropy = torch.cumsum(entropy, dim=1)
        cummax_entropy = torch.cummax(entropy, dim=0).values
        num_transfer_tokens = (acc_entropy - cummax_entropy <= gamma).sum(dim=1)

    num_transfer_tokens = torch.clamp(
        num_transfer_tokens,
        min=min_transfer_tokens,
        max=max_transfer_tokens,
    )

    if fallback_indices := [
        i for i, t in enumerate(transfer_index) if t.numel() < num_transfer_tokens[i]
    ]:
        confidence_subset = confidence[fallback_indices]
        # 2. select the tokens that have top-num_transfer_tokens highest probs as generated tokens
        topk_transfer_index = [
            torch.topk(
                confidence_subset[i],
                int(
                    torch.min(
                        transfer_index_mask[fallback_indices[i]].sum(),
                        num_transfer_tokens[fallback_indices[i]],
                    )
                ),
                dim=-1,
            ).indices
            for i in range(confidence_subset.size(0))
        ]
        source_iter = iter(topk_transfer_index)
        transfer_index = [
            next(source_iter) if i in fallback_indices else t
            for i, t in enumerate(transfer_index)
        ]

    return tuple(transfer_index)


@register("vanilla")
def vanilla_generate(
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
    # EB sampler
    gamma: float | None = None,
    # PC sampler
    debias: bool = False,
    clip_alpha: float | None = None,
    # parallel decoding
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache: dCache | None = None,
) -> DecodeRecord:
    """
    Vanilla generation for diffusion large language models.
    Args:
        model: Mask predictor.
        input_ids: Prompt token ids.
        max_new_tokens: Maximum number of tokens to generate. If None, block diffusion decoding grows until EOS.
        block_length: Block length, less than or equal to max_new_tokens. If less than max_new_tokens, it means using semi_autoregressive remasking.
        num_transfer_tokens: Minimum number of tokens to transfer per decoding step.
        temperature: Categorical distribution sampling temperature.
        mask_token_id: The token id of [MASK].
        eos_token_id: The token id of [EOS]. It must be provided if stop_until_eos is set.
        top_k: The number of highest probability tokens to keep for one generation step.
        top_p: The cumulative probability threshold for nucleus sampling.
        sigma: The standard deviation of the Gaussian kernel used in decoding with certainty prior, see https://arxiv.org/abs/2509.23094.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        output_hidden_states: Whether to return the hidden states of all decoded tokens from layers.
        output_probs: Whether to return the probs of all tokens.
    """
    assert isinstance(mask_token_id, int)
    if stop_until_eos:
        assert isinstance(eos_token_id, int)
    if max_new_tokens is None:
        assert stop_until_eos and isinstance(eos_token_id, int)

    block_aligned = is_block_diffusion(model)
    initial_frame = (
        Frame.create_initial_frame(
            input_ids,
            num_new_tokens=get_initial_new_tokens(
                input_ids.size(-1),
                block_length,
                max_new_tokens,
                block_aligned=block_aligned,
            ),
            mask_token_id=mask_token_id,
        )
        .to(device=model.device, dtype=model.dtype)
        .as_batch()
    )
    frame = initial_frame

    def unmasking_fn(
        active_seq_idx: torch.Tensor,
        scores: torch.Tensor,
        probs: torch.Tensor,
        transfer_index_mask: torch.Tensor,
        block_mask: torch.Tensor,
        num_transfer_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        return (
            # select tokens to transfer based on probs and mask
            confidence_unmasking(
                scores=scores,
                transfer_index_mask=transfer_index_mask & block_mask,
                min_transfer_tokens=num_transfer_tokens,
                threshold=threshold,
                factor=factor,
                gamma=gamma,
                p=probs,
            ),
            {},
        )

    deltas = []
    block_idx = 0

    while True:
        block_mask = get_block_mask(
            frame, block_idx, block_length, block_aligned=block_aligned
        )
        if not torch.any(block_mask):
            break
        total_length = frame.prompts.size(-1) + frame.generated_tokens.size(-1)
        if attention_mask is not None and attention_mask.size(-1) < total_length:
            attention_mask = F.pad(
                attention_mask, (0, total_length - attention_mask.size(-1)), value=1
            )

        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        block_deltas = []
        while True:
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=num_transfer_tokens,
                unmasking_fn=unmasking_fn,
                block_length=block_length,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
                cache=cache,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sigma=sigma,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                debias=debias,
                clip_alpha=clip_alpha,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                # if no more mask tokens are left, break the loop
                break
            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            prev_length = frame.generated_tokens.size(-1)
            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta, mask_token_id=mask_token_id)
            # a new block is appended, break to start the next block
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
