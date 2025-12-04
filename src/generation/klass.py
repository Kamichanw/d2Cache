import os
import torch
import torch.nn.functional as F

from typing import Type

from src.cache import dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import prepare_logits_for_generation, sample_tokens
from src.utils import certainty_density, register
from src.third_party import get_token_freq
from src.generation.vanilla import get_num_transfer_tokens, confidence_unmasking


@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor | int,
    # klass
    prev_probs: torch.Tensor,
    kl_history: torch.Tensor,
    kl_threshold: float = 0.02,
    attention_mask: torch.Tensor | None = None,
    past_key_values: dCache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int = None,  # type: ignore
    sigma: float | None = None,
    # parallel decoding
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    """
    Generate a single step from a given frame. If all mask tokens have been filled, return None.
    For `num_transfer_tokens`, a generation step selects a fixed number of tokens to transfer.
    For `threshold`, a generation step selects all tokens whose confidence is above the threshold.

    Args:
        model: Mask predictor.
        frame: The current frame containing the prompt and generated tokens.
        block_mask: A mask of shape [B, gen_length] indicating which block of tokens are being processed.
        num_transfer_tokens: The number of tokens to transfer at this step. It can be a tensor with shape (batch_size,) or
            a single integer that represents the number of tokens to transfer for all batches.
        temperature: Categorical distribution sampling temperature.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        output_hidden_states: Whether to return the hidden states of all decoded tokens.
        output_probs: Whether to return the probabilities of the decoded tokens.
    """
    if threshold is not None and factor is not None:
        raise ValueError(
            "Only one of `threshold` or `factor` can be specified, not both."
        )

    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape

    if isinstance(num_transfer_tokens, torch.Tensor):
        if num_transfer_tokens.numel() != batch_size or num_transfer_tokens.dim() != 1:
            raise ValueError(
                f"`num_transfer_tokens` must be a tensor of shape ({batch_size},) or a single integer, "
                f"but got shape of {num_transfer_tokens.shape}."
            )
    else:
        num_transfer_tokens = torch.full(
            (batch_size,),
            num_transfer_tokens,
            device=block_mask.device,
            dtype=torch.long,
        )

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()
    can_generate = (block_mask & transfer_index_mask).any(dim=-1)
    if not torch.any(can_generate):
        return None
    # skip sequence that doesn't require to generate
    can_generate &= num_transfer_tokens > 0

    if past_key_values is not None:
        past_key_values.active_seq_mask = can_generate

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
    attention_mask = (
        attention_mask[can_generate] if attention_mask is not None else None
    )
    num_transfer_tokens = num_transfer_tokens[can_generate]
    outputs = model(
        x,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)
    if past_key_values is not None and past_key_values.active_q_mask is not None:
        if model.config.model_type.lower() == "dream":
            valid_mask = past_key_values.active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = past_key_values.active_q_mask[:, prompt_length:]
        transfer_index_mask[can_generate].logical_and_(valid_mask)
    logits = logits[:, prompt_length:].to(torch.float64)

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    # sampling tokens for all generated positions
    confidence, x0, p = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
    )
    scores = confidence = torch.where(
        transfer_index_mask[can_generate], confidence, -torch.inf
    )
    if sigma is not None and sigma > 0:
        scores = confidence * certainty_density(~remaining_mask, sigma=sigma)

    # ----------------------------- Klass-specific logic -----------------------------
    eps = 1e-12
    kl_current_prev = (p * (torch.log(p + eps) - torch.log(prev_probs + eps))).sum(
        dim=-1
    )
    # Shift kl_history and insert new KL at the end
    active_index = torch.nonzero(can_generate, as_tuple=True)[0]
    kl_history[active_index] = kl_history[active_index].roll(shifts=-1, dims=-1)
    kl_history[active_index, ..., -1] = kl_current_prev[active_index]

    stable_mask = torch.zeros_like(transfer_index_mask)
    stable_mask[active_index] = torch.all(
        kl_history[active_index] < kl_threshold, dim=-1
    )
    failback_transfer_mask = transfer_index_mask & block_mask
    stable_transfer_mask = failback_transfer_mask & stable_mask

    # Case 1: select based on KL stability & confidence
    ta = confidence_unmasking(
        scores=scores,
        transfer_index_mask=stable_transfer_mask[can_generate],
        num_transfer_tokens=torch.zeros_like(num_transfer_tokens),  # disable fallback
        threshold=threshold,
        factor=factor,
    )

    # Case 2 (failback): select based on top-k confidence
    tb = confidence_unmasking(
        scores=scores,
        transfer_index_mask=failback_transfer_mask[can_generate],
        num_transfer_tokens=num_transfer_tokens,
        threshold=None,  # disable parallel decoding
        factor=None,
    )

    transfer_index = tuple(
        ta_idx if ta_idx.numel() > 0 else tb_idx for ta_idx, tb_idx in zip(ta, tb)
    )

    # ----------------------------- Klass-specific end -----------------------------
    transfer_index_iter = iter(transfer_index)
    transfer_index = tuple(
        (
            next(transfer_index_iter)
            if is_not_finished
            else torch.tensor([], dtype=torch.long, device=x0.device)
        )
        for is_not_finished in can_generate
    )

    delta = FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=torch.where(
            transfer_index_mask[can_generate], x0, INVALID_TOKEN_ID
        ),
        confidence=confidence,
        probs=(
            torch.where(transfer_index_mask[can_generate].unsqueeze(-1), p, -torch.inf)
            if output_probs
            else None
        ),
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
        extra=dict(curr_probs=p, active_index=active_index),
    ).to(model.dtype)
    return delta.unbatch() if not frame.is_batched else delta


@register.gen_strategy("klass")
def klass_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 128,
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    sigma: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    # klass
    kl_threshold: float = 0.01,
    kl_history_length: int = 2,
    # parallel decoding
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    """
    KLASS generation strategy: KL-Adaptive Stability Sampling.
    """

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask is not None:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame
    batch_size, prompt_length = input_ids.shape

    deltas = []
    kl_history = torch.zeros(
        (batch_size, gen_length, kl_history_length),
        dtype=torch.float64,
        device=model.device,
    )
    prev_probs = torch.zeros(
        (batch_size, gen_length, model.config.vocab_size),
        dtype=torch.float64,
        device=model.device,
    )

    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (batch_size, gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps)
        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        block_deltas = []
        for i in range(steps):
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=num_transfer_tokens[:, i],
                prev_probs=prev_probs,
                kl_history=kl_history,
                kl_threshold=kl_threshold,
                attention_mask=attention_mask,
                past_key_values=cache,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sigma=sigma,
                mask_token_id=mask_token_id,
                threshold=threshold,
                factor=factor,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                # if no more mask tokens are left, break the loop
                break
            prev_probs[delta.extra["active_index"]] = delta.extra["curr_probs"].to(torch.float64)
            delta.extra.pop("curr_probs")
            delta.extra.pop("active_index")
            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

        if cache is not None:
            cache.on_block_end(block_mask, start_frame, block_deltas)

        deltas.extend(block_deltas)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
