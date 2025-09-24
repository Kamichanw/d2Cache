import os
import torch
import torch.nn.functional as F

from typing import Type

from src.cache import d2Cache
from src.frame import Frame, FrameDelta, DecodeRecord, Intermediate, INVALID_TOKEN_ID
from src.generation.utils import prepare_logits_for_generation
from src.generation.vanilla import sample_tokens, confidence_unmasking
from src.third_party import get_token_freq
from src.utils import register, certainty_density

_token_freq: torch.Tensor | None = None


@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    num_transfer_tokens: int | torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: d2Cache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    debias: bool = False,
    mask_token_id: int = None,  # type: ignore
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
            device=model.device,
            dtype=torch.long,
        )

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()
    can_generate = transfer_index_mask.any(dim=-1)
    if not torch.any(can_generate):
        return None

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
    attention_mask = (
        attention_mask[can_generate] if attention_mask is not None else None
    )
    num_transfer_tokens = (
        num_transfer_tokens[can_generate]
        if isinstance(num_transfer_tokens, torch.Tensor)
        else num_transfer_tokens
    )
    outputs = model(
        x,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
    )
    # main difference (1) with LLaDA, see https://github.com/DreamLM/Dream/issues/31
    logits = prepare_logits_for_generation(model, outputs.logits)
    if past_key_values is not None and past_key_values.active_q_mask is not None:
        if model.config.model_type.lower() == "dream":
            valid_mask = past_key_values.active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = past_key_values.active_q_mask[:, prompt_length:]
        transfer_index_mask &= valid_mask
    logits = logits[:, prompt_length:][transfer_index_mask].view(
        batch_size, -1, logits.size(-1)
    )

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    if debias:
        global _token_freq
        _token_freq = get_token_freq(model, device=logits.device, dtype=logits.dtype)

    # sampling tokens for all generated positions
    if alg == "maskgit_plus":
        # same as LLaDA
        confidence, x0, p = sample_tokens(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, debias=debias
        )
    elif alg == "topk_margin":
        # main difference (2) with LLaDA, the confidence may not be the token probabilities
        confidence, x0, p = sample_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=True,
            debias=debias,
        )
    elif alg == "entropy":
        confidence, x0, p = sample_tokens(
            logits,
            temperature,
            top_p=top_p,
            top_k=top_k,
            neg_entropy=True,
            debias=debias,
        )
    else:
        raise RuntimeError(f"Unknown alg: {alg}")

    if past_key_values is not None:
        sigma = past_key_values.sigma
    else:
        sigma = 10.0
    confidence = torch.full_like(
        frame.generated_tokens, -float("inf"), dtype=confidence.dtype
    ).masked_scatter_(transfer_index_mask, confidence)

    density = certainty_density(~remaining_mask, sigma=sigma)
    # select tokens to transfer based on probs and mask
    transfer_index = confidence_unmasking(
        scores=confidence * density,
        transfer_index_mask=transfer_index_mask[can_generate],
        num_transfer_tokens=num_transfer_tokens,
        threshold=threshold,
        factor=factor,
    )
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
        decoded_tokens=torch.full_like(
            frame.generated_tokens, INVALID_TOKEN_ID
        ).masked_scatter_(transfer_index_mask, x0),
        confidence=confidence,
        probs=p if output_probs else None,
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
    )
    return delta.unbatch() if not frame.is_batched else delta


@register.gen_strategy("dyna")
def dyna_generate(
    model,
    input_ids: torch.Tensor,
    alg: str = "maskgit_plus",
    gen_length: int = 128,
    steps: int = 64,
    temperature: float = 0.0,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    debias: bool = False,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[d2Cache] | None = None,
    tokenizer=None,
) -> DecodeRecord:
    """
    Vanilla generation for diffusion large language models.
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        steps: Sampling steps, less than or equal to gen_length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_token_id: The token id of [MASK]. It can be `None` if "MASK_TOKEN_ID" is specified in the environment variables.
        top_k: The number of highest probability tokens to keep for one generation step.
        top_p: The cumulative probability threshold for nucleus sampling.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        output_hidden_states: Whether to return the hidden states of all decoded tokens from layers.
        output_probs: Whether to return the probs of all tokens.
    """

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore
    num_transfer_tokens = gen_length // steps
    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)
    if pad_token_id is not None:
        attention_mask = torch.cat(
            [
                (input_ids != pad_token_id).long().to(device=model.device),
                torch.ones(input_ids.size(0), gen_length, device=model.device).long(),
            ],
            dim=-1,
        )
    else:
        attention_mask = None
    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    deltas = []

    if cache is not None:
        cache.on_block_start(torch.ones_like(frame.generated_tokens), frame)
    while True:

        if cache is not None:
            cache.on_step_start(None, frame)
        delta = generate_step(
            model=model,
            frame=frame,
            num_transfer_tokens=num_transfer_tokens,
            attention_mask=attention_mask,
            past_key_values=cache,
            alg=alg,
            temperature=temperature,
            mask_token_id=mask_token_id,
            top_p=top_p,
            top_k=top_k,
            debias=debias,
            threshold=threshold,
            factor=factor,
            output_hidden_states=output_hidden_states,
            output_probs=output_probs,
        )
        if delta is None:
            # if no more mask tokens are left, break the loop
            break
        if cache is not None:
            cache.on_step_end(None, frame, delta)
        frame = frame.apply_delta(delta)
        deltas.append(delta.to("cpu"))

    if cache is not None:
        cache.on_block_end(
            torch.ones_like(frame.generated_tokens), initial_frame, deltas
        )

    return DecodeRecord(initial_frame=initial_frame.to("cpu"), deltas=deltas)
