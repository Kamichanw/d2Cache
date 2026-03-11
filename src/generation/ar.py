import os
import torch
import torch.nn.functional as F
from typing import Any, Type

from src.cache import dCache
from src.frame import Frame, DecodeRecord
from src.generation.vanilla import (
    generate_step,
)
from src.generation.utils import register


def get_ar_prefix_mask(transfer_index_mask: torch.Tensor) -> torch.Tensor:
    """
    Restrict autoregressive decoding to the contiguous transferable span that
    immediately follows the generated prefix. This method masks out any tokens
    after the first non-transferable token.
    """
    batch_size, seq_len = transfer_index_mask.shape
    positions = torch.arange(seq_len, device=transfer_index_mask.device).expand(
        batch_size, -1
    )
    has_transferable = transfer_index_mask.any(dim=-1)
    start = torch.where(
        has_transferable,
        transfer_index_mask.int().argmax(dim=-1),
        torch.full((batch_size,), seq_len, device=transfer_index_mask.device),
    )
    after_start = positions >= start.unsqueeze(-1)
    first_invalid = (~transfer_index_mask) & after_start
    end = torch.where(
        first_invalid.any(dim=-1),
        first_invalid.int().argmax(dim=-1),
        torch.full((batch_size,), seq_len, device=transfer_index_mask.device),
    )
    return after_start & (positions < end.unsqueeze(-1))


def autoregressive_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    min_transfer_tokens: torch.Tensor | int,
    threshold: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Select tokens to fix based on confidence while preserving left-to-right order.
    Unlike vanilla confidence-based unmasking, autoregressive decoding restricts
    selection to the contiguous transferable prefix immediately following the
    generated prefix, so any multi-token update still advances sequentially.

    Args:
        scores: A tensor of shape [B, gen_length] containing token confidence.
        transfer_index_mask: A boolean tensor of shape [B, gen_length] indicating
            which contiguous prefix tokens can be transferred.
        min_transfer_tokens: A tensor of shape [B,] indicating the minimum number
            of tokens to be transferred at each step.
        threshold: A threshold for parallel decoding. If provided, all prefix
            tokens whose confidence stays above this threshold will be kept until
            the first failure.
    """
    batch_size, seq_len = scores.shape
    device = scores.device
    if isinstance(min_transfer_tokens, int):
        min_transfer_tokens = torch.full(
            (batch_size,), min_transfer_tokens, device=device, dtype=torch.long
        )
    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
    allowed_count = transfer_index_mask.sum(dim=-1)
    start = torch.where(
        allowed_count > 0,
        transfer_index_mask.int().argmax(dim=-1),
        torch.zeros(batch_size, dtype=torch.long, device=device),
    )
    gather_index = (start.unsqueeze(-1) + positions).clamp(max=seq_len - 1)
    prefix_scores = torch.gather(scores, dim=1, index=gather_index)
    valid_prefix_mask = positions < allowed_count.unsqueeze(-1)

    if threshold is not None:
        # Select the longest prefix whose confidence stays above the threshold.
        fail_mask = valid_prefix_mask & (prefix_scores < threshold)
        selected_count = torch.where(
            fail_mask.any(dim=-1),
            fail_mask.int().argmax(dim=-1),
            allowed_count,
        )
    else:
        selected_count = allowed_count

    # Preserve the scheduled minimum while staying inside the valid AR prefix.
    selected_count = torch.minimum(
        torch.maximum(min_transfer_tokens.to(allowed_count.dtype), selected_count),
        allowed_count,
    )
    selected_mask = valid_prefix_mask & (positions < selected_count.unsqueeze(-1))
    flat_indices = gather_index[selected_mask]
    split_sizes = selected_mask.sum(dim=-1).tolist()
    return tuple(torch.split(flat_indices, split_sizes))


@register("ar")
def ar_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    block_length: int = 32,
    gen_length: int = 128,
    num_transfer_tokens: int = 1,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
    # PC sampler
    debias: bool = False,
    # parallel decoding
    threshold: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    """
    Autoregressive generation for diffusion large language models.
    Args:
        model: Mask predictor.
        input_ids: A tensor of shape (B, prompt_len).
        block_length: Block length, less than or equal to gen_length. Each block
            is decoded autoregressively from left to right.
        gen_length: Generated answer length.
        num_transfer_tokens: Minimum number of tokens to transfer per decoding step.
        temperature: Categorical distribution sampling temperature.
        mask_token_id: The token id of [MASK]. It can be `None` if "MASK_TOKEN_ID" is specified in the environment variables.
        pad_token_id: The token id of [PAD].
        eos_token_id: The token id of [EOS]. It must be provided if stop_until_eos is set.
        top_k: The number of highest probability tokens to keep for one generation step.
        top_p: The cumulative probability threshold for nucleus sampling.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        output_hidden_states: Whether to return the hidden states of all decoded tokens from layers.
        output_probs: Whether to return the probs of all tokens.
    """

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore
    if stop_until_eos:
        if eos_token_id is None and os.environ.get("EOS_TOKEN_ID", None) is None:
            raise ValueError(
                "eos_token_id must be provided either as an argument or an environment variable if stop_until_eos is set to True."
            )
        eos_token_id = eos_token_id or int(os.environ.get("EOS_TOKEN_ID"))  # type: ignore

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    if num_transfer_tokens <= 0:
        raise ValueError(f"{num_transfer_tokens=} must be > 0")

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    def unmasking_fn(
        *,
        active_seq_idx: torch.Tensor,
        scores: torch.Tensor,
        probs: torch.Tensor,
        transfer_index_mask: torch.Tensor,
        block_mask: torch.Tensor,
        num_transfer_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        active_transfer_mask = transfer_index_mask & block_mask
        ar_transfer_mask = get_ar_prefix_mask(active_transfer_mask)
        return (
            autoregressive_unmasking(
                scores=scores,
                transfer_index_mask=ar_transfer_mask,
                min_transfer_tokens=num_transfer_tokens,
                threshold=threshold,
            ),
            {},
        )

    deltas = []

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
                attention_mask=attention_mask,
                past_key_values=cache,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                debias=debias,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                break
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
