import os
import torch
import torch.nn.functional as F

from typing import Any, Type

from src.cache import dCache
from src.frame import Frame, DecodeRecord
from src.generation.vanilla import (
    confidence_unmasking,
    generate_step,
)
from src.generation.utils import register


@register("klass")
def klass_generate(
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
    sigma: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
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
    batch_size, gen_length = frame.generated_tokens.shape

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

        eps = 1e-12
        kl_current_prev = (
            probs
            * (torch.log(probs + eps) - torch.log(prev_probs[active_seq_idx] + eps))
        ).sum(dim=-1)

        # shift kl_history and insert new KL at the end
        kl_history[active_seq_idx] = kl_history[active_seq_idx].roll(shifts=-1, dims=-1)
        kl_history[active_seq_idx, ..., -1] = kl_current_prev

        stable_mask = torch.all(kl_history[active_seq_idx] < kl_threshold, dim=-1)
        stable_transfer_mask = active_transfer_mask & stable_mask

        # case 1: select based on KL stability & confidence
        stable_transfer_index = confidence_unmasking(
            scores=scores,
            transfer_index_mask=stable_transfer_mask,
            min_transfer_tokens=0,
            threshold=threshold,
            factor=factor,
        )

        # case 2 (fallback): select based on top-k confidence
        fallback_transfer_index = confidence_unmasking(
            scores=scores,
            transfer_index_mask=active_transfer_mask,
            min_transfer_tokens=num_transfer_tokens,
            threshold=None,
            factor=None,
        )
        transfer_index = tuple(
            stable_idx if stable_idx.numel() > 0 else fallback_idx
            for stable_idx, fallback_idx in zip(
                stable_transfer_index, fallback_transfer_index
            )
        )

        return (
            transfer_index,
            {"curr_probs": probs, "active_index": active_seq_idx},
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
                sigma=sigma,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                # if no more mask tokens are left, break the loop
                break

            prev_probs[delta.extra.pop("active_index")] = delta.extra.pop("curr_probs")
            delta = delta.to(dtype=model.dtype)
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
