import torch

from typing import Any

from src.cache import dCache
from src.frame import Frame, DecodeRecord
from src.generation.vanilla import (
    confidence_unmasking,
    generate_step,
)
from src.generation.utils import (
    get_block_mask,
    get_initial_new_tokens,
    register,
)
from src.utils import is_block_diffusion


@register("klass")
def klass_generate(
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
    # klass
    kl_threshold: float = 0.01,
    kl_history_length: int = 2,
    # parallel decoding
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache: dCache | None = None,
) -> DecodeRecord:
    """
    KLASS generation strategy: KL-Adaptive Stability Sampling.
    """

    assert isinstance(mask_token_id, int)
    if stop_until_eos:
        assert isinstance(eos_token_id, int)
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
    batch_size, gen_length = initial_frame.generated_tokens.shape
    frame = initial_frame

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

    block_idx = 0

    while True:
        block_mask = get_block_mask(
            frame, block_idx, block_length, block_aligned=block_aligned
        )
        if not torch.any(block_mask):
            break
        total_length = frame.prompts.size(-1) + frame.generated_tokens.size(-1)
        if attention_mask is not None and attention_mask.size(-1) < total_length:
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, total_length - attention_mask.size(-1)), value=1
            )

        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(model, block_mask, frame)
        block_deltas = []
        while True:
            if cache is not None:
                cache.on_step_start(model, block_mask, frame)
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
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                # if no more mask tokens are left, break the loop
                break

            prev_probs[delta.extra.pop("active_index")] = delta.extra.pop("curr_probs")
            delta = delta.to(dtype=model.dtype)
            if cache is not None:
                cache.on_step_end(model, block_mask, frame, delta)

            prev_length = frame.generated_tokens.size(-1)
            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta, mask_token_id=mask_token_id)
            new_length = frame.generated_tokens.size(-1)
            if new_length > prev_length:
                kl_history = torch.cat(
                    [
                        kl_history,
                        torch.zeros(
                            (batch_size, new_length - prev_length, kl_history_length),
                            dtype=kl_history.dtype,
                            device=kl_history.device,
                        ),
                    ],
                    dim=1,
                )
                prev_probs = torch.cat(
                    [
                        prev_probs,
                        torch.zeros(
                            (
                                batch_size,
                                new_length - prev_length,
                                model.config.vocab_size,
                            ),
                            dtype=prev_probs.dtype,
                            device=prev_probs.device,
                        ),
                    ],
                    dim=1,
                )
                break

        if cache is not None:
            cache.on_block_end(
                model,
                block_mask,
                start_frame,
                block_deltas,
            )

        deltas.extend(block_deltas)
        block_idx += 1

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
