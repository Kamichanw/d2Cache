import os
import torch

from src.frame import Frame, DecodeRecord, FrameDelta
from src.generation.vanilla import confidence_unmasking, generate_step
from src.generation.utils import prepare_logits_for_generation, register


def calculate_eos_conf(
    gen_probs: torch.Tensor, num_check_eos_tokens: int, eos_token_id: int
):
    """
    Calculate the average confidence of the last k EOS tokens.

    Args:
        gen_probs: Probs tensor of shape (batch_size, sequence_length, vocab_size).
        num_check_eos_tokens: The number (k) of last EOS tokens to check.
        eos_token_id: The token ID for the EOS token.

    Returns:
        A tensor of shape (batch_size,) with the average confidences.
    """
    x0 = torch.argmax(gen_probs, dim=-1)

    eos_mask_reversed = torch.flip(x0 == eos_token_id, dims=[1])
    eos_counts_reversed = torch.cumsum(eos_mask_reversed.int(), dim=1)

    final_mask = torch.flip(
        (eos_counts_reversed <= num_check_eos_tokens) & eos_mask_reversed, dims=[1]
    )

    return (
        torch.sum(gen_probs[:, :, eos_token_id] * final_mask, dim=1)
        / num_check_eos_tokens
    )


@register("daedal")
def daedal_generate(
    model,
    input_ids: torch.Tensor,
    alg: str = "maskgit_plus",
    block_length: int = 32,
    initial_gen_length: int = 128,
    max_gen_length: int = 2048,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    initial_eos_expand_thres=0.5,
    decode_eos_expand_thres=0.9,
    low_conf_expand_thres: float = 0.1,
    num_check_last_eos: int = 32,
    expansion_factor: int = 8,
    threshold: float = 0.9,
    factor: float | None = None,
    output_hidden_states: bool = False,
) -> DecodeRecord:
    """
    DAEDAL generation for LLaDA, see https://arxiv.org/abs/2508.00819.
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_token_id: The token id of [MASK].
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        output_hidden_states: Whether to return the hidden states of all decoded tokens from layers.
        output_probs: Whether to return the probs of all tokens.
    """

    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID", -1))
    pad_token_id = pad_token_id or int(os.environ.get("PAD_TOKEN_ID", -1))
    eos_token_id = eos_token_id or int(os.environ.get("EOS_TOKEN_ID", -1))

    if -1 in [mask_token_id, pad_token_id, eos_token_id]:
        raise ValueError(
            "mask_token_id, pad_token_id, and eos_token_id must be provided either as arguments or environment variables."
        )

    batch_size, prompt_length = input_ids.shape
    gen_lengths = torch.full(
        (batch_size,),
        initial_gen_length,
        dtype=torch.long,
        device=model.device,
    )

    # stage 1: Initial Length Adjustment
    frame = Frame.create_initial_frame(
        input_ids,
        gen_length=initial_gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)
    while True:
        x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)
        with torch.no_grad():
            logits = model(x, attention_mask=(x != pad_token_id).long()).logits
        logits = prepare_logits_for_generation(model, logits)
        need_expand = (
            calculate_eos_conf(
                logits[:, prompt_length:].softmax(dim=-1),
                num_check_last_eos,
                eos_token_id,
            )
            < initial_eos_expand_thres
        )
        if not need_expand.any():
            # all sequences have expanded to adequate length
            break
        gen_lengths = torch.clamp_max(
            gen_lengths + expansion_factor * need_expand, max_gen_length
        )
        if (gen_lengths == max_gen_length).all():
            # all sequences have reached max length
            break
        frame = Frame.create_initial_frame(
            frame.prompts, int(gen_lengths.max()), mask_token_id
        ).to(device=model.device, dtype=model.dtype)

    initial_frame = frame.clone()

    def unmasking_fn(
        *,
        active_seq_idx: torch.Tensor,
        scores: torch.Tensor,
        probs: torch.Tensor,
        transfer_index_mask: torch.Tensor,
        block_mask: torch.Tensor,
        num_transfer_tokens: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], dict]:
        return (
            confidence_unmasking(
                scores=scores,
                transfer_index_mask=transfer_index_mask & block_mask,
                min_transfer_tokens=num_transfer_tokens,
                threshold=threshold,
                factor=factor,
                p=probs,
            ),
            {"active_index": active_seq_idx},
        )

    # stage 2: Iterative Denoising and Mask Insertion
    deltas = []
    row_indices = torch.arange(batch_size, dtype=torch.long, device=model.device)
    prompt_attn_mask = (frame.prompts != pad_token_id).long()
    block_idx = 0
    while block_idx * block_length < gen_lengths.max().item():
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=model.device)
        block_deltas = []

        while not finished.all():
            block_mask = torch.zeros(
                (batch_size, int(gen_lengths.max())),
                dtype=torch.bool,
                device=model.device,
            )
            block_mask[
                ~finished, block_idx * block_length : (block_idx + 1) * block_length
            ] = True
            attention_mask = torch.cat(
                [prompt_attn_mask, torch.ones_like(block_mask, dtype=torch.long)], dim=1
            )
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=1,
                unmasking_fn=unmasking_fn,
                attention_mask=attention_mask,
                alg=alg,
                temperature=temperature,
                mask_token_id=mask_token_id,
                top_p=top_p,
                top_k=top_k,
                output_hidden_states=output_hidden_states,
                output_probs=True,
            )
            if delta is None:
                break
            assert delta.probs is not None and delta.confidence is not None
            active_index = delta.extra.pop("active_index")

            full_decoded_tokens = frame.generated_tokens.clone()
            full_decoded_tokens[active_index] = delta.decoded_tokens
            if frame.confidence is None:
                raise RuntimeError("DAEDAL requires frame confidence to be tracked.")
            full_confidence = frame.confidence.clone()
            full_confidence[active_index] = delta.confidence

            full_probs = torch.full(
                (batch_size, int(gen_lengths.max()), delta.probs.size(-1)),
                -torch.inf,
                device=model.device,
                dtype=delta.probs.dtype,
            )
            full_probs[active_index] = delta.probs

            # select sequences with 1) low eos confidence and 2) not exceeding max length
            # (B,)
            need_expand = torch.zeros(
                (batch_size,), dtype=torch.bool, device=model.device
            )
            need_expand[active_index] = (
                calculate_eos_conf(delta.probs, num_check_last_eos, eos_token_id)
                < decode_eos_expand_thres
            ) & (gen_lengths[active_index] < max_gen_length)
            # select tokens with 1) is a valid masked token and 2) low confidence
            # (B,)
            masked_confidence = torch.full(
                (batch_size, int(gen_lengths.max())),
                torch.inf,
                device=model.device,
                dtype=delta.confidence.dtype,
            )
            masked_confidence[active_index] = torch.where(
                block_mask[active_index]
                & frame.generated_tokens[active_index].eq(mask_token_id)
                & delta.confidence.less(low_conf_expand_thres),
                delta.confidence,
                torch.inf,
            )
            expand_indices = masked_confidence.argmin(dim=-1)
            need_expand &= masked_confidence[row_indices, expand_indices].isfinite()
            if need_expand.any():
                gen_lengths = torch.clamp_max(
                    gen_lengths + need_expand * (expansion_factor - 1), max_gen_length
                )
                expanded_generated_tokens = torch.full(
                    (batch_size, int(gen_lengths.max())),
                    eos_token_id,
                    dtype=torch.long,
                    device=model.device,
                )
                expanded_confidence = torch.full_like(
                    expanded_generated_tokens, -torch.inf, dtype=masked_confidence.dtype
                )
                transfer_index = list(delta.transfer_index)
                transfer_src_index = list(delta.transfer_index)
                insert_index, insert_src_index = [], []
                # copy from previous decoded tokens
                for i in range(batch_size):
                    if need_expand[i]:
                        transfer_src_index[i] = torch.where(
                            transfer_src_index[i] > expand_indices[i],
                            transfer_src_index[i] + (expansion_factor - 1),
                            transfer_src_index[i],
                        )
                        # add expand index to both transfer_index and transfer_src_index
                        transfer_src_index[i] = (
                            torch.cat(
                                [transfer_src_index[i], expand_indices[i : i + 1]]
                            )
                            .sort()
                            .values
                        )
                        transfer_index[i] = (
                            torch.cat([transfer_index[i], expand_indices[i : i + 1]])
                            .sort()
                            .values
                        )
                        insert_index.append(
                            torch.full(
                                (expansion_factor - 1,),
                                int(expand_indices[i]),
                                device=model.device,
                            )
                        )
                        insert_src_index.append(
                            torch.arange(
                                int(expand_indices[i]) + 1,
                                int(expand_indices[i]) + expansion_factor,
                                device=model.device,
                            )
                        )
                        # copy the part before expand_indices[i]
                        expanded_generated_tokens[i, : expand_indices[i]] = (
                            full_decoded_tokens[i, : expand_indices[i]]
                        )
                        expanded_confidence[i, : expand_indices[i] + 1] = (
                            full_confidence[i, : expand_indices[i] + 1]
                        )  # keep the confidence of the expanded token
                        # copy the part after expand_indices[i]
                        expanded_generated_tokens[
                            i, expand_indices[i] + expansion_factor :
                        ] = full_decoded_tokens[i, expand_indices[i] + 1 :]
                        expanded_confidence[
                            i, expand_indices[i] + expansion_factor :
                        ] = full_confidence[i, expand_indices[i] + 1 :]
                        # fill <mask_token_id>
                        expanded_generated_tokens[
                            i, expand_indices[i] : expand_indices[i] + expansion_factor
                        ] = mask_token_id
                    else:
                        expanded_generated_tokens[i, : gen_lengths[i]] = (
                            full_decoded_tokens[i, : gen_lengths[i]]
                        )
                        expanded_confidence[i, : gen_lengths[i]] = full_confidence[
                            i, : gen_lengths[i]
                        ]
                        # even we don't need to expand, we still insert tokens for padding
                        insert_index.append(
                            torch.full(
                                (expansion_factor - 1,),
                                int(gen_lengths[i]),
                                device=model.device,
                            )
                        )
                        insert_src_index.append(
                            torch.arange(
                                int(gen_lengths[i]) + 1,
                                int(gen_lengths[i]) + expansion_factor,
                                device=model.device,
                            )
                        )

                delta = FrameDelta(
                    transfer_index=tuple(transfer_index),
                    transfer_src_index=tuple(transfer_src_index),
                    insert_index=torch.stack(insert_index),
                    insert_src_index=torch.stack(insert_src_index),
                    decoded_tokens=expanded_generated_tokens,
                    confidence=expanded_confidence,
                )

            frame = frame.apply_delta(delta)
            block_deltas.append(delta.to("cpu"))

            finished = (
                torch.sum(
                    frame.generated_tokens[
                        :, block_idx * block_length : (block_idx + 1) * block_length
                    ].view(batch_size, -1)
                    == mask_token_id,
                    dim=-1,
                )
                == 0
            )

        deltas.extend(block_deltas)
        block_idx += 1

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
