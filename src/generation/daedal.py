import os
import torch
import torch.nn.functional as F

from src.frame import Frame, DecodeRecord, FrameDelta
from src.generation.vanilla import generate_step
from src.generation.utils import prepare_logits_for_generation
from src.utils import register


def calculate_eot_conf(
    gen_probs: torch.Tensor, num_check_eot_tokens: int, eot_token_id: int
):
    """
    Calculate the average confidence of the last k EOT tokens.

    Args:
        gen_probs: Probs tensor of shape (batch_size, sequence_length, vocab_size).
        num_check_eot_tokens: The number (k) of last EOT tokens to check.
        eot_token_id: The token ID for the EOT token.

    Returns:
        A tensor of shape (batch_size,) with the average confidences.
    """
    x0 = torch.argmax(gen_probs, dim=-1)

    eot_mask_reversed = torch.flip(x0 == eot_token_id, dims=[1])
    eot_counts_reversed = torch.cumsum(eot_mask_reversed.int(), dim=1)

    final_mask = torch.flip(
        (eot_counts_reversed <= num_check_eot_tokens) & eot_mask_reversed, dims=[1]
    )

    return (
        torch.sum(gen_probs[:, :, eot_token_id] * final_mask, dim=1)
        / num_check_eot_tokens
    )


@register.gen_strategy("daedal")
def daedal_generate(
    model,
    input_ids: torch.Tensor,
    alg: str = "maskgit_plus",
    steps: int = 128,
    block_length: int = 32,
    initial_gen_length: int = 128,
    max_gen_length: int = 2048,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    eot_token_id: int | None = None,
    pad_token_id: int | None = None,
    initial_eot_expand_thres=0.5,
    decode_eot_expand_thres=0.9,
    low_conf_expand_thres: float = 0.1,
    num_check_last_eot: int = 32,
    expansion_factor: int = 8,
    threshold: float = 0.9,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
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
    eot_token_id = eot_token_id or int(os.environ.get("EOT_TOKEN_ID", -1))

    if -1 in [mask_token_id, pad_token_id, eot_token_id]:
        raise ValueError(
            "mask_token_id, pad_token_id, and eot_token_id must be provided either as arguments or environment variables."
        )

    assert initial_gen_length % block_length == 0

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
            calculate_eot_conf(
                logits[:, prompt_length:].softmax(dim=-1),
                num_check_last_eot,
                eot_token_id,
            )
            < initial_eot_expand_thres
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
                attention_mask=attention_mask,
                alg=alg,
                temperature=temperature,
                mask_token_id=mask_token_id,
                top_p=top_p,
                top_k=top_k,
                threshold=threshold,
                factor=factor,
                output_hidden_states=output_hidden_states,
                output_probs=True,
            )
            if delta is None:
                break
            assert delta.probs is not None and delta.confidence is not None
            # select sequences with 1) low eot confidence and 2) not exceeding max length
            # (B,)
            need_expand = (
                calculate_eot_conf(delta.probs, num_check_last_eot, eot_token_id)
                < decode_eot_expand_thres
            ) & (gen_lengths < max_gen_length)
            # select tokens with 1) is a valid masked token and 2) low confidence
            # (B,)
            masked_confidence = torch.where(
                block_mask
                & frame.generated_tokens.eq(mask_token_id)
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
                    eot_token_id,
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
                            delta.decoded_tokens[i, : expand_indices[i]]
                        )
                        expanded_confidence[i, : expand_indices[i] + 1] = (
                            delta.confidence[i, : expand_indices[i] + 1]
                        )  # keep the confidence of the expanded token
                        # copy the part after expand_indices[i]
                        expanded_generated_tokens[
                            i, expand_indices[i] + expansion_factor :
                        ] = delta.decoded_tokens[i, expand_indices[i] + 1 :]
                        expanded_confidence[
                            i, expand_indices[i] + expansion_factor :
                        ] = delta.confidence[i, expand_indices[i] + 1 :]
                        # fill <mask_token_id>
                        expanded_generated_tokens[
                            i, expand_indices[i] : expand_indices[i] + expansion_factor
                        ] = mask_token_id
                    else:
                        expanded_generated_tokens[i, : gen_lengths[i]] = (
                            delta.decoded_tokens[i, : gen_lengths[i]]
                        )
                        expanded_confidence[i, : gen_lengths[i]] = delta.confidence[
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
