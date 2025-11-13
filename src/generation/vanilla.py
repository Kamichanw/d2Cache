import os
import torch
import torch.distributions as dists
import torch.nn.functional as F

from typing import Type

from src.cache import dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import prepare_logits_for_generation
from src.dream.generation_utils import top_k_logits, top_p_logits
from src.utils import certainty_density, register
from src.third_party import get_token_freq


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.

    Args:
        mask_index: A boolean tensor of shape (B, L) indicating the positions of mask
        steps: The number of steps in a block to sample.

    Returns:
        A tensor of shape (B, steps) indicating the number of tokens to be transferred at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def confidence_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    threshold: float | torch.Tensor | None = None,
    factor: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Select tokens to be fixed based on token probs, i.e., confidence.
    It consists of parallel decoding and low-confidence remasking.
    Args:
        token_probs: A tensor of shape [B, gen_length] containing the probabilities of each token.
        transfer_index_mask: A boolean tensor of shape [B, gen_length] indicating which tokens can be transferred.
        num_transfer_tokens: A tensor of shape [B,] indicating the number of tokens to be transferred at each step.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
    """
    batch_size, _ = scores.shape

    confidence = torch.where(transfer_index_mask, scores, -torch.inf)
    transfer_index = [torch.tensor([]) for _ in range(batch_size)]
    if threshold is not None or factor is not None:
        if threshold is not None:
            # 1.a select all tokens whose confidence is above the threshold
            col_indices = torch.nonzero(confidence >= threshold, as_tuple=False)[:, 1]
            counts = torch.sum(confidence >= threshold, dim=-1).cpu().tolist()
            transfer_index = torch.split(col_indices, counts)
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
                transfer_index[i] = torch.topk(confidence[i], n - 1, dim=-1).indices  # type: ignore

        # check if there are too few tokens to be decoded in any sequence
        # in this case, fall back to topk selection
        confidence = confidence[
            [
                i
                for i, t in enumerate(transfer_index)
                if t.numel() < num_transfer_tokens[i]
            ]
        ]
    if confidence.size(0) > 0:
        # 2. select the tokens that have top-num_transfer_tokens highest probs as generated tokens
        topk_transfer_index = [
            torch.topk(
                confidence[i],
                int(torch.min(transfer_index_mask[i].sum(), num_transfer_tokens[i])),
                dim=-1,
            ).indices
            for i in range(confidence.size(0))
        ]
        # put topk_transfer_index to rows that have fewer than num_transfer_tokens
        source_iter = iter(topk_transfer_index)
        transfer_index = [
            next(source_iter) if t.numel() < num_transfer_tokens[i] else t
            for i, t in enumerate(transfer_index)
        ]

    return tuple(transfer_index)


_token_freq: torch.Tensor | None = None


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    debias=False,
    alg="maskgit_plus",
):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    epsilon = 1e-10
    if debias:
        global _token_freq
        if _token_freq is None:
            raise ValueError("Token frequency not initialized for debiasing.")
        confidence = torch.clamp_max(
            -confidence * torch.log(_token_freq[x0] + epsilon), max=10
        )

    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs
    elif alg == "entropy":
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    elif alg == "random":
        confidence = torch.rand_like(confidence)

    return confidence, x0, probs


@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor | int,
    attention_mask: torch.Tensor | None = None,
    past_key_values: dCache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    sigma: float | None = None,
    debias: bool = True,
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

    logits = prepare_logits_for_generation(model, outputs.logits)
    if past_key_values is not None and past_key_values.active_q_mask is not None:
        if model.config.model_type.lower() == "dream":
            valid_mask = past_key_values.active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = past_key_values.active_q_mask[:, prompt_length:]
        transfer_index_mask[can_generate].logical_and_(valid_mask)
    logits = logits[:, prompt_length:]

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    if debias:
        global _token_freq
        _token_freq = get_token_freq(model, device=logits.device, dtype=logits.dtype)

    # sampling tokens for all generated positions
    confidence, x0, p = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        alg=alg,
    )
    scores = confidence = torch.where(
        transfer_index_mask[can_generate], confidence, -torch.inf
    )
    if sigma is not None and sigma > 0:
        scores = confidence * certainty_density(~remaining_mask, sigma=sigma)

    # select tokens to transfer based on probs and mask
    transfer_index = confidence_unmasking(
        scores=scores,
        transfer_index_mask=(transfer_index_mask & block_mask)[can_generate],
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
        decoded_tokens=torch.where(
            transfer_index_mask[can_generate], x0, INVALID_TOKEN_ID
        ),
        confidence=confidence,
        probs=(
            torch.where(transfer_index_mask[can_generate], p, -torch.inf)
            if output_probs
            else None
        ),
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
    )
    return delta.unbatch() if not frame.is_batched else delta


@register.gen_strategy("vanilla")
def vanilla_generate(
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
    debias: bool = False,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    """
    Vanilla generation for diffusion large language models.
    Args:
        model: Mask predictor.
        input_ids: A tensor of shape (B, prompt_len).
        steps: Sampling steps, less than or equal to gen_length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        mask_token_id: The token id of [MASK]. It can be `None` if "MASK_TOKEN_ID" is specified in the environment variables.
        top_k: The number of highest probability tokens to keep for one generation step.
        top_p: The cumulative probability threshold for nucleus sampling.
        sigma: The standard deviation of the Gaussian kernel used in decoding with certainty prior, see https://arxiv.org/abs/2509.23094.
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
                attention_mask=attention_mask,
                past_key_values=cache,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sigma=sigma,
                mask_token_id=mask_token_id,
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
