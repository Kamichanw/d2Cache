import os
import re
import torch
import torch.distributions as dists

from src.frame import PLACEHOLDER_STEP, Frame
from src.models.dream.generation_utils import top_k_logits, top_p_logits


def prepare_logits_for_generation(model, logits: torch.Tensor):
    """Prepare logits for unmasking."""
    from src.models import LLaDAModelLM, DreamModel

    if isinstance(model, LLaDAModelLM):
        ...
    elif isinstance(model, DreamModel):
        # main difference with LLaDA, see https://github.com/DreamLM/Dream/issues/31
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def decode_final_frame(
    tokenizer, final_frame: Frame, stop_words: list[str] | None = None, **kwargs
) -> str | list[str]:
    """
    Decode the final frame to a string or a list of strings, removing tokens after the first <|endoftext|>.
    If `stop_words` is provided, it will trim the generated text at the first occurrence of any stop word.

    Args:
        tokenizer: The tokenizer to decode the frame.
        final_frame: The final frame to decode.
        stop_words: A list of stop words to trim the generated text. Defaults to eos token.
        kwargs: Additional keyword arguments to pass to the tokenizer's decode method.

    Returns:
        A string or a list of strings.
    """
    frame = final_frame.as_batch()

    can_generate = check_can_generate(
        frame,
        stop_until_eot=True,  # we only want to check tokens before the first EOS in this case
        mask_token_id=tokenizer.mask_token_id,
        eot_token_id=tokenizer.eos_token_id,
    )
    if can_generate.any():
        raise ValueError(
            "The frame contains mask tokens, indicating that the generation has not completed."
        )

    stop_words = stop_words or []
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    if tokenizer.eos_token not in stop_words:
        stop_words.append(tokenizer.eos_token)

    # trim until stop words
    filtered_tokens = frame.generated_tokens.clone()
    filtered_tokens[frame.generated_tokens > len(tokenizer.get_vocab())] = (
        tokenizer.eos_token_id
    )
    texts = tokenizer.batch_decode(filtered_tokens, skip_special_tokens=False, **kwargs)

    texts = [
        (
            text[: match.start()]
            if (match := re.search(r"|".join(re.escape(sw) for sw in stop_words), text))
            else text
        )
        for text in texts
    ]

    # remove special tokens
    texts = tokenizer.batch_decode(
        tokenizer(texts).input_ids, skip_special_tokens=skip_special_tokens, **kwargs
    )

    return texts if final_frame.is_batched else texts[0]


_token_freq: torch.Tensor | None = None


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    debias=False,
    clip_alpha=None,
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
        alpha = clip_alpha if clip_alpha is not None else 10.0
        if _token_freq is None:
            raise ValueError("Token frequency not initialized for debiasing.")
        confidence = torch.clamp_max(
            -confidence * torch.log(_token_freq[x0] + epsilon), max=alpha
        )

    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs
    elif alg == "entropy":
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    elif alg == "random":
        confidence = torch.rand_like(confidence)
    elif alg == "maskgit_plus":
        pass  # default behavior
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    return confidence, x0, probs


def check_can_generate(
    frame: Frame,
    eligible_mask: torch.Tensor | None = None,
    num_transfer_tokens: int | torch.Tensor | None = None,
    stop_until_eot: bool = False,
    mask_token_id: int | None = None,
    eot_token_id: int | None = None,
):
    """
    Check whether a frame can perform generation. A frame can continue generate if positions where eligible_mask is True:
    1. all tokens before the first EOS are unmasked if stop_until_eot is true, else
    2. all masked tokens are converted to certain tokens.
    3. num_transfer_tokens > 0 if num_transfer_tokens is provided.
    """
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    gen_length = frame.generated_tokens.size(1)
    device = frame.generated_tokens.device

    if isinstance(num_transfer_tokens, torch.Tensor):
        if num_transfer_tokens.numel() != batch_size or num_transfer_tokens.dim() != 1:
            raise ValueError(
                f"`num_transfer_tokens` must be a tensor of shape ({batch_size},) or a single integer, "
                f"but got shape of {num_transfer_tokens.shape}."
            )
    elif isinstance(num_transfer_tokens, int):
        num_transfer_tokens = torch.full(
            (batch_size,), num_transfer_tokens, device=device, dtype=torch.long
        )

    if eligible_mask is None:
        eligible_mask = torch.ones_like(frame.generated_tokens)

    # condition 1
    if stop_until_eot:
        if eot_token_id is None:
            raise ValueError(
                "eot_token_id must be specified if stop_until_eos is True."
            )
        # check if all tokens before EOT have been generated
        eot_mask = frame.generated_tokens == eot_token_id
        first_eot_idx = torch.where(
            eot_mask.any(dim=-1, keepdim=True),
            eot_mask.int().argmax(dim=-1, keepdim=True),
            gen_length,
        )
        eligible_mask &= (
            torch.arange(gen_length, device=device).unsqueeze(0).expand(batch_size, -1)
        ) < first_eot_idx

    # condition 2
    if mask_token_id is not None:
        eligible_mask &= frame.generated_tokens == mask_token_id

    can_generate = eligible_mask.any(dim=-1)
    # condition 3
    if num_transfer_tokens is not None:
        # skip sequence that doesn't require to generate
        can_generate &= num_transfer_tokens > 0

    return can_generate
