import re
import torch

from src.frame import PLACEHOLDER_STEP, Frame
from src.llada import LLaDAModelLM
from src.dream import DreamModel


def prepare_logits_for_generation(model, logits: torch.Tensor):
    """Prepare logits for unmasking."""
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
    if (final_frame.steps == PLACEHOLDER_STEP).any():
        raise ValueError(
            "The frame contains mask tokens, indicating that the generation has not completed."
        )
    frame = final_frame.as_batch()
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
