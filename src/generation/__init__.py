import os
import torch
import inspect

from loguru import logger
from collections.abc import Callable
from typing import Literal
from transformers.modeling_utils import PreTrainedModel

from src.cache import BlockdCache, dCache
from src.utils import Registry, is_block_diffusion
from src.generation.utils import decode_final_frame, register

Registry.trigger(os.path.dirname(__file__), __name__)


def generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    strategy: str,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int | None = None,
    block_length: int | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
    cache_cls: Callable[..., dCache] | None = None,
    num_transfer_tokens: int = 1,
    ignore_unknown_args: Literal["ignore", "warn", "forbid"] = "warn",
    **kwargs,
):
    """
    Generate text using the specified generation strategy.
    To register a new generation strategy, use the `register` decorator.
    This method also expands attention_mask and position_ids to match the generation length if they are provided.

    Args:
        model: The model to use for generation.
        input_ids: A tensor of shape (B, L) containing the input IDs.
        strategy: The name of the generation strategy to use.
        ignore_unknown_args: How to handle unknown arguments:
            - "ignore": Ignore unknown arguments.
            - "warn": Log a warning for unknown arguments.
            - "forbid": Raise an error for unknown arguments.

    Example:
    ```python
    @register("my_strategy")
    def my_generation(model, input_ids, max_new_tokens, **kwargs):
        # Your generation logic here
        ...
    ```
    Then you can call this function with the strategy name:
    ```python
    outputs = generate(model, input_ids, strategy="my_strategy", ...)
    ```
    """

    def find_incompatible_kwargs(input_kwargs: dict, target_fn: Callable) -> tuple:
        """
        Returns a tuple of keyword arguments in `input_kwargs` that are not compatible
        with the signature of `target_fn`.
        """
        sig = inspect.signature(target_fn)
        params = sig.parameters
        if all(p.kind != p.VAR_KEYWORD for p in params.values()) and (
            unknown_args := set(input_kwargs) - set(params)
        ):
            return tuple(unknown_args)
        return tuple()

    def resolve_token_id(
        name: str,
        env_name: str,
        explicit: int | None,
        *,
        required: bool,
    ) -> int | None:
        if explicit is not None:
            return int(explicit)
        if env_name in os.environ and os.environ[env_name] not in {"", "None"}:
            return int(os.environ[env_name])
        if required:
            raise ValueError(
                f"{name} must be provided as an argument or environment variable {env_name}."
            )
        return None

    try:
        gen_fn = register.get(strategy)
    except ValueError as e:
        raise NotImplementedError(
            f"Generation strategy '{strategy}' is not implemented."
        ) from e

    mask_token_id = resolve_token_id(
        "mask_token_id", "MASK_TOKEN_ID", mask_token_id, required=True
    )
    pad_token_id = resolve_token_id(
        "pad_token_id", "PAD_TOKEN_ID", pad_token_id, required=False
    )
    eos_token_id = resolve_token_id(
        "eos_token_id", "EOS_TOKEN_ID", eos_token_id, required=False
    )

    if max_new_tokens is None:
        if not is_block_diffusion(model):
            raise ValueError(
                "max_new_tokens=None is only supported for block diffusion generation."
            )
        if block_length is None:
            raise ValueError("block_length must be provided when max_new_tokens=None.")
        block_length = int(block_length)
    else:
        max_new_tokens = int(max_new_tokens)
        block_length = int(block_length or max_new_tokens)
    num_transfer_tokens = int(num_transfer_tokens)
    if block_length <= 0 or (max_new_tokens is not None and max_new_tokens < 0):
        raise ValueError(
            f"{max_new_tokens=} and {block_length=} must be non-negative, with block_length > 0."
        )
    if num_transfer_tokens <= 0:
        raise ValueError(f"{num_transfer_tokens=} must be > 0.")
    if stop_until_eos and eos_token_id is None:
        raise ValueError("eos_token_id must be available when stop_until_eos is True.")
    if max_new_tokens is None and not stop_until_eos:
        raise ValueError("stop_until_eos must be True when max_new_tokens=None.")

    cache: dCache | None = None
    if cache_cls is not None:
        cache = cache_cls(model.config)

    if (
        is_block_diffusion(model)
        and cache is not None
        and not isinstance(cache, BlockdCache)
    ):
        raise ValueError(
            "Block diffusion generation requires BlockdCache as past_key_values."
        )

    if attention_mask is None:
        attention_mask = (
            (input_ids != pad_token_id).long()
            if pad_token_id is not None
            else torch.ones_like(input_ids, dtype=torch.long)
        )
    attention_mask = attention_mask.to(model.device)

    kwargs.update(
        {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "attention_mask": attention_mask,
            "block_length": block_length,
            "num_transfer_tokens": num_transfer_tokens,
            "mask_token_id": mask_token_id,
            "eos_token_id": eos_token_id,
            "stop_until_eos": stop_until_eos,
            "cache": cache,
        }
    )

    unknown_args = find_incompatible_kwargs(kwargs, gen_fn)
    if len(unknown_args) > 0:
        msg = f"The arguments {unknown_args} are not supported by the generation strategy '{strategy}'."
        if ignore_unknown_args == "warn":
            logger.warning(msg, once=True, rank_zero_only=True)
        elif ignore_unknown_args == "forbid":
            raise ValueError(msg)
    kwargs = {k: v for k, v in kwargs.items() if k not in unknown_args}
    return gen_fn(model, **kwargs)


__all__ = ["generate", "register", "decode_final_frame"]
