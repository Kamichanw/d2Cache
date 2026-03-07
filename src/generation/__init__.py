import os
import torch
import inspect

from loguru import logger
from typing import Callable, Literal

from src.utils import Registry
from src.generation.utils import decode_final_frame, register

Registry.trigger(os.path.dirname(__file__), __name__)


def generate(
    model,
    input_ids: torch.Tensor,
    *,
    strategy: str,
    ignore_unknown_args: Literal["ignore", "warn", "forbit"] = "warn",
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
    def my_generation(model, input_ids, **kwargs):
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

    try:
        gen_fn = register.get(strategy)
    except ValueError as e:
        raise NotImplementedError(
            f"Generation strategy '{strategy}' is not implemented."
        ) from e

    unknown_args = find_incompatible_kwargs(kwargs, gen_fn)
    if len(unknown_args) > 0:
        msg = f"The arguments {unknown_args} are not supported by the generation strategy '{strategy}'."
        if ignore_unknown_args == "warn":
            logger.warning(msg, once=True, rank_zero_only=True)
        elif ignore_unknown_args == "forbid":
            raise ValueError(msg)
    kwargs = {k: v for k, v in kwargs.items() if k not in unknown_args}
    return gen_fn(model, input_ids, **kwargs)


__all__ = ["generate", "register", "decode_final_frame"]
