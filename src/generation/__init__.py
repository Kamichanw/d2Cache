import os
import torch

from loguru import logger
from typing import Literal

from src.utils import register, Registry, find_incompatible_kwargs
from src.generation.utils import decode_final_frame

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

    if (gen_fn := register.get(strategy)) is not None:
        unknown_args = find_incompatible_kwargs(kwargs, gen_fn)
        if len(unknown_args) > 0:
            msg = f"The arguments {unknown_args} are not supported by the generation strategy '{strategy}'."
            if ignore_unknown_args == "warn":
                logger.warning(msg, once=True, rank_zero_only=True)
            elif ignore_unknown_args == "forbid":
                raise ValueError(msg)
        kwargs = {k: v for k, v in kwargs.items() if k not in unknown_args}
        return gen_fn(model, input_ids, **kwargs)
    else:
        raise NotImplementedError(
            f"Generation strategy '{strategy}' is not implemented."
        )


__all__ = ["generate", "register", "decode_final_frame"]
