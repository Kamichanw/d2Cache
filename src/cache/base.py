import os
import torch
import torch.nn as nn

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from transformers.cache_utils import Cache, CacheLayerMixin, StaticLayer
from typing import Any

from src.frame import Frame, FrameDelta


class CacheState(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class AttentionContext:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor | None = None  # assigned from model

    # if config._attn_implementation == "eager", this will be the attention weights
    # of shape (B, nh, q_len, seq_len)
    attn_weights: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @classmethod
    def convert_attention_mask(
        cls,
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
        query_length: int | None = None,
        key_value_length: int | None = None,
    ):
        """
        Convert masks to the form expected by attention kernels.

        - Boolean mask: True means *keep* (attend), False means mask out. We convert to additive mask
          with 0 for keep and -inf for mask, using the provided dtype.
        - Float mask: assumed already additive; returned as-is (after any required expansion).
        Shapes: accept (B, L) or (B, 1, Q, K).
        """
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # (B, kv_len) -> (B, 1, q_len, kv_len)
                try:
                    attention_mask = attention_mask[:, None, None, :].expand(
                        attention_mask.size(0),
                        1,
                        query_length or attention_mask.size(1),
                        key_value_length or attention_mask.size(1),
                    )
                except Exception:
                    # if there is an exception raised, we assume the subclass will process attention mask properly
                    return attention_mask
            elif attention_mask.dim() != 4:
                raise ValueError(
                    f"Expected attention_mask to have 2 or 4 dimensions, but got {attention_mask.dim()}."
                )

            if torch.any(attention_mask < 0):
                # already an additive mask, just convert dtype if needed
                attention_mask = attention_mask.to(dtype)
            else:
                attention_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(
                    dtype
                ).min

        return attention_mask


@dataclass
class FFNContext:
    hidden_states: torch.Tensor
    ffn_out: torch.Tensor | None = None  # assigned from model


@dataclass
class ModelForwardContext:
    input_embeds: torch.Tensor
    position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    logits: torch.Tensor | None = None  # assigned from model


class NoCacheLayer(CacheLayerMixin):
    """
    A dummy cache layer that does not cache anything, used for class dCache.
    """

    def lazy_initialization(self, key_states: torch.Tensor): ...

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return 0, 0

    def get_seq_length(self) -> int:
        return 0

    def get_max_cache_shape(self) -> int:
        return -1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return key_states, value_states


class StaticdCacheLayer(StaticLayer):

    def __init__(self):
        super().__init__(
            max_cache_len=-1  # this will be set during lazy initialization
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization.
        if not self.is_initialized:
            self.max_cache_len = key_states.size(2)
            self.lazy_initialization(key_states)

        active_rows = (
            cache_kwargs.get("active_rows") if cache_kwargs is not None else None
        )
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        if active_rows is None:
            active_rows = torch.arange(key_states.size(0), device=key_states.device)
        if cache_position is None:
            cache_position = (
                torch.arange(key_states.size(-2), device=key_states.device)
                .unsqueeze(0)
                .repeat(active_rows.size(0), 1)
            )

        assert self.keys is not None and self.values is not None
        row_indices = active_rows[:, None]
        self.keys[row_indices, :, cache_position, :] = key_states.transpose(1, 2)
        self.values[row_indices, :, cache_position, :] = value_states.transpose(1, 2)
        return self.keys[active_rows], self.values[active_rows]


class dCache(Cache):
    """
    A cache structure used during diffusion language models decoding to reuse intermediate states.
    """

    def __init__(
        self,
        model_config,
        layer_class_to_replicate: type[CacheLayerMixin] | None = None,
    ):
        super().__init__(
            layer_class_to_replicate=layer_class_to_replicate or NoCacheLayer
        )
        self.model_config = model_config
        self.state = CacheState.PREFILL
        self.active_q_mask: torch.Tensor | None = None  # (B_active, gen_len)

        self._active_seq_mask: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.state = CacheState.PREFILL
        self.active_q_mask = None
        self._active_seq_mask = None

    @staticmethod
    def split_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        if tensor.dim() == 4:
            return tensor
        return (
            tensor.view(tensor.size(0), tensor.size(1), num_heads, -1)
            .transpose(1, 2)
            .contiguous()
        )

    @property
    def active_seq_mask(self):
        """
        A boolean tensor indicates which sequences can generate new tokens in current step.
        It should be assigned outside the cache before model forward.
        """
        if self._active_seq_mask is None:
            raise RuntimeError("The active_seq_mask is not set.")
        return self._active_seq_mask

    @active_seq_mask.setter
    def active_seq_mask(self, mask: torch.Tensor):
        self._active_seq_mask = mask

    @property
    def cache_kwargs(self) -> dict[str, torch.Tensor] | None:
        """
        Per-forward cache update metadata passed to cache layers.

        The returned dict tells layer caches which batch rows are active and,
        when only part of a sequence is recomputed, which token positions should
        be updated. `None` means the cache layer can use its default dense update.
        Subclasses may extend this with cache-specific position metadata.
        """
        if self._active_seq_mask is None:
            return None
        active_rows = torch.where(self.active_seq_mask)[0]
        cache_kwargs = {
            "active_rows": active_rows,
        }
        if self.active_q_mask is not None:
            cache_kwargs["cache_position"] = torch.nonzero(
                self.active_q_mask, as_tuple=True
            )[1].view(active_rows.size(0), -1)
        return cache_kwargs

    @contextmanager
    def model_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        A context manager that modifies the input/output tensors for the forward pass of model layers. In this function,
        it can select a subset to feed into model layers, but it must recover the final logits to be the shape of (batch_size, seq_len, vocab_size).

        Args:
            x (torch.Tensor): The input tensor after embedding layers, with shape (batch_size, seq_len, d_model).
            position_ids (torch.Tensor, *optional*): Position IDs for the input, with shape (batch_size, seq_len).
            attention_mask (torch.Tensor, *optional*): Attention mask for the input.
        """
        input_shape = x.shape
        ctx = ModelForwardContext(
            input_embeds=x,
            position_ids=position_ids,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=x.dtype,
                query_length=x.size(1),
                key_value_length=x.size(1),
            ),
        )

        yield ctx

        if ctx.logits is None:
            raise RuntimeError("The logits are not set in the context.")

        if ctx.logits.shape[:2] != input_shape[:2]:
            raise RuntimeError(
                f"The logits shape {ctx.logits.shape!r} is not compatible with the input shape {input_shape!r}."
            )

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        A context manager that modifies the input/output tensors for attention computation. In this function, it should
        compute query, key, and value projections, and yield a `AttentionContext` object that stores `q`, `k`, `v` tensors.
        The outer code should handle the actual attention computation, then add `o` to the context object.

        Args:
            layer_idx (int): The index of the layer to update.
            x (torch.Tensor): The normalized input tensor before QKV projections, with shape (batch_size, seq_len, d_model).
            q_proj (nn.Linear): The query projection layer.
            k_proj (nn.Linear): The key projection layer.
            v_proj (nn.Linear): The value projection layer.
            attention_mask (torch.Tensor, *optional*): An optional attention mask prepared by model_forward.
        """
        input_shape = x.shape
        q_heads = int(self.model_config.num_attention_heads)
        kv_heads = int(self.model_config.num_key_value_heads)
        if x.numel() > 0:
            q = self.split_heads(q_proj(x), q_heads)
            k = self.split_heads(k_proj(x), kv_heads)
            v = self.split_heads(v_proj(x), kv_heads)
        else:
            q = x.new_empty(x.size(0), q_heads, 0, q_proj.out_features // q_heads)
            k = x.new_empty(x.size(0), kv_heads, 0, k_proj.out_features // kv_heads)
            v = x.new_empty(x.size(0), kv_heads, 0, v_proj.out_features // kv_heads)

        ctx = AttentionContext(q=q, k=k, v=v, attention_mask=attention_mask)
        yield ctx

        if ctx.o is None:
            raise RuntimeError("The attention output is not set in the context.")

        if input_shape != ctx.o.shape:
            raise RuntimeError(
                f"The attention output shape {ctx.o.shape!r} is not compatible with the input shape {input_shape!r}."
            )

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor):
        """
        A context manager that modifies the input/output tensors for feed-forward network computation. In this function,
        it should yield a `FFNContext` object that stores hidden states. The outer code should handle the
        actual feed-forward network computation, then add `ffn_out` to the context object.

        Args:
            layer_idx (int): The index of the layer to update.
            x (torch.Tensor): The input tensor after self-attention, with shape (batch_size, seq_len, d_model).
        """
        input_shape = x.shape
        ctx = FFNContext(hidden_states=x)
        yield ctx

        if ctx.ffn_out is None:
            raise RuntimeError(
                "The feed-forward network output is not set in the context."
            )

        if input_shape != ctx.ffn_out.shape:
            raise RuntimeError(
                f"The feed-forward network output shape {ctx.ffn_out.shape!r} is not compatible with the input shape {input_shape!r}."
            )

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        """
        Called at the start of each generation step to update the cache with the current frame.

        Args:
            block_mask (torch.Tensor): A boolean mask indicating which positions in the block are active.
            frame (Frame): The frame before applying the delta.
        """
        ...

    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        """
        Called at the end of each generation step to update the cache with the current frame and delta.

        Args:
            block_mask (torch.Tensor): A boolean mask indicating which positions in the block are active.
            frame (Frame): The frame before applying the delta.
            delta (FrameDelta): The delta to apply to the frame.
        """
        self.state = CacheState.DECODE

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        """
        Called at the start of each block to update the cache with the current frame.

        Args:
            block_mask (torch.Tensor): A boolean mask indicating which positions in the block are active.
            frame (Frame): The frame before applying any deltas in the block.
        """
        ...

    def on_block_end(
        self,
        block_mask: torch.Tensor,
        frame: Frame,
        deltas: list[FrameDelta],
    ):
        """
        Called at the end of each block to update the cache with the current frame and deltas.

        Args:
            block_mask (torch.Tensor): A boolean mask indicating which positions in the block are active.
            frame (Frame): The frame before applying any deltas in the block.
            deltas (list[FrameDelta]): The list of deltas applied in the block.
        """
        ...

    @property
    def mask_token_id(self):
        return int(os.environ["MASK_TOKEN_ID"])
