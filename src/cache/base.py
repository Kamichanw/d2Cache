import os
import torch
import torch.nn as nn

from contextlib import contextmanager
from dataclasses import dataclass

from src.frame import Frame, FrameDelta



@dataclass
class AttentionContext:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    residual: torch.Tensor
    o: torch.Tensor | None = None  # assigned from model

    # if config._attn_implementation == "eager", this will be the attention weights
    # of shape (B, nh, q_len, seq_len)
    attn_weight: torch.Tensor | None = None

    # if you select a subset of qkv, you must also provide these properties
    q_position_ids: torch.Tensor | None = None
    kv_position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @classmethod
    def select_position_ids(
        cls,
        position_ids: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ):
        q_position_ids, kv_position_ids = position_ids, position_ids
        if position_ids is not None:
            if q_mask is not None:
                q_position_ids = position_ids[q_mask].view(q_mask.size(0), -1)
            if kv_mask is not None:
                kv_position_ids = position_ids[kv_mask].view(kv_mask.size(0), -1)
        return q_position_ids, kv_position_ids

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
        Shapes: accept (B, L) or (B, 1, Q, K) / (B, Q, K).
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

            # attention_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

        return attention_mask


@dataclass
class FFNContext:
    x: torch.Tensor
    residual: torch.Tensor
    ffn_out: torch.Tensor | None = None  # assigned from model


@dataclass
class ModelForwardContext:
    x: torch.Tensor
    logits: torch.Tensor | None = None  # assigned from model


class dCache:
    """
    A cache structure used during diffusion language models decoding to reuse intermediate states.
    """

    def __init__(self, model_config):
        self.model_config = model_config
        self.active_q_mask: torch.Tensor | None = None

        self._active_seq_mask: torch.Tensor | None = None

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

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        """
        A context manager that modifies the input/output tensors for the forward pass of model layers. In this function,
        it can select a subset to feed into model layers, but it must recover the final logits to be the shape of (batch_size, seq_len, vocab_size).

        Args:
            x (torch.Tensor): The input tensor after embedding layers, with shape (batch_size, seq_len, d_model).

        """
        input_shape = x.shape
        ctx = ModelForwardContext(x=x)

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
        attn_norm: nn.Module,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """
        A context manager that modifies the input/output tensors for attention computation. In this function, it should
        compute query, key, and value projections, and yield a `AttentionContext` object that stores `q`, `k`, `v`, `residual` tensors.
        The outer code should handle the actual attention computation, then add `o` to the context object.

        Args:
            layer_idx (int): The index of the layer to update.
            x (torch.Tensor): The input tensor after pre-layer norm, with shape (batch_size, seq_len, d_model).
            attn_norm (nn.Module): The layer normalization module before attention.
            q_proj (nn.Linear): The query projection layer.
            k_proj (nn.Linear): The key projection layer.
            v_proj (nn.Linear): The value projection layer.
            attention_mask (torch.Tensor, *optional*): An optional attention mask, with shape (batch_size, seq_len, seq_len).
            position_ids (torch.Tensor, *optional*): An optional tensor of position IDs, with shape (batch_size, seq_len).
        """
        residual = x
        x = attn_norm(x)
        if x.numel() > 0:
            q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
            q, k, v = x[:, 0:0], x[:, 0:0], x[:, 0:0]

        if layer_idx == 0:
            self._attention_mask = AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=q.dtype,
                query_length=q.shape[1],
                key_value_length=k.shape[1],
            )

        ctx = AttentionContext(
            q=q,
            k=k,
            v=v,
            residual=residual,
            attention_mask=self._attention_mask,
            q_position_ids=position_ids,
            kv_position_ids=position_ids,
        )
        yield ctx

        if ctx.o is None:
            raise RuntimeError("The attention output is not set in the context.")

        if ctx.residual.shape != ctx.o.shape:
            raise RuntimeError(
                f"The attention output shape {ctx.o.shape!r} is not compatible with the residual shape {ctx.residual.shape!r}."
            )

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor):
        """
        A context manager that modifies the input/output tensors for feed-forward network computation. In this function,
        it should yield a `FFNContext` object that stores `x` and `residual` tensors. The outer code should handle the
        actual feed-forward network computation, then add `ffn_out` to the context object.

        Args:
            layer_idx (int): The index of the layer to update.
            x (torch.Tensor): The input tensor after self-attention, with shape (batch_size, seq_len, d_model).
        """
        ctx = FFNContext(x=x, residual=x)
        yield ctx

        if ctx.ffn_out is None:
            raise RuntimeError(
                "The feed-forward network output is not set in the context."
            )

        if ctx.residual.shape != ctx.ffn_out.shape:
            raise RuntimeError(
                f"The feed-forward network output shape {ctx.ffn_out.shape!r} is not compatible with the residual shape {ctx.residual.shape!r}."
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
        ...

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        """
        Called at the start of each block to update the cache with the current frame.

        Args:
            block_mask (torch.Tensor): A boolean mask indicating which positions in the block are active.
            frame (Frame): The frame before applying any deltas in the block.
        """
        ...

    def on_block_end(
        self, block_mask: torch.Tensor, frame: Frame, deltas: list[FrameDelta]
    ):
        """
        Called at the end of each block to update the cache with the current frame and deltas.

        Args:
            block_mask (torch.Tensor): A boolean mask indicating which positions in the block are active.
            frame (Frame): The frame before applying all deltas in the block.
            deltas (list[FrameDelta]): The list of deltas applied in the block.
        """
        ...

    @property
    def mask_token_id(self):
        return int(os.environ["MASK_TOKEN_ID"])
