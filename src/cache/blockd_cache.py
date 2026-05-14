from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicLayer

from src.frame import Frame, FrameDelta
from src.cache.base import AttentionContext, CacheState, ModelForwardContext, dCache


class BlockDynamicLayer(DynamicLayer):
    """
    Dynamic cache layer that writes block diffusion KV states by absolute position.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization.
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        assert (
            self.keys is not None
            and self.values is not None
            and cache_kwargs is not None
        )

        cache_position = cache_kwargs["cache_position"]
        active_rows = cache_kwargs["active_rows"]

        length = int(cache_position.max().item()) + 1
        cached_length = self.get_seq_length()
        if cached_length < length:
            pad_length = length - cached_length
            self.keys = F.pad(self.keys, (0, 0, 0, pad_length))
            self.values = F.pad(self.values, (0, 0, 0, pad_length))

        row_indices = active_rows[:, None]
        self.keys[row_indices, :, cache_position, :] = key_states.transpose(1, 2)
        self.values[row_indices, :, cache_position, :] = value_states.transpose(1, 2)
        return self.keys[active_rows], self.values[active_rows]


class BlockdCache(dCache):
    """
    Cache for block diffusion language model decoding.

    The KV storage itself is inherited from Transformers' dynamic Cache through
    dCache. This class only tracks block-level generation state used by the
    shared decoding loop.
    """

    def __init__(self, model_config):
        super().__init__(model_config, layer_class_to_replicate=BlockDynamicLayer)
        self._block_mask: torch.Tensor
        self._pending_refresh_mask: torch.Tensor | None = None

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        if self._pending_refresh_mask is not None:
            pending_refresh_mask = F.pad(
                self._pending_refresh_mask,
                (0, block_mask.size(-1) - self._pending_refresh_mask.size(-1)),
                value=False,
            )
            self._pending_refresh_mask = None
            block_mask = block_mask | pending_refresh_mask
        self._block_mask = block_mask

    def on_block_end(
        self,
        block_mask: torch.Tensor,
        frame: Frame,
        deltas: list[FrameDelta],
    ):
        super().on_block_end(block_mask, frame, deltas)
        if deltas:
            # The last step samples after forward; refresh this block in the next forward.
            self._pending_refresh_mask = block_mask

    def reset(self) -> None:
        super().reset()
        self._pending_refresh_mask = None

    @contextmanager
    def model_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len, hidden_size = x.shape

        block_mask = self._block_mask
        prompt_length = seq_len - block_mask.size(-1)
        self.active_q_mask = F.pad(
            block_mask, (prompt_length, 0), value=(self.state is CacheState.PREFILL)
        )

        q_mask = self.active_q_mask
        if q_mask.size(0) != batch_size:
            q_mask = q_mask[self.active_seq_mask]

        q_len = int(q_mask[0].sum().item())

        x = x[q_mask].view(batch_size, q_len, hidden_size)

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        elif position_ids.size(0) != batch_size:
            position_ids = position_ids[self.active_seq_mask]
        position_ids = position_ids[q_mask].view(batch_size, q_len)

        if attention_mask is not None:
            if attention_mask.size(0) != batch_size:
                attention_mask = attention_mask[self.active_seq_mask]
            if attention_mask.dim() == 4 and attention_mask.size(-2) == seq_len:
                q_idx = torch.nonzero(q_mask, as_tuple=True)[1].view(batch_size, q_len)
                attention_mask = attention_mask.gather(
                    2,
                    q_idx[:, None, :, None].expand(-1, 1, -1, attention_mask.size(-1)),
                )

        ctx = ModelForwardContext(
            input_embeds=x,
            position_ids=position_ids,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=x.dtype,
                query_length=q_len,
                key_value_length=max(
                    self.get_seq_length(), int(position_ids.max().item()) + 1
                ),
            ),
        )

        yield ctx

        assert ctx.logits is not None
        if q_len != seq_len:
            left_pad = int(q_mask[0].int().argmax().item())
            right_pad = seq_len - left_pad - q_len
            ctx.logits = F.pad(ctx.logits, (0, 0, left_pad, right_pad))
