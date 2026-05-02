import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import contextmanager
from typing import Any

from transformers.cache_utils import DynamicLayer

from src.frame import Frame, FrameDelta
from src.cache.base import dCache


class BlockDynamicLayer(DynamicLayer):
    """
    Dynamic cache layer that writes block diffusion KV states by absolute position.
    """

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        batch_size, num_heads, _, head_dim = key_states.shape
        self.keys = torch.empty(
            batch_size, num_heads, 0, head_dim, dtype=self.dtype, device=self.device
        )
        self.values = torch.empty_like(self.keys)
        self.filled_mask = torch.zeros(
            batch_size, 0, dtype=torch.bool, device=self.device
        )
        self.is_initialized = True

    def _ensure_length(self, length: int) -> None:
        assert self.keys is not None and self.values is not None
        if self.keys.shape[-2] >= length:
            return
        pad_length = length - self.keys.shape[-2]
        key_padding = torch.zeros(
            self.keys.size(0),
            self.keys.size(1),
            pad_length,
            self.keys.size(-1),
            dtype=self.keys.dtype,
            device=self.keys.device,
        )
        value_padding = torch.zeros_like(key_padding)
        self.keys = torch.cat(
            [self.keys, key_padding],
            dim=-2,
        )
        self.values = torch.cat(
            [self.values, value_padding],
            dim=-2,
        )
        self.filled_mask = torch.cat(
            [
                self.filled_mask,
                torch.zeros(
                    self.filled_mask.size(0),
                    pad_length,
                    dtype=torch.bool,
                    device=self.filled_mask.device,
                ),
            ],
            dim=-1,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        active_seq_mask = (
            cache_kwargs.get("active_seq_mask") if cache_kwargs is not None else None
        )
        if cache_position is None:
            return super().update(key_states, value_states, cache_kwargs)

        if not self.is_initialized:
            self.lazy_initialization(key_states)

        assert self.keys is not None and self.values is not None
        if cache_position.dim() == 1:
            cache_position = cache_position.unsqueeze(0).expand(key_states.size(0), -1)
        cache_position = cache_position.to(self.keys.device)
        self._ensure_length(int(cache_position.max().item()) + 1)
        cache_rows = torch.arange(key_states.size(0), device=self.keys.device)
        if active_seq_mask is not None and self.keys.size(0) != key_states.size(0):
            cache_rows = torch.where(active_seq_mask.to(self.keys.device))[0]
            if cache_rows.numel() != key_states.size(0):
                raise ValueError(
                    "active_seq_mask does not match the cache update batch size."
                )
        for src_idx, cache_row in enumerate(cache_rows.tolist()):
            positions = cache_position[src_idx]
            self.keys[cache_row, :, positions, :] = key_states[src_idx]
            self.values[cache_row, :, positions, :] = value_states[src_idx]
            self.filled_mask[cache_row, positions] = True
        return self.keys, self.values

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.filled_mask.numel() == 0:
            return 0
        filled_positions = self.filled_mask.any(dim=0).nonzero(as_tuple=False)
        if filled_positions.numel() == 0:
            return 0
        return int(filled_positions[-1].item()) + 1

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "filled_mask"):
            self.filled_mask.zero_()


class BlockdCache(dCache):
    """
    Cache for block diffusion language model decoding.

    The KV storage itself is inherited from Transformers' dynamic Cache through
    dCache. This class only tracks block-level generation state used by the
    shared decoding loop.
    """

    def __init__(self, model_config):
        super().__init__(model_config)
        self._block_length: int | None = None

    def on_block_start(self, model, block_mask: torch.Tensor, frame: Frame):
        block_width = block_mask.sum(dim=-1)
        active_width = block_width[block_width > 0]
        if active_width.numel() > 0:
            self._block_length = int(active_width.max().item())
        self.active_q_mask = F.pad(block_mask, (frame.prompts.size(-1), 0), value=False)

    def on_block_end(
        self,
        model,
        block_mask: torch.Tensor,
        frame: Frame,
        deltas: list[FrameDelta],
    ):
        commit_frame = frame
        for delta in deltas:
            commit_frame = commit_frame.apply_delta(delta)
        commit_frame = commit_frame.as_batch().to(
            device=model.device, dtype=model.dtype
        )
        batch_size, prompt_length = commit_frame.prompts.shape
        x = torch.cat([commit_frame.prompts, commit_frame.generated_tokens], dim=-1)
        position_ids = (
            torch.arange(x.size(1), device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        self.active_seq_mask = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        self.active_q_mask = F.pad(
            block_mask,
            (
                prompt_length,
                commit_frame.generated_tokens.size(-1) - block_mask.size(-1),
            ),
            value=False,
        )
        with torch.no_grad():
            model(
                x,
                attention_mask=torch.ones_like(x, dtype=torch.long, device=x.device),
                position_ids=position_ids,
                output_hidden_states=False,
                past_key_values=self,
                use_cache=True,
            )
        self.active_q_mask = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(BlockDynamicLayer())
        cache_kwargs = dict(cache_kwargs or {})
        if self._active_seq_mask is not None:
            cache_kwargs.setdefault("active_seq_mask", self._active_seq_mask)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def reset(self) -> None:
        super().reset()
        self._block_length = None
        self.active_q_mask = None
        self._active_seq_mask = None

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            batch_size, seq_len, hidden_size = x.shape
            q_mask = self.active_q_mask
            if q_mask is not None:
                if q_mask.size(0) != batch_size:
                    q_mask = q_mask[self.active_seq_mask]
                selected = q_mask.sum(dim=-1)
                if torch.unique(selected).numel() != 1:
                    raise ValueError(
                        "Block diffusion cache requires the same number of active query tokens per active sequence."
                    )
                ctx.x = x[q_mask].view(batch_size, int(selected[0].item()), hidden_size)

            yield ctx

            if q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (batch_size, seq_len, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(q_mask.unsqueeze(-1), ctx.logits)

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
        with super().attention(
            layer_idx,
            x,
            attn_norm,
            q_proj,
            k_proj,
            v_proj,
            attention_mask=None,
            position_ids=position_ids,
        ) as ctx:
            if self.active_q_mask is None:
                if attention_mask is not None and attention_mask.dim() == 2:
                    q_len = ctx.q.size(-2)
                    kv_len = ctx.k.size(-2)
                    if attention_mask.size(-1) < kv_len:
                        attention_mask = F.pad(
                            attention_mask,
                            (0, kv_len - attention_mask.size(-1)),
                            value=1,
                        )
                    ctx.attention_mask = (
                        attention_mask[:, None, None, :kv_len]
                        .to(device=x.device, dtype=torch.bool)
                        .expand(-1, 1, q_len, -1)
                    )
                else:
                    ctx.attention_mask = attention_mask
                yield ctx
                return

            if position_ids is None:
                position_ids = (
                    torch.arange(x.size(1), device=x.device)
                    .unsqueeze(0)
                    .expand(x.size(0), -1)
                )

            block_length = self._block_length or x.size(1)
            q_idx = position_ids.to(x.device)
            q_len = q_idx.size(1)
            layer = self.layers[layer_idx] if layer_idx < len(self.layers) else None
            if layer is not None and layer.is_initialized:
                assert layer.keys is not None
                cached_length = layer.keys.size(-2)
                filled_mask = layer.filled_mask.to(x.device)  # type: ignore
                if filled_mask.size(0) != x.size(0):
                    filled_mask = filled_mask[self.active_seq_mask]
                filled_mask = filled_mask[:, :cached_length]
            else:
                cached_length = 0
                filled_mask = torch.zeros(
                    x.size(0), 0, dtype=torch.bool, device=x.device
                )

            key_value_length = max(
                cached_length,
                int(q_idx.max().item()) + 1 if q_idx.numel() > 0 else cached_length,
            )
            kv_idx = (
                torch.arange(key_value_length, device=x.device)
                .unsqueeze(0)
                .expand(x.size(0), -1)
            )
            valid_kv = torch.zeros(
                x.size(0), key_value_length, dtype=torch.bool, device=x.device
            )
            valid_kv[:, :cached_length] = filled_mask
            if q_len > 0:
                valid_kv.scatter_(1, q_idx.clamp_max(key_value_length - 1), True)
            q_block = q_idx[:, :, None] // block_length
            kv_block = kv_idx[:, None, :] // block_length
            ctx.attention_mask = (
                (kv_block <= q_block) & valid_kv[:, None, :]
            ).unsqueeze(1)
            ctx.q_position_ids = q_idx
            ctx.kv_position_ids = kv_idx
            yield ctx
