import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import contextmanager

from transformers.cache_utils import StaticLayer

from src.frame import Frame, FrameDelta
from src.cache.base import dCache, AttentionContext
from src.utils import is_adapted_from_ar


class PrefixCache(dCache):

    def __init__(self, model_config, use_dual: bool = False):
        super().__init__(model_config)
        self.use_dual = use_dual
        self.active_q_mask: torch.Tensor | None = None

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.active_q_mask is not None:
                if B != self.active_q_mask.size(0):
                    # if some sequences in the batch have ended, we need to resize
                    # the active_q_mask accordingly.
                    self.active_q_mask = self.active_q_mask[0].expand(B, -1)
                ctx.x = x[self.active_q_mask].view(B, -1, C)

            yield ctx

            if self.active_q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

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
            attention_mask,
            position_ids,
        ) as ctx:
            if (
                layer_idx >= len(self.layers)
                or not self.layers[layer_idx].is_initialized
            ):
                # the first forward pass, store states as cache
                while len(self.layers) <= layer_idx:
                    self.layers.append(StaticLayer(x.shape[1]))
                layer = self.layers[layer_idx]
                if layer.keys is None or layer.values is None:
                    self.update(ctx.k, ctx.v, layer_idx)
                    layer = self.layers[layer_idx]
                    assert layer.keys is not None and layer.values is not None
                else:
                    layer.keys[self.active_seq_mask] = ctx.k
                    layer.values[self.active_seq_mask] = ctx.v
                    layer.is_initialized = True
            else:
                assert self.active_q_mask is not None
                layer = self.layers[layer_idx]
                assert layer.keys is not None and layer.values is not None
                if layer_idx == 0:
                    active_seq_idx = torch.where(self.active_seq_mask)[0]
                    m_nonzero = self.active_q_mask.nonzero(as_tuple=False)
                    self._active_q_indices = (
                        active_seq_idx[m_nonzero[:, 0]],
                        m_nonzero[:, 1],
                    )

                rows, cols = self._active_q_indices
                layer.keys[rows, :, cols, :] = ctx.k.transpose(1, 2).flatten(0, 1)
                layer.values[rows, :, cols, :] = ctx.v.transpose(1, 2).flatten(0, 1)
                ctx.k = layer.keys[self.active_seq_mask]
                ctx.v = layer.values[self.active_seq_mask]

            if layer_idx == 0:
                # cache common variables sharing among layers
                self._q_position_ids, self._kv_position_ids = (
                    AttentionContext.select_position_ids(
                        position_ids, self.active_q_mask
                    )
                )
                self._attention_mask = AttentionContext.convert_attention_mask(
                    attention_mask,
                    dtype=ctx.k.dtype,
                    query_length=ctx.q.shape[-2],
                    key_value_length=layer.values.shape[-2],
                )

            ctx.q_position_ids = self._q_position_ids
            ctx.kv_position_ids = self._kv_position_ids
            ctx.attention_mask = self._attention_mask

            yield ctx

    def on_step_end(
        self, model, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta
    ):
        if self.active_q_mask is None:
            q_mask = F.pad(block_mask, (frame.prompts.size(-1), 0), value=False)
            if not self.use_dual:
                block_start = int(block_mask[0].int().argmax() + 1)
                q_mask[:, frame.prompts.size(-1) + block_start :] = True

            if is_adapted_from_ar(self.model_config):
                q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)

            self.active_q_mask = q_mask

    def on_block_start(self, model, block_mask: torch.Tensor, frame: Frame):
        for layer in self.layers:
            layer.is_initialized = False
        self.active_q_mask = None
