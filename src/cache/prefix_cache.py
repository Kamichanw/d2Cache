import torch
import torch.nn.functional as F

from contextlib import contextmanager

from src.frame import Frame, FrameDelta
from src.cache.base import AttentionContext, CacheState, dCache, StaticdCacheLayer


class PrefixCache(dCache):

    def __init__(self, model_config, use_dual: bool = False):
        super().__init__(model_config, layer_class_to_replicate=StaticdCacheLayer)
        self.use_dual = use_dual
        self.active_q_mask: torch.Tensor | None = None

    @contextmanager
    def model_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        with super().model_forward(
            x=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ) as ctx:
            B, T, C = x.shape
            if self.state is CacheState.DECODE:
                assert self.active_q_mask is not None
                if B != self.active_q_mask.size(0):
                    # if some sequences in the batch have ended, we need to resize
                    # the active_q_mask accordingly.
                    self.active_q_mask = self.active_q_mask[0].expand(B, -1)
                ctx.input_embeds = x[self.active_q_mask].view(B, -1, C)
                if position_ids is not None:
                    ctx.position_ids = position_ids[self.active_q_mask].view(B, -1)
            ctx.attention_mask = AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=x.dtype,
                query_length=ctx.input_embeds.size(1),
                key_value_length=T,
            )

            yield ctx

            if self.state is CacheState.DECODE:
                assert ctx.logits is not None and self.active_q_mask is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        if self.state is CacheState.PREFILL:
            q_mask = F.pad(block_mask, (frame.prompts.size(-1), 0), value=False)
            if not self.use_dual:
                block_start = int(block_mask[0].int().argmax() + 1)
                q_mask[:, frame.prompts.size(-1) + block_start :] = True

            if self.model_config.model_type.lower() == "dream":
                q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)

            self.active_q_mask = q_mask

        super().on_step_end(block_mask, frame, delta)

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        self.reset()
