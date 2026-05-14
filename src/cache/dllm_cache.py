import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import contextmanager
from typing import Any

from src.frame import Frame
from src.cache.base import (
    AttentionContext,
    CacheState,
    FFNContext,
    StaticdCacheLayer,
    dCache,
)


class dLLMCacheLayer(StaticdCacheLayer):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.max_cache_len = key_states.size(2)
            self.lazy_initialization(key_states)

        assert (
            self.keys is not None
            and self.values is not None
            and cache_kwargs is not None
        )

        active_rows = cache_kwargs["active_rows"]
        cache_position = cache_kwargs["cache_position"]
        value_cache_position = cache_kwargs["value_cache_position"]

        row_indices = active_rows[:, None]
        self.keys[row_indices, :, cache_position, :] = key_states.transpose(1, 2)
        self.values[row_indices, :, value_cache_position, :] = value_states.transpose(
            1, 2
        )

        return self.keys[active_rows], self.values[active_rows]


class dLLMCache(dCache):
    def __init__(self, model_config, kp: int = 50, kr: int = 2, rou: float = 0.25):
        super().__init__(model_config, layer_class_to_replicate=dLLMCacheLayer)
        self.attn_cache: list[torch.Tensor] = []
        self.ffn_cache: list[torch.Tensor] = []
        self.kp = kp
        self.kr = kr
        self.rou = rou
        self._cache_kwargs: dict[str, Any] | None = None

    @property
    def cache_kwargs(self) -> dict[str, Any] | None:
        return self._cache_kwargs

    @contextmanager
    def model_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        with super().model_forward(
            x=x, position_ids=position_ids, attention_mask=attention_mask
        ) as ctx:
            yield ctx
            # for dLLMCache, active_q_mask is only used during forward
            # all positions are regarded as active during decoding
            self.active_q_mask = None
            self._cache_kwargs = None

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
        refresh_prompt = (
            self.refresh_prompt or layer_idx == 0 or self.state is CacheState.PREFILL
        )
        refresh_response = (
            self.refresh_response or layer_idx == 0 or self.state is CacheState.PREFILL
        )
        x_prompt = x[:, : self._prompt_length]
        x_response = x[:, self._prompt_length :]
        x = x[:, 0:0]  # make it empty
        refresh_index = torch.tensor([], device=x.device, dtype=torch.long)
        if refresh_prompt:
            x = x_prompt
            refresh_index = torch.arange(self._prompt_length, device=x.device)

        if self.rou > 0 or refresh_response:
            x = torch.cat([x, x_response], dim=1)
            if refresh_response:
                refresh_index = torch.cat(
                    [
                        refresh_index,
                        self._prompt_length
                        + torch.arange(x_response.size(1), device=x.device),
                    ]
                )
        refresh_index = refresh_index.unsqueeze(0).expand(x.size(0), -1)

        B, _, C = x.shape
        q_heads = int(self.model_config.num_attention_heads)
        kv_heads = int(self.model_config.num_key_value_heads)
        # if response part needs to be refreshed or adaptive refreshing is disabled or it is the first
        # forward pass, we calculate all projections
        q = torch.empty((B, 0, q_proj.out_features), dtype=x.dtype, device=x.device)
        k = torch.empty((B, 0, k_proj.out_features), dtype=x.dtype, device=x.device)
        v = torch.empty((B, 0, v_proj.out_features), dtype=x.dtype, device=x.device)
        refresh_index_response = None
        if refresh_response or self.rou == 0 or self.state is CacheState.PREFILL:
            if x.numel() > 0:
                q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
            value_cache = self.layers[layer_idx].values
            assert value_cache is not None
            value_cache = value_cache[self.active_seq_mask].transpose(1, 2)
            value_cache = value_cache.contiguous().view(B, value_cache.size(1), -1)
            if refresh_prompt:
                x_prompt = x[:, : self._prompt_length]
                x_response = x[:, self._prompt_length :]
                q, k, v = q_proj(x_prompt), k_proj(x_prompt), v_proj(x_prompt)
            else:
                x_response = x

            # refresh response part adaptively
            v_response = v_proj(x_response)
            num_replace = int(x_response.size(1) * self.rou)
            cos_sim = F.cosine_similarity(
                v_response,
                value_cache[:, self._prompt_length :],
                dim=-1,
            )
            refresh_index_response = torch.topk(
                cos_sim, largest=False, k=num_replace
            ).indices
            refresh_index_response = torch.sort(refresh_index_response, dim=-1).values

            selected_x_response = torch.gather(
                x_response, 1, refresh_index_response.unsqueeze(-1).expand(-1, -1, C)
            )
            q = torch.cat([q, q_proj(selected_x_response)], dim=1)
            k = torch.cat([k, k_proj(selected_x_response)], dim=1)
            v = torch.cat([v, v_response], dim=1)

        q = (
            self.split_heads(q, q_heads)
            if q.numel() > 0
            else q.new_empty(B, q_heads, 0, q_proj.out_features // q_heads)
        )
        k = (
            self.split_heads(k, kv_heads)
            if k.numel() > 0
            else k.new_empty(B, kv_heads, 0, k_proj.out_features // kv_heads)
        )
        v = (
            self.split_heads(v, kv_heads)
            if v.numel() > 0
            else v.new_empty(B, kv_heads, 0, v_proj.out_features // kv_heads)
        )

        if self.state is CacheState.DECODE:
            layer = self.layers[layer_idx]
            assert layer.keys is not None and layer.values is not None
            if self.rou > 0 or refresh_response:
                if refresh_response:
                    # if adaptive refreshing is disabled, we refresh all response
                    refresh_index_response = (
                        torch.arange(x_response.size(1), device=x.device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                refresh_index_response = refresh_index_response + self._prompt_length  # type: ignore

                if not refresh_response:
                    # we've concatenated index before if refresh_response is true
                    refresh_index = torch.cat(
                        [refresh_index, refresh_index_response], dim=-1
                    )

        value_cache_position = refresh_index
        if self.rou > 0 and not refresh_response:
            value_cache_position = (
                (
                    self._prompt_length
                    + torch.arange(x_response.size(1), device=x.device)
                )
                .unsqueeze(0)
                .expand(B, -1)
            )
            if refresh_prompt:
                prompt_position = (
                    torch.arange(self._prompt_length, device=x.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                value_cache_position = torch.cat(
                    [prompt_position, value_cache_position], dim=-1
                )

        active_rows = torch.where(self.active_seq_mask)[0]
        self.active_q_mask = torch.zeros(
            B,
            self._prompt_length + x_response.size(1),
            dtype=torch.bool,
            device=x.device,
        ).scatter_(1, refresh_index, True)
        self._refresh_index = refresh_index
        self._cache_kwargs = {
            "active_rows": active_rows,
            "cache_position": refresh_index,
            "value_cache_position": value_cache_position,
        }

        ctx = AttentionContext(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
        )
        if (
            attention_mask is not None
            and attention_mask.dim() == 4
            and attention_mask.size(-2) != q.size(-2)
        ):
            ctx.attention_mask = attention_mask[:, 0][self.active_q_mask].view(
                q.size(0), q.size(-2), attention_mask.size(-1)
            )[:, None]

        yield ctx
        self._cache_kwargs = None

        assert ctx.o is not None
        if self.state is CacheState.PREFILL:
            self.attn_cache.append(ctx.o)
        else:
            if ctx.o.numel() > 0:
                row_indices = active_rows.unsqueeze(-1).expand_as(refresh_index)
                self.attn_cache[layer_idx][row_indices, refresh_index] = ctx.o

        ctx.o = self.attn_cache[layer_idx][self.active_seq_mask]

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor):
        row_indices = (
            torch.arange(x.size(0), device=x.device)
            .unsqueeze(-1)
            .expand_as(self._refresh_index)
        )
        x = x[row_indices, self._refresh_index]
        ctx = FFNContext(hidden_states=x)

        yield ctx

        assert ctx.ffn_out is not None
        if self.state is CacheState.PREFILL:
            self.ffn_cache.append(ctx.ffn_out)
        else:
            active_rows = torch.where(self.active_seq_mask)[0]
            active_rows = active_rows.unsqueeze(-1).expand_as(self._refresh_index)
            self.ffn_cache[layer_idx][active_rows, self._refresh_index] = ctx.ffn_out
        ctx.ffn_out = self.ffn_cache[layer_idx][self.active_seq_mask]

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        current_steps = frame.steps.max(-1, keepdim=True).values
        refresh_prompt = (current_steps + 1) % self.kp == 0
        refresh_response = (current_steps + 1) % self.kr == 0
        self._prompt_length = frame.prompts.size(-1)

        if self.state is CacheState.PREFILL:
            self.refresh_prompt = self.refresh_response = True
        else:
            assert (
                torch.unique(refresh_prompt[self.active_seq_mask]).numel() <= 1
                and torch.unique(refresh_response[self.active_seq_mask]).numel() <= 1
            ), "All unfinished sequences must have the same refresh schedule."

            self.refresh_prompt = refresh_prompt[self.active_seq_mask][0].item()
            self.refresh_response = refresh_response[self.active_seq_mask][0].item()

    def reset(self) -> None:
        super().reset()
        self.attn_cache = []
        self.ffn_cache = []
        self._cache_kwargs = None
