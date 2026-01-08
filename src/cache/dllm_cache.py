import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from src.frame import Frame
from src.cache.base import dCache, AttentionContext, FFNContext


class dLLMCache(dCache):
    def __init__(self, model_config, kp: int = 50, kr: int = 2, rou: float = 0.25):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.attn_cache: list[torch.Tensor] = []
        self.ffn_cache: list[torch.Tensor] = []
        self.kp = kp
        self.kr = kr
        self.rou = rou

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
        refresh_prompt = self.refresh_prompt or layer_idx == 0
        refresh_response = self.refresh_response or layer_idx == 0
        residual = x
        x = attn_norm(x)
        # select prompt or/and response part to feed into the projections
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

        B, T, C = x.shape
        # if response part needs to be refreshed or adaptive refreshing is disabled or it is the first
        # forward pass, we calculate all projections
        q = torch.empty((B, 0, q_proj.out_features), dtype=x.dtype, device=x.device)
        k = torch.empty((B, 0, k_proj.out_features), dtype=x.dtype, device=x.device)
        v = torch.empty((B, 0, v_proj.out_features), dtype=x.dtype, device=x.device)
        if refresh_response or self.rou == 0 or len(self.key_cache) <= layer_idx:
            if x.numel() > 0:
                q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
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
                self.value_cache[layer_idx][
                    self.active_seq_mask, self._prompt_length :
                ],
                dim=-1,
            )
            refresh_index_response = torch.topk(
                cos_sim, largest=False, k=num_replace
            ).indices

            selected_x_response = torch.gather(
                x_response, 1, refresh_index_response.unsqueeze(-1).expand(-1, -1, C)
            )
            q = torch.cat([q, q_proj(selected_x_response)], dim=1)
            k = torch.cat([k, k_proj(selected_x_response)], dim=1)
            v = torch.cat([v, v_response], dim=1)

        # update cache
        if len(self.key_cache) <= layer_idx:
            # the first forward pass, store states as cache
            self.key_cache.append(k)
            self.value_cache.append(v)
            q_position_ids = position_ids
        else:
            if refresh_prompt:
                self.key_cache[layer_idx][
                    self.active_seq_mask, : self._prompt_length
                ] = k[:, : self._prompt_length]
                self.value_cache[layer_idx][
                    self.active_seq_mask, : self._prompt_length
                ] = v[:, : self._prompt_length]
                prompt_offset = self._prompt_length
            else:
                prompt_offset = 0

            q_position_ids = (
                position_ids[:, :prompt_offset] if position_ids is not None else None
            )

            if self.rou > 0 or refresh_response:
                if refresh_response:
                    # if adaptive refreshing is disabled, we refresh all response
                    refresh_index_response = (
                        torch.arange(x_response.size(1)).unsqueeze(0).expand(B, -1)
                    )

                refresh_index_response: torch.Tensor = refresh_index_response + self._prompt_length  # type: ignore
                self.key_cache[layer_idx][
                    self.active_seq_mask.nonzero(), refresh_index_response
                ] = k[:, prompt_offset:]
                # note that for value states, we recompute all even we are using adaptive refreshing
                self.value_cache[layer_idx][
                    self.active_seq_mask, self._prompt_length :
                ] = v[:, prompt_offset:]

                if not refresh_response:
                    # we've concatenated index before if refresh_response is true
                    refresh_index = torch.cat([refresh_index, refresh_index_response], dim=-1)  # type: ignore

                if q_position_ids is not None:
                    assert position_ids is not None
                    row_indices = (
                        torch.arange(B).unsqueeze(-1).expand_as(refresh_index_response)
                    )
                    q_position_ids = torch.cat(
                        [
                            q_position_ids,
                            position_ids[row_indices, refresh_index_response],
                        ],
                        dim=-1,
                    )

        self._refresh_index = refresh_index
        ctx = AttentionContext(
            q=q,
            k=self.key_cache[layer_idx][self.active_seq_mask],
            v=self.value_cache[layer_idx][self.active_seq_mask],
            residual=residual,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=q.dtype,
                query_length=q.shape[1],
                key_value_length=self.key_cache[layer_idx].shape[1],
            ),
            q_position_ids=q_position_ids,
            kv_position_ids=position_ids,
        )

        yield ctx

        assert ctx.o is not None
        if len(self.attn_cache) <= layer_idx:
            self.attn_cache.append(ctx.o)
        else:
            if ctx.o.numel() > 0:
                self.attn_cache[layer_idx][
                    self.active_seq_mask.nonzero(), refresh_index
                ] = ctx.o

        ctx.o = self.attn_cache[layer_idx][self.active_seq_mask]

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor):
        B, _, C = x.shape
        row_indices = torch.arange(B).unsqueeze(-1).expand_as(self._refresh_index)
        residual = x
        x = x[row_indices, self._refresh_index]
        ctx = FFNContext(x=x, residual=residual)

        yield ctx

        assert ctx.ffn_out is not None
        if len(self.ffn_cache) <= layer_idx:
            self.ffn_cache.append(ctx.ffn_out)
        else:
            self.ffn_cache[layer_idx][
                self.active_seq_mask.nonzero(), self._refresh_index
            ] = ctx.ffn_out
        ctx.ffn_out = self.ffn_cache[layer_idx][self.active_seq_mask]

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        current_steps = frame.steps.max(-1, keepdim=True).values
        refresh_prompt = (current_steps + 1) % self.kp == 0
        refresh_response = (current_steps + 1) % self.kr == 0
        B, self._prompt_length = frame.prompts.shape

        try:
            active_seq_mask = self.active_seq_mask
        except RuntimeError:
            # ignore runtime error may caused by accessing active_seq_mask at the first step
            active_seq_mask = torch.ones(
                (B,), dtype=torch.bool, device=current_steps.device
            )

        assert (
            torch.unique(refresh_prompt[active_seq_mask]).numel() <= 1
            and torch.unique(refresh_response[active_seq_mask]).numel() <= 1
        ), "All unfinished sequences must have the same refresh schedule."

        if refresh_prompt[active_seq_mask].numel() > 0:
            self.refresh_prompt = refresh_prompt[active_seq_mask][0].item()
            self.refresh_response = refresh_response[active_seq_mask][0].item()
