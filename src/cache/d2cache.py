import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from src.frame import Frame, FrameDelta
from src.utils import certainty_density, nucleus_select, top_up_mask_
from src.cache.base import dCache, AttentionContext


class d2Cache(dCache):

    def __init__(
        self,
        model_config,
        rollout_p: float = 0.1,
        current_k: int = 32,
        sigma: float = 10.0,
        inflate_w: int = 4,
    ):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._conf_cache: torch.Tensor | None = None  # shape (B, G)
        self._full_q_mask: torch.Tensor | None = None  # shape (B, T)
        self._density_score: torch.Tensor  # shape (B, G)
        self._global_importance: torch.Tensor  # shape (B, T)
        self.rollout_p = rollout_p
        self.current_k = current_k
        self.sigma = sigma
        self.inflate_w = inflate_w

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self._full_q_mask is not None:
                self.active_q_mask = self.top_up_mask(
                    self._full_q_mask[self.active_seq_mask]
                )
                ctx.x = x[self.active_q_mask].view(B, -1, C)
            yield ctx

            if self._full_q_mask is not None:
                assert ctx.logits is not None and self.active_q_mask is not None
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
            if len(self.key_cache) <= layer_idx:
                # the first forward pass, store states as cache
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                assert self.active_q_mask is not None
                if layer_idx == 0:
                    active_seq_idx = torch.where(self.active_seq_mask)[0]
                    m_nonzero = self.active_q_mask.nonzero(as_tuple=False)
                    self._active_q_indices = (
                        active_seq_idx[m_nonzero[:, 0]],
                        m_nonzero[:, 1],
                    )

                self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)
                self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(
                    0, 1
                )
                ctx.k = self.key_cache[layer_idx][self.active_seq_mask]
                ctx.v = self.value_cache[layer_idx][self.active_seq_mask]

            if layer_idx == 0:
                self._q_position_ids, self._kv_position_ids = (
                    AttentionContext.select_position_ids(
                        position_ids, self.active_q_mask
                    )
                )
                self._attention_mask = AttentionContext.convert_attention_mask(
                    attention_mask,
                    dtype=ctx.k.dtype,
                    query_length=ctx.q.shape[1],
                    key_value_length=self.value_cache[layer_idx].shape[1],
                )

            ctx.q_position_ids = self._q_position_ids
            ctx.kv_position_ids = self._kv_position_ids
            ctx.attention_mask = self._attention_mask
            yield ctx

            assert (
                ctx.attn_weight is not None
            ), 'The attention weights must be outputed, make sure you\'ve set attn_implementation="eager"'

            if layer_idx == 0:
                # shape: (B, pooled_size, pooled_size)
                self._attn_rollout = torch.eye(
                    self.key_cache[layer_idx].size(1), device=x.device, dtype=x.dtype
                ).expand(x.size(0), -1, -1)
            self.accumulate_attn_rollout(ctx.attn_weight)

    def top_up_mask(self, q_mask: torch.Tensor):
        q_mask = q_mask.clone()
        num_selected_per_seq = q_mask.sum(dim=-1)
        _, G = self._density_score.shape
        if torch.any(num_selected_per_seq != num_selected_per_seq.max()):
            # prioritize selection from masked tokens with higher certainty density.
            # if all masked tokens have been selected, select from the remaining tokens based on the rollout values.
            combined_scores = torch.where(
                q_mask, -torch.inf, self._global_importance[self.active_seq_mask]
            )
            combined_scores[:, -G:] += (
                combined_scores.max() + self._density_score[self.active_seq_mask]
            )

            top_up_mask_(q_mask, int(num_selected_per_seq.max()), combined_scores)
        return q_mask

    def accumulate_attn_rollout(self, attn_scores: torch.Tensor):
        """
        Computes one step of the Attention Rollout for attention maps.
        In this setup, only a subset of tokens act as queries.

        Args:
            attn_scores (torch.Tensor):
                Attention scores for the current layer, with shape of (B, num_heads, q_len, seq_len).
        """
        B, n_heads, q_len, seq_len = attn_scores.shape
        device, dtype = attn_scores.device, attn_scores.dtype

        # inject the rectangular attention map into the rows
        if self.active_q_mask is None:
            effective_attn = attn_scores.mean(dim=1)
        else:
            effective_attn = torch.eye(seq_len, device=device, dtype=dtype).repeat(
                B, 1, 1
            )
            effective_attn[self.active_q_mask] = attn_scores.mean(dim=1).reshape(
                -1, seq_len
            )

        residual_attn = effective_attn + torch.eye(seq_len, device=device, dtype=dtype)
        # re-normalize the matrix so that each row sums to 1
        residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)

        self._attn_rollout = residual_attn @ self._attn_rollout

    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        confidence = delta.confidence
        assert confidence is not None
        B, P = frame.prompts.shape
        B_active, G = confidence.shape
        T = G + P
        block_mask = block_mask[self.active_seq_mask]
        new_frame = frame.apply_delta(delta)
        device = confidence.device

        if self._conf_cache is None:
            self._conf_cache = confidence

        # prepare active mask for query.
        # 1. for the masked positions, we only calculate those have large certainty
        # (B_active, G)
        remaining_mask = (
            new_frame.generated_tokens[self.active_seq_mask] == self.mask_token_id
        )
        if self.active_q_mask is not None:
            # only position where are selected at previous step and are still masked
            # can produce valid confidence scores
            valid_mask = (
                self.active_q_mask[:, P:] & frame.generated_tokens[self.active_seq_mask]
                == self.mask_token_id
            )
            self._conf_cache[self.active_seq_mask][valid_mask] = confidence[valid_mask]

        block_size = block_mask.sum(dim=1, keepdim=True)

        # find the minimal end index that contains at least k candidates.
        meets_target = torch.cumsum(remaining_mask.int(), dim=1) >= self.current_k
        min_search_end = torch.argmax(meets_target.int(), dim=1, keepdim=True)
        min_search_end[~meets_target.any(dim=1, keepdim=True)] = G - 1

        # round this minimal end index up to the next block boundary
        search_end = (((min_search_end // block_size) + 1) * block_size) - 1

        block_start_indices = torch.argmax(block_mask.int(), dim=1, keepdim=True)
        col_indices = torch.arange(G, device=device)
        search_mask = (col_indices >= block_start_indices) & (col_indices <= search_end)

        scores = self._conf_cache[self.active_seq_mask] * certainty_density(
            ~remaining_mask, self.sigma
        )

        # add a bias to tokens in block to ensure at least one token is selected in block
        scores[block_mask] += scores.max()
        _, indices = torch.topk(
            torch.where(search_mask & remaining_mask, scores, -torch.inf),
            k=min(self.current_k, remaining_mask.size(-1)),
            dim=-1,
        )
        selected_mask = (
            torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(
                1, indices, True
            )
            & remaining_mask
        )
        # if model is dream, we need to retain the token before masked tokens
        if self.model_config.model_type.lower() == "dream":
            response_mask = F.pad(selected_mask[:, 1:], (0, 1), value=False)
        else:
            response_mask = selected_mask
        # 2. recompute all new generated tokens, as they transform from mask to real tokens
        transfer_src_index = (
            delta.transfer_src_index
            if delta.transfer_src_index is not None
            else delta.transfer_index
        )
        lengths = torch.tensor(
            [ti.numel() for ti in transfer_src_index if ti.numel() > 0], device=device
        )
        row_indices = torch.repeat_interleave(
            torch.arange(B_active, device=confidence.device), lengths
        )
        col_indices = torch.cat(transfer_src_index)  # type: ignore
        response_mask[row_indices, col_indices] = True

        q_mask = F.pad(response_mask, (P, 0), value=False)

        # 3. for other tokens, select top-k tokens based on attention rollout
        global_importance = self._attn_rollout.sum(dim=1)
        q_mask |= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)

        if self.model_config.model_type.lower() == "dream":
            # if the first mask token is selected, we need to select the token before it
            # i.e., the last prompt token
            q_mask[:, P - 1] = selected_mask[:, 0]  # type: ignore

        # 4. inflate the mask: if two selected tokens are within a window, select all tokens between them.
        if self.inflate_w > 0:
            arange_t = torch.arange(T, device=device).expand(B_active, -1)

            # find distance to the next selected token for each position
            masked_indices_next = torch.where(q_mask, arange_t, T)
            next_selected_indices = torch.cummin(
                torch.flip(masked_indices_next, dims=[-1]), dim=-1
            ).values
            next_selected_indices = torch.flip(next_selected_indices, dims=[-1])
            dist_to_next_true = next_selected_indices - arange_t

            # find distance to the previous selected token for each position
            masked_indices_prev = torch.where(q_mask, arange_t, -1)
            prev_selected_indices = torch.cummax(masked_indices_prev, dim=-1).values
            dist_to_prev_true = arange_t - prev_selected_indices

            # inflate if the gap is smaller than or equal to the window size.
            gap_len = dist_to_next_true + dist_to_prev_true
            q_mask |= (
                (gap_len <= self.inflate_w)
                & (prev_selected_indices >= 0)
                & (next_selected_indices < T)
            )

        if self._full_q_mask is None:
            self._full_q_mask = q_mask
            self._global_importance = global_importance
            self._density_score = scores
        else:
            self._full_q_mask[self.active_seq_mask] = q_mask
            self._global_importance[self.active_seq_mask] = global_importance
            self._density_score[self.active_seq_mask] = scores
