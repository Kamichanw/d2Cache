import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import triton
import triton.language as tl

from dataclasses import dataclass
from contextlib import contextmanager

from src.frame import Frame, FrameDelta
from src.utils import certainty_density, nucleus_select, pad_mask_


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

            attention_mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min

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


class d2Cache(dCache):

    def __init__(
        self,
        model_config,
        rollout_p: float = 0.1,
        current_k: int = 32,
        sigma: float = 10.0,
        inflate_w: int = 8,
    ):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._conf_cache: torch.Tensor | None = None
        self.rollout_p = rollout_p
        self.current_k = current_k
        self.sigma = sigma
        self.inflate_w = inflate_w

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.active_q_mask is not None:
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
            if len(self.key_cache) <= layer_idx:
                # the first forward pass, store states as cache
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                self.key_cache[layer_idx][self.active_q_mask] = ctx.k
                self.value_cache[layer_idx][self.active_q_mask] = ctx.v
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]

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
            B, T, C = self.key_cache[layer_idx].shape

            if layer_idx == 0:
                # shape: (B, pooled_size, pooled_size)
                self._attn_rollout = torch.eye(
                    T, device=x.device, dtype=x.dtype
                ).expand(B, -1, -1)

            self.accumulate_attn_rollout(ctx.attn_weight)

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
        effective_attn = torch.eye(seq_len, device=device, dtype=dtype).expand(
            B, -1, -1
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
        B, G_old = frame.generated_tokens.shape
        B, P = frame.prompts.shape
        T = G_old + P
        new_frame = frame.apply_delta(delta)
        device = confidence.device

        if self._conf_cache is None:
            self._conf_cache = confidence

        # prepare active mask for query.
        # 1. for the masked positions, we only calculate those have large certainty
        # (B, G_old)
        remaining_mask = new_frame.generated_tokens == self.mask_token_id
        if self.active_q_mask is not None:
            # only position where are selected at previous step and are still masked
            # can produce valid confidence scores
            self._conf_cache = torch.where(
                self.active_q_mask[:, P:] & frame.generated_tokens
                == self.mask_token_id,
                confidence,
                self._conf_cache,
            )

        block_size = block_mask.sum(dim=1, keepdim=True)

        # find the minimal end index that contains at least k candidates.
        meets_target = torch.cumsum(remaining_mask.int(), dim=1) >= self.current_k
        min_search_end = torch.argmax(meets_target.int(), dim=1, keepdim=True)
        min_search_end[~meets_target.any(dim=1, keepdim=True)] = G_old - 1

        # round this minimal end index up to the next block boundary
        search_end = (((min_search_end // block_size) + 1) * block_size) - 1

        block_start_indices = torch.argmax(block_mask.int(), dim=1, keepdim=True)
        col_indices = torch.arange(G_old, device=device)
        search_mask = (col_indices >= block_start_indices) & (col_indices <= search_end)

        scores = self._conf_cache * certainty_density(~remaining_mask, self.sigma)

        # add a bias to tokens in block to ensure at least one token is selected in block
        scores[block_mask] += scores.max()
        _, indices = torch.topk(
            torch.where(search_mask & remaining_mask, scores, -torch.inf),
            k=min(self.current_k, remaining_mask.size(-1)),
            dim=-1,
        )
        response_mask = (
            torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(
                1, indices, True
            )
            & remaining_mask
        )
        # 2. recompute all new generated tokens, as they transform from mask to real tokens
        transfer_src_index = (
            delta.transfer_src_index
            if delta.transfer_src_index is not None
            else delta.transfer_index
        )
        lengths = torch.tensor([ti.numel() for ti in transfer_src_index], device=device)
        row_indices = torch.repeat_interleave(
            torch.arange(B, device=confidence.device), lengths
        )
        col_indices = torch.cat(transfer_src_index)  # type: ignore
        response_mask[row_indices, col_indices] = True

        q_mask = F.pad(response_mask, (P, 0), value=False)

        # 3. for other tokens, select top-k tokens based on attention rollout
        global_importance = self._attn_rollout.sum(dim=1)
        q_mask |= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)

        # 4. inflate the mask: if two selected tokens are within a window, select all tokens between them.
        if self.inflate_w > 0:
            arange_t = torch.arange(T, device=device).expand(B, -1)

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

        # 5. pad for batch
        num_selected_per_seq = q_mask.sum(dim=-1)
        if torch.any(num_selected_per_seq != num_selected_per_seq.max()):
            # prioritize selection from masked tokens with higher certainty density.
            # if all masked tokens have been selected, select from the remaining tokens based on the rollout values.
            combined_scores = torch.where(q_mask, -torch.inf, global_importance)
            combined_scores[:, P:] += combined_scores.max() + scores

            pad_mask_(q_mask, int(num_selected_per_seq.max()), combined_scores)

        if self.model_config.model_type.lower() == "dream":
            # if model is dream, we need to retain the token before masked tokens.
            # to ensure each sequence has the same number of selected tokens,
            # we only move the selected positions leftward if the position before it is not selected.
            selected_mask = q_mask[:, P:] & remaining_mask

            source_rows, source_cols = torch.where(selected_mask)

            source_cols = source_cols + P
            target_cols = source_cols - 1

            target_is_false_mask = ~q_mask[source_rows, target_cols]

            source_rows = source_rows[target_is_false_mask]

            q_mask[source_rows, source_cols[target_is_false_mask]] = False
            q_mask[source_rows, target_cols[target_is_false_mask]] = True

        # now, each sequence has the same number of selected tokens
        self.active_q_mask = q_mask


class PrefixCache(dCache):

    def __init__(self, model_config, use_dual: bool = False):
        super().__init__(model_config)
        self.use_dual = use_dual

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self.active_q_mask is not None:
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
            if len(self.key_cache) <= layer_idx:
                # the first forward pass, store states as cache
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                self.key_cache[layer_idx][self.active_q_mask] = ctx.k.flatten(0, 1)
                self.value_cache[layer_idx][self.active_q_mask] = ctx.v.flatten(0, 1)
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]

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

    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        if self.active_q_mask is None:
            q_mask = torch.cat(
                [
                    torch.zeros_like(frame.prompts, dtype=torch.bool),
                    block_mask,
                ],
                dim=-1,
            )
            if not self.use_dual:
                block_start = int(block_mask[0].int().argmax() + 1)
                q_mask[:, frame.prompts.size(-1) + block_start :] = True

            if self.model_config.model_type.lower() == "dream":
                q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)

            self.active_q_mask = q_mask

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.active_q_mask: torch.Tensor | None = None


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
                self.value_cache[layer_idx][:, self._prompt_length :],
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
                self.key_cache[layer_idx][:, : self._prompt_length] = k[
                    :, : self._prompt_length
                ]
                self.value_cache[layer_idx][:, : self._prompt_length] = v[
                    :, : self._prompt_length
                ]
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

                refresh_index_response = refresh_index_response + self._prompt_length  # type: ignore

                row_indices = torch.arange(B).unsqueeze(-1).expand_as(refresh_index_response)  # type: ignore
                self.key_cache[layer_idx][row_indices, refresh_index_response] = k[
                    :, prompt_offset:
                ]
                # note that for value states, we recompute all even we are using adaptive refreshing
                self.value_cache[layer_idx][:, self._prompt_length :] = v[
                    :, prompt_offset:
                ]

                if not refresh_response:
                    # we've concatenated index before if refresh_response is true
                    refresh_index = torch.cat([refresh_index, refresh_index_response], dim=-1)  # type: ignore

                if q_position_ids is not None:
                    assert position_ids is not None
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
            k=self.key_cache[layer_idx],
            v=self.value_cache[layer_idx],
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
                self.attn_cache[layer_idx].scatter_(
                    1, refresh_index.unsqueeze(-1).expand(-1, -1, C), ctx.o
                )

        ctx.o = self.attn_cache[layer_idx]

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
            self.ffn_cache[layer_idx].scatter_(
                1, self._refresh_index.unsqueeze(-1).expand(-1, -1, C), ctx.ffn_out
            )
        ctx.ffn_out = self.ffn_cache[layer_idx]

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        current_steps = frame.steps.max(-1, keepdim=True).values
        can_generate = torch.sum(
            frame.generated_tokens == self.mask_token_id, dim=-1, keepdim=True
        ).bool()
        refresh_prompt = (current_steps + 1) % self.kp == 0
        refresh_response = (current_steps + 1) % self.kr == 0
        self._prompt_length = frame.prompts.size(-1)

        assert (
            torch.unique(refresh_prompt[can_generate]).numel() <= 1
            and torch.unique(refresh_response[can_generate]).numel() <= 1
        ), "All unfinished sequences must have the same refresh schedule."

        if refresh_prompt[can_generate].numel() > 0:
            # all fisequences are
            self.refresh_prompt = refresh_prompt[can_generate][0].item()
            self.refresh_response = refresh_response[can_generate][0].item()
            self.can_generate = can_generate
