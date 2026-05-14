# This file is adapted from the SDAR model implementation released by JetAstra.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os

import torch
from torch import nn
from einops import rearrange

from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub  # type: ignore
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils.auto_docstring import (
    HARDCODED_CONFIG_FOR_MODELS,
    auto_docstring,
)
from transformers.utils.generic import can_return_tuple
from transformers.utils.import_utils import is_torch_flex_attn_available
from transformers.utils import logging
from .configuration_sdar import SDARConfig
from .fused_linear_diffusion_cross_entropy import FusedLinearDiffusionCrossEntropyLoss
from ...cache import BlockdCache

import torch.nn.functional as F

HARDCODED_CONFIG_FOR_MODELS.setdefault("sdar", "SDARConfig")
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401

    liger_kernel_is_available = True
except ImportError:
    liger_kernel_is_available = False


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention


logger = logging.get_logger(__name__)


def modify_padded_position_ids_2d(position_ids: torch.LongTensor) -> torch.LongTensor:
    """
    Adjust packed position ids with vectorized PyTorch operations.
    The input is expected to be a 2D tensor with shape
    `(batch_size, sequence_length)`, and each row is processed independently.

    Args:
        position_ids: A 2D tensor with shape `(batch_size, sequence_length)`.

    Returns:
        The adjusted position ids with shape `(batch_size, sequence_length)`.
    """
    if position_ids.dim() != 2:
        raise ValueError(
            f"Input tensor must be 2D, but got {position_ids.dim()} dimensions."
        )

    batch_size, seq_len = position_ids.shape
    device = position_ids.device

    col_indices = torch.arange(seq_len, device=device, dtype=position_ids.dtype).expand(
        batch_size, -1
    )
    mask = position_ids != 0

    masked_indices = col_indices * mask
    last_nonzero_idx = torch.max(masked_indices, dim=1).values
    has_nonzero = torch.any(mask, dim=1)
    pad_start_idx = torch.where(
        has_nonzero,
        last_nonzero_idx + 1,
        torch.tensor(0, device=device, dtype=position_ids.dtype),
    )

    padding_mask = col_indices >= pad_start_idx.unsqueeze(1)
    new_pad_values = col_indices - pad_start_idx.unsqueeze(1)
    position_ids = torch.where(padding_mask, new_pad_values, position_ids)  # type: ignore[assignment]

    return position_ids


def calculate_token_nums(position_ids: torch.Tensor):
    """
    Compute the length of each packed segment in a batch.

    Args:
        position_ids: A 2D tensor with shape `(batch_size, sequence_length)`.
            For example: `tensor([[0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,0,0,0]])`.
    Returns:
        A list containing one tensor of packed segment lengths per batch item.
        For example: `[tensor([5, 6, 4, 1, 1, 1])]`.
    """
    if position_ids.dim() != 2:
        raise ValueError(f"Input must be a 2D tensor, but got {position_ids.dim()}D.")

    all_lengths = []

    # Rows can have different numbers of packed segments, so batch-level
    # iteration keeps the ragged structure simple while each row remains
    # vectorized.
    for pids_row in position_ids:
        seq_len = pids_row.shape[0]

        # Each segment starts where the packed position id resets to 0.
        zero_indices = torch.nonzero(pids_row == 0).flatten()

        # Add the full row length as a sentinel so the last segment length can
        # be computed with the same adjacent difference operation.
        split_points = torch.cat(
            [
                zero_indices,
                torch.tensor(
                    [seq_len], device=pids_row.device, dtype=zero_indices.dtype
                ),
            ]
        )

        lengths = torch.diff(split_points)

        all_lengths.append(lengths)

    return all_lengths


def forward_add_noise_packed(
    inputs_ids: torch.Tensor,
    num_tokens_list: list[torch.Tensor],
    prompt_mask: torch.Tensor,
    mask_id: int,
    eps: float = 1e-3,
    max_tries: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add mask noise to a batch of packed token id sequences.

    Each logical sample inside a packed batch item receives an independent
    random noise rate. Tokens selected for noising are replaced by `mask_id`,
    while prompt positions are left unchanged.

    Args:
        inputs_ids (torch.Tensor):
            Token ids with shape `(bsz, total_tokens)`.
        num_tokens_list (list[torch.Tensor]):
            One tensor per batch item. Each tensor stores the lengths of the
            logical samples packed into that row, for example
            `[tensor([len1, len2]), tensor([len3, len4, len5])]`.
        prompt_mask (torch.Tensor):
            Boolean tensor with shape `(bsz, total_tokens)`. True positions are
            prompt tokens and must not be noised.
        mask_id (int):
            Token id used to replace noised tokens.
        eps (float):
            Small floor that prevents the noise rate from becoming exactly zero.
        max_tries (int):
            Maximum retries per batch item to ensure at least one non-prompt
            token is noised.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - noisy_input_ids (torch.Tensor):
            Token ids after noising, with shape `(bsz, total_tokens)`.
        - final_masked_indices (torch.Tensor):
            Boolean mask indicating which positions were noised, with shape
            `(bsz, total_tokens)`.
        - p_masks (torch.Tensor):
            A 1D tensor containing the noise rate for each noised token.
    """
    bsz, total_tokens = inputs_ids.shape
    device = inputs_ids.device
    if max_tries < 1:
        raise ValueError(f"max_tries must be >= 1, got {max_tries}.")

    assert (
        len(num_tokens_list) == bsz
    ), f"num_tokens_list length ({len(num_tokens_list)}) must equal bsz ({bsz})."
    assert prompt_mask.shape == (
        bsz,
        total_tokens,
    ), f"prompt_mask shape mismatch, expected {(bsz, total_tokens)}, got {prompt_mask.shape}."

    noisy_ids_list = []
    final_masked_indices_list = []
    p_masks_per_token_list = []

    # Iterate over batch rows because each row can have a different packed layout.
    for i in range(bsz):
        current_ids = inputs_ids[i : i + 1]  # shape: (1, total_tokens)
        current_num_tokens = num_tokens_list[i]
        current_prompt_mask = prompt_mask[i : i + 1]  # shape: (1, total_tokens)

        num_samples_in_item = len(current_num_tokens)
        assert total_tokens == torch.sum(
            current_num_tokens
        ), f"Batch item {i} has packed token sum {torch.sum(current_num_tokens)}, expected {total_tokens}."

        eligible_for_masking = ~current_prompt_mask

        if not eligible_for_masking.any():
            noisy_ids_list.append(current_ids)
            final_masked_indices_list.append(
                torch.zeros_like(current_prompt_mask, dtype=torch.bool)
            )
            p_masks_per_token_list.append(
                torch.full((1, total_tokens), eps, device=device, dtype=torch.float)
            )
            continue

        final_masked_indices_item = torch.zeros_like(
            current_prompt_mask, dtype=torch.bool
        )
        p_mask_per_token = None

        for _ in range(max_tries):
            t = torch.rand(num_samples_in_item, device=device)
            p_mask_per_sample = (1 - eps) * t + eps

            p_mask_per_token_1d = torch.repeat_interleave(
                p_mask_per_sample, current_num_tokens
            )
            p_mask_per_token = p_mask_per_token_1d.unsqueeze(
                0
            )  # shape: (1, total_tokens)

            masked_indices = torch.rand_like(p_mask_per_token) < p_mask_per_token
            final_masked_indices_item = masked_indices & eligible_for_masking

            if final_masked_indices_item.any():
                break

        # Extremely rarely all retries can miss; force one eligible token then.
        if not final_masked_indices_item.any():
            eligible_indices = torch.nonzero(
                eligible_for_masking.squeeze(0), as_tuple=True
            )[0]
            if len(eligible_indices) > 0:
                random_choice = torch.randint(0, len(eligible_indices), (1,)).item()
                force_mask_idx = eligible_indices[random_choice]  # type: ignore[index]
                final_masked_indices_item[0, force_mask_idx] = True

        noisy_ids_item = torch.where(final_masked_indices_item, mask_id, current_ids)

        noisy_ids_list.append(noisy_ids_item)
        final_masked_indices_list.append(final_masked_indices_item)
        p_masks_per_token_list.append(p_mask_per_token)  # type: ignore[arg-type]

    noisy_input_ids = torch.cat(noisy_ids_list, dim=0)
    final_masked_indices = torch.cat(final_masked_indices_list, dim=0)
    p_mask_full = torch.cat(p_masks_per_token_list, dim=0)

    p_masks = p_mask_full[final_masked_indices]

    return noisy_input_ids, final_masked_indices, p_masks


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


def block_attn_mask(num_tokens, block_size, device):
    masks = []
    for i in range(len(num_tokens)):
        cur_masks = []
        for num in num_tokens[i]:
            # Return the full 2n x 2n mask for each packed sample.
            single_mask = block_diff_mask(
                b=None,
                h=None,
                q_idx=torch.arange(num * 2, device=device)[:, None],
                kv_idx=torch.arange(num * 2, device=device)[None, :],
                block_size=block_size,
                n=num,
            )
            cur_masks.append(single_mask)
        masks.append(torch.block_diag(*cur_masks))
    masks = torch.stack(masks, dim=0)
    return masks


if is_torch_flex_attn_available():

    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def fused_flex_attention(query, key, value, attention_mask, **kwargs):
        return flex_attention(query, key, value, block_mask=attention_mask, **kwargs)

else:  # pragma: no cover

    def fused_flex_attention(*_args, **_kwargs):
        raise ImportError(
            "SDAR requires PyTorch Flex Attention. "
            "Upgrade PyTorch to a build that includes `torch.nn.attention.flex_attention`."
        )


@use_kernel_forward_from_hub("RMSNorm")
class SDARRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        SDARRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SDARMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if liger_kernel_is_available:
            return self.down_proj(
                LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
            )
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SDARAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SDARConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        # unlike olmo, only on the head dim!
        self.q_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # thus post q_norm does not need reshape
        self.k_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BlockdCache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is None:
            past_key_values = BlockdCache(self.config)

        with past_key_values.attention(
            self.layer_idx,
            hidden_states,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            attention_mask=attention_mask,
        ) as ctx:
            attention_mask = ctx.attention_mask

            query_states = self.q_norm(ctx.q)
            key_states = self.k_norm(ctx.k)
            value_states = ctx.v

            cos, sin = position_embeddings
            active_q_mask = (
                past_key_values.active_q_mask if past_key_values is not None else None
            )
            if cos.size(-2) != query_states.size(-2):
                cos = cos[active_q_mask].view(  # type: ignore[index]
                    query_states.size(0), query_states.size(-2), cos.size(-1)
                )
                sin = sin[active_q_mask].view(  # type: ignore[index]
                    query_states.size(0), query_states.size(-2), sin.size(-1)
                )
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_values is not None and kwargs.get("use_cache", False):
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs=past_key_values.cache_kwargs,
                )

            if self.training:
                attn_output, attn_weights = fused_flex_attention(  # type: ignore
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    attention_mask=attention_mask,
                    enable_gqa=True,
                    scale=self.scaling,
                    return_lse=True,
                )
                attn_weights = (
                    attn_weights.to(value_states.dtype)
                    if attn_weights is not None
                    else None
                )
                attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
            else:
                attn_weights = None
                if attention_mask is None:
                    can_use_flash = True
                elif attention_mask.dtype == torch.bool:
                    can_use_flash = bool(torch.all(attention_mask))
                else:
                    can_use_flash = bool(torch.all(attention_mask == 0))
                if can_use_flash and flash_attn_func is not None:
                    query_states = query_states.transpose(1, 2)
                    key_states = key_states.transpose(1, 2)
                    value_states = value_states.transpose(1, 2)
                    attn_output = flash_attn_func(
                        query_states,
                        key_states,
                        value_states,
                        causal=False,
                        softmax_scale=self.scaling,
                    )
                    attn_output = rearrange(attn_output, "b l h d -> b l (h d)")
                else:
                    attn_output = F.scaled_dot_product_attention(
                        query=query_states,
                        key=key_states,
                        value=value_states,
                        attn_mask=attention_mask,
                        is_causal=False,
                        scale=self.scaling,
                        enable_gqa=True,
                    )
                    attn_output = rearrange(attn_output, "b h l d -> b l (h d)")

            attn_output = self.o_proj(attn_output)
            ctx.o = attn_output
            ctx.attn_weights = attn_weights

        return attn_output, attn_weights


class SDARDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SDARConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = SDARAttention(config=config, layer_idx=layer_idx)
        self.mlp = SDARMLP(config)
        self.input_layernorm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SDARRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BlockdCache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # Self Attention
        residual = hidden_states
        if past_key_values is None:
            past_key_values = BlockdCache(self.self_attn.config)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,  # type: ignore[arg-type]
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        with past_key_values.ffn(self.layer_idx, hidden_states) as ctx:
            hidden_states = self.post_attention_layernorm(ctx.hidden_states)
            hidden_states = self.mlp(hidden_states)
            ctx.ffn_out = hidden_states

        hidden_states = residual + ctx.ffn_out  # type: ignore[operator]

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)  # type: ignore[assignment]

        return outputs  # type: ignore[return-value]


@auto_docstring
class SDARPreTrainedModel(PreTrainedModel):
    config_class = SDARConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SDARDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SDARRMSNorm):
            module.weight.data.fill_(1.0)


class SDARRotaryEmbedding(nn.Module):
    def __init__(self, config: SDARConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    # power user: used with advanced RoPE types (e.g. dynamic rope)
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class SDARModel(SDARPreTrainedModel):
    def __init__(self, config: SDARConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                SDARDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SDARRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BlockdCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), BlockdCache)):
            raise ValueError("SDAR only supports BlockdCache as `past_key_values`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = BlockdCache(self.config)

        if position_ids is None:
            position_ids = (
                torch.arange(  # type: ignore[assignment]
                    inputs_embeds.shape[1], device=inputs_embeds.device
                )
                .unsqueeze(0)
                .expand(inputs_embeds.shape[0], -1)
            )  # type: ignore[assignment]

        cm = ctx = None
        if use_cache:
            cm = past_key_values.model_forward(
                inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            ctx = cm.__enter__()  # type: ignore[union-attr]
            hidden_states = ctx.input_embeds  # type: ignore[union-attr]
            position_ids = ctx.position_ids  # type: ignore[union-attr]
            attention_mask = ctx.attention_mask  # type: ignore[union-attr]
        else:
            hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # type: ignore[arg-type]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # type: ignore[operator]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        if cm is not None:
            output.cm = cm  # type: ignore[attr-defined]
            output.ctx = ctx  # type: ignore[attr-defined]
        return output


@auto_docstring
class SDARForCausalLM(SDARPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = SDARModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_for_bd_training(
        self, inputs_ids, position_ids, prompt_mask, mask_token_id: int
    ):
        bsz, seq_len = inputs_ids.shape
        num_tokens = calculate_token_nums(position_ids)  # list[torch.Tensor]
        noisy_inputs_ids, logits_to_keep_half, p_mask = forward_add_noise_packed(
            inputs_ids=inputs_ids,
            num_tokens_list=num_tokens,
            prompt_mask=prompt_mask,
            mask_id=mask_token_id,
        )
        router_noisy_part_list = []
        for i in range(bsz):
            cur_router_noisy_part = (
                torch.arange(num_tokens[i].shape[0] * 2) % 2 == 0
            ).to(inputs_ids.device)
            cur_router_noisy_part = cur_router_noisy_part.repeat_interleave(
                num_tokens[i].repeat_interleave(2)
            )
            router_noisy_part_list.append(cur_router_noisy_part)
        router_noisy_part = torch.stack(router_noisy_part_list, dim=0)

        # concated inputs_ids: (bzs, seq_len x 2)
        concat_inputs_ids = inputs_ids.repeat(1, 2)
        # concated logits_to_keep: (bsz, seq_len x 2)
        logits_to_keep = torch.zeros(
            bsz, 2 * seq_len, dtype=torch.bool, device=inputs_ids.device
        )
        # concated position_ids: (bsz, seq_len x 2)
        concat_position_ids = torch.zeros(
            bsz, 2 * seq_len, dtype=position_ids.dtype, device=position_ids.device
        )
        for i in range(bsz):
            concat_inputs_ids[i][router_noisy_part[i]] = noisy_inputs_ids[i]
            concat_inputs_ids[i][~router_noisy_part[i]] = inputs_ids[i]

            logits_to_keep[i][router_noisy_part[i]] = logits_to_keep_half[i]

            concat_position_ids[i][router_noisy_part[i]] = position_ids[i]
            concat_position_ids[i][~router_noisy_part[i]] = position_ids[i]

        # create flex_attention mask
        attention_mask = block_attn_mask(
            num_tokens, self.config.block_size, inputs_ids.device
        )
        flex_attention_mask_3d = create_block_mask(
            lambda b, h, q_idx, kv_idx: attention_mask[b, q_idx, kv_idx],
            B=attention_mask.size(0),
            H=None,
            Q_LEN=attention_mask.size(1),
            KV_LEN=attention_mask.size(2),
        )

        return (
            concat_inputs_ids,
            concat_position_ids,
            flex_attention_mask_3d,
            logits_to_keep_half,
            logits_to_keep,
            p_mask,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: BlockdCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, SDARForCausalLM

        >>> model = SDARForCausalLM.from_pretrained("DiffuOpen/SDAR-1.7B-Chat")
        >>> tokenizer = AutoTokenizer.from_pretrained("DiffuOpen/SDAR-1.7B-Chat")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not isinstance(past_key_values, (type(None), BlockdCache)):
            raise ValueError("SDAR only supports BlockdCache as `past_key_values`.")
        if self.training:
            assert inputs_embeds is None, "only support input_ids during training"
            assert labels is not None, "Labels must be provided for training."
            assert (
                position_ids is not None
            ), "position_ids must be provided for training."
            mask_token_id = kwargs.pop("mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = int(os.environ["MASK_TOKEN_ID"])
            prompt_mask = labels == -100
            position_ids = modify_padded_position_ids_2d(position_ids)
            (
                concat_inputs_ids,
                concat_position_ids,
                flex_attention_mask_3d,
                logits_to_keep_half,
                logits_to_keep,
                p_mask,
            ) = self.prepare_for_bd_training(
                input_ids, position_ids, prompt_mask, mask_token_id
            )
            outputs = self.model(
                input_ids=concat_inputs_ids,
                attention_mask=flex_attention_mask_3d,
                position_ids=concat_position_ids,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            hidden_states = hidden_states[logits_to_keep].contiguous()
            answer_len = (labels != -100).sum()
            loss_fct = FusedLinearDiffusionCrossEntropyLoss(reduction="sum")
            loss = loss_fct(  # it will return (sum_loss, unreduced_loss)
                # conduct `view(-1, V)` inside the function
                x=hidden_states,
                target=labels[logits_to_keep_half].contiguous(),
                weight=self.lm_head.weight,
                bias=self.lm_head.bias,
                p_mask=p_mask,
            )
            loss = loss / answer_len
            logits = None
        else:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs: BaseModelOutputWithPast = self.model(  # type: ignore[no-redef]
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            hidden_states = hidden_states[:, slice_indices, :].contiguous()
            fuse_linear_and_cross_entropy = (
                self.config.fuse_cross_entropy and self.training
            )
            if fuse_linear_and_cross_entropy:
                # When using fused_linear_ce_loss, we do not compute the whole logits on HBM
                logits = None
            else:
                logits = self.lm_head(hidden_states)
                if use_cache and hasattr(outputs, "cm"):
                    outputs.ctx.logits = logits  # type: ignore[attr-defined]
                    outputs.cm.__exit__(None, None, None)  # type: ignore[attr-defined]
                    logits = outputs.ctx.logits  # type: ignore[attr-defined]

            loss = None
            if labels is not None:
                # FusedLinearCrossEntropyLoss will be implemented by monkey patch when training
                # We don't use it when inferencing
                loss_fct = nn.CrossEntropyLoss()  # type: ignore[assignment]  # nn.CE
                loss = loss_fct(
                    logits.view(-1, self.config.vocab_size), labels.view(-1)  # type: ignore[union-attr]
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "SDARForCausalLM",
    "SDARModel",
    "SDARPreTrainedModel",
]
