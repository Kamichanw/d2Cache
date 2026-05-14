from __future__ import annotations

import math
from abc import abstractmethod
from typing import cast
from dataclasses import dataclass

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput
from transformers.models.auto import AutoModel
from transformers.utils import logging

from ...cache import dCache, d2Cache
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
)

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAOutputWithPast",
    "LLaDAGenerateOutput",
    "LLaDAPreTrainedModel",
    "LLaDAModelLM",
]


log = logging.get_logger(__name__)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: LLaDAConfig,
        *,
        size: int | None = None,
        elementwise_affine: bool | None = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (
            elementwise_affine is None and self.config.layer_norm_with_affine
        ):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(
        cls, config: LLaDAConfig, size: int | None = None, **kwargs
    ) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(
                f"Unknown LayerNorm type: '{config.layer_norm_type}'"
            )

    def _cast_if_autocast_enabled(
        self, tensor: torch.Tensor, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype()
            )
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype()
            )
        else:
            return tensor


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: LLaDAConfig,
        size: int | None = None,
        low_precision: bool = False,
        elementwise_affine: bool | None = None,
        eps: float = 1e-05,
    ):
        super().__init__(
            config, size=size, elementwise_affine=elementwise_affine, eps=eps
        )
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight)
                if self.weight is not None
                else self.weight
            )
            downcast_bias = (
                self._cast_if_autocast_enabled(self.bias)
                if self.bias is not None
                else self.bias
            )
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x,
                    self.normalized_shape,
                    weight=downcast_weight,
                    bias=downcast_bias,
                    eps=self.eps,
                )
        else:
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: LLaDAConfig,
        size: int | None = None,
        elementwise_affine: bool | None = None,
        eps: float = 1e-5,
    ):
        super().__init__(
            config,
            size=size,
            elementwise_affine=elementwise_affine,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: LLaDAConfig,
        size: int | None = None,
        elementwise_affine: bool | None = None,
        eps: float = 1e-5,
    ):
        super().__init__(
            config,
            size=size,
            elementwise_affine=elementwise_affine,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: LLaDAConfig, device: torch.device | None = None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        if not hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = config.max_sequence_length
        max_position_embeddings = config.max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids: torch.Tensor, device):
        """
        Dynamic RoPE recomputes `inv_freq` when the sequence grows beyond the
        cached length, and resets when it returns to the original scale.
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)  # type: ignore
            self.max_seq_len_cached = self.original_max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)

        if "dynamic" in self.rope_type and position_ids.numel() > 0:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None]  # type: ignore
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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


class Activation(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: LLaDAConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


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


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_idx: int, config: LLaDAConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size
            if config.mlp_hidden_size is not None
            else config.mlp_ratio * config.d_model
        )
        assert config.d_model % config.n_heads == 0

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: LayerNormBase | None = None
        self.q_norm: LayerNormBase | None = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(
                config, elementwise_affine=config.attention_layer_norm_with_affine
            )

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.include_bias,
        )
        setattr(self.attn_out, "layer_idx", layer_idx)
        setattr(self.attn_out, "type_of_module", ModuleType.out_module)

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
        )
        setattr(self.ff_out, "_is_residual", True)
        setattr(self.ff_out, "layer_idx", layer_idx)
        setattr(self.ff_out, "type_of_module", ModuleType.out_module)

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config)

        self.flash_attn_func = None
        if config.flash_attention:
            from flash_attn import flash_attn_func  # type: ignore

            self.flash_attn_func = flash_attn_func

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        num_kv_heads = k.size(1)
        num_heads = q.size(1)
        k = repeat_kv(k, n_rep=num_heads // num_kv_heads)
        v = repeat_kv(v, n_rep=num_heads // num_kv_heads)

        if self.config._attn_implementation == "eager":
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))
            if attn_mask is not None:  # no matter the length, we just slice it
                causal_mask = attn_mask[:, :, :, : k.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=dropout_p, training=self.training
            )
            return torch.matmul(attn_weights, v), (
                attn_weights if output_attentions else None
            )
        else:
            if output_attentions:
                log.warning_once(  # type: ignore
                    "`sdpa` attention does not support `output_attentions=True`."
                    " Please set your attention to `eager` if you want any of these features."
                )
            # Modify: MDM set causal to False, and with no attn_mask.
            return (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                ),
                None,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: dCache | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = q.size(0), q.size(-2)
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = (
                self.q_norm(q.transpose(1, 2).contiguous().view(B, T, -1))
                .to(dtype=dtype)
                .view(B, T, self.config.n_heads, q.size(-1))
                .transpose(1, 2)
            )
            k = (
                self.k_norm(
                    k.transpose(1, 2).contiguous().view(k.size(0), k.size(-2), -1)
                )
                .to(dtype=dtype)
                .view(
                    k.size(0), k.size(-2), self.config.effective_n_kv_heads, k.size(-1)
                )
                .transpose(1, 2)
            )

        if self.config.rope and position_embeddings is not None:
            # Apply rotary embeddings.
            cos, sin = position_embeddings
            active_q_mask = (
                past_key_values.active_q_mask if past_key_values is not None else None
            )
            if cos.size(-2) != q.size(-2):
                assert active_q_mask is not None
                cos = cos[active_q_mask].view(q.size(0), q.size(-2), cos.size(-1))
                sin = sin[active_q_mask].view(q.size(0), q.size(-2), sin.size(-1))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(
                k, v, self.layer_idx, cache_kwargs=past_key_values.cache_kwargs
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att, attn_weights = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
            output_attentions=output_attentions,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, -1, self.config.d_model)

        # Apply output projection.
        return self.attn_out(att), attn_weights

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: dCache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_idx: int, config: LLaDAConfig) -> LLaDABlock:
        if config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_idx, config)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_idx: int, config: LLaDAConfig):
        super().__init__(layer_idx, config)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model,
            q_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
        )

        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
        )

        # Add metadata for init
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.ff_proj, self.up_proj]:
            setattr(proj, "type_of_module", ModuleType.in_module)
            setattr(proj, "layer_idx", layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: dCache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # create a dummy cache to simplify code
        if past_key_values is None:
            past_key_values = dCache(self.config)
        residual = x
        x = self.attn_norm(x)
        with past_key_values.attention(
            self.layer_idx,
            x,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            attention_mask=attention_mask,
        ) as ctx:
            if ctx.q.numel() > 0:
                ctx.o, ctx.attn_weights = self.attention(
                    ctx.q,
                    ctx.k,
                    ctx.v,
                    ctx.attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    output_attentions=bool(output_attentions)
                    or isinstance(past_key_values, d2Cache),
                )
            else:
                ctx.o = residual.new_empty(
                    ctx.q.size(0), ctx.q.size(-2), self.config.d_model
                )
                ctx.attn_weights = None

        q, k, v, o = ctx.q, ctx.k, ctx.v, ctx.o  # keep them for visualization
        attn_weights = ctx.attn_weights
        x = residual + self.dropout(ctx.o)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        residual = x
        with past_key_values.ffn(self.layer_idx, x) as ctx:
            x = self.ff_norm(ctx.hidden_states)
            x, x_up = self.ff_proj(x), self.up_proj(x)  # new add
            x = self.act(x)
            x = x * x_up  # new add
            x = self.ff_out(x)
            ctx.ffn_out = x

        x = residual + self.dropout(ctx.ffn_out)
        return x, attn_weights


@dataclass
class LLaDAOutput(ModelOutput):
    """

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None  # never used, but kept for compatibility
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class LLaDAOutputWithPast(LLaDAOutput):
    """
    Output class for LLaDA model with past key values.

    Args:
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            head_size)`.

            Contains pre-computed key and value hidden states of the attention blocks that can be used to speed up
            decoding.
    """

    past_key_values: dCache | None = None


@dataclass
class LLaDAGenerateOutput(ModelOutput):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class LLaDAPreTrainedModel(PreTrainedModel):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["LLaDABlock", "LLaDALlamaBlock"]

    def _init_weights(self, module):
        """
        Initialize the weights according to the HF best practices and the original model's scheme.
        """
        config = self.config

        if isinstance(module, LayerNormBase):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return

        # Extract metadata attached to the module
        layer_idx = getattr(module, "layer_idx", None)
        type_of_module = getattr(module, "type_of_module", None)

        d = config.d_model
        if isinstance(module, nn.Linear):
            d = module.in_features

        std_factor = 1.0
        if isinstance(module, nn.Embedding) and type_of_module == ModuleType.emb:
            if self.config.scale_logits:
                std_factor = 0.5 * math.sqrt(self.config.d_model)

        if config.init_fn == InitFnType.normal:
            std = config.init_std * std_factor
            if config.init_cutoff_factor is not None:
                cutoff = config.init_cutoff_factor * std
                nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff
                )
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif config.init_fn == InitFnType.mitchell:
            std = std_factor / math.sqrt(d)
            if layer_idx is not None:
                std /= math.sqrt(2 * (layer_idx + 1))
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
        elif config.init_fn == InitFnType.kaiming_normal:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif config.init_fn == InitFnType.fan_in:
            std = std_factor / math.sqrt(d)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif config.init_fn == InitFnType.full_megatron:
            cutoff_factor = config.init_cutoff_factor or 3
            if type_of_module == ModuleType.in_module:
                std = config.init_std
            elif type_of_module == ModuleType.out_module:
                std = config.init_std / math.sqrt(2.0 * config.n_layers)
            elif type_of_module == ModuleType.emb:
                std = config.init_std
            elif type_of_module == ModuleType.final_out:
                std = config.d_model**-0.5
            else:
                raise RuntimeError(
                    f"Unknown module type '{type_of_module}' for megatron init"
                )
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )
        else:
            raise NotImplementedError(config.init_fn)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if config.init_fn == InitFnType.normal and getattr(
                module, "_is_residual", False
            ):
                with torch.no_grad():
                    module.weight.div_(math.sqrt(2 * config.n_layers))


class LLaDAModel(LLaDAPreTrainedModel):

    def __init__(self, config: LLaDAConfig):
        super().__init__(config)
        self.config = config
        if (
            self.config.embedding_size is not None
            and self.config.embedding_size != self.config.vocab_size
        ):
            if self.config.embedding_size < self.config.vocab_size:
                raise ValueError(
                    "embedding size should be at least as big as vocab size"
                )
            elif self.config.embedding_size % 128 != 0:
                log.warning(
                    "Embedding size is not a multiple of 128! This could hurt performance."
                )

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise ValueError("n_layers must be divisible by block_group_size")

        self.gradient_checkpointing = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )
        setattr(self.transformer.wte, "type_of_module", ModuleType.emb)

        blocks = [LLaDABlock.build(i, config) for i in range(config.n_layers)]
        assert self.config.block_group_size == 1
        self.transformer.update({"blocks": nn.ModuleList(blocks)})
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config)

        if not (self.config.alibi or self.config.rope):
            wpe = nn.Embedding(
                config.max_sequence_length,
                config.d_model,
            )
            setattr(wpe, "type_of_module", ModuleType.emb)
            self.transformer.update({"wpe": wpe})

        if not config.weight_tying:
            ff_out = nn.Linear(
                config.d_model,
                config.embedding_size or config.vocab_size,
                bias=config.include_bias,
            )
            setattr(ff_out, "type_of_module", ModuleType.final_out)
            self.transformer.update({"ff_out": ff_out})

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: dCache | None = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeds: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        # Add Basic MDM Model config check
        assert (
            not self.config.alibi
        ), "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        assert self.config.block_group_size == 1

        # create a dummy cache to simplify code
        if past_key_values is None:
            past_key_values = dCache(self.config)
        batch_size, seq_len = (
            input_ids.size()  # type: ignore
            if inputs_embeds is None
            else inputs_embeds.size()[:2]
        )

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if inputs_embeds is None else inputs_embeds  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if position_ids is None:
            # Create position IDs.
            # shape: (batch_size, seq_len)
            if attention_mask is None:
                position_ids = (
                    torch.arange(
                        seq_len,
                        dtype=torch.long,
                        device=x.device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            else:
                position_ids = attention_mask.cumsum(dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

        if hasattr(self.transformer, "wpe"):
            pos_emb = self.transformer.wpe(position_ids)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # decoder layers
        all_attentions = []
        all_hidden_states = []

        with past_key_values.model_forward(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ) as ctx:
            x = ctx.input_embeds
            position_ids = ctx.position_ids
            attention_mask = ctx.attention_mask
            position_embeddings = (
                self.rotary_emb(x, position_ids) if self.config.rope else None
            )
            # Apply blocks one-by-one.
            for block in self.transformer.blocks:  # type: ignore
                if output_hidden_states:
                    all_hidden_states.append(x)
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        block.__call__,
                        x,
                        attention_mask,
                        None,  # past_key_values must be None for checkpointing
                        position_embeddings,
                        output_attentions,
                    )
                else:
                    layer_outputs = block(
                        x,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        position_embeddings=position_embeddings,
                    )

                x = layer_outputs[0]
                if output_attentions:
                    all_attentions.append(layer_outputs[1])

            # Apply final layer norm.
            # shape: (batch_size, seq_len, d_model)
            x = self.transformer.ln_f(x)  # type: ignore
            if output_hidden_states:
                # add final hidden state post-final-layernorm, following HuggingFace's convention
                all_hidden_states.append(x)

            # Get logits.
            # shape: (batch_size, seq_len, vocab_size)
            if self.config.weight_tying:
                logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
            else:
                logits = self.transformer.ff_out(x)  # type: ignore
            if self.config.scale_logits:
                logits.mul_(1 / math.sqrt(self.config.d_model))
            ctx.logits = logits

        return LLaDAOutput(
            logits=cast(torch.FloatTensor, ctx.logits),
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )


class LLaDAModelLM(LLaDAPreTrainedModel):
    """
    LLaDA Model with a language modeling head.
    """

    def __init__(self, config: LLaDAConfig):
        super().__init__(config)
        self.model = LLaDAModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.transformer.wte  # type: ignore

    def set_input_embeddings(self, value: nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self) -> nn.Module:
        if self.config.weight_tying:
            return self.model.transformer.wte  # type: ignore
        else:
            return self.model.transformer.ff_out  # type: ignore

    def set_output_embeddings(self, new_embeddings: nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = new_embeddings
        else:
            self.model.transformer.ff_out = new_embeddings

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.get_output_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: dCache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | LLaDAOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if use_cache and past_key_values is None:
            past_key_values = dCache(self.config)

        outputs = self.model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,  # type: ignore[arg-type]
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            log.warning(
                "Loss calculation within the LLaDA model is not standard. External calculation is recommended."
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LLaDAOutputWithPast(
            loss=loss,
            logits=logits,
            attentions=outputs.attentions if output_attentions else None,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states,
        )

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: dCache | None = None,
        **kwargs,
    ):
        if past_key_values is None:
            past_key_values = dCache(self.config)

        position_ids = kwargs.get("position_ids", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": kwargs.get("attention_mask"),
        }


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)
