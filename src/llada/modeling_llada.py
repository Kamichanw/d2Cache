from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from functools import partial
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    cast,
)
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput
from transformers.models.auto import AutoModel

from ..cache import dCache
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ActivationCheckpointingStrategy,
)

from ..utils import Timer

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

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
]


log = logging.getLogger(__name__)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: LLaDAConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value
            )
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)  # type: ignore
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)  # type: ignore
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(
                f"When using the {InitFnType.full_megatron} init, every module must have a type."
            )

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
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


def activation_checkpoint_function(cfg: LLaDAConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0)
        and (cfg.embedding_dropout == 0.0)
        and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):  # type: ignore
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: LLaDAConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (
            elementwise_affine is None and self.config.layer_norm_with_affine
        ):
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=config.init_device)
            )
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, device=config.init_device)
                )
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
        cls, config: LLaDAConfig, size: Optional[int] = None, **kwargs
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
        self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None
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

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: LLaDAConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
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
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
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
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
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

    def __init__(self, config: LLaDAConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(
            config.max_sequence_length, _non_meta_init_device(config)
        )

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin, pos_cos

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = (
                positions.sin()[None, None, :, :],
                positions.cos()[None, None, :, :],
            )
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_position_ids: torch.Tensor | None = None,
        kv_position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        if q_position_ids is None:
            q_len = q.shape[2]
            q_position_ids = torch.arange(q_len, device=q.device).unsqueeze(0)

        if kv_position_ids is None:
            k_len = k.shape[2]
            kv_position_ids = torch.arange(k_len, device=k.device).unsqueeze(0)

        max_pos = int(torch.max(q_position_ids.max(), kv_position_ids.max()).item())
        pos_sin, pos_cos = self.get_rotary_embedding(max_pos + 1, q_.device)

        pos_sin = pos_sin.squeeze(0).squeeze(0)
        pos_cos = pos_cos.squeeze(0).squeeze(0)

        sin_q = pos_sin[q_position_ids].type_as(q_)
        cos_q = pos_cos[q_position_ids].type_as(q_)
        q_ = self.apply_rotary_pos_emb(sin_q.unsqueeze(1), cos_q.unsqueeze(1), q_)

        sin_k = pos_sin[kv_position_ids].type_as(k_)
        cos_k = pos_cos[kv_position_ids].type_as(k_)
        k_ = self.apply_rotary_pos_emb(sin_k.unsqueeze(1), cos_k.unsqueeze(1), k_)

        return q_.type_as(q), k_.type_as(k)


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

    def __init__(self, layer_id: int, config: LLaDAConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size
            if config.mlp_hidden_size is not None
            else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
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
            device=config.init_device,
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.flash_attention:
            from flash_attn import flash_attn_func  # type: ignore

            self.flash_attn_func = flash_attn_func

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            return torch.matmul(attn_weights, v), attn_weights
        else:
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
        attention_mask: Optional[torch.Tensor] = None,
        q_position_ids: Optional[torch.Tensor] = None,
        kv_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, -1, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(
            B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(
            B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k, q_position_ids, kv_position_ids)

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att, attn_weight = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, -1, C)

        # Apply output projection.
        return self.attn_out(att), attn_weight

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[dCache] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(
        cls, layer_id: int, config: LLaDAConfig, cache: BufferCache
    ) -> LLaDABlock:
        if config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: LLaDAConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
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
            device=config.init_device,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(
            self.config, self.up_proj, d=self.config.d_model, layer_id=None
        )  # new add

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[dCache] = None,
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # create a dummy cache to simplify code
        past_key_values = past_key_values or dCache(self.config)
        with past_key_values.attention(
            self.layer_id,
            x,
            self.attn_norm,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ) as ctx:
            q_mismatch = ctx.q.shape != x.shape and (
                ctx.q_position_ids is None
                or ctx.q_position_ids.shape != ctx.q.shape[:2]
            )
            kv_mismatch = (ctx.k.shape != x.shape or ctx.v.shape != x.shape) and (
                ctx.kv_position_ids is None
                or ctx.kv_position_ids.shape != ctx.k.shape[:2]
            )
            if q_mismatch or kv_mismatch:
                raise ValueError(
                    "If you select a subset of the qkv in past_key_values, "
                    "the q, k, v must match the shape of corresponding position_ids."
                )

            if ctx.q.numel() > 0:
                ctx.o, ctx.attn_weight = self.attention(
                    ctx.q,
                    ctx.k,
                    ctx.v,
                    ctx.attention_mask,
                    q_position_ids=ctx.q_position_ids,
                    kv_position_ids=ctx.kv_position_ids,
                )
            else:
                ctx.o, ctx.attn_weight = torch.empty_like(ctx.q), None

        q, k, v, o = ctx.q, ctx.k, ctx.v, ctx.o  # keep them for visualization
        x = ctx.residual  + self.dropout(ctx.o)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        with past_key_values.ffn(self.layer_id, x) as ctx:
            x = self.ff_norm(ctx.x)
            x, x_up = self.ff_proj(x), self.up_proj(x)  # new add
            x = self.act(x)
            x = x * x_up  # new add
            x = self.ff_out(x)
            ctx.ffn_out = x

        x = ctx.residual + self.dropout(ctx.ffn_out)
        return x


@dataclass
class LLaDAOutput(ModelOutput):
    """

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None  # never used, but kept for compatibility
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class LLaDAOutputWithPast(LLaDAOutput):
    """
    Output class for LLaDA model with past key values.

    Args:
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            head_size)`.

            Contains pre-computed key and value hidden states of the attention blocks that can be used to speed up
            decoding.
    """

    past_key_values: Optional[dCache] = None


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


class LLaDAModel(nn.Module):
    def __init__(self, config: LLaDAConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        if (
            self.config.embedding_size is not None
            and self.config.embedding_size != self.config.vocab_size
        ):
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception(
                    "embedding size should be at least as big as vocab size"
                )
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.",
                    UserWarning,
                )

        self.activation_checkpointing_strategy: Optional[
            ActivationCheckpointingStrategy
        ] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(
            self.config
        )

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(
            False
        )  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                    device=config.init_device,
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [
            LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)
        ]
        assert self.config.block_group_size == 1
        self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {
                    "wpe": nn.Embedding(
                        config.max_sequence_length,
                        config.d_model,
                        device=config.init_device,
                    )
                }
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self.transformer.blocks:  # type: ignore
            block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(
                (0.5 * math.sqrt(self.config.d_model))
                if self.config.scale_logits
                else 1.0
            ),
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        assert self.config.block_group_size == 1
        for block in self.transformer.blocks:  # type: ignore
            block.reset_parameters()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[dCache] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
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
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide input_ids or inputs_embeds.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        # create a dummy cache to simplify code
        past_key_values = past_key_values or dCache(self.config)
        batch_size, seq_len = (
            input_ids.size() # type: ignore
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

        if not (self.config.alibi or self.config.rope):
            pos_emb = self.transformer.wpe(position_ids)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # decoder layers
        all_hidden_states = []

        with past_key_values.model_forward(x) as ctx:
            x = ctx.x
            # Apply blocks one-by-one.
            for block_idx, block in enumerate(self.transformer.blocks):  # type: ignore
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)
                # shape: (batch_size, seq_len, d_model)
                x = block(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )

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

        return LLaDAOutput(logits=ctx.logits, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]


class LLaDAModelLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _no_split_modules = ["LLaDABlock", "LLaDALlamaBlock"]

    def __init__(
        self,
        config: LLaDAConfig,
        model: Optional[LLaDAModel] = None,
        init_params: bool = False,
    ):
        super().__init__(config)

        if not model:
            self.model = LLaDAModel(config, init_params=init_params)
        else:
            self.model = model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[dCache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LLaDAOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache and past_key_values is None:
            raise ValueError(
                "use_cache is True but past_key_values is None. "
                "You must pass a specific dCache to use cache."
            )

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,  # type: ignore[arg-type]
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            import warnings

            warnings.warn(
                "Note that for LLaDA, you cannot calculate the loss here.", UserWarning
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LLaDAOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple]] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte  # type: ignore

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self) -> torch.nn.Module:
        if self.config.weight_tying:
            return self.model.transformer.wte  # type: ignore
        else:
            return self.model.transformer.ff_out  # type: ignore

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)
