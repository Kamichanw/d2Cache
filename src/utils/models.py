import transformers

from omegaconf import DictConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


def load_pretrained_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    """
    Load a pretrained model based on the configuration.
    """
    from ..models import LLaDAModelLM, DreamModel

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        model = LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dream":
        model = DreamModel.from_pretrained(cfg.model.path, **model_kwargs)
    else:
        raise ValueError(f"Unsupported pretrained model: {cfg.model.name}")

    return model


def load_eval_model(cfg: DictConfig, **model_kwargs):
    from ..models import LLaDAEval, DreamEval

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == "dream":
        eval_model = DreamEval(cfg, **model_kwargs)
    else:
        raise NotImplementedError(
            f"Model family {model_family} is not implemented for evaluation."
        )

    return eval_model


def load_tokenizer(cfg: DictConfig, **tokenizer_kwargs):

    # ---------------- Tokenizer loading ----------------
    tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.path, **tokenizer_kwargs
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Model-specific customization ----------------
    model_family = cfg.model.name.split("-")[0]
    match model_family:
        case "llada":
            tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
            tokenizer.eot_token = "<|eot_id|>"

            # fix bugs in chat template
            tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
        case "dream":
            tokenizer.eot_token = "<|im_end|>"
            tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.eot_token
            )
    return tokenizer


def is_adapted_from_ar(
    model_or_config: PreTrainedModel | PretrainedConfig,
) -> bool:
    """
    Check if a model or its configuration is adapted from an autoregressive architecture.
    """

    if isinstance(model_or_config, PreTrainedModel):
        config = model_or_config.config
    elif isinstance(model_or_config, PretrainedConfig):
        config = model_or_config
    else:
        raise ValueError(
            f"Expected model_or_config to be an instance of PreTrainedModel or PretrainedConfig, but got {type(model_or_config)}"
        )

    if config.model_type.lower() == "llada":
        return False
    elif config.model_type.lower() == "dream":
        return True
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
