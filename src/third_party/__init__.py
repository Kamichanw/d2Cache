import json
import torch

from functools import lru_cache
from pathlib import Path
from transformers.modeling_utils import PreTrainedModel


def get_model_family(model: PreTrainedModel) -> str:
    from src.llada import LLaDAModelLM
    from src.dream import DreamModel

    if isinstance(model, LLaDAModelLM):
        model_family = "llada"
    elif isinstance(model, DreamModel):
        model_family = "dream"
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    return model_family


@lru_cache
def get_token_freq(
    model: PreTrainedModel,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Get the token frequency for a given model to debias trivial tokens.

    Modified from https://github.com/NEUIR/PC-Sampler/blob/master/src/generate.py.
    """
    model_family = get_model_family(model)

    corpus_path = Path(__file__).parent / f"{model_family}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    num_token = corpus["num_token"]
    raw_counts_dict = corpus["p_baseline_dict"]

    token_frequencies = {
        int(token_id): count / num_token for token_id, count in raw_counts_dict.items()
    }

    background_freq = 1 / num_token
    debiased_freqs = torch.full(
        (model.config.vocab_size,), background_freq, device=device, dtype=dtype
    )

    indices = torch.tensor(
        list(token_frequencies.keys()), device=device, dtype=torch.long
    )
    values = torch.tensor(list(token_frequencies.values()), device=device, dtype=dtype)

    return debiased_freqs.scatter_(0, indices, values)
