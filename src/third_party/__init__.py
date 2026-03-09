import json
import torch

from functools import lru_cache
from pathlib import Path


@lru_cache
def get_token_freq(model_family: str, model_vocab_size: int) -> torch.Tensor | None:
    """
    Get the token frequency for a given model to debias trivial tokens.

    Modified from https://github.com/NEUIR/PC-Sampler/blob/master/src/generate.py.
    """

    corpus_path = Path(__file__).parent / f"{model_family}_corpus.json"
    if not corpus_path.exists():
        return None

    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    num_token = corpus["num_token"]
    raw_counts_dict = corpus["p_baseline_dict"]

    token_frequencies = {
        int(token_id): count / num_token for token_id, count in raw_counts_dict.items()
    }

    background_freq = 1 / num_token
    debiased_freqs = torch.full((model_vocab_size,), background_freq)

    indices = torch.tensor(list(token_frequencies.keys()), dtype=torch.long)
    values = torch.tensor(list(token_frequencies.values()))

    return debiased_freqs.scatter_(0, indices, values)
