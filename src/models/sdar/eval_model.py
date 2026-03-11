import torch
import torch.nn.functional as F

from omegaconf import DictConfig
from tqdm import tqdm

from ..eval_mdlm import EvalMDLM


class SDAREval(EvalMDLM):
    """
    Evaluation wrapper for SDAR (Synergy of Diffusion and AutoRegression).

    - Generation is handled by `src/generation/sdar.py` (strategy name: "sdar").
    - Loglikelihood uses standard left-to-right causal LM scoring, so lm-eval
      tasks that rely on loglikelihood can run.
    """

    def __init__(self, cfg: DictConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        # Ensure tokenizer exposes mask_token_id for Frame decoding utilities.
        if not hasattr(self.tokenizer, "mask_token_id") or self.tokenizer.mask_token_id is None:  # type: ignore[attr-defined]
            setattr(self.tokenizer, "mask_token_id", cfg.generation.mask_token_id)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        context_enc: list[int] = self.tokenizer(context).input_ids
        continuation_enc: list[int] = self.tokenizer(continuation).input_ids

        if len(context_enc) == 0:
            # For empty context, prepend the model prefix/BOS token so the first continuation token is scoreable.
            context_enc = [int(self.prefix_token_id)]

        return context_enc, continuation_enc

    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm: bool = False):
        out = []
        for instance in tqdm(
            requests,
            desc="Computing likelihood...",
            disable=disable_tqdm or not self.accelerator.is_main_process,
        ):
            context, continuation = self._encode_pair(*instance.args)
            input_ids = torch.tensor([context + continuation], device=self.device)

            if input_ids.size(1) > self.cfg.max_length:
                # Truncate on the left (keep tail, which contains continuation).
                input_ids = input_ids[:, -self.cfg.max_length :]
                # If we truncated into the continuation boundary, we can't reconstruct the exact split;
                # fall back to scoring everything except the first token.
                context_len = max(1, input_ids.size(1) - len(continuation))
            else:
                context_len = len(context)

            logits = self.model(input_ids).logits  # (1, L, V)
            log_probs = F.log_softmax(logits[:, :-1].to(torch.float32), dim=-1)
            target_ids = input_ids[:, 1:]

            cont_start = max(context_len - 1, 0)
            cont_log_probs = log_probs[:, cont_start:]
            cont_targets = target_ids[:, cont_start:]

            token_log_probs = cont_log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
            logprob = token_log_probs.sum().item()

            greedy = cont_log_probs.argmax(dim=-1)
            is_greedy = bool(torch.all(greedy == cont_targets).item())
            out.append((logprob, is_greedy))

        return out

