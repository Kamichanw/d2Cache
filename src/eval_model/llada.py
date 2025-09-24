import torch
import torch.nn.functional as F

from omegaconf import DictConfig
from tqdm import tqdm

from src.eval_model.model_base import EvalModelBase


class LLaDAEval(EvalModelBase):
    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ):
        """
        Args:
            mask_token_id: The token id of [MASK] is 126336.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
        """

        super().__init__(cfg, **kwargs)

        self.mc_num = cfg.mc_num
        self.batch_size = cfg.batch_size
        self.is_check_greedy = cfg.is_check_greedy
        self.mask_token_id = cfg.generation.mask_token_id
        

        if self.mc_num is not None:
            assert self.mc_num % self.batch_size == 0

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, prompt_index.sum(), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_token_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        mc_num = self.mc_num or 1
        for _ in range(mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_token_id

            logits = self.model(perturbed_seq).logits

            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_token_id, device=self.device
        )
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_token_id
            logits = self.model(seq).logits[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_token_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context: str, continuation: str):
        """
        Move spaces at the end of context to the beginning of continuation, and
        encode both context and continuation into token ids. This is modified from
        `lm_eval.api.model.TemplateLM._encode_pair`.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc: list[int] = self.tokenizer(context + continuation).input_ids
        context_enc: list[int] = self.tokenizer(context).input_ids

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """

        out = []
        for instance in tqdm(requests, desc="Computing likelihood..."):
            context, continuation = self._encode_pair(*instance.args)
            assert len(context) + len(continuation) <= self.cfg.max_length, (
                f"Context + continuation length exceeds {self.cfg.max_length} tokens: "
                f"{len(context)} + {len(continuation)}"
            )
            context = torch.tensor(context, device=self.device)
            continuation = torch.tensor(continuation, device=self.device)

            logprob = self.get_loglikelihood(context, continuation)
            isgreedy = self.suffix_greedy_prediction(context, continuation)
            out.append((logprob, isgreedy))

        return out