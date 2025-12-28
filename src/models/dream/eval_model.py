import torch
import torch.nn.functional as F
import torch.amp

from typing import List
from loguru import logger
from omegaconf import DictConfig
from lm_eval.api.instance import Instance
from tqdm import tqdm

from ..eval_mdlm import EvalMDLM


class DreamEval(EvalMDLM):
    def __init__(
        self,
        cfg: DictConfig,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)

        self.add_bos_token = self.cfg.add_bos_token
        self.batch_size = cfg.batch_size
        self.nll_type = cfg.nll_type
        self.log_type = cfg.log_type
        self.mc_num = cfg.mc_num
        self.max_length = cfg.max_length
        self.mask_token_id: int = getattr(self.tokenizer, "mask_token_id")
        self.sampling_eps = getattr(self.model.generation_config, "eps")

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        """
        prompt_index : 1D bool tensor, length=batch.shape[1]
        """
        input = batch
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore
            logits = self.model(input).logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == "ftb":
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            # eval_logger.info("before noising")
            perturbed_seq_, p_mask = self._forward_process(seq)
            # eval_logger.info("end noising")
            if self.log_type == "ftb":
                perturbed_seq[:, -len(target) :] = perturbed_seq_[:, -len(target) :]
            elif self.log_type == "btf":
                perturbed_seq[:, : len(prefix)] = perturbed_seq_[:, : len(prefix)]
            elif self.log_type == "union":
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0)  # 1*l1, 1*l2
        assert self.log_type in ["ftb", "btf"]
        assert self.nll_type in ["ar_ftb", "ar_btf"]

        if self.log_type == "ftb":
            prompt_index = (
                torch.arange(prefix.shape[1] + target.shape[1], device=self.device)
                < prefix.shape[1]
            )
        else:
            prompt_index = (
                torch.arange(prefix.shape[1] + target.shape[1], device=self.device)
                >= prefix.shape[1]
            )

        if self.log_type == "ftb":
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous()  # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous()  # l1*l1

        mask_index = torch.ones(
            (perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool
        )
        if self.nll_type == "ar_ftb":
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id
        if self.log_type == "ftb":
            perturbed_seq = torch.cat(
                [prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1
            )
        else:
            perturbed_seq = torch.cat(
                [perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1
            )

        logits_ = []
        num = (
            len(perturbed_seq) // self.batch_size
            if len(perturbed_seq) % self.batch_size == 0
            else len(perturbed_seq) // self.batch_size + 1
        )
        for i in range(num):
            end = (
                (i + 1) * self.batch_size
                if (i + 1) * self.batch_size < len(perturbed_seq)
                else len(perturbed_seq)
            )
            perturbed_seq_ = perturbed_seq[i * self.batch_size : end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones(
            (perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool
        )
        if self.nll_type == "ar_ftb":
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == "ftb":
            logits_index = torch.cat(
                [
                    torch.zeros(
                        (perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool
                    ),
                    mask_index,
                ],
                dim=-1,
            )
        else:
            logits_index = torch.cat(
                [
                    mask_index,
                    torch.zeros(
                        (perturbed_.shape[1], target.shape[1]), dtype=torch.bool
                    ),
                ],
                dim=-1,
            )

        if self.log_type == "ftb":
            loss = (
                F.cross_entropy(logits[logits_index], target[0], reduction="sum")
                .cpu()
                .item()
            )
        else:
            loss = (
                F.cross_entropy(logits[logits_index], prefix[0], reduction="sum")
                .cpu()
                .item()
            )
        return loss

    def _encode_pair(self, context: str, continuation: str):
        """
        Move spaces at the end of context to the beginning of continuation, and
        encode both context and continuation into token ids. This is modified from
        `lm_eval.api.model.TemplateLM._encode_pair`.
        """
        if self.add_bos_token:
            assert isinstance(self.tokenizer.bos_token, str)
            context = self.tokenizer.bos_token + context

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation).input_ids + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer(context).input_ids

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        # by default truncate on the left
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            logger.warning(
                f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side"
            )
            context_remain = context_enc_len - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                logger.warning(f"All context (prompt) is truncated.")
                context_enc = []
                continuation_enc = whole_enc[-self.max_length :]

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
        with torch.no_grad():
            for elem in tqdm(requests, desc="Computing likelihood..."):
                context, continuation = self._encode_pair(*elem.args)
                # likelihood calculations are modified from https://github.com/ML-GSAI/SMDM/blob/main/evaluate_diff.py
                if self.nll_type == "mc":
                    logprob = -self._eval_target_nll_mc(context, continuation)
                    if self.log_type == "union":
                        logprob = logprob / (len(continuation) + len(context))
                elif self.nll_type == "ar_ftb" or self.nll_type == "ar_btf":
                    logprob = -self._eval_target_nll_ar(context, continuation)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                isgreedy = False
                out.append((logprob, isgreedy))
        return out

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError
