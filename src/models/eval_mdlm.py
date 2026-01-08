import os
import accelerate
import torch
import itertools

from typing import Iterable, cast
from datetime import timedelta
from omegaconf import DictConfig
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm

from src.frame import Frame
from src.utils import Timer, load_pretrained_model, load_tokenizer


class EvalMDLM(TemplateLM):
    """
    Base class for evaluating masked denoising language models (MDLMs) using the LM Evaluation Harness.
    """

    def __init__(self, cfg: DictConfig, **kwargs):

        # setup facilities...
        accelerator_kwargs = accelerate.InitProcessGroupKwargs(
            timeout=timedelta(weeks=52)
        )
        self.accelerator = accelerate.Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.model = (
            load_pretrained_model(
                cfg,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
            )
            .eval()
            .to(self.accelerator.device)
        )
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = load_tokenizer(
            cfg, trust_remote_code=True
        )

        # setup properties from LM
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        # setup custom properties
        self.cfg = cfg
        self.throughput = None
        self.tps = None
        self.full_throughput = None
        self.full_tps = None
        self.latency = None
        self.total_time = None
        self.input_length = None
        self.device = self.accelerator.device
        self.extra_gen_kwargs = kwargs.get("extra_gen_kwargs", {})

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        NOTE: This method is expected to handle strings which already contain the BOS token (when add_special_tokens=None).
        Otherwise, will use add_special_tokens if specified.
        """
        add_special_tokens = add_special_tokens or self.cfg.get("add_bos_token")
        # set add_special_tokens=False if the string already starts with BOS token.
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.decode(self.prefix_token_id)
        ):
            add_special_tokens = False
        if add_special_tokens is not None:
            kwargs["add_special_tokens"] = add_special_tokens
        return self.tokenizer.encode(string, **kwargs)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, **kwargs):
        raise NotImplementedError

    @property
    def eot_token_id(self) -> int:  # type: ignore
        try:
            return int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore
        except TypeError:
            return self.tokenizer.eos_token_id  # type: ignore

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        from src.generation import generate, decode_final_frame

        out, throughput, tps = [], [], []
        full_throughput, full_tps = [], []
        latency = []
        input_length = []

        batch_size = self.cfg.get("batch_size", 1)

        def batched(iterable):
            """
            A quick implementation of itertools.batched for Python versions < 3.12.
            """
            it = iter(iterable)
            while batch := tuple(itertools.islice(it, batch_size)):
                yield batch

        for instances in tqdm(
            batched(requests),
            total=(len(requests) + batch_size - 1) // batch_size,
            desc="Generating...",
            disable=disable_tqdm or not self.accelerator.is_main_process,
        ):
            context, until = map(list, zip(*(instance.args for instance in instances)))
            if self.cfg.get("add_bos_token", False):
                context = [
                    (
                        self.tokenizer.bos_token + ctx
                        if not has_bos_prefix(ctx, self.tokenizer.bos_token)
                        else ctx
                    )
                    for ctx in context
                ]
            inputs = self.tokenizer(
                context, return_tensors="pt", padding=True, padding_side="left"
            )
            until = [u["until"] for u in until]
            try:
                with Timer("eval") as timer:
                    decode_record = generate(
                        self.model,
                        **inputs,
                        **self.cfg.generation,
                        **self.extra_gen_kwargs,
                        ignore_unknown_args="ignore",
                    )

                input_length.append(
                    torch.sum(inputs["attention_mask"]).item() / batch_size  # type: ignore
                )
                throughput.append(timer.token_per_second(decode_record))
                full_throughput.append(
                    timer.token_per_second(
                        decode_record, self.cfg.generation.stop_until_eot
                    )
                )
                tps.append(timer.token_per_step(decode_record))
                full_tps.append(
                    timer.token_per_step(
                        decode_record, self.cfg.generation.stop_until_eot
                    )
                )
                latency.append(timer.elapsed_time_s / batch_size)
                final_frame: Frame = decode_record[-1]
                generated_answer = [
                    cast(
                        str,
                        decode_final_frame(
                            self.tokenizer,
                            final_frame[i],
                            stop_words=(
                                u if not self.cfg.dataset.name == "humaneval" else None
                            ),
                            skip_special_tokens=True,
                        ),
                    )
                    for i, u in enumerate(until)
                ]

                out.extend(generated_answer)
            except torch.cuda.OutOfMemoryError:
                out.append("[out-of-memory]")

            # if you got a watchdog timeout error, you can uncomment this line to avoid it.
            # it will slow down the evaluation though.
            # self.accelerator.wait_for_everyone()

        throughput = self.accelerator.gather_for_metrics(throughput)
        tps = self.accelerator.gather_for_metrics(tps)
        full_throughput = self.accelerator.gather_for_metrics(full_throughput)
        full_tps = self.accelerator.gather_for_metrics(full_tps)
        latency = self.accelerator.gather_for_metrics(latency)
        input_length = self.accelerator.gather_for_metrics(input_length)

        if self.accelerator.is_main_process:
            self.tps = sum(tps) / len(tps)
            self.throughput = sum(throughput) / len(throughput)
            self.full_tps = sum(full_tps) / len(full_tps)
            self.full_throughput = sum(full_throughput) / len(full_throughput)
            self.latency = sum(latency) / len(latency)
            self.total_time = Timer.get_cumulative_s("eval")
            self.input_length = sum(input_length) / len(input_length)

        return out

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        assert isinstance(chat_templated, str)
        return chat_templated


def has_bos_prefix(sequence: str, bos_str: str | Iterable[str] | None = None):
    if bos_str is None:
        return False
    elif isinstance(bos_str, str):
        return sequence.startswith(bos_str)
    else:
        return any(sequence.startswith(x) for x in bos_str)
