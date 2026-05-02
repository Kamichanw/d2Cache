from loguru import logger
from pydantic import BaseModel, Field, model_validator
from typing import Literal


class GenerationArgs(BaseModel):
    max_new_tokens: int | None = Field(default=None, ge=0)
    block_length: int | None = Field(default=None, ge=0)

    alg: Literal["maskgit_plus", "entropy", "topk_margin"] = Field(
        default="maskgit_plus"
    )
    temperature: float | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    sigma: float | None = Field(default=None, gt=0)

    cache_args: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_constraints(self):
        if self.max_new_tokens is None or self.block_length is None:
            return self
        if self.block_length > self.max_new_tokens:
            raise ValueError(
                f"{self.block_length=} must be <= {self.max_new_tokens=}"
            )
        if self.max_new_tokens % self.block_length != 0:
            raise ValueError(
                f"{self.max_new_tokens=} must be divisible by {self.block_length=}"
            )

        return self


def get_generation_args(task: str, model: str, cache: str | None = None) -> dict:
    cache_args = {}
    alg = "maskgit_plus"
    block_length = None
    sigma = None
    temperature, top_p, top_k = 0.0, None, None

    # set based on task
    match task:
        case (
            "gsm8k"
            | "gsm8k_cot"
            | "math-500"
            | "gpqa_main_generative_n_shot"
            | "mmlu_pro"
        ):
            max_new_tokens = 256
        case "humaneval" | "mbpp":
            max_new_tokens = 512
        case task if "longbench" in task:
            max_new_tokens = 512
        case _:
            logger.info(
                f"Unsupported task {task}, you should specify in {__file__}."
                " Using default max_new_tokens=512."
            )
            max_new_tokens = 512

    block_length = 32 if model.endswith("inst") else max_new_tokens

    # set cache args
    match cache:
        case "d2cache":
            sigma = 10.0
            cache_args = {
                "rollout_p": 0.1,
                "current_k": 32,
                "sigma": sigma,
                "inflate_w": 0,
            }
            # when using certainty prior (CP) guided decoding, block-wise semi-ar is no longer needed.
            block_length = max_new_tokens
            # but it is also possible to use CP guided decoding and block-wise semi-ar.
            # to achieve this, pass `generation.block_length=32 cache.inflate_w=4` in cli
        case "prefix":
            block_length = 32
        case "dllm":
            kp, kr = 50, 4
            match task:
                case "gsm8k" | "gsm8k_cot":
                    match model:
                        case "llada-base":
                            kp, kr = 25, 5
                        case "llada-inst":
                            kp, kr = 50, 7
                        case "dream-base":
                            kp, kr = 100, 8
                        case "dream-inst":
                            kp, kr = 25, 2
                case "humaneval":
                    match model:
                        case "llada-base":
                            kp, kr = 50, 5
                        case "llada-inst":
                            kp, kr = 25, 5
                        case "dream-base":
                            kp, kr = 5, 1
                        case "dream-inst":
                            kp, kr = 50, 1
                case "math-500":
                    match model:
                        case "llada-base":
                            kp, kr = 50, 8
                        case "llada-inst":
                            kp, kr = 50, 1
                        case "dream-base":
                            kp, kr = 100, 4
                        case "dream-inst":
                            kp, kr = 50, 1
                case "mbpp":
                    match model:
                        case "llada-base":
                            kp, kr = 25, 4
                        case "llada-inst":
                            kp, kr = 100, 5
                        case "dream-base":
                            kp, kr = 25, 8
                        case "dream-inst":
                            kp, kr = 10, 8
                case "gpqa_main_generative_n_shot":
                    match model:
                        case "llada-base":
                            kp, kr = 100, 8
                        case "llada-inst":
                            kp, kr = 50, 6
                        case "dream-base":
                            kp, kr = 100, 8
                        case "dream-inst":
                            kp, kr = 10, 8 
                case "mmlu_pro":
                    match model:
                        case "llada-base":
                            kp, kr = 100, 6
                        case "llada-inst":
                            kp, kr = 50, 3
                        case "dream-base":
                            kp, kr = 25, 2
                        case "dream-inst":
                            kp, kr = 5, 1 
            cache_args = {"kp": kp, "kr": kr}

    # set based on model
    match model:
        case "dream-base" | "dream-inst":
            top_p = 0.9
        case model if model.startswith("sdar"):
            block_length = 4

    return GenerationArgs(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        alg=alg,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sigma=sigma,
        cache_args=cache_args,
    ).model_dump()
