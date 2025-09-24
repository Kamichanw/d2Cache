from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal


class GenerationArgs(BaseModel):
    gen_length: int = Field(ge=0)
    block_length: int = Field(ge=0)
    steps: int = Field(ge=0)

    alg: Literal["maskgit_plus", "entropy", "topk_margin"] = Field(
        default="maskgit_plus"
    )
    temperature: float | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    threshold: float | None = Field(default=None)
    debias: bool = Field(default=False)

    cache_args: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_constraints(self):
        if self.block_length > self.gen_length or self.steps > self.gen_length:
            raise ValueError(
                f"{self.block_length=} and {self.steps=} must be <= {self.gen_length=}"
            )
        num_blocks = self.gen_length // self.block_length
        if self.gen_length % self.block_length != 0 or self.steps % num_blocks != 0:
            raise ValueError(
                f"{self.gen_length=} must be divisible by {self.block_length=} and {self.steps=} must be divisible by number of blocks {num_blocks=}"
            )

        return self


def get_generation_args(task: str, model: str, cache: str | None = None):
    cache_args = {}
    alg = "maskgit_plus"
    block_length = None
    debias = False
    temperature, top_p, top_k = 0.0, None, None
    threshold = None

    # set based on task
    match task:
        case "gsm8k" | "gsm8k_cot":
            gen_length = 256
            if cache == "dllm":
                match model:
                    case "llada-base":
                        kp, kr = 25, 5
                    case "llada-inst":
                        kp, kr = 50, 7
                    case "dream-base":
                        kp, kr = 100, 8
                    case "dream-inst":
                        kp, kr = 25, 2
                    case _:
                        raise ValueError(f"Unsupported model {model} for dllm cache.")
                cache_args = {"kp": kp, "kr": kr}
        case "humaneval":
            gen_length = 512
            if cache == "dllm":
                match model:
                    case "llada-base":
                        kp, kr = 50, 5
                    case "llada-inst":
                        kp, kr = 25, 5
                    case "dream-base":
                        kp, kr = 5, 1
                    case "dream-inst":
                        kp, kr = 50, 1
                    case _:
                        raise ValueError(f"Unsupported model {model} for dllm cache.")
                cache_args = {"kp": kp, "kr": kr}
        case "math-500":
            gen_length = 256
            if cache == "dllm":
                match model:
                    case "llada-base":
                        kp, kr = 50, 8
                    case "llada-inst":
                        kp, kr = 50, 1
                    case "dream-base":
                        kp, kr = 100, 4
                    case "dream-inst":
                        kp, kr = 50, 1
                    case _:
                        raise ValueError(f"Unsupported model {model} for dllm cache.")
                cache_args = {"kp": kp, "kr": kr}
        case "mbpp":
            gen_length = 512
            if cache == "dllm":
                match model:
                    case "llada-base":
                        kp, kr = 25, 4
                    case "llada-inst":
                        kp, kr = 100, 5
                    case "dream-base":
                        kp, kr = 25, 8
                    case "dream-inst":
                        kp, kr = 10, 8
                    case _:
                        raise ValueError(f"Unsupported model {model} for dllm cache.")
                cache_args = {"kp": kp, "kr": kr}

        case _:
            raise ValueError(
                f"Unsupported task {task}, you should specify in {__file__}."
            )

    # set based on cache
    match cache:
        case "heat":
            sigma = 10.0
            cache_args = {
                "rollout_p": 0.1,
                "current_k": 32,
                "sigma": sigma,
            }

    # set based on model
    match model:
        case "dream-base" | "dream-inst":
            top_p = 0.9

    block_length = 32 if model.endswith("inst") else gen_length
    steps = gen_length

    return GenerationArgs(
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
        alg=alg,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        threshold=threshold,
        debias=debias,
        cache_args=cache_args,
    )
