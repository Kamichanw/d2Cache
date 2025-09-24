import inspect
import os
import hydra
import json
import torch

from contextlib import nullcontext
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import cast
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import ConfigurableTask, get_task_dict, TaskManager

from src.eval_model import load_eval_model
from src.utils import pre_initialize, Timer, sympy_antlr_patcher


def serializer(o):
    if inspect.isfunction(o):
        try:
            source_code = inspect.getsource(o)
            return source_code
        except (TypeError, OSError):
            return f"<uninspectable function: {o.__name__}>"

    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.item()
        else:
            return o.tolist()

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


dream_inst_humaneval = {
    "doc_to_text": "Write a solution to the following problem and make sure that it passes the tests:\n```{{prompt}}",
    "gen_prefix": "Here is the completed function:\n```python\n{{prompt}}\n",
}


def overwrite_eval_task(cfg: DictConfig):
    eval_args = cast(dict, OmegaConf.to_container(cfg.eval_args, resolve=True))
    if cfg.dataset.name == "humaneval" and cfg.model.name == "dream-inst":
        # see https://github.com/DreamLM/Dream/blob/main/eval_instruct/lm_eval/tasks/humaneval/humaneval_instruct.yaml
        task: ConfigurableTask = get_task_dict("humaneval")["humaneval"]
        task.config.doc_to_text = "Write a solution to the following problem and make sure that it passes the tests:\n```{{prompt}}"
        task.config.gen_prefix = (
            "Here is the completed function:\n```python\n{{prompt}}\n"
        )
        eval_args["tasks"] = [task]

    if cfg.dataset.name == "math-500":
        task_manager = TaskManager(
            include_path=str(Path(__file__).parent / "tasks" / "math-500")
        )
        eval_args["task_manager"] = task_manager

    return eval_args


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    extra_cfg = pre_initialize(cfg)
    model = load_eval_model(cfg, extra_gen_kwargs=extra_cfg.get("extra_gen_kwargs"))
    output_dir = HydraConfig.get().runtime.output_dir

    patcher_ctx = sympy_antlr_patcher if cfg.dataset.name == "math-500" else nullcontext
    with patcher_ctx():
        results = simple_evaluate(
            model=model,
            use_cache=os.path.join(output_dir, "response") if cfg.use_eval_cache else None,
            apply_chat_template=cfg.model.name.endswith("inst"),
            **overwrite_eval_task(cfg),
        )

    results_path = os.path.join(output_dir, "results.json")

    if model.accelerator.is_main_process:
        results = results or {}
        if model.tps is not None and model.throughput is not None:
            logger.info(
                f"Throughput: {model.throughput:.2f} tokens/sec, "
                f"Tokens per step: {model.tps:.2f} tokens/step "
                f"(full: {model.full_throughput:.2f} tokens/sec, {model.full_tps:.2f} tokens/step), "
                f"Latency: {model.latency:.2f} s, "
                f"Total time: {model.total_time:.2f} s"
            )
            results["tps"] = model.tps
            results["throughput"] = model.throughput
            results["total_time"] = model.total_time
            results["full_tps"] = model.full_tps
            results["full_throughput"] = model.full_throughput
            results["latency"] = model.latency

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=serializer)

        logger.info(f"Results saved to {results_path}")

        for timer in Timer.cumulative:
            logger.info(f"{timer} time: {Timer(timer).cumulative_s:.2f} seconds")


if __name__ == "__main__":
    main()
