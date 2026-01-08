import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import omegaconf
import warnings
from importlib.metadata import version

from pathlib import Path
from typing import Callable
from accelerate.utils import set_seed
from contextlib import contextmanager
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from dotenv import load_dotenv
from hydra import compose
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from dill import PickleWarning

from .models import *
from .common import *


@contextmanager
def sympy_antlr_patcher(target_version: str = "4.11.0"):
    """
    The `hydra` requires `antlr4-python3-runtime` version 4.9.*, but when evaluating the MATH dataset, the `sympy` used requires
    `antlr4-python3-runtime` version 4.11, which caused a conflict. This context manager solves the conflict by dynamically
    loading the required version at runtime without altering the base environment.
    """
    current_version = version("antlr4-python3-runtime")
    logger.info(
        f"Detected antlr4-python3-runtime version {current_version}. Temporarily switching to {target_version}..."
    )

    temp_dir = tempfile.mkdtemp(prefix="isolated_antlr_")
    temp_dir_path = Path(temp_dir)

    original_sys_path = sys.path[:]
    original_modules = {k: v for k, v in sys.modules.items() if k.startswith("antlr4")}

    try:
        logger.info(
            f"Downloading antlr4-python3-runtime=={target_version} to {temp_dir}..."
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                f"antlr4-python3-runtime=={target_version}",
                "--no-deps",
                "-d",
                temp_dir,
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
            ],
            capture_output=True,
            text=True,
        )

        wheel_files = list(temp_dir_path.glob("*.whl"))
        if not wheel_files or result.returncode != 0:
            raise RuntimeError(
                f"Failed to download antlr4-python3-runtime=={target_version}"
                f" (return code: {result.returncode}): {result.stderr}"
            )

        logger.info(f"Unpacking {wheel_files[0].name}...")
        with zipfile.ZipFile(wheel_files[0], "r") as whl:
            whl.extractall(temp_dir_path)

        for k in list(sys.modules.keys()):
            if k.startswith("antlr4"):
                del sys.modules[k]

        sys.path.insert(0, str(temp_dir_path))

        yield

    finally:
        logger.info("Restoring original environment...")
        sys.path[:] = original_sys_path

        for k in list(sys.modules.keys()):
            if k.startswith("antlr4"):
                del sys.modules[k]

        sys.modules.update(original_modules)
        shutil.rmtree(temp_dir)
        logger.info("Environment restored.")


def find_incompatible_kwargs(input_kwargs: dict, target_fn: Callable) -> tuple:
    """
    Returns a tuple of keyword arguments in `input_kwargs` that are not compatible
    with the signature of `target_fn`.
    """
    sig = inspect.signature(target_fn)
    params = sig.parameters
    if all(p.kind != p.VAR_KEYWORD for p in params.values()) and (
        unknown_args := set(input_kwargs) - set(params)
    ):
        return tuple(unknown_args)
    return tuple()


def get_config_diff(d1: dict, d2: dict) -> dict:
    """Compare dict d1 and d2 recursively, and returns the d1 - d2."""
    diff = {}
    for key, value in d1.items():
        if key not in d2:
            diff[key] = value
        elif isinstance(value, dict) and isinstance(d2.get(key), dict):
            nested_diff = get_config_diff(value, d2[key])
            if nested_diff:
                diff[key] = nested_diff
        elif value != d2.get(key):
            diff[key] = value

    return diff


def pre_initialize(cfg: DictConfig) -> dict:
    """
    Pre-initialize the environment and configuration. Returns a dictionary with additional configurations.
    """
    sys.path.insert(0, str(Path(__file__).parents[2] / "configs"))
    import src.generation # triger registration of all generation methods
    from gen_args import get_generation_args  # type: ignore

    # basic environment settings
    load_dotenv()
    set_seed(cfg.seed)
    logger.remove()
    logger.add(sys.stderr, filter=LoggerFilter())
    warnings.filterwarnings("ignore", category=PickleWarning)
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="No device id is provided via.*"
    )

    # process additional configs
    cache_choice = HydraConfig.get().runtime.choices.get("cache", None)
    gen_strategy_choice = HydraConfig.get().runtime.choices.get("gen_strategy", None)
    generation_args = get_generation_args(
        cfg.dataset.name,
        cfg.model.name,
        cache_choice,
    ).model_dump()
    cache_args = generation_args.pop("cache_args", {})
    default_overrides = []
    if gen_strategy_choice is not None:
        default_overrides.append(f"generation={gen_strategy_choice}")
    if cache_choice is not None:
        default_overrides.append(f"cache={cache_choice}")
    default_cfg = compose(
        HydraConfig.get().job.config_name, overrides=default_overrides
    )
    with omegaconf.open_dict(cfg):
        model_gen_args = OmegaConf.create(cfg.model.generation, parent=cfg.model)
        OmegaConf.resolve(model_gen_args)
        # order: default cfg -> gen args from model -> predefined args -> cli overrided args
        cfg.generation = OmegaConf.merge(
            OmegaConf.to_container(default_cfg.generation, resolve=True),
            model_gen_args,
            generation_args,
            get_config_diff(cfg.generation, default_cfg.generation),
        )
        logger.info(
            re.sub(r"{", "{{", re.sub(r"}", "}}", str(cfg.generation))),
            rank_zero_only=True,
        )

    os.environ["MASK_TOKEN_ID"] = str(cfg.generation.mask_token_id)
    os.environ["EOT_TOKEN_ID"] = str(cfg.generation.eot_token_id)
    os.environ["PAD_TOKEN_ID"] = str(cfg.generation.pad_token_id)

    extra_gen_kwargs = {}

    if cfg.get("cache") is not None:
        # order: predefined args -> cli overrided args
        cache_args.update(get_config_diff(cfg["cache"], default_cfg.get("cache", {})))
        extra_gen_kwargs["cache_cls"] = instantiate(
            cfg.cache, **cache_args, _partial_=True
        )
        logger.info(
            re.sub(r"{", "{{", re.sub(r"}", "}}", str(extra_gen_kwargs["cache_cls"]))),
            rank_zero_only=True,
        )

    if attn_cfg := cfg.get("attention"):
        if not set(attn_cfg.type).issubset(set("qkvo")):
            raise ValueError(
                f"The attention type to be recorded should be a combination of 'qkvo', but got {attn_cfg.type}"
            )

    return {
        "extra_gen_kwargs": extra_gen_kwargs,
    }
