import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import omegaconf
import torch
import torch.nn.functional as F
import warnings
import importlib
import importlib.metadata

from collections.abc import Sequence, Mapping
from pathlib import Path
from typing import Callable, TYPE_CHECKING, overload
from accelerate.utils import set_seed
from contextlib import contextmanager
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from dotenv import load_dotenv
from hydra import compose
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from dill import PickleWarning
from transformers.modeling_utils import PreTrainedModel

if TYPE_CHECKING:
    from src.frame import DecodeRecord


class Registry:
    def __init__(self):
        self._gen_strategy = {}

    def gen_strategy(self, strategy: str, alias: list[str] | None = None):
        def decorator(obj):
            if not callable(obj):
                raise TypeError(f"{obj} is not callable")
            self._gen_strategy[strategy] = obj
            for a in alias or []:
                self._gen_strategy[a] = obj
            return obj

        return decorator

    def get(self, key: str):
        ret = self._gen_strategy.get(key)
        if ret is None:
            raise ValueError(f"Unknown key: {key}")
        return ret

    @classmethod
    def trigger(cls, module_dir: str, module_name: str):
        """
        Trigger the registration from a module.
        """
        for filename in os.listdir(module_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                try:
                    importlib.import_module(f"{module_name}.{filename[:-3]}")
                except ImportError:
                    pass

    @property
    def all_gen_strategies(self):
        return list(self._gen_strategy.keys())


register = Registry()


class Timer:
    """
    Basic timer context manager for measuring CPU or GPU time.
    """

    cumulative = dict()

    def __init__(self, name: str | None = None):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.start_time.record()  # type: ignore
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            assert isinstance(self.start_time, torch.cuda.Event)
            self.end_time.record()  # type: ignore
            torch.cuda.synchronize()
            self.elapsed_time_s = self.start_time.elapsed_time(self.end_time) / 1000
            self.elapsed_time_ms = self.elapsed_time_s * 1000
        else:
            assert isinstance(self.start_time, float)
            self.end_time = time.time()
            self.elapsed_time_s = self.end_time - self.start_time
            self.elapsed_time_ms = self.elapsed_time_s * 1000

        if self.name is not None:
            Timer.cumulative[self.name] = (
                Timer.cumulative.get(self.name, 0.0) + self.elapsed_time_s
            )

    @classmethod
    def get_cumulative_s(cls, name: str) -> float:
        return cls.cumulative.get(name, 0.0)

    @classmethod
    def get_cumulative_ms(cls, name: str) -> float:
        return cls.get_cumulative_s(name) * 1000

    @property
    def cumulative_s(self) -> float:
        if self.name is None:
            return 0.0
        return self.get_cumulative_s(self.name)

    @property
    def cumulative_ms(self) -> float:
        return self.cumulative_s * 1000

    def token_per_second(self, record: "DecodeRecord", until_eot: bool = True) -> float:
        """
        Calculate the number of tokens processed per second.
        """
        if not hasattr(self, "elapsed_time_s"):
            raise RuntimeError("Timer has not been started or stopped.")
        eot_token_id = int(
            os.environ.get("EOT_TOKEN_ID", "126081")
        )  # 126081 for llada, 151643 for dream
        if until_eot:
            final_token_seqs = record[-1].generated_tokens
            num_tokens = (final_token_seqs == eot_token_id).int().argmax()
            num_tokens = torch.where(
                num_tokens > 0, num_tokens, final_token_seqs.size(-1)
            )
            return torch.sum(num_tokens).item() / self.elapsed_time_s
        else:
            total_tokens = [
                (
                    torch.cat(delta.transfer_index).numel()
                    if isinstance(delta.transfer_index, tuple)
                    else delta.transfer_index.numel()
                )
                for delta in record.deltas
            ]
            return sum(total_tokens) / self.elapsed_time_s

    def token_per_step(self, record: "DecodeRecord", until_eot: bool = True) -> float:
        """
        Calculate the number of tokens processed per step.
        """
        if until_eot:
            eot_token_id = int(
                os.environ["EOT_TOKEN_ID"]  # 126081 for llada, 151643 for dream
            )
            final_token_seqs = record[-1].generated_tokens
            num_tokens = (final_token_seqs == eot_token_id).int().argmax(keepdim=True)
            num_tokens = torch.where(
                num_tokens > 0, num_tokens, final_token_seqs.size(-1)
            )
            total_steps = 0
            for batch_idx, eot_idx in enumerate(num_tokens):
                # add 1, since steps are 0-indexed
                total_steps += record[-1].steps[batch_idx, :eot_idx].max().item() + 1
            return torch.sum(num_tokens).item() / total_steps
        else:
            total_tokens = 0
            for delta in record.deltas:
                new_tokens = delta.transferred_tokens
                if isinstance(new_tokens, tuple):
                    total_tokens += sum(t.numel() for t in new_tokens)
                elif isinstance(new_tokens, torch.Tensor):
                    total_tokens += new_tokens.numel()
            return total_tokens / record.num_steps


class LoggerFilter:
    def __init__(self):
        self.msg_history = defaultdict(set)

    def __call__(self, record):
        if "once" in record["extra"]:
            level = record["level"].no
            message = record["message"]
            if message in self.msg_history[level]:
                return False
            self.msg_history[level].add(message)
        elif record["extra"].get("rank_zero_only", False):
            if os.environ.get("LOCAL_RANK", "0") != "0":
                return False
        return True


def apply_fn(
    obj: Mapping | Sequence,
    fn: Callable,
    check_cycles: bool = False,
):
    """
    Recursively traverse a mapping or sequence, and apply `fn` to each non-built-in element.

    Args:
        obj (Mapping | Sequence):
            The object to process. Can be a dict, list, tuple, set, etc.
        fn (Callable):
            The function to apply to each element. It should take a single argument and return a value to replace it.
        check_cycles (bool, optional):
            Whether to check for cycles during the traversal to prevent infinite recursion. Default is False.
    """

    def traverse(obj, visited, depth):
        if check_cycles:
            if id(obj) in visited:
                return obj
            visited.add(id(obj))

        if isinstance(obj, Mapping):
            return {
                key: traverse(value, visited, depth + 1) for key, value in obj.items()
            }

        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            return type(obj)(traverse(item, visited, depth + 1) for item in obj)  # type: ignore[call-arg]

        return fn(obj)

    return traverse(obj, set() if check_cycles else None, 0)


@overload
def tensor_insert(
    dest_tensor: torch.Tensor,
    insert_index: torch.Tensor,
    src: int | float,
) -> torch.Tensor: ...


@overload
def tensor_insert(
    dest_tensor: torch.Tensor,
    insert_index: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor: ...


def tensor_insert(dest_tensor, insert_index, src) -> torch.Tensor:
    """
    Inserts a source tensor or value into a destination tensor along dimension 1.

    This function is specialized for sequence data with shapes like (B, L, ...),
    where B is the batch size and L is the sequence length.

    Args:
        dest_tensor (torch.Tensor): The destination tensor, e.g., of shape (B, L, D).
        insert_index (torch.Tensor): A LongTensor of shape (B, K) specifying insertion
            points along dimension 1.
        src (torch.Tensor | Number): The source to insert.
            - If a Tensor, its shape must match `dest_tensor` but with size K at dim 1,
              e.g., (B, K, D).
            - If a Number, it will be broadcasted to fill the new entries.

    Returns:
        torch.Tensor: The resulting tensor with insertions, shape (B, L+K, ...).
    """

    B, L = dest_tensor.shape[:2]
    K = insert_index.shape[1]

    if K == 0:
        return dest_tensor.clone()

    # reshape inputs to a 3D view: (B, L, rest)
    trailing_dims = dest_tensor.shape[2:]
    T = int(torch.prod(torch.tensor(trailing_dims)).item()) if trailing_dims else 1
    dest_view = dest_tensor.contiguous().view(B, L, T)

    L_new = L + K
    sorted_indices, sort_perm = torch.sort(insert_index, dim=1)

    dest_indices_new = sorted_indices + torch.arange(
        K, device=dest_tensor.device
    ).unsqueeze(0)
    old_indices = torch.arange(L, device=dest_tensor.device).expand(B, -1)
    shifts = (sorted_indices.unsqueeze(2) <= old_indices.unsqueeze(1)).sum(dim=1)
    dest_indices_old = old_indices + shifts

    result_view = torch.empty(
        (B, L_new, T), dtype=dest_tensor.dtype, device=dest_tensor.device
    )

    dest_indices_old = dest_indices_old.unsqueeze(2).expand(B, L, T)
    dest_indices_new = dest_indices_new.unsqueeze(2).expand(B, K, T)

    result_view.scatter_(1, dest_indices_old, dest_view)

    if isinstance(src, (int, float)):
        result_view.scatter_(1, dest_indices_new, src)
    elif isinstance(src, torch.Tensor):
        src_view = src.contiguous().view(B, K, T)
        inv_sort_perm = torch.argsort(sort_perm, dim=1).unsqueeze(2).expand(B, K, T)
        gathered_src = torch.gather(src_view, 1, inv_sort_perm)
        result_view.scatter_(1, dest_indices_new, gathered_src)
    else:
        raise TypeError(f"src must be a Tensor or a number, but got {type(src)}")

    return result_view.view((B, L_new) + trailing_dims)


def tensor_delete(src_tensor: torch.Tensor, delete_index: torch.Tensor) -> torch.Tensor:
    """
    Deletes elements from each row of a (B, L) tensor.

    Args:
        src_tensor (torch.Tensor): The tensor to delete from, shape (B, L).
        delete_index (torch.Tensor): The indices to delete, shape (B, K).

    Returns:
        torch.Tensor: The resulting tensor of shape (B, L - K).
    """
    B, L_current = src_tensor.shape
    K_del = delete_index.shape[1]

    if K_del == 0:
        return src_tensor.clone()

    L_new = L_current - K_del
    device = src_tensor.device

    keep_mask = torch.ones(B, L_current, dtype=torch.bool, device=device)

    row_indices = torch.arange(B, device=device).repeat_interleave(K_del)
    col_indices = delete_index.flatten()

    keep_mask[row_indices, col_indices] = False

    return src_tensor[keep_mask].view(B, L_new)


def certainty_density(
    mask: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Calculates the certainty density using a Gaussian kernel via FFT.

    This version implements a custom padding rule for handling boundaries during normalization:
    - The conceptual signal to the left of the boundary is always considered True.
    - The conceptual signal to the right is considered True if the mask's last
      element is True, otherwise it's considered False.

    Args:
        mask (torch.Tensor): A 2D boolean tensor of shape (B, L), which typically represents
            the generated tokens where True indicates a generated token and False indicates mask tokens.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        torch.Tensor: A 2D float tensor of shape (B, L) with the density values.
    """
    assert sigma > 0, "sigma must be a positive number."
    B, L = mask.shape
    device = mask.device
    float_mask = mask.float()

    # create an extended canvas with custom padding baked in.
    # [L_pad_left | L_mask | L_pad_right]
    padded_mask = F.pad(float_mask, (L, L), "constant", 1.0)
    padded_mask[mask[:, -1] == False, 2 * L :] = 0.0

    extended_L = 3 * L
    padded_len = 2 * extended_L

    dist = torch.cat(
        (
            torch.arange(extended_L, device=device),
            torch.arange(-extended_L, 0, device=device),
        )
    )
    kernel_fft = torch.fft.fft(torch.exp(-(dist**2) / (2 * sigma**2)), n=padded_len)

    weighted_sum_ext = torch.fft.ifft(
        torch.fft.fft(F.pad(padded_mask, (0, extended_L)), n=padded_len) * kernel_fft,
        n=padded_len,
    ).real

    kernel_sum_ext = torch.fft.ifft(
        torch.fft.fft(torch.ones(B, extended_L * 2, device=device), n=padded_len)
        * kernel_fft,
        n=padded_len,
    ).real

    weighted_sum = weighted_sum_ext[..., L : 2 * L]
    kernel_sum_at_pos = kernel_sum_ext[..., L : 2 * L].clamp_min(1e-8)

    return weighted_sum / kernel_sum_at_pos


def nucleus_select(
    scores: torch.Tensor, top_p: float, min_k: int = 1, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Selects elements from a score tensor using a combination of top-p (nucleus)
    and min_k sampling logic, with an optional mask for valid positions.

    This version is corrected to handle the edge case where top_p=1.0, ensuring
    that no elements outside the mask are ever selected.

    Args:
        scores (torch.Tensor): A 2D tensor of non-negative scores of shape (B, L).
        top_p (float): The cumulative probability threshold for nucleus sampling.
        min_k (int): The minimum number of tokens to select.
        mask (Optional[torch.Tensor]): A boolean tensor of shape (B, L). `True` indicates
                                       a valid position to consider for selection.

    Returns:
        torch.Tensor: A boolean mask of shape (B, L) where `True` indicates selection.
    """
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be between 0.0 and 1.0, but got {top_p}.")
    if not isinstance(min_k, int) or min_k < 0:
        raise ValueError(f"min_k must be a non-negative integer, but got {min_k}.")

    scores = torch.where(mask, scores, 0.0) if mask is not None else scores

    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs <= top_p

    k = min(min_k, scores.shape[-1])
    top_k_mask = torch.arange(nucleus_mask.shape[-1], device=nucleus_mask.device) < k

    combined_mask = nucleus_mask | top_k_mask

    if mask is not None:
        combined_mask &= torch.gather(mask, 1, sorted_indices)

    return torch.zeros_like(scores, dtype=torch.bool).scatter_(
        dim=1, index=sorted_indices, src=combined_mask
    )


def pad_mask_(
    mask: torch.Tensor, target_count: int, scores: torch.Tensor
) -> torch.Tensor:
    """
    Pads a boolean mask to ensure each row has a target number of True values.
    It selects pad positions which `False` positions to flip to `True` based on the provided `scores`.

    Args:
        mask (torch.Tensor): The boolean mask to be padded.
                             Shape: (B, L), where B is batch size and L is sequence length.
        target_count (int): The desired total number of `True` values for each row
                            after padding.
        scores (torch.Tensor): The scores used to select which `False` positions to
                               flip to `True`. Higher scores are selected first.
                               Should have the same shape as `mask`.

    Returns:
        torch.Tensor: The modified mask tensor with padded `True` values.
                      Note: The modification is done in-place.
    """
    B, _ = mask.shape
    device = mask.device

    num_selected_per_seq = mask.sum(dim=-1)
    num_to_pad_per_seq = (target_count - num_selected_per_seq).clamp(min=0)

    if num_to_pad_per_seq.sum() == 0:
        return mask

    max_num_to_pad = int(num_to_pad_per_seq.max())
    scores = torch.where(mask, -torch.inf, scores)

    _, indices = torch.topk(
        scores,
        k=max_num_to_pad,
        dim=-1,
    )

    # select the indices that really need to be set to true
    pad_indices = indices.masked_select(
        torch.arange(max_num_to_pad, device=device).expand(B, -1)
        < num_to_pad_per_seq.unsqueeze(-1)
    )

    row_indices = torch.repeat_interleave(
        torch.arange(B, device=device),
        num_to_pad_per_seq.long(),
    )
    mask[row_indices, pad_indices] = True

    return mask


@contextmanager
def sympy_antlr_patcher(target_version: str = "4.11.0"):
    """
    The `hydra` requires `antlr4-python3-runtime` version 4.9.*, but when evaluating the MATH dataset, the `sympy` used requires
    `antlr4-python3-runtime` version 4.11, which caused a conflict. This context manager solves the conflict by dynamically
    loading the required version at runtime without altering the base environment.
    """
    current_version = importlib.metadata.version("antlr4-python3-runtime")
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
                "https://pypi.tuna.tsinghua.edu.cn/simple"
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
    sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))
    import src.generation
    from gen_args import get_generation_args  # type: ignore

    # basic environment settings
    load_dotenv()
    set_seed(cfg.seed)
    logger.remove()
    logger.add(
        sys.stderr,
        filter=LoggerFilter(),
    )
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


def load_pretrained_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    """
    Load a pretrained model based on the configuration.
    """
    from src.llada import LLaDAModelLM
    from src.dream import DreamModel

    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        return LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dream":
        return DreamModel.from_pretrained(cfg.model.path, **model_kwargs)

    raise ValueError(f"Unsupported pretrained model: {cfg.model.name}")
