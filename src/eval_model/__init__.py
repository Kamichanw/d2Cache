from omegaconf import DictConfig
from src.eval_model.llada import LLaDAEval
from src.eval_model.dream import DreamEval

def load_eval_model(cfg: DictConfig, **model_kwargs):
    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        return LLaDAEval(cfg, **model_kwargs)
    elif model_family == "dream":
        return DreamEval(cfg, **model_kwargs)
    
    raise NotImplementedError(
        f"Model family {model_family} is not implemented for evaluation."
    )
