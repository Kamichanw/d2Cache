from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
