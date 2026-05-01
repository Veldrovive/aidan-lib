from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os

def load_hydra_config_by_path(config_path: str | Path, base_path: str | Path | None = None) -> DictConfig:
    config_path = Path(config_path)
    assert config_path.exists(), f"Config path {config_path} does not exist"

    # Get the relative path of the parent to the base path or to the cwd if the base path is None
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    # We use os.path.relpath to get the relative path of the parent to the base path to properly handle
    # the case where the base path is not the current working directory
    relative_path = os.path.relpath(config_path.parent, base_path)
    

    with initialize(version_base=None, config_path=str(relative_path)):
        cfg = compose(config_name=config_path.name, overrides=[])
    return cfg

    