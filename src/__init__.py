import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every random source used in the project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)