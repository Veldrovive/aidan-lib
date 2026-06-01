import random

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int) -> None:
    """Set global seed for reproducibility across random, numpy, and torch.

    Args:
        seed: The seed value to set.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
