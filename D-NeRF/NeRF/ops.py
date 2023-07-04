import numpy as np


def to_8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)
