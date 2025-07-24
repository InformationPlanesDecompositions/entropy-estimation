import numpy as np

import math

from distributions import binomial


def generate_samples(p: float, size: int | tuple[int, ...]) -> np.ndarray:
    return binomial.generate_samples(p, size, n_trials=1)


def compute_entropy(p: float) -> float:
    if p < 0 or p > 1:
        raise ValueError(f'Invalid success probability provided, was {p:3f}')

    q = 1 - p

    if math.isclose(p, 0) or math.isclose(p, 1):
        return 0

    return -(p * np.log2(p) + q * np.log2(q))


def compute_joint_entropy(p: float, d: int) -> float:
    if p < 0 or p > 1:
        raise ValueError(f'Invalid success probability provided, was {p:3f}')
    
    if d <= 0:
        raise ValueError(f'Invalid Bernoulli RV dimension provided, was {d}')
    
    return d * compute_entropy(p)
