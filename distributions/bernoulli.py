import numpy as np

import math

from distributions import binomial


def generate_samples(p: float, n_samples: int) -> np.ndarray:
    return binomial.generate_samples(p, n_samples, n_trials=1)


def compute_true_entropy(p: float) -> float:
    if p < 0 or p > 1:
        raise ValueError(f'Invalid success probability provided, was {p:3f}')

    q = 1 - p

    if math.isclose(p, 0) or math.isclose(p, 1):
        return 0

    return -(p * np.log2(p) + q * np.log2(q))
