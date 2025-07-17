import numpy as np
import scipy as sp

import math

from distributions import rng


def generate_samples(p: float, n_samples: int, n_trials: int) -> np.ndarray:
    if p < 0 or p > 1:
        raise ValueError(f'Invalid success probability provided, was {p:3f}')

    return rng.binomial(n=n_trials, p=p, size=n_samples)


def compute_true_entropy(p: float, n_trials: int) -> float:
    if p < 0 or p > 1:
        raise ValueError(f'Invalid success probability provided, was {p:3f}')

    eta = 0

    for k in range(0, n_trials + 1):
        p_k = sp.stats.binom.pmf(k, n_trials, p)

        if math.isclose(p_k, 0):
            continue

        eta += p_k * np.log2(p_k)

    return -1 * eta
