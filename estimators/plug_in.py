from collections import Counter

import numpy as np


# See Antos and Kontoyiannis, 2001
def estimate_entropy(samples: np.ndarray, use_fast_estimate: bool = False) -> float:
    if len(samples.shape) > 2:
        raise ValueError(f'Currently only supported for 1 or 2-dim arrays')
    
    if len(samples.shape) == 0:
        return 0

    # TODO: Rewrite to check if multi-dim and bitwise
    if use_fast_estimate:
        p = fast_empirical_distribution(samples)
    else:
        p = _compute_observed_rates(samples)
        p = p[p != 0]

    return -np.sum(p * np.log2(p))


# Miller and Madow 1954 (through Luce, 1960)
def compute_entropy_variance(p: float, d: int, h_true: np.floating | float):
    import scipy.special

    q = 1 - p

    lhs = np.sum([
        scipy.special.binom(d, k) * p ** k * q ** (d - k) * np.pow(k * np.log2(p) + (d - k) * np.log2(q), 2)
        for k in range(0, d + 1)
    ])

    return lhs - np.pow(h_true, 2)


# Ricci et al., 2021
def estimate_entropy_variance(
    samples: np.ndarray,
    entropy_hat: float | np.floating | None = None,
    use_fast_estimate: bool = False,
) -> float:
    if len(samples.shape) > 2:
        raise ValueError(f'Currently only supported for 1 or 2-dim arrays')
    
    if len(samples.shape) == 0:
        return 0

    # A bit inefficient but not important for now
    if entropy_hat is None:
        entropy_hat = estimate_entropy(samples, use_fast_estimate)

    if use_fast_estimate:
        p = fast_empirical_distribution(samples)
    else:
        p = _compute_observed_rates(samples)
        p = p[p != 0]

    variance = np.sum(p * np.log2(p) ** 2)
    variance -= entropy_hat ** 2

    return variance / samples.shape[0]


def fast_joint_probabilitiy_estimation(x: np.ndarray, y: np.ndarray):
    n = x.shape[0]

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(f'Only supported for arrays of shape (N,)')

    joint_count = Counter(zip(x, y))

    return {xy: count / n for xy, count in joint_count.items()}


def fast_empirical_distribution(samples: np.ndarray) -> np.ndarray:
    if not issubclass(samples.dtype.type, np.integer):
        raise ValueError(f'Only supported for integer arrays')

    n = samples.shape[0]
    
    return np.divide(list(Counter(samples).values()), n)


def _compute_observed_rates(samples: np.ndarray) -> np.ndarray:
    _, c = np.unique(samples, axis=0, return_counts=True)

    return c / samples.shape[0]
