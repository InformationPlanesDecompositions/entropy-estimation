import numpy as np


# TODO: Adjust to multi-dim array to return entropy per experiment (i.e. 0 index)
def estimate_entropy(samples: np.ndarray) -> float:
    if len(samples.shape) > 2:
        raise ValueError(f'Currently only supported for 1 or 2-dim arrays')
    
    if len(samples.shape) == 0:
        return 0

    p = _compute_observed_rates(samples)
    p = p[p != 0]

    return -np.sum(p * np.log2(p))


def estimate_entropy_variance(
    samples: np.ndarray,
    entropy_hat: float | None = None
) -> float:
    if len(samples.shape) > 2:
        raise ValueError(f'Currently only supported for 1 or 2-dim arrays')
    
    if len(samples.shape) == 0:
        return 0

    # A bit inefficient but not important for now
    if entropy_hat is None:
        entropy_hat = estimate_entropy(samples)

    p = _compute_observed_rates(samples)
    p = p[p != 0]

    variance = np.sum(p * np.log2(p) ** 2)
    variance -= entropy_hat ** 2

    return variance / samples.shape[0]


def _compute_observed_rates(samples: np.ndarray) -> np.ndarray:
    _, c = np.unique(samples, axis=0, return_counts=True)

    return c / samples.shape[0]
