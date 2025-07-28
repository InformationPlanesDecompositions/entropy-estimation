import numpy as np


# TODO: Adjust to multi-dim array to return entropy per experiment (i.e. 0 index)
def estimate_entropy(samples: np.ndarray) -> float:
    if len(samples.shape) > 2:
        raise ValueError(f'Currently only supported for 1 or 2-dim arrays')
    
    if len(samples.shape) == 0:
        return 0

    _, c = np.unique(samples, axis=0, return_counts=True)

    p = c / samples.shape[0]

    p = p[p != 0]

    return -np.sum(p * np.log2(p))


# TODO: Adjust to multidimensional RV/joint entropy?
def estimate_entropy_variance(
    samples: np.ndarray,
    entropy_hat: float | None = None
) -> float:
    if entropy_hat is None:
        entropy_hat = estimate_entropy(samples)

    variance = -entropy_hat ** 2

    for _, count in zip(*np.unique_counts(samples)):
        p = count / len(samples)
        variance += p * np.log2(p) ** 2

    return variance / len(samples)
