import numpy as np


def estimate_entropy(samples: np.ndarray) -> float:
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
