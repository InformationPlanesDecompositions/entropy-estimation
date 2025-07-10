import numpy as np
import scipy as sp


def estimate_entropy(samples: np.ndarray) -> float:
    entropy = 0

    for _, count in zip(*np.unique_counts(samples)):
        p = count / len(samples)
        entropy += p * np.log2(p)
    
    return -1 * entropy


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


def compute_entropy_confidence_interval(
    samples: np.ndarray,
    confidence: float = 0.95,
    as_dict: bool = False
) -> tuple[float, tuple[float, float]] | dict[str, float | tuple]:
    h_hat = estimate_entropy(samples)
    var_hat = estimate_entropy_variance(samples, h_hat)
    std_hat = np.sqrt(var_hat)

    # TODO: Kinda temp, check which distribution to use
    t_score = sp.stats.t.pdf((1 + confidence) / 2, len(samples) - 1)

    interval = (
        h_hat - t_score * std_hat,
        h_hat + t_score * std_hat
    )

    if as_dict:
        res = {
            'entropy': h_hat,
            'confidence': interval
        }
    else:
        res = (h_hat, interval)

    return res
