import numpy as np

from estimators import plug_in


def estimate_entropy(samples: np.ndarray) -> float:
    n = len(samples)
    entropy = plug_in.estimate_entropy(samples)

    offset_entropy = 0

    for offset in range(n):
        jackknife_samples = np.concat([samples[:offset], samples[offset + 1:]])
        offset_entropy += plug_in.estimate_entropy(jackknife_samples)

    return n * entropy - (n - 1) * offset_entropy / n


def estimate_entropy_variance(samples: np.ndarray) -> float:
    raise NotImplementedError()
