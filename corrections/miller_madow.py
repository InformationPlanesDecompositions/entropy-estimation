import numpy as np


def first_order(samples: np.ndarray, n_classes: int | None = None) -> float:
    if n_classes is None:
        n_classes = len(np.unique(samples))

    return (n_classes - 1) / (2 * len(samples))


def second_order(
    samples: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int | None = None
) -> float:
    if n_classes is None:
        n_classes = len(probabilities)
    elif n_classes != len(probabilities):
        raise ValueError(
            f'Mismatch between number of classes ({n_classes})' 
            + f'and probabilities ({len(probabilities)})'
        )

    correction = first_order(samples, n_classes=n_classes)

    correction -= (1 - sum(1 / probabilities)) / (12 * len(samples) ** 2)

    return correction
