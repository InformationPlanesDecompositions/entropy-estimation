"""
Compute the Miller-Madow bias correction of the plug-in estimator, as defined in "The Statistical
Estimation of Entropy in the Non-Parametric Case" (Harris, 1975) and "Note on the bias of
information estimates" (Miller, 1955).

Applied as H^MM = H^hat + MM
"""
import numpy as np


_nats_to_bits = np.log2(np.e)


# TODO:
#   - Add axis keyword to work on (M, N, D) arrays
def first_order(samples: np.ndarray, n_classes: int | None = None) -> float:
    if len(samples.shape) > 2:
        raise ValueError('Currently only supported for 2-dim arrays')
    
    if len(samples.shape) == 0:
        return 0

    if n_classes is None:
        n_classes = len(np.unique(samples, axis=0))

    return _nats_to_bits * (n_classes - 1) / (2 * samples.shape[0])


# TODO:
#   - Check if this even works for higher dimensions
#   - Adjust for new first order method
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

    correction -= _nats_to_bits * (1 - sum(1 / probabilities)) / (12 * len(samples) ** 2)

    return correction
