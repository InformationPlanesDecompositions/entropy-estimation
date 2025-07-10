import numpy as np

def estimate_entropy(samples: np.ndarray) -> float:
    entropy = 0

    for _, count in zip(*np.unique_counts(samples)):
        p = count / len(samples)
        entropy += p * np.log2(p)
    
    return -1 * entropy
