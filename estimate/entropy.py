import numpy as np

def compute_plug_in_estimate(samples: np.ndarray) -> float:
    entropy = 0

    for _, count in zip(*np.unique_counts(samples)):
        p = count / len(samples)
        entropy += p * np.log2(p)
    
    return -1 * entropy
