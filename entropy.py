import numpy as np

def estimate_entropy(alphabet: np.ndarray) -> float:
    entropy = 0

    for _, count in zip(*np.unique_counts(alphabet)):
        p = count / len(alphabet)
        entropy += p * np.log2(p)
    
    return -1 * entropy
