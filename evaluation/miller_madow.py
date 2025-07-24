import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from collections import defaultdict

import distributions.bernoulli
from corrections import mm
from estimators import plug_in


# TODO: Make possible for "ternary bernoulli" (binomial in general)
def evaluate(n_experiments: int, p: float, n_samples_space: np.ndarray):
    if n_samples_space.dtype != np.int64:
        n_samples_space = n_samples_space.astype(np.int64)

    data = defaultdict(list)

    h_true = distributions.bernoulli.compute_true_entropy(p)

    for n_samples in n_samples_space:
        experiments = distributions.bernoulli.generate_samples(p, size=(n_experiments, n_samples))

        mm_first = mm.first_order(experiments[0], n_classes=2)
        mm_second = mm.second_order(experiments[0], probabilities=np.array([1 - p, p]), n_classes=2)

        h_estimates = [plug_in.estimate_entropy(sample) for sample in experiments]

        for h_hat in h_estimates:
            data['N'].append(n_samples)
            data['H^'].append(h_hat)
            data['MM1'].append(mm_first)
            data['H^_1'].append(h_hat + mm_first)
            data['MM2'].append(mm_second)
            data['H^_2'].append(h_hat + mm_second)
    
    df_data = pd.DataFrame(data=data, columns=['N', 'H^', 'MM1', 'H^_1', 'MM2', 'H^_2'])

    fig, ax = plt.subplots()

    ax.axhline(y=h_true, ls='--', c='r', label=r'$H_\text{true}$')

    sns.lineplot(data=df_data, x='N', y='H^_1', ax=ax, errorbar=('sd', 2), label=r'$\hat{H}_\text{MM}$')
    sns.scatterplot(data=df_data, x='N', y='H^_1', ax=ax, marker='x', s=25)

    fig.tight_layout()
    fig.show()
