import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from collections import defaultdict

from distributions import bernoulli
from corrections import mm
from estimators import plug_in


# TODO: Implement for ternary Bernoulli / binomial


def onedimensional_bernoulli(n_experiments: int, p: float, n_samples_space: np.ndarray):
    if n_samples_space.dtype != np.int64:
        n_samples_space = n_samples_space.astype(np.int64)

    data = defaultdict(list)

    h_true = bernoulli.compute_entropy(p)

    for n_samples in n_samples_space:
        experiments = bernoulli.generate_samples(p, size=(n_experiments, n_samples))

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
    plt.show(block=True)


# TODO:
#   - Compute 2nd order MM correction
def multidimensional_bernoulli(
    n_experiments: int,
    p: float,
    n_samples: int,
    *,
    d_steps: np.ndarray | list | None = None,
    d_range: tuple[int, int] | None = None,
    d_n_steps: int = 10,
) -> None:
    if d_steps is None:
        if d_range is None:
            raise KeyError(f'Neither dimension steps nor range provided')

        d_min, d_max = d_range

        if d_min > d_max or d_min < 0:
            raise ValueError(f'Invalid dimension range provided, was [{d_min}, {d_max}]')
        
        d_steps = np.linspace(d_min, d_max, d_n_steps)

    if type(d_steps) is not np.ndarray:
        d_steps = np.array(d_steps)

    if d_steps.dtype != np.int64:
        d_steps = d_steps.astype(np.int64)

    data = defaultdict(list)

    for d in d_steps:
        h_true = bernoulli.compute_joint_entropy(p=p, d=d)
        
        experiments = bernoulli.generate_samples(p, size=(n_experiments, n_samples, d))

        estimates = [
            (plug_in.estimate_entropy(samples), plug_in.estimate_entropy_variance(samples), mm.first_order(samples))
            for samples in experiments
        ]

        for h_hat, var_hat, mm_first in estimates:
            data['D'].append(d)
            data['H_true'].append(h_true)
            data['H^'].append(h_hat)
            data['Var^'].append(var_hat)
            data['MM1'].append(mm_first)
            data['H^1'].append(h_hat + mm_first)
            # data['MM2'].append(mm_second)
            # data['H^2'].append(h_hat + mm_second)

    df_data = pd.DataFrame(data=data)

    fig, ax = plt.subplots()

    sns.lineplot(data=df_data, x='D', y='H^1', ax=ax, errorbar=None, label=r'$\hat{H}_\text{MM}$')
    sns.lineplot(x=df_data['D'], y=df_data['H^1'] + np.sqrt(df_data['Var^']), ax=ax, c='g', ls=':', errorbar=None, label=None)
    sns.lineplot(x=df_data['D'], y=df_data['H^1'] - np.sqrt(df_data['Var^']), ax=ax, c='g', ls=':', errorbar=None, label=None)
    sns.lineplot(data=df_data, x='D', y='H_true', ax=ax, label=r'$H_\text{true}$', c='r', ls='--', errorbar=None)
    sns.scatterplot(data=df_data, x='D', y='H^1', ax=ax, marker='x', s=25)

    ax.set_yscale('log', base=2)

    fig.tight_layout()
    plt.show()
