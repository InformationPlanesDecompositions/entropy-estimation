from os import path

import seaborn as sns
import matplotlib.axes
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from collections import defaultdict

from distributions import bernoulli
from corrections import mm
from estimators import plug_in


def _generate_evaluation_data(
    p: float,
    n_samples: int = 1000,
    n_experiments: int = 20,
    max_d: int = 20
) -> dict[str, list[int | float]]:
    data = defaultdict(list)

    for d in range(1, max_d + 1):
        h_true = bernoulli.compute_joint_entropy(p=p, d=d)

        experiments = bernoulli.generate_samples(p=p, size=(n_experiments, n_samples, d))

        for experiment in experiments:
            samples_as_int = experiment.dot(1 << np.arange(d - 1, -1, -1))

            h_est = plug_in.estimate_entropy(samples_as_int, use_fast_estimate=True)
            mm_corr = mm.first_order(samples_as_int)
            mm_corr_hat = mm.first_order(samples_as_int, n_classes=2 ** d)

            data['p'].append(p)
            data['N'].append(n_samples)
            data['D'].append(d)
            data['H'].append(h_true)
            data['H^'].append(h_est)
            data['MM'].append(mm_corr)
            data['H^_MM'].append(h_est + mm_corr)
            data['MM^'].append(mm_corr_hat)
            data['H^_MM^'].append(h_est + mm_corr_hat)
    
    return data


def evaluate_plugin_estimate(
    n_experiments: int = 20,
    n_samples: int = 1000,
    max_d: int = 20,
    use_existing_data: bool = True,
    save: bool = True,
    output_dir: str = '',
):
    file_prefix = f'Bernoulli_N{f"{n_samples:.0e}".replace('+', '')}_D{max_d}'
    
    data_path = path.join(output_dir, f'{file_prefix}_data.csv')
    plot_path = path.join(output_dir, f'{file_prefix}_plot.png')

    n_experiments = 20
    ps = [0.5, 0.7, 0.9]

    if use_existing_data and (path.exists(data_path) and path.isfile(data_path)):
        df_data = pd.read_csv(data_path, index_col=0, decimal=',', sep=';')
    else:
        data = defaultdict(list)

        for p in ps:
            _data = _generate_evaluation_data(p, n_samples, n_experiments=n_experiments)

            for k, vs in _data.items():
                data[k].extend(vs)

        df_data = pd.DataFrame(data)

        if save:
            df_data.to_csv(data_path, sep=';', decimal=',')

    axes_grid_kws = {
        'visible': True,
        'alpha': 0.75,
        'ls': '--',
    }
    axes_xticks = np.arange(2, max_d + 2, 2)

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex='col', sharey=False, figsize=(10, 12))

    for idx, p in enumerate(ps):
        axes_p: list[matplotlib.axes.Axes] = axes[idx]
        ax_h, ax_err = axes_p

        axes_title = rf'$p = {p}$'
        
        df_p = df_data[df_data['p'] == p]

        sns.lineplot(data=df_p, x='D', y='H', ax=ax_h, c='tab:red', ls=':', alpha=0.75, errorbar=None, label=r'$H_\text{true}$')
        sns.lineplot(data=df_p, x='D', y='H^', c='tab:green', ls='-', ax=ax_h, label=r'$\hat{H}$')
        # sns.scatterplot(data=df_p, x='D', y='H^', c='tab:green', ax=ax_h, marker='x', s=25, label=None)

        sns.lineplot(data=df_p, x='D', y='H^_MM', c='tab:blue', ls='-.', ax=ax_h, errorbar=None, label=r'$\hat{H}_\text{MM}$')
        sns.lineplot(data=df_p, x='D', y='H^_MM^', c='tab:orange', ls='--', ax=ax_h, errorbar=None, label=r'$\hat{H}_\text{MM}$, $\hat{k} = |\mathcal{X}|$')

        ax_h.get_legend().remove()

        ax_h.set_title(axes_title)

        ax_h.set_xticks(axes_xticks)
        ax_h.set_xlabel(r'RV vector size $D$')
        ax_h.set_ylabel(r'Entropy Estimate $H$')
        ax_h.set_ylim((0, df_p['H'].max() + 0.5))
        
        ax_h.grid(**axes_grid_kws)

        df_err = df_p[['H^', 'H^_MM', 'H^_MM^']].sub(df_p['H'], axis=0).abs()
        df_err['D'] = df_p['D']

        sns.lineplot(data=df_err, x='D', y='H^', c='tab:green', ls='-', ax=ax_err)
        sns.lineplot(data=df_err, x='D', y='H^_MM', c='tab:blue', ls='-.', ax=ax_err)
        sns.lineplot(data=df_err, x='D', y='H^_MM^', c='tab:orange', ls='--', ax=ax_err)

        ax_err.set_title(axes_title)

        ax_err.set_xticks(axes_xticks)
        ax_err.set_xlabel(r'RV Vector size $D$')
        ax_err.set_ylabel(r'Absolute Error $|\varepsilon|$')
        ax_err.set_yscale('log', base=2)

        ax_err.grid(**axes_grid_kws)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    if save:
        plt.savefig(plot_path)

    plt.show(block=True)
