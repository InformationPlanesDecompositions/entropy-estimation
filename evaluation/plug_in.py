from collections import defaultdict
from os import path

import seaborn as sns
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import h5py

import pandas as pd
import numpy as np

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
        var_true = plug_in.compute_entropy_variance(p=p, d=d, h_true=h_true)

        experiments = bernoulli.generate_samples(p=p, size=(n_experiments, n_samples, d))

        for experiment in experiments:
            samples_as_int = experiment.dot(1 << np.arange(d - 1, -1, -1))

            h_est = plug_in.estimate_entropy(samples_as_int, use_fast_estimate=True)
            mm_corr = mm.first_order(samples_as_int)

            var_est = plug_in.estimate_entropy_variance(samples_as_int, h_est, use_fast_estimate=True)

            data['p'].append(p)
            data['N'].append(n_samples)
            data['D'].append(d)
            data['H'].append(h_true)
            data['H^'].append(h_est)
            data['MM'].append(mm_corr)
            data['H^_MM'].append(h_est + mm_corr)
            data['Var'].append(var_true / n_samples)
            data['Var^'].append(var_est)
    
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
    plot_prefix = path.join(output_dir, f'{file_prefix}_plot')

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

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex='col', sharey=False, figsize=(8, 12))
    fig_var, axes_var = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(12, 4))
    axes_var = axes_var.ravel()

    for idx, p in enumerate(ps):
        axes_p: list[matplotlib.axes.Axes] = axes[idx]
        ax_h, ax_err = axes_p
        ax_var: matplotlib.axes.Axes = axes_var[idx]

        axes_title = rf'$p = {p}$'
        
        df_p = df_data[df_data['p'] == p]

        sns.lineplot(data=df_p, x='D', y='H', ax=ax_h, c='tab:red', ls=':', alpha=0.75, errorbar=None, label=r'Truth')
        sns.lineplot(data=df_p, x='D', y='H^', c='tab:green', ls='-', ax=ax_h, label=r'Plug-in')
        sns.lineplot(data=df_p, x='D', y='H^_MM', c='tab:blue', ls='-.', ax=ax_h, errorbar=None, label=r'Miller-Madow')

        ax_h.get_legend().remove()

        ax_h.set_title(axes_title)

        ax_h.set_xticks(axes_xticks)
        ax_h.set_xlabel(r'RV vector size $D$')
        ax_h.set_ylabel(r'Entropy Estimate $H$')
        ax_h.set_ylim((0, df_p['H'].max() + 0.5))
        
        ax_h.grid(**axes_grid_kws)

        df_err = df_p[['H^', 'H^_MM']].sub(df_p['H'], axis=0).abs()
        df_err['D'] = df_p['D']

        sns.lineplot(data=df_err, x='D', y='H^', c='tab:green', ls='-', ax=ax_err)
        sns.lineplot(data=df_err, x='D', y='H^_MM', c='tab:blue', ls='-.', ax=ax_err)

        ax_err.set_title(axes_title)

        ax_err.set_xticks(axes_xticks)
        ax_err.set_xlabel(r'RV Vector size $D$')
        ax_err.set_ylabel(r'Absolute Error $|\varepsilon|$')
        ax_err.set_yscale('log', base=2)

        ax_err.grid(**axes_grid_kws)

        sns.lineplot(data=df_p, x='D', y='Var', c='tab:red', ls=':', ax=ax_var, alpha=0.75, errorbar=None)
        sns.lineplot(data=df_p, x='D', y='Var^', c='tab:green', ls='-', ax=ax_var)

        ax_var.grid(**axes_grid_kws)

        ax_var.set_title(axes_title)

        ax_var.set_xticks(axes_xticks)
        ax_var.set_xlabel(r'RV vector size $D$')
        ax_var.set_ylabel(r'Entropy Variance Estimate $\sigma^2_H$')

        ax_var.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{float(x * 1e3):.2f}'))

        ax_var.text(0.01, 1.02, r'$1e{-3}$', transform=ax_var.transAxes, va='bottom', ha='left')
        
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    fig_var.tight_layout()

    if save:
        fig.savefig(f'{plot_prefix}_entropy.pdf')
        fig_var.savefig(f'{plot_prefix}_variance.pdf')

    plt.show(block=True)


def evaluate_entropy_subaddivity(
    activation_data: h5py.Group | h5py.Dataset,
    output_dir: str,
    file_postfix: str = '',
    save: bool = False,
):
    data = defaultdict(list)
    rng = np.random.default_rng(2620)

    if file_postfix != '' and not file_postfix.startswith('_'):
        file_postfix = '_' + file_postfix

    layer_widths = dict()

    for epoch_data in tqdm(activation_data.values(), ncols=100, ascii=True):  # type: ignore
        epoch_data: h5py.Group
        epoch_idx = epoch_data.attrs['epoch_idx']

        for _, layer_data in epoch_data.items():
            layer_idx = layer_data.attrs['layer_idx']

            if not layer_data.attrs['is_packed']:
                continue

            data['Epoch'].append(epoch_idx)
            data['Layer'].append(layer_idx)

            t = np.unpackbits(layer_data[:])
            t = t.reshape(-1, *layer_data.attrs['shape'])

            _, dim = t.shape

            if layer_idx not in layer_widths.keys():
                layer_widths[layer_idx] = dim

            t_int = t.dot(1 << np.arange(dim - 1, -1, -1))
            h_t = plug_in.estimate_entropy(t_int, use_fast_estimate=True)

            t_shuffle = np.apply_along_axis(rng.permutation, 0, t)
            t_shuffle_int = t_shuffle.dot(1 << np.arange(dim - 1, -1, -1))
            h_t_shuffle = plug_in.estimate_entropy(t_shuffle_int, use_fast_estimate=True)

            ps_hat = t.mean(axis=0)
            h_neurons = -np.sum(ps_hat * np.log2(ps_hat) + (1 - ps_hat) * np.log2(1 - ps_hat))

            data['HT'].append(h_t)
            data['HShuffle'].append(h_t_shuffle)
            data['HNeurons'].append(h_neurons)

    df_data = pd.DataFrame.from_dict(data, orient='columns')

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
    axes = axes.ravel()

    for layer_idx in range(4):
        ax: matplotlib.axes.Axes = axes[layer_idx]

        df_layer = df_data[df_data['Layer'] == layer_idx]
        sns.lineplot(data=df_layer, x='Epoch', y='HT', label=r'$\hat{H}(T_\ell)$', ls='-', ax=ax, legend=False)
        sns.lineplot(data=df_layer, x='Epoch', y='HShuffle', label=r'$\hat{H}(T_\ell^\text{shuffle})$', ls=':', alpha=0.75, ax=ax, legend=False)
        sns.lineplot(data=df_layer, x='Epoch', y='HNeurons', label=r'$\sum_i \hat{H}(T_{\ell, i})$', ls='--', alpha=0.75, ax=ax, legend=False)

        ax.set_title(rf'$d_\ell = {layer_widths[layer_idx]}$')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$H(T)$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.925))

    if save:
        plt.savefig(path.join(output_dir, f'Entropy_Subadditivity{file_postfix}.pdf'), dpi=300)

    plt.show(block=True)
