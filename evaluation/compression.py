from collections import defaultdict
import os
from os import path
import pathlib

import matplotlib.axes
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats


def plot_compression(
    df_rho: pd.DataFrame,
    dataset_order: list[str],
    save: bool,
    output_dir: pathlib.Path,
    show_plt: bool,
):
    palette = 'cividis'
    cmap = plt.cm.ScalarMappable(cmap=palette)
    cmap.set_array([df_rho['WD'].min(), df_rho['WD'].max()])

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4))
    axes = np.ravel(axes)

    for idx, dataset in enumerate(dataset_order):
        df_dataset = df_rho[df_rho['Dataset'] == dataset]
        ax: matplotlib.axes.Axes = axes[idx]

        plt_ax = sns.swarmplot(
            df_dataset,
            x='#X-Axis', y='Rho',
            hue='WD', palette=palette,
            s=5,
            edgecolor="auto",
            linewidth=0.01,
            ax=ax,
        )

        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel(r'$\varrho$')
        ax.set_title(dataset)

        ax.grid(True, alpha=0.85, ls='--', axis='y')

        if (lg := plt_ax.get_legend()) is not None:
            lg.remove()

    cbar = fig.colorbar(cmap, ax=axes[1::2])
    cbar.ax.set_xlabel(r'$\lambda$')
    fig.subplots_adjust(right=0.85)
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    if save:
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(
            output_dir.joinpath('compression_factors.pdf'),
            dpi=300,
            format='pdf',
        )

    if show_plt:
        plt.show(block=True)
    else:
        plt.close()


def compute_compression_rank_correlation(
    df: pd.DataFrame,
    to_latex: bool,
    output_dir: pathlib.Path,
):
    df_grouped = df.groupby(by=['Dataset', 'Group', 'Experiment', 'Run', 'Layer'])
    df_agg = df_grouped.aggregate({
        'MI_x': ['mean'],
        'Val. Acc': ['mean']
    }).reset_index()

    data: defaultdict[str, list] = defaultdict(list)

    for (dataset, group, layer_idx), df_group in df_agg.groupby(by=['Dataset', 'Group', 'Layer']):
        r, p = scipy.stats.spearmanr(df_group[[('MI_x', 'mean'), ('Val. Acc', 'mean')]])

        data['Dataset'].append(dataset)
        data['Group'].append(group)
        data['Layer'].append(layer_idx)
        data['SpearmanR'].append(r)
        data['p-Value'].append(p)

    os.makedirs(output_dir, exist_ok=True)

    df_result = pd.DataFrame(data)
    df_result.to_csv(path.join(output_dir, 'rank_corr_data.csv'), sep=';', decimal=',')

    if not to_latex:
        return
    
    lines = []

    lines.append('Dataset & Experiment Group & Layer & Spearman $r_s$ & $p$-value \\\\\n')
    lines.append('\\midrule\n')

    last_ds = ''

    for (_, ds, grp, layer_idx, r, p) in df_result.itertuples():
        if last_ds != '' and ds != last_ds:
            lines.append('\\midrule\n')
        
        last_ds = ds

        lines.append(
            f'{ds} & {grp} & {layer_idx} & $\\numprint{{{r}}}$ & ${f"\\numprint{{{p}}}" if p >= 0.0005 else "< 0.001"}$ \\\\\n'
        )

    with open(path.join(output_dir, 'rank_corr_table.tex'), 'w') as f:
        f.writelines(lines)
