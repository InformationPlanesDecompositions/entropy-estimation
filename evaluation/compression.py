from collections import defaultdict
import os
from os import path
import pathlib

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats

from utility.data import concat_experiment_files


def quantify_compression(
    experiment_groups: dict[str, dict[str, list[str]]],
    included_layer_indices: dict[str, list[int]],
    included_groups: list[str],
    dataset_order: list[str],
    n_epochs: int,
    use_existing: bool,
    dir_mi: pathlib.Path,
    save: bool,
    output_dir: pathlib.Path,
    show_plt: bool,
):
    data_file_name = 'compression_factor_data.csv'

    if use_existing:
        if not (data_path := output_dir.joinpath(data_file_name)).is_file():
            raise FileNotFoundError(f'Prompted to use existing data, but did not find <{data_path}>')
        
        df_rho = pd.read_csv(data_path, decimal=',', sep=';', index_col=0)

        plot_compression(
            df_rho,
            dataset_order=dataset_order,
            save=save, output_dir=output_dir,
            show_plt=show_plt,
        )

        return
        
    if not dir_mi.is_dir():
        raise FileNotFoundError(f'Provided argument <dir_mi> is not a directory <{dir_mi}>')
    
    experiments = {
        exp: exp
        for groups in experiment_groups.values()
        for group_name, group in groups.items()
        if group_name in included_groups
        for exp in group
    }

    df_groupings = pd.DataFrame(
        [
            (dataset, group_name, exp) for dataset, groups in experiment_groups.items()
            for group_name, group in groups.items()
            for exp in group
        ],
        columns=['Dataset', 'Group', 'Experiment'],
    )

    df, *_ = concat_experiment_files(
        experiments,
        files=['mi_data.csv'],
        dirs=[dir_mi],
        is_key_path=True,
    )

    max_layer_indices = df.groupby(by='Experiment')['Layer'].max()

    df['Layer'] = df.apply(
        lambda row: row['Layer'] - max_layer_indices[row['Experiment']],
        axis=1
    ).astype(int)

    df = df_groupings.merge(df, on='Experiment', how='right')

    df.drop(
        index=df[df[['Group', 'Layer']].apply(
            lambda row: row['Layer'] not in included_layer_indices[row['Group']],
            axis=1
        )].index,
        inplace=True,
    )

    n_layers_per_group: pd.Series[int] = df.groupby(by=['Dataset', 'Group'])['Layer'].nunique()

    ref_x = df.groupby(by=['Dataset', 'Group', 'Experiment', 'Run', 'Layer'])['MI_x'].max()
    end_x = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)]\
        .groupby(by=['Dataset', 'Group', 'Experiment', 'Run', 'Layer'])\
        .mean()['MI_x']

    rho = (ref_x - end_x) / ref_x
    df_rho = rho.reset_index(name='Rho')

    df_rho['WD'] = df_rho['Experiment'].str.extract(r'-wd-(\d+(?:\.\d+)?e[-+]\d)')

    df_rho.dropna(subset=['WD'], inplace=True)
    df_rho['WD'] = df_rho['WD'].astype(float)

    df_rho['#X-Axis'] = df_rho[['Dataset', 'Group', 'Layer']].apply(
        lambda row: f'{row['Group']}\nLayer {row['Layer']}'
            if n_layers_per_group[(row['Dataset'], row['Group'])] > 1
            else row['Group'],
        axis=1
    )

    if save:
        df_rho.to_csv(output_dir.joinpath(data_file_name), decimal=',', sep=';')

    plot_compression(
        df_rho,
        dataset_order=dataset_order,
        save=save, output_dir=output_dir,
        show_plt=show_plt
    )


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


def compare_compressions(
    experiments: dict,
    dir_exp: pathlib.Path,
    dir_mi: pathlib.Path,
    layer_offset_idx: int,
    n_epochs: int,
    agg_func: str,
    legend_title: str,
    as_cbar: bool,
    is_discrete_cbar: bool,
    cbar_minimum: int,
    save: bool,
    output_dir: pathlib.Path,
    show_plt: bool,
):
    def err_low(col: pd.Series):
        err = col.mean() - col.min()

        return err if not np.isclose(err, 0) else 0

    def err_high(col: pd.Series):
        err = col.max() - col.mean()
    
        return err if not np.isclose(err, 0) else 0
    
    if len(experiments) == 0:
        raise ValueError(f'Please provide a dict of experiments')
    
    n_exp = len(experiments)
    
    if not dir_exp.is_dir():
        raise FileNotFoundError(f'Invalid data directory for experiments provided, could not find {dir_exp}')

    if not dir_mi.is_dir():
        raise FileNotFoundError(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')
    
    if layer_offset_idx > 0:
        layer_offset_idx *= -1

    df_metrics, df_mis, *_ = concat_experiment_files(
        experiments,
        files=['metrics.csv', 'mi_data.csv'],
        dirs=[dir_exp, dir_mi],
        is_key_path=True
    )

    df = pd.merge(df_mis, df_metrics, on=['Experiment', 'Run', 'Epoch'], how='left')

    max_layer_indices = df.groupby(by='Experiment')['Layer'].max()
    df['Layer_RevOffset'] = df.apply(lambda row: row['Layer'] - max_layer_indices[row['Experiment']], axis=1)

    df = df[df.apply(lambda row: row['Layer_RevOffset'] == layer_offset_idx, axis=1)]
    df = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)]

    df_grouped = df.groupby(by=['Experiment', 'Run'])
    df_agg = df_grouped.aggregate({
        'MI_x': [agg_func, err_low, err_high],
        'Val. Acc': [agg_func, err_low, err_high]
    }).reset_index()

    # df_agg.drop(index=27, inplace=True)  # Hard-coded; outlier point for fashion-1024-20-20-20-10-wd-1.7e-0

    fig, ax = plt.subplots(figsize=(6, 4.8))

    palette_name = 'cividis'

    if is_discrete_cbar:
        palette_name += '_r'

    categories = sorted(experiments.values()) if as_cbar else list(experiments.values())

    if is_discrete_cbar or not as_cbar:
        palette = sns.color_palette(palette=palette_name, n_colors=n_exp)
        c_per_exp = {exp: palette[idx] for idx, exp in enumerate(categories)}
    else:
        min_val, max_val = df['Experiment'].min(), df['Experiment'].max()

        palette = plt.cm.ScalarMappable(cmap=palette_name)
        palette.set_array([min_val, max_val])

        c_per_exp = {exp: palette.to_rgba(float(exp), norm=True) for exp in categories}  # type: ignore
        
    for _, row in df_agg.iterrows():
        plt.errorbar(
            x=row[('MI_x', agg_func)], y=row[('Val. Acc', agg_func)],
            xerr=[[row[('MI_x', err_low.__name__)]], [row[('MI_x', err_high.__name__)]]],
            yerr=[[row[('Val. Acc', err_low.__name__)]], [row[('Val. Acc', err_high.__name__)]]],
            color=c_per_exp[row['Experiment'].values[0]],
            fmt='none', capsize=5, alpha=0.5, zorder=-1,
        )

    sct_ax = sns.scatterplot(
        data=df_agg,
        x=('MI_x', agg_func), y=('Val. Acc', agg_func),
        hue='Experiment', style='Experiment', hue_order=experiments.values(),
        palette=palette_name if as_cbar else c_per_exp,  # type: ignore
        s=50,
        ax=ax,
        zorder=2,
    )
    
    if not as_cbar:
        handles, labels = sct_ax.get_legend_handles_labels()

    if (lg := sct_ax.get_legend()) is not None:
        lg.remove()

    ax.set_xlabel(r'$I(X;T_\ell)$')
    ax.set_ylabel('Accuracy')

    ax.grid(True, alpha=0.75, ls=':')

    if as_cbar:
        bounds = [cbar_minimum] + categories if cbar_minimum not in categories else categories

        if is_discrete_cbar:
            cmap = sns.color_palette(palette=palette_name, as_cmap=True)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
            value_array = []
        else:
            norm = None
            value_array = [df['Experiment'].min(), df['Experiment'].max()]
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=palette_name)
        sm.set_array(value_array)

        cbar = ax.figure.colorbar(sm, ax=sct_ax)
        cbar.ax.set_xlabel(legend_title)

        if is_discrete_cbar:
            cbar.ax.set_yticks(
                ticks=[high - (high - low) / 2 for low, high in zip(bounds, bounds[1:])],
                labels=[str(cat) for cat in categories]
            )

        fig.tight_layout()
    else:
        fig.legend(handles, labels, title=legend_title, loc='upper right', ncols=1)  # type: ignore
        fig.subplots_adjust(right=0.8)
        fig.tight_layout(rect=(0, 0, 0.8, 1))

    if save:
        if output_dir.suffix == '':
            output_dir = output_dir.with_name('tmp.pdf')
        elif output_dir.suffix != '.pdf':
            output_dir = output_dir.with_suffix('.pdf')

        os.makedirs(output_dir.parent, exist_ok=True)
        plt.savefig(output_dir, dpi=300, format='pdf')

    if show_plt:
        plt.show(block=True)


def compute_compression_rank_correlation(
    experiment_groups: dict[str, dict[str, list[str]]],
    dir_exp: pathlib.Path,
    dir_mi: pathlib.Path,
    n_epochs: int,
    to_latex: bool,
    output_dir: pathlib.Path,
):
    if not dir_exp.is_dir():
        raise FileNotFoundError(f'Invalid data directory for experiments provided, could not find {dir_exp}')

    if not dir_mi.is_dir():
        raise FileNotFoundError(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')
    
    experiments = {exp: exp for groups in experiment_groups.values() for group in groups.values() for exp in group}

    df_groupings = pd.DataFrame(
        [
            (ds_name, grp_name, exp) for ds_name, ds in experiment_groups.items()
            for grp_name, grp in ds.items()
            for exp in grp
        ],
        columns=['Dataset', 'Group', 'Experiment'],
    )

    df_metrics, df_mis, *_ = concat_experiment_files(
        experiments,
        ['metrics.csv', 'mi_data.csv'],
        [dir_exp, dir_mi],
        is_key_path=True
    )

    df = pd.merge(df_mis, df_metrics, on=['Experiment', 'Run', 'Epoch'], how='left')

    max_layer_indices = df.groupby(by='Experiment')['Layer'].max()
    df['Layer'] = df.apply(lambda row: row['Layer'] - max_layer_indices[row['Experiment']], axis=1).astype(int)
    df = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)]
    df.drop(index=df[df['Layer'] == 0].index, inplace=True)

    df = df_groupings.merge(df, on='Experiment', how='right')

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
