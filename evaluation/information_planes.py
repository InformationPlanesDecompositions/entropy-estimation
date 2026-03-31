import itertools
import os
from os import path
import pathlib

import h5py

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from mi.information_plane import estimate_mi_data
from utility.data import concat_experiment_files


def generate_information_planes(
    data_dir: pathlib.Path,
    run_idx: int | None,
    show_plots: bool,
    compute_mi: bool,
    save: bool,
    as_pdf: bool,
):
    if data_dir.is_dir():
        raise FileNotFoundError(f'Please provide an existing directory, did not find {data_dir}')

    dir_name = path.join(data_dir.parent.name, data_dir.name)

    output_dir = path.join('output/mi/', dir_name)

    if save:
        os.makedirs(output_dir, exist_ok=True)

    mi_data_path = path.join(output_dir, 'mi_data.csv')

    if not compute_mi:
        if not path.isfile(mi_data_path):
            raise FileNotFoundError(f'Could not find existing MI data in {output_dir}')

        df_data = pd.read_csv(path.join(output_dir, 'mi_data.csv'), decimal=',', sep=';')
    else:
        activation_path = data_dir.joinpath('activations.h5')
        data_path = data_dir.joinpath('data.h5')

        if not activation_path.is_file():
            raise FileNotFoundError(f'No <activations.h5> found in given directory')

        if not data_path.is_file():
            raise FileNotFoundError(f'No <data.h5> found in given directory')

        activation_file = h5py.File(activation_path, 'r')
        data_file = h5py.File(data_path, 'r')

        if not activation_file.attrs.get('has_top_group', False):
            activation_iter = enumerate([activation_file])
        else:
            activation_iter = activation_file.items()

        dfs: list[pd.DataFrame] = []

        for _, run_data in activation_iter:
            run_idx = run_data.attrs.get('group_idx', 0)

            if run_idx is not None and run_idx != run_idx:
                continue

            df = estimate_mi_data(run_data, data_file)
            df.set_index('Epoch', drop=True, inplace=True)
            df['Run'] = run_idx

            dfs.append(df)

            if run_idx is not None:
                break
        
        df_data = pd.concat(dfs)

        if save:
            df_data.to_csv(mi_data_path, decimal=',', sep=';')
        
        df_data.reset_index(drop=False, inplace=True)
    
    if not show_plots and not save:
        return
    
    run_indices = df_data['Run'].unique() if run_idx is None else [run_idx]
    last_run = df_data['Run'].max() if run_idx is None else run_idx

    for run_idx in run_indices:
        df_run = df_data[df_data['Run'] == run_idx]

        plot_information_plane(
            df_run, show_plt=show_plots, block_plt=run_idx == last_run,
            save=save, output_dir=output_dir, postfix=f'_run_{run_idx}',
            as_pdf=as_pdf,
        )


def plot_information_plane(
    df_data: pd.DataFrame,
    show_plt: bool = True,
    block_plt: bool = True,
    save: bool = True,
    output_dir: str = '',
    postfix: str = '',
    as_pdf: bool = False,
    palette: str = 'cividis',
    ax: matplotlib.axes.Axes | None = None,
    cmap: plt.cm.ScalarMappable | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.8))

    if cmap is None:
        min_epoch, max_epoch = df_data['Epoch'].min(), df_data['Epoch'].max()

        norm = matplotlib.colors.Normalize(min_epoch, max_epoch)
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=palette)
        cmap.set_array([])

    sct_ax = sns.scatterplot(
        data=df_data,
        x='MI_x', y='MI_y',
        hue='Epoch', style='Layer', palette=palette,
        s=15,
        linewidth=0.01,
        edgecolor='#9999990F',
        ax=ax,
    )

    ax.set_xlabel(r'$I(X;T_\ell)$')
    ax.set_ylabel(r'$I(T_\ell;Y)$')

    if (lg := ax.get_legend()) is not None:
        lg.remove()

    if (not show_plt and not save):
        return ax
        
    cbar = ax.figure.colorbar(cmap, ax=sct_ax)
    cbar.ax.set_xlabel('Epoch')

    if save:
        plt.savefig(
            path.join(output_dir, f'information_plane{postfix}.{"pdf" if as_pdf else "png"}'),
            dpi=300,
            bbox_inches='tight',
        )

    if show_plt:
        if ax is None:
            fig.tight_layout()

        plt.show(block=block_plt)
    else:
        plt.close()

    return ax


def compare_information_planes(
    experiments: dict[str, str],
    run_idx: int,  # NOTE
    dir_mi: pathlib.Path,
    dir_exp: pathlib.Path | None,
    show_plots: bool,
    plot_layout: tuple[int, int],
    name_as_wd: bool,
    plot_losses: bool,  # NOTE
    plot_accuracy: bool,  # NOTE
    save: bool,
    output: pathlib.Path,
):
    if not dir_mi.is_dir():
        raise FileNotFoundError(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    if (n_exp := len(experiments)) == 0:
        raise ValueError('No experiments provided')
    
    n_rows, n_cols = plot_layout

    if n_cols > n_rows:
        h_ratio = 1
        w_ratio = n_cols / n_rows
    else:
        w_ratio = 1
        h_ratio = n_rows / n_cols

    max_n_exp = n_rows * n_cols

    if n_exp > max_n_exp:
        print(f'WARNING: {n_exp} experiments were provided, only {max_n_exp} are supported and will be plotted')
        experiments = dict(itertools.islice(experiments.items(), max_n_exp))
    elif n_exp < max_n_exp:
        print(f'INFO: Program is designed for {max_n_exp} plots, but only {n_exp} experiments were provided')

    if name_as_wd:
        experiments = {key: rf'$\lambda = {val}$' for key, val in experiments.items()}

    experiment_names = list(experiments.values())

    n_exp = min(n_exp, max_n_exp)

    files = ['mi_data.csv']
    dirs = [dir_mi]

    if plot_losses or plot_accuracy:
        if dir_exp is None:
            raise AttributeError('No data directory for experiments provided')

        if not dir_exp.is_dir():
            raise FileNotFoundError(f'Invalid data directory for experiments provided, could not find {dir_exp}')

        files.append('metrics.csv')
        dirs.append(dir_exp)

    concated_dfs = concat_experiment_files(
        experiments,
        files=files,
        dirs=dirs,  # type: ignore
        is_key_path=True
    )

    if plot_accuracy or plot_losses:
        df_mis, df_metrics, *_ = concated_dfs
    else:
        df_mis, *_ = concated_dfs
        df_metrics = None

    figsize = (5 * w_ratio, 5 * h_ratio)

    palette = 'cividis'

    # --------------------
    # Plot information planes
    # --------------------

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)
    axes = axes.ravel() if n_rows * n_cols > 1 else [axes]

    norm = matplotlib.colors.Normalize(df_mis['Epoch'].min(), df_mis['Epoch'].max())
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=palette)
    cmap.set_array([])

    for idx, exp_name in enumerate(experiment_names):
        ax: matplotlib.axes.Axes = axes[idx]

        df = df_mis[(df_mis['Experiment'] == exp_name) & (df_mis['Run'] == run_idx)]

        plot_information_plane(
            df, show_plt=False, save=False, palette=palette, cmap=cmap, ax=ax,
        )

        ax.set_title(exp_name)

    cbar = fig.colorbar(cmap, ax=axes[n_cols - 1::n_cols])
    cbar.ax.set_xlabel('Epoch')
    fig.subplots_adjust(right=0.85)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    
    # --------------------
    # Plot metrics
    # --------------------

    if not (plot_losses or plot_accuracy) or df_metrics is None:
        if save:
            os.makedirs(output.parent, exist_ok=True)

            plt.savefig(
                output.with_stem(output.stem + '-ips').with_suffix('.pdf'),
                dpi=300,
                format='pdf',
            )

        if show_plots:
            plt.show(block=True)

        return

    for plt_type, show_plt in [('Loss', plot_losses), ('Accuracy', plot_accuracy)]:
        if not show_plt:
            continue

        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=plt_type == 'Accuracy', figsize=figsize)
        axes = axes.ravel()

        for idx, exp_name in enumerate(experiment_names):
            ax: matplotlib.axes.Axes = axes[idx]

            df = df_metrics[(df_metrics['Experiment'] == exp_name) & (df_metrics['Run'] == run_idx)]

            if plt_type == 'Loss':
                sns.lineplot(df, x='Epoch', y='Train Loss', ax=ax, label='Train', alpha=0.75, legend=False)
                sns.lineplot(df, x='Epoch', y='Val. Loss', ax=ax, label='Validation', ls=':', alpha=0.75, legend=False)
            else:
                sns.lineplot(df, x='Epoch', y='Val. Acc', ax=ax)

            ax.set_title(exp_name)
        
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel(plt_type)

        if plt_type == 'Loss':
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout(rect=(0, 0, 1, 0.925))
        else:
            fig.tight_layout()

        if save:
            fig.savefig(
                output.with_stem(f'{output.stem}-{plt_type}').with_suffix('.pdf'),
                dpi=300,
                format='pdf'
            )
    
    if show_plots:
        plt.show(block=True)
