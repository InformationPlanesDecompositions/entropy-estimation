import os
from os import path
import pathlib


import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns


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
    df_mis: pd.DataFrame,
    df_metrics: pd.DataFrame | None,
    experiment_names: list[str],
    run_idx: int,
    n_rows: int,
    n_cols: int,
    figsize: tuple[float, float],
    plot_losses: bool,
    plot_accuracy: bool,
    save: bool,
    output: pathlib.Path,
    show_plots: bool,
):
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
