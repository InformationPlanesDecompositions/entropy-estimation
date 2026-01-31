import argparse
import itertools

import os
from os import path

from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colors

import numpy as np
import pandas as pd

import h5py

import cli.parser
import cli.configure

import evaluation
import evaluation.plug_in

from mi import information_plane


def _perform_evaluation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.evaluation_type != 'plug-in':
        parser.error(f'Unknown evaluation target <{args.evaluation_type}>')

    n_experiments: int = args.n_experiments
    n_samples: int = args.n_samples
    max_d: int = args.max_dimensions

    save: bool = args.save
    use_existing: bool = args.use_existing
    output_dir = args.output

    if save:
        os.makedirs(output_dir, exist_ok=True)

    evaluation.plug_in.evaluate_plugin_estimate(
        n_experiments, n_samples, max_d,
        use_existing, save, output_dir
    )


def _evaluate_subadditivity(parser: argparse.ArgumentParser, args: argparse.Namespace):
    data_dir = args.data

    if not path.isdir(data_dir):
        parser.error(f'Please provide an existing directory, did not find {data_dir}')
    
    dir_name = path.basename(path.dirname(data_dir) if path.basename(data_dir) == '' else data_dir)

    output_dir = f'output/ee/'

    if args.save:
        os.makedirs(output_dir, exist_ok=True)

    activation_path = path.join(data_dir, 'activations.h5')

    if not (path.exists(activation_path) and path.isfile(activation_path)):
        parser.error(f'No <activations.h5> found in given directory')

    activation_file = h5py.File(activation_path, 'r')

    run_idx = args.run

    run_data = activation_file.get(f'run_{run_idx}', None) if activation_file.attrs['has_top_group'] else activation_file

    if run_data is None:
        parser.error(f'No run with index {run_idx} found in activation data')

    evaluation.plug_in.evaluate_entropy_subadditivity(
        run_data,  # type: ignore
        output_dir,
        file_postfix=f'_{dir_name}_run_{run_idx}',
        save=args.save
    )


def _perform_mi_estimation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    data_dir = args.data

    if not path.isdir(data_dir):
        parser.error(f'Please provide an existing directory, did not find {data_dir}')

    prefix, dir_name = path.split(path.dirname(data_dir) if path.basename(data_dir) == '' else data_dir)

    if (prefix := path.basename(prefix)) != 'output':
        dir_name = path.join(prefix, dir_name)

    output_dir = f'output/mi/{dir_name}'

    if args.save:
        os.makedirs(output_dir, exist_ok=True)

    show_plt = bool(args.show_plots)
    compute_mi = bool(args.compute_mi)

    run_selection: None | int = args.run

    mi_data_path = path.join(output_dir, 'mi_data.csv')

    if not compute_mi:
        if not path.isfile(mi_data_path):
            parser.error(f'Could not find existing MI data in {output_dir}')

        df_data = pd.read_csv(path.join(output_dir, 'mi_data.csv'), decimal=',', sep=';')
    else:
        activation_path = path.join(data_dir, 'activations.h5')
        data_path = path.join(data_dir, 'data.h5')

        if not path.isfile(activation_path):
            parser.error(f'No <activations.h5> found in given directory')

        if not path.isfile(data_path):
            parser.error(f'No <data.h5> found in given directory')

        activation_file = h5py.File(activation_path, 'r')
        data_file = h5py.File(data_path, 'r')

        if not activation_file.attrs.get('has_top_group', False):
            activation_iter = enumerate([activation_file])
        else:
            activation_iter = activation_file.items()

        dfs: list[pd.DataFrame] = []

        for _, run_data in activation_iter:
            run_idx = run_data.attrs.get('group_idx', 0)

            if run_selection is not None and run_idx != run_selection:
                continue

            df = information_plane.estimate_mi_data(run_data, data_file)
            df.set_index('Epoch', drop=True, inplace=True)
            df['Run'] = run_idx

            dfs.append(df)

            if run_selection is not None:
                break
        
        df_data = pd.concat(dfs)

        if args.save:
            df_data.to_csv(mi_data_path, decimal=',', sep=';')
        
        df_data.reset_index(drop=False, inplace=True)
    
    if not show_plt and not args.save:
        return
    
    run_indices = df_data['Run'].unique() if run_selection is None else [run_selection]
    last_run = df_data['Run'].max() if run_selection is None else run_selection

    for run_idx in run_indices:
        df_run = df_data[df_data['Run'] == run_idx]

        information_plane.plot_information_plane(
            df_run, show_plt, block_plt=run_idx == last_run,
            save=args.save, output_dir=output_dir, postfix=f'_run_{run_idx}',
            as_pdf=args.plot_as_pdf,
        )


def _compare_experiments(parser: argparse.ArgumentParser, args: argparse.Namespace):
    config = cli.configure.read_config(args.config)
    config = config.get('comparison', config)

    experiments = config.get('experiments', {})

    if type(experiments) != dict or len(experiments) == 0:
        parser.error(f'Please provide a dict of experiments')

    n_exp = len(experiments)

    n_rows: int
    n_cols: int
    n_rows, n_cols = args.plot_layout

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

    n_exp = min(n_exp, max_n_exp)

    dir_exp = args.dir_experiments
    dir_mi = args.dir_mi

    if not path.isdir(dir_exp):
        parser.error(f'Invalid data directory for experiments provided, could not find {dir_exp}')

    if not path.isdir(dir_mi):
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    run_idx = config.get('run_idx', 0) if args.run is None else args.run
    plot_accuracy = config.get('accuracy_plot', True) if args.accuracy_plot is None else args.accuracy_plot
    plot_losses = config.get('loss_plot', False) if args.loss_plot is None else args.loss_plot

    dfs_metrics = []
    dfs_mis = []

    for p, n in experiments.items():
        df_metrics = pd.read_csv(path.join(dir_exp, p,'metrics.csv'), sep=';', decimal=',')
        df_mi = pd.read_csv(path.join(dir_mi, p, 'mi_data.csv'), sep=';', decimal=',')

        df_metrics['Experiment'] = n
        df_mi['Experiment'] = n

        dfs_metrics.append(df_metrics)
        dfs_mis.append(df_mi)

    df_metrics = pd.concat(dfs_metrics, ignore_index=True)
    df_mis = pd.concat(dfs_mis, ignore_index=True)

    figsize = (5 * w_ratio, 5 * h_ratio)

    # --------------------
    # Plot information planes
    # --------------------

    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)
    axes = axes.ravel()

    norm = matplotlib.colors.Normalize(df_mis['Epoch'].min(), df_mis['Epoch'].max())
    cmap = plt.cm.ScalarMappable(norm=norm, cmap='flare_r')
    cmap.set_array([])

    for idx, exp_name in enumerate(experiments.values()):
        ax: matplotlib.axes.Axes = axes[idx]

        df = df_mis[(df_mis['Experiment'] == exp_name) & (df_mis['Run'] == run_idx)]

        sns.scatterplot(
            data=df, x='MI_x', y='MI_y',
            hue='Epoch', style='Layer', ax=ax,
            palette='flare_r', alpha=0.75, s=25
        )
        ax.set_title(exp_name)

        if (lg := ax.get_legend()) is not None:
            lg.remove()

    for ax in axes:
        ax.set_xlabel(r'$I(X;T_\ell)$')
        ax.set_ylabel(r'$I(T_\ell;Y)$')


    cbar = fig.colorbar(cmap, ax=axes[n_cols - 1::n_cols])
    cbar.ax.set_xlabel('Epoch')
    fig.subplots_adjust(right=0.85)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    # --------------------
    # Plot metrics
    # --------------------

    for plt_type, show_plt in [('Loss', plot_losses), ('Accuracy', plot_accuracy)]:
        if not show_plt:
            continue

        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=plt_type == 'Accuracy', figsize=figsize)
        axes = axes.ravel()

        for idx, exp_name in enumerate(experiments.values()):
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
    
    plt.show(block=True)


def _compare_compression(parser: argparse.ArgumentParser, args: argparse.Namespace):
    def err_low(col: pd.Series):
        return col.mean() - col.min()

    def err_high(col: pd.Series):
        return col.max() - col.mean()

    config = cli.configure.read_config(args.config)
    config = config.get('comparison', config)

    experiments: dict = config.get('experiments', {})

    if type(experiments) != dict or len(experiments) == 0:
        parser.error(f'Please provide a dict of experiments')

    n_exp = len(experiments)

    dir_exp = str(args.dir_experiments)
    dir_mi = str(args.dir_mi)

    if not path.isdir(dir_exp):
        parser.error(f'Invalid data directory for experiments provided, could not find {dir_exp}')

    if not path.isdir(dir_mi):
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    layer_revoffset_idx = int(args.layer_offset_idx)
    if layer_revoffset_idx > 0:
        layer_revoffset_idx *= -1

    n_epochs = int(args.n_epochs)
    agg_func: str = str(args.agg_func)

    exp_as_cbar: bool = bool(args.exp_as_cbar)
    legend_title: str = str(args.legend_title)

    df_metrics, df_mis, *_ = _concat_experiment_files(
        experiments,
        files=['metrics.csv', 'mi_data.csv'],
        dirs=[dir_exp, dir_mi],
        is_key_path=True
    )

    df = pd.merge(df_mis, df_metrics, on=['Experiment', 'Run', 'Epoch'], how='left')

    max_layer_indices = df.groupby(by='Experiment')['Layer'].max()
    df['Layer_RevOffset'] = df.apply(lambda row: row['Layer'] - max_layer_indices[row['Experiment']], axis=1)

    df = df[df.apply(lambda row: row['Layer_RevOffset'] == layer_revoffset_idx, axis=1)]
    df = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)]

    df_grouped = df.groupby(by=['Experiment', 'Run'])
    df_agg = df_grouped.aggregate({
        'MI_x': [agg_func, err_low, err_high],
        'Val. Acc': [agg_func, err_low, err_high]
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.8))

    palette = 'plasma'

    exp_name_order = {str(name): idx for idx, name in enumerate(experiments.values())}

    if exp_as_cbar:
        min_val, max_val = df['Experiment'].min(), df['Experiment'].max()

        cmap = plt.cm.ScalarMappable(cmap=palette)
        cmap.set_array([min_val, max_val])

        c_per_exp = {exp: cmap.to_rgba(float(exp), norm=True) for exp in df_agg['Experiment'].unique()}  # type: ignore
    else:
        cmap = sns.color_palette(palette=palette, n_colors=n_exp)
        c_per_exp = {exp: cmap[idx] for idx, exp in enumerate(df_agg['Experiment'].unique())}

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
        hue='Experiment', style='Experiment', palette=palette if exp_as_cbar else cmap,  # type: ignore
        s=50,
        ax=ax,
        zorder=2,
    )
    
    if not exp_as_cbar:
        handles, labels = sct_ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda tup: exp_name_order[str(tup[0])]))

    if (lg := sct_ax.get_legend()) is not None:
        lg.remove()

    ax.set_xlabel(r'$I(X;T_\ell)$')
    ax.set_ylabel('Accuracy')

    ax.grid(True, alpha=0.75, ls=':')

    if exp_as_cbar:
        cbar = ax.figure.colorbar(cmap, ax=sct_ax)  # type: ignore
        cbar.ax.set_xlabel(legend_title)  # type: ignore

        fig.tight_layout()
    else:
        fig.legend(handles, labels, title=legend_title, loc='center right')  # type: ignore
        fig.subplots_adjust(right=0.75)
        fig.tight_layout(rect=(0, 0, 0.75, 1))

    if args.save:
        output = str(args.output)

        if not output.lower().endswith('.pdf'):
            output += '.pdf'

        os.makedirs(path.dirname(output), exist_ok=True)
        plt.savefig(output, dpi=300, format='pdf')

    plt.show(block=True)


def _quantify_compression(parser: argparse.ArgumentParser, args: argparse.Namespace):
    config = cli.configure.read_config(args.config)
    config = config.get('comparison', config)

    experiments: dict = config.get('experiments', {})

    exp_as_cbar = bool(args.exp_as_cbar)
    legend_title = str(args.legend_title)
    ref = str(args.reference_func)
    n_epochs = int(args.n_epochs)
    dir_mi = str(args.dir_mi)

    if not path.isdir(dir_mi):
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    [df] = _concat_experiment_files(experiments, ['mi_data.csv'],  [dir_mi], is_key_path=True)

    max_layer_indices = df.groupby(by='Experiment')['Layer'].max()
    df['Layer'] = df.apply(lambda row: row['Layer'] - max_layer_indices[row['Experiment']], axis=1).astype(int)

    df.drop(index=df[df['Layer'] == 0].index, inplace=True)

    palette = 'plasma'

    if ref == 'max':
        ref_x = df.groupby(by=['Experiment', 'Run', 'Layer'])['MI_x'].max()
    else:
        ref_x = df[df['Epoch'] == 0].groupby(by=['Experiment', 'Run', 'Layer'])['MI_x'].min()
    
    end_x = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)].groupby(by=['Experiment', 'Run', 'Layer']).mean()['MI_x']

    df_rho = (ref_x - end_x) / ref_x
    df_rho = df_rho.reset_index(name='Rho')

    fig, ax = plt.subplots()

    sns.boxplot(df_rho, x='Layer', y='Rho', fill=False, color='grey')
    strp_ax = sns.swarmplot(
        df_rho, x='Layer', y='Rho',
        hue='Experiment', palette=palette, hue_order=experiments.values(),
        ax=ax, legend=True,
    )

    strp_ax.legend().remove()

    if exp_as_cbar:
        min_val, max_val = df['Experiment'].min(), df['Experiment'].max()

        cmap = plt.cm.ScalarMappable(cmap=palette)
        cmap.set_array([min_val, max_val])

        cbar = fig.colorbar(cmap, ax=strp_ax)
        cbar.ax.set_xlabel(legend_title)
    else:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title=legend_title)
    
    ax.grid(True, axis='y', alpha=0.75, ls='--')

    fig.tight_layout()

    plt.show(block=True)


def _concat_experiment_files(
    experiments: dict,
    files: list[str],
    dirs: list[str],
    is_key_path: bool = True,
) -> list[pd.DataFrame]:
    if len(files) != len(dirs):
        raise ValueError(f'Please provide equal number of files and directories, was {len(files)}--{len(dirs)}')

    dfs: defaultdict[str, list[pd.DataFrame]] = defaultdict(list)
    
    for exp_path, exp_name in experiments.items():
        if not is_key_path:
            exp_path, exp_name = exp_name, exp_path

        exp_path = str(exp_path)

        for d, f in zip(dirs, files):
            df = pd.read_csv(path.join(d, exp_path, f), sep=';', decimal=',')

            df['Experiment'] = exp_name

            dfs[f].append(df)

    return [pd.concat(df_list, ignore_index=True) for df_list in dfs.values()]


def main():
    parser = cli.parser.build_parser()
    args = parser.parse_args()

    match args.command:
        case 'evaluate':
            if args.eval_target == 'toy':
                _perform_evaluation(parser, args)
            else:
                _evaluate_subadditivity(parser, args)
        case 'mi' | 'mutual-information':
            _perform_mi_estimation(parser, args)
        case 'compare':
            if args.comparison_target in ['ip', 'information_plane']:
                _compare_experiments(parser, args)
            elif args.comparison_target == 'q1':
                _quantify_compression(parser, args)
            else:
                _compare_compression(parser, args)


if __name__ == '__main__':
    main()
