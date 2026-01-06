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

from tqdm import tqdm

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


def _perform_mi_estimation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    data_dir = args.data

    if not path.isdir(data_dir):
        parser.error(f'Please provide an existing directory, did not find {data_dir}')

    prefix, dir_name = path.split(path.dirname(data_dir) if path.basename(data_dir) == '' else data_dir)

    if (prefix := path.basename(prefix)) != 'output':
        dir_name = path.join(prefix, dir_name)

    output_dir = f'output/mi/{dir_name}'

    show_plt = bool(args.show_plots)
    compute_mi = bool(args.compute_mi)

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

        run_selection: None | int = args.run

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
    
    run_indices = df_data['Run'].unique()
    last_run = df_data['Run'].max()

    for run_idx in run_indices:
        df_run = df_data[df_data['Run'] == run_idx]

        information_plane.plot_information_plane(
            df_run, show_plt, block_plt=run_idx == last_run,
            save=args.save, output_dir=output_dir, postfix=f'_{run_idx}',
        )


# TODO: Move to package evaluation, either into plug_in.py or model.py
def _compare_entropy(parser: argparse.ArgumentParser, args: argparse.Namespace):
    data_dir = args.data

    if not path.isdir(data_dir):
        parser.error(f'Please provide an existing directory, did not find {data_dir}')

    activation_path = path.join(data_dir, 'activations.h5')

    if not (path.exists(activation_path) and path.isfile(activation_path)):
        parser.error(f'No <activations.h5> found in given directory')

    activation_file = h5py.File(activation_path, 'r')

    run_idx = args.run

    data = defaultdict(list)
    rng = np.random.default_rng(2620)

    run_data = activation_file.get(f'run_{run_idx}', None) if activation_file.attrs['has_top_group'] else activation_file

    if run_data is None:
        parser.error(f'No run with index {run_idx} found in activation data')

    for epoch_data in tqdm(run_data.values(), ncols=100, ascii=True):  # type: ignore
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

            n, dim = t.shape

            t_int = t.dot(1 << np.arange(dim - 1, -1, -1))
            p_t = np.bincount(t_int) / n
            h_t = -np.sum(p_t * np.log2(p_t + 1e-12), where=~np.isclose(p_t, 0))

            t_shuffle = np.apply_along_axis(rng.permutation, 0, t)
            t_shuffle_int = t_shuffle.dot(1 << np.arange(dim - 1, -1, -1))
            p_shuffle = np.bincount(t_shuffle_int) / n
            h_t_shuffle = -np.sum(p_shuffle * np.log2(p_shuffle + 1e-12), where=~np.isclose(p_shuffle, 0))

            p_neurons = np.apply_along_axis(np.bincount, 1, t.T) / n
            h_neurons = -np.sum(p_neurons * np.log2(p_neurons + 1e-12), where=~np.isclose(p_neurons, 0))

            data['HT'].append(h_t)
            data['HShuffle'].append(h_t_shuffle)
            data['HNeurons'].append(h_neurons.sum())

    df_data = pd.DataFrame.from_dict(data, orient='columns')

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = axes.ravel()

    for layer_idx in range(4):
        ax = axes[layer_idx]

        df_layer = df_data[df_data['Layer'] == layer_idx]
        sns.lineplot(data=df_layer, x='Epoch', y='HT', label=r'normal', ls='-', ax=ax, legend=False)
        sns.lineplot(data=df_layer, x='Epoch', y='HShuffle', label=r'shuffled', ls=':', ax=ax, legend=False)
        sns.lineplot(data=df_layer, x='Epoch', y='HNeurons', label=r'$\sum_\ell H(T_\ell)$', ls='--', ax=ax, legend=False)

        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$H(T)$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    plt.subplots_adjust(right=0.8)
    plt.show(block=True)


def _compare_experiments(parser: argparse.ArgumentParser, args: argparse.Namespace):
    config = cli.configure.read_config(args.config)
    config = config.get('comparison', config)

    experiments = config.get('experiments', {})

    if type(experiments) != dict or len(experiments) == 0:
        parser.error(f'Please provide a dict of experiments')

    n_exp = len(experiments)

    if n_exp > 4:
        print(f'WARNING: {n_exp} experiments were provided, only 4 are supported and will be plotted')
        experiments = dict(itertools.islice(experiments.items(), 4))
    elif n_exp < 4:
        print(f'INFO: Program is designed for 4 plots, but only {n_exp} experiments were provided')

    n_exp = min(n_exp, 4)

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

    # --------------------
    # Plot information planes
    # --------------------

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(6.4 * 1.5, 4.8 * 1.5))
    axes = axes.ravel()

    norm = matplotlib.colors.Normalize(df_mis['Epoch'].min(), df_mis['Epoch'].max())
    cmap = plt.cm.ScalarMappable(norm=norm, cmap='flare_r')
    cmap.set_array([])

    epoch_spacing = np.unique(np.logspace(
        0, np.log10(3000),
        num=300, base=10,
        dtype=int
    ))
    epoch_spacing = np.append(epoch_spacing, [0, 2999])

    for idx, exp_name in enumerate(experiments.values()):
        ax: matplotlib.axes.Axes = axes[idx]

        df = df_mis[(df_mis['Experiment'] == exp_name) & (df_mis['Run'] == run_idx)]

        df = df[df['Epoch'].isin(epoch_spacing)]

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

    fig.colorbar(cmap, ax=axes[1::2])
    fig.subplots_adjust(right=0.8)
    
    # --------------------
    # Plot metrics
    # --------------------

    for plt_type, show_plt in [('Loss', plot_losses), ('Accuracy', plot_accuracy)]:
        if not show_plt:
            continue

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=plt_type == 'Accuracy', figsize=(6.4 * 1.5, 4.8 * 1.5))
        axes = axes.ravel()

        for idx, exp_name in enumerate(experiments.values()):
            ax: matplotlib.axes.Axes = axes[idx]

            df = df_metrics[(df_metrics['Experiment'] == exp_name) & (df_metrics['Run'] == run_idx)]

            if plt_type == 'Loss':
                sns.lineplot(df, x='Epoch', y='Train Loss', ax=ax, label='Train', alpha=0.75)
                sns.lineplot(df, x='Epoch', y='Val. Loss', ax=ax, label='Validation', ls=':', alpha=0.75)
            else:
                sns.lineplot(df, x='Epoch', y='Val. Acc', ax=ax)

            ax.set_title(exp_name)
        
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel(plt_type)

        fig.tight_layout()
    
    plt.show(block=True)


def main():
    parser = cli.parser.build_parser()
    args = parser.parse_args()

    match args.command:
        case 'evaluate':
            if args.eval_target == 'toy':
                _perform_evaluation(parser, args)
            else:
                _compare_entropy(parser, args)
        case 'mi' | 'mutual-information':
            _perform_mi_estimation(parser, args)
        case 'compare':
            _compare_experiments(parser, args)


if __name__ == '__main__':
    main()
