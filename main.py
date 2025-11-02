import os
from os import path

from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

import numpy as np
import pandas as pd

import h5py

from tqdm import tqdm

import cli.parser

import evaluation
import evaluation.plug_in

from mi import information_plane


def _perform_evaluation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.evaluation_type == 'plug-in':
        p: float = args.success_prob
        n_experiments: int = args.n_experiments

        n_sample_steps: list[int] = args.n_samples

        if (dimensions := args.dimensions) is None:
            if len(n_sample_steps) < 3:
                parser.error(f'Please provide 3 parameters für argument -N if -D is not set')

            n_start, n_stop, n_steps = n_sample_steps[:3]
            n_sample_space = np.linspace(n_start, n_stop, n_steps).astype(np.int64)

            evaluation.plug_in.onedimensional_bernoulli(n_experiments, p, n_sample_space)

            return
        
        n_samples = n_sample_steps[0]
        d_steps = np.linspace(*dimensions)

        evaluation.plug_in.multidimensional_bernoulli(
            n_experiments, p, n_samples,
            d_steps=d_steps,
            save=args.save, output_dir=args.output
        )


def _perform_mi_estimation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    data_dir = args.data

    if not path.isdir(data_dir):
        parser.error(f'Please provide an existing directory, did not find {data_dir}')

    activation_path = path.join(data_dir, 'activations.h5')
    data_path = path.join(data_dir, 'data.h5')

    dir_name = path.basename(path.dirname(activation_path))

    if not path.isfile(activation_path):
        parser.error(f'No <activations.h5> found in given directory')

    if not path.isfile(data_path):
        parser.error(f'No <data.h5> found in given directory')

    activation_file = h5py.File(activation_path, 'r')
    data_file = h5py.File(data_path, 'r')

    output_dir = f'./output/mi/{dir_name}'

    if args.save:
        os.makedirs(output_dir, exist_ok=True)

    if not activation_file.attrs['has_top_group']:
        df_data = information_plane.generate_information_plane(
            activation_file,
            data_file,
            save=args.save,
            output_dir=output_dir
        )

        df_data.set_index('Epoch', drop=True, inplace=True)
        df_data.to_csv(path.join(output_dir, 'mi_data.csv'), sep=';', decimal=',')

        return
    
    mode = None

    run_selection: None | int = args.run

    for run_key, run_data in activation_file.items():
        run_idx = run_data.attrs['group_idx']

        if run_selection is not None and run_idx != run_selection:
            continue

        if mode is None:
            mode = 'w'
        else:
            mode = 'a'

        df_data = information_plane.generate_information_plane(
            run_data,
            data_file,
            save=args.save,
            output_dir=output_dir,
            postfix=f'_{run_key}',
            block_plt=run_idx == (len(activation_file) - 1),
        )

        if not args.save:
            continue

        df_data.set_index('Epoch', drop=True, inplace=True)
        df_data['Run'] = run_data.attrs['group_idx']
        df_data.to_csv(
            path.join(output_dir, 'mi_data.csv'),
            sep=';',
            decimal=',',
            mode=mode,
            header=mode == 'w'
        )

        if run_selection is not None:
            break


# TODO: Move to package evaluation, either into plug_in.py or model.py
# TODO: Might not work if activation file is grouped into runs
def _compare_entropy(parser: argparse.ArgumentParser, args: argparse.Namespace):
    data_dir = args.data

    if not path.isdir(data_dir):
        parser.error(f'Please provide an existing directory, did not find {data_dir}')

    activation_path = path.join(data_dir, 'activations.h5')

    if not (path.exists(activation_path) and path.isfile(activation_path)):
        parser.error(f'No <activations.h5> found in given directory')

    activation_file = h5py.File(activation_path, 'r')

    data = defaultdict(list)
    rng = np.random.default_rng(2620)

    # type: ignore
    for epoch_data in tqdm(activation_file.values(), ncols=100, ascii=True):
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


if __name__ == '__main__':
    main()
