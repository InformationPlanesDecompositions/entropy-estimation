from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors

import time

import argparse

import numpy as np

import h5py

import scipy.special

import cli.parser

import evaluation
import evaluation.plug_in


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


def _fast_probability_estimation(latent: np.ndarray, target: np.ndarray):
    n, *dim = latent.shape

    if len(dim) > 1:
        raise NotImplementedError(
            f'Probability estimation not implemented for multi-dim latent variables'
        )
    
    dim = dim[0]

    if dim >= 64:
        raise ValueError(
            f'Dimension of latent variable above limit of 64! (was {dim}). Fast'
            + 'probability counting is not supported.'
        )
    
    latent_as_integer = latent.dot(1 << np.arange(dim - 1, -1, -1))

    joint_count = Counter(zip(latent_as_integer, target))
    p_joint = {t: count / n for t, count in joint_count.items()}
    p_latent = np.bincount(latent_as_integer) / n

    return p_joint, p_latent


def _probability_counting(latent: np.ndarray, target: np.ndarray):
    n = len(latent)

    joint_count = defaultdict(int)
    latent_count = defaultdict(int)

    for t, y in zip(latent, target):
        joint_count[(tuple(t), y)] += 1
        latent_count[tuple(t)] += 1

    p_joint = {k: count / n for k, count in joint_count.items()}
    p_latent = {t: v / n for t, v in latent_count.items()}

    return p_joint, p_latent


def _perform_mi_estimation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    activation_file = h5py.File('../ma-bnn-training/output/szt-8000/activations.h5', 'r')
    data_file = h5py.File('../ma-bnn-training/output/szt-8000/data.h5', 'r')

    x_shape = data_file['data/X'].attrs.get('shape', (1,))
    n = x_shape[0]

    y = np.reshape(data_file['data/Y'], (-1, 2)) # type: ignore
    y_reduced = y[:, 1].reshape(-1, 1)
    target: np.ndarray = y_reduced.flatten().astype(np.int64)

    # TODO: Beautify, obviously
    u, c = np.unique(y[:, 0], return_counts=True, axis=0)
    p_y = {_u: _c / n for _u, _c in zip(u, c)}

    data = defaultdict(list)

    now = time.time()

    for _, epoch_data in activation_file.items():
        epoch_data: h5py.Group
        epoch_idx = epoch_data.attrs['epoch_idx']
        
        for _, layer_data in epoch_data.items():
            layer_idx = layer_data.attrs['layer_idx']
            is_layer_packed = layer_data.attrs['is_packed']

            data['Epoch'].append(epoch_idx)
            data['Layer'].append(layer_idx)

            t = layer_data[:]

            if not is_layer_packed:
                t = t.reshape(-1, *layer_data.attrs['shape'])

                sm = scipy.special.softmax(t, axis=-1)
                sm_mean = np.mean(sm, axis=0)

                h_yhat = -sum(sm_mean * np.log2(sm_mean))
                h_yhat_given_x = np.mean(-np.sum(sm * np.log2(sm), axis=1))

                h_yhat_given_y = 0

                for cls in range(2):
                    p_bar = np.mean(sm[target == cls], axis=0)
                    h_cls = -np.sum(p_bar * np.log2(p_bar))

                    h_yhat_given_y += (target == cls).mean() * h_cls

                data['MI_x'].append(h_yhat - h_yhat_given_x)
                data['MI_y'].append(h_yhat - h_yhat_given_y)
                continue

            t = np.unpackbits(t)
            t = t.reshape(-1, *layer_data.attrs['shape'])

            if t.shape[1] < 64:
                p_joint, p_latent = _fast_probability_estimation(t, target)

                mi_x = -np.sum(p_latent * np.log2(p_latent + 1e-12), where=~np.isclose(p_latent, 0))
            else:
                p_joint, p_latent = _probability_counting(t, target)

                mi_x = -np.sum(p * np.log2(p) for p in p_latent.values())  # type: ignore

            # Only applicable for SZT!! ==> replace with call to plug_in.estimate?
            data['MI_x'].append(mi_x)

            mi_y = 0

            for (latent, label), p_ty in p_joint.items():
                mi_y += p_ty * np.log2(p_ty / (p_latent[latent] * p_y[label]))

            data['MI_y'].append(mi_y)

    print(f'Done after {time.time() - now}...')

    df_data = pd.DataFrame.from_dict(data, orient='columns')
    df_data.to_csv('./data-8000.csv', sep=';', decimal=',')

    fig, ax = plt.subplots()

    norm = matplotlib.colors.Normalize(df_data['Epoch'].min(), df_data['Epoch'].max())
    cmap = plt.cm.ScalarMappable(norm=norm, cmap='flare_r')
    cmap.set_array([])

    sct_ax = sns.scatterplot(data=df_data, x='MI_x', y='MI_y', hue='Epoch', style='Layer', ax=ax, palette='flare_r')
    ax.set_xlabel(r'$I(X;T_\ell)$')
    ax.set_ylabel(r'$I(T_\ell;Y)$')

    ax.get_legend().remove()
    ax.figure.colorbar(cmap, ax=sct_ax)

    fig.tight_layout()
    plt.show(block=True)


def main():
    parser = cli.parser.build_parser()
    args = parser.parse_args()

    match args.command:
        case 'evaluate':
            _perform_evaluation(parser, args)
        case 'mi' | 'mutual-information':
            _perform_mi_estimation(parser, args)


if __name__ == '__main__':
    main()

    pass
