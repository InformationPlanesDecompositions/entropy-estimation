from collections import defaultdict
import pathlib
import os
from os import path

import h5py

import numpy as np

import pandas as pd

from tqdm import tqdm


# Taken from "Geometric and Information Compression of Representations in Deep Learning" (Adilova et al., 2026)
# Translated to numpy from https://github.com/link-er/information_geometric_compression/blob/master/gaus_dropout/NC_regularizer.py
def compute_neural_collapse(latents: np.ndarray, labels: np.ndarray):
    n_classes: int = labels.max() + 1

    counts_classes = np.bincount(labels, minlength=n_classes)

    # Compute per-class means
    sums = np.zeros((n_classes, latents.shape[1]))
    np.add.at(sums, labels, latents)
    means = sums / counts_classes.reshape(-1, 1).clip(min=1)

    # Compute per-class variances
    diffs = latents - means[labels]
    sq_norms = np.pow(diffs, 2).sum(axis=1)

    var_sums = np.zeros(n_classes)
    np.add.at(var_sums, labels, sq_norms)
    variances = var_sums / counts_classes.clip(min=1)

    # Compute pair-wise distance
    diff = means[:, None, :] - means[None, :, :]
    sq_distance = np.pow(diff, 2).sum(axis=2)

    # Avoid divide-by-zero and self-pairs
    mask = sq_distance < 1e-12
    sq_distance[mask] = np.inf

    # Compute NC
    res = (variances[:, None] + variances[None, :]) / (2 * sq_distance)

    nc_score = res[~mask].mean()

    return nc_score, means, variances


def compute_nc_from_file(data_dir: pathlib.Path, n_epochs: int):
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Please provide an existing directory, did not find {data_dir}')

    dir_name = path.join(data_dir.parent.name, data_dir.name)

    output_dir = path.join('output/nc/', dir_name)

    os.makedirs(output_dir, exist_ok=True)

    activation_path = data_dir.joinpath('activations.h5')
    data_path = data_dir.joinpath('data.h5')

    if not activation_path.is_file():
        raise FileNotFoundError(f'No <activations.h5> found in given directory')

    if not data_path.is_file():
        raise FileNotFoundError(f'No <data.h5> found in given directory')

    activation_file = h5py.File(activation_path, 'r')
    data_file = h5py.File(data_path, 'r')

    labels = data_file['data/Y']
    labels_shape = labels.attrs.get('shape', (1, 1))
    labels = np.reshape(labels, labels_shape)  # type: ignore

    if len(labels_shape) > 1:
        labels = np.argmax(labels, axis=1)

    if not activation_file.attrs.get('has_top_group', False):
        activation_iter = enumerate([activation_file])
    else:
        activation_iter = activation_file.items()

    data = defaultdict(list)

    for _, run_data in activation_iter:
        run_idx = run_data.attrs.get('group_idx', 0)

        max_epoch_idx = np.max([int(key.split('_')[-1]) for key in run_data.keys()])  # A bit hack-y

        for epoch_data in tqdm(run_data.values(), ncols=100, ascii=True):
            epoch_idx = epoch_data.attrs['epoch_idx']

            if epoch_idx < max_epoch_idx - n_epochs + 1:
                continue

            for layer_data in epoch_data.values():
                is_layer_packed = layer_data.attrs['is_packed']

                if not is_layer_packed:
                    # We skip output layers
                    continue

                layer_idx = layer_data.attrs['layer_idx']

                t = np.unpackbits(layer_data[:])
                t = t.reshape(-1, *layer_data.attrs['shape'])

                score, *_ = compute_neural_collapse(t, labels)

                data['Run'].append(run_idx)
                data['Epoch'].append(epoch_idx)
                data['Layer'].append(layer_idx)
                data['NC'].append(score)

    df_data = pd.DataFrame.from_dict(data, orient='columns')
    df_data.sort_values(by=['Run', 'Epoch', 'Layer'], inplace=True)
    df_data.set_index(['Run', 'Epoch', 'Layer'], drop=True, inplace=True)

    df_data.to_csv(f'{output_dir}/scores.csv', decimal=',', sep=';')

    return df_data
