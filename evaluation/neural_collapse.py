from collections import defaultdict
import pathlib
import os
from os import path

import h5py
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from utility.data import concat_experiment_files


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


def compute_nc_rank_correlations(
    experiment_groups: dict[str, dict[str, list[str]]],
    dir_nc: pathlib.Path,
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

    df_metrics, df_mis, df_ncs, *_ = concat_experiment_files(
        experiments,
        files=['metrics.csv', 'mi_data.csv', 'scores.csv'],
        dirs=[dir_exp, dir_mi, dir_nc],
        is_key_path=True
    )

    df = pd.merge(df_ncs, df_mis, on=['Experiment', 'Run', 'Epoch', 'Layer'], how='left')
    df = pd.merge(df, df_metrics, on=['Experiment', 'Run', 'Epoch'], how='left')

    max_layer_indices = df_mis.groupby(by='Experiment')['Layer'].max()
    df['Layer'] = df.apply(lambda row: row['Layer'] - max_layer_indices[row['Experiment']], axis=1).astype(int)
    df = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)]
    df.drop(index=df[df['Layer'] == 0].index, inplace=True)

    df = df_groupings.merge(df, on='Experiment', how='right')
    
    df_grouped = df.groupby(by=['Dataset', 'Group', 'Experiment', 'Run', 'Layer'])
    df_agg = df_grouped.aggregate({
        'NC': ['mean'],
        'MI_x': ['mean'],
        'Val. Acc': ['mean']
    }).reset_index()

    data: defaultdict[str, list] = defaultdict(list)

    for (dataset, group, layer_idx), df_group in df_agg.groupby(by=['Dataset', 'Group', 'Layer']):
        data['Dataset'].append(dataset)
        data['Group'].append(group)
        data['Layer'].append(layer_idx)

        r, p = scipy.stats.spearmanr(df_group[[('NC', 'mean'), ('Val. Acc', 'mean')]])

        data['R NC-Acc'].append(r)
        data['p NC-Acc'].append(p)

        r, p = scipy.stats.spearmanr(df_group[[('NC', 'mean'), ('MI_x', 'mean')]])
        
        data['R NC-MI'].append(r)
        data['p NC-MI'].append(p)

    
    df_result = pd.DataFrame(data)
    df_result.to_csv(path.join(output_dir, 'rank_corr_nc_data.csv'), sep=';', decimal=',')

    if not to_latex:
        return
    
    lines = []

    lines.append('Dataset & Experiment Group & Layer & $r_s$ NC-Acc & $p$ NC-Acc & $r_s$ NC-MI & $p$ NC-MI \\\\\n')
    lines.append('\\midrule\n')

    last_ds = ''

    for (_, ds, grp, layer_idx, r_acc, p_acc, r_mi, p_mi) in df_result.itertuples():
        if last_ds != '' and ds != last_ds:
            lines.append('\\midrule\n')
        
        last_ds = ds

        lines.append(
            f'{ds} & {grp} & {layer_idx} & $\\numprint{{{r_acc}}}$ & ${f"\\numprint{{{p_acc}}}" if p_acc >= 0.0005 else "< 0.001"}$ & $\\numprint{{{r_mi}}}$ & ${f"\\numprint{{{p_mi}}}" if p_mi >= 0.0005 else "< 0.001"}$ \\\\\n'
        )

    with open(path.join(output_dir, 'rank_corr_nc_table.tex'), 'w') as f:
        f.writelines(lines)
