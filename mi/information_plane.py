from collections import defaultdict
from os import path

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import seaborn as sns

from tqdm import tqdm

from estimators import plug_in


# TODO: Make this work for MNIST/other variants
def generate_information_plane(
    activation_data: h5py.File | h5py.Group,
    data_file: h5py.File,
    save: bool = False,
    output_dir: str = '',
    postfix: str = '',
    block_plt: bool = True
):
    x_shape = data_file['data/X'].attrs.get('shape', (1, 1))
    n = x_shape[0]

    y = data_file['data/Y']
    y_shape = y.attrs.get('shape', (1, 1))
    y = np.reshape(y, y_shape)  # type: ignore

    if n != y_shape[0]:
        raise ValueError(f'Provided data for X and Y does not have same number of samples, was {n} =/= {y_shape[0]}')
    
    if y_shape[1] > 2:
        raise NotImplementedError('Currently only supported for bi-class targets')
    
    target = np.argmax(y, axis=1)
    
    p_y = np.bincount(target) / n

    data = defaultdict(list)

    for epoch_data in tqdm(activation_data.values(), ncols=100, ascii=True):
        epoch_data: h5py.Group
        epoch_idx = epoch_data.attrs['epoch_idx']
        
        for layer_data in epoch_data.values():
            layer_idx = layer_data.attrs['layer_idx']
            is_layer_packed = layer_data.attrs['is_packed']

            data['Epoch'].append(epoch_idx)
            data['Layer'].append(layer_idx)

            t = layer_data[:]

            if not is_layer_packed:
                t = t.reshape(-1, *layer_data.attrs['shape'])

                mi_x, mi_y = _estimate_output_layer_mi(t, target)

                data['MI_x'].append(mi_x)
                data['MI_y'].append(mi_y)

                continue

            t = np.unpackbits(t)
            t = t.reshape(-1, *layer_data.attrs['shape'])

            if t.shape[1] >= 64:
                raise ValueError(f'Activations for layer {layer_idx} have too high dimension, was {t.shape[1]}, only support up to 63 bits')
                
            # TODO: Move to estimators/plug_in.py
            t_int = _bit_array_to_integer(t)
            p_joint = plug_in.fast_joint_probabilitiy_estimation(t_int, target)
            p_latent = np.bincount(t_int) / n

            mi_x = -np.sum(p_latent * np.log2(p_latent + 1e-12), where=~np.isclose(p_latent, 0))

            data['MI_x'].append(mi_x)

            mi_y = 0

            for (latent, label), p_ty in p_joint.items():
                mi_y += p_ty * np.log2(p_ty / (p_latent[latent] * p_y[label]))

            data['MI_y'].append(mi_y)

    df_data = pd.DataFrame.from_dict(data, orient='columns')
    df_data.sort_values(by=['Epoch', 'Layer'], inplace=True)

    fig, ax = plt.subplots()

    norm = matplotlib.colors.Normalize(df_data['Epoch'].min(), df_data['Epoch'].max())
    cmap = plt.cm.ScalarMappable(norm=norm, cmap='flare_r')
    cmap.set_array([])

    sct_ax = sns.scatterplot(data=df_data, x='MI_x', y='MI_y', hue='Epoch', style='Layer', ax=ax, palette='flare_r', alpha=0.75)
    ax.set_xlabel(r'$I(X;T_\ell)$')
    ax.set_ylabel(r'$I(T_\ell;Y)$')

    if (lg := ax.get_legend()) is not None:
        lg.remove()
        
    ax.figure.colorbar(cmap, ax=sct_ax)

    fig.tight_layout()

    if save:
        plt.savefig(path.join(output_dir, f'information_plane{postfix}.png'))

    plt.show(block=block_plt)

    return df_data


def _estimate_output_layer_mi(latent: np.ndarray, target: np.ndarray) -> tuple[np.floating, ...]:
    sm = scipy.special.softmax(latent, axis=-1)
    sm_mean = np.mean(sm, axis=0)

    h_yhat = -sum(sm_mean * np.log2(sm_mean))
    h_yhat_given_x = np.mean(-np.sum(sm * np.log2(sm), axis=1))

    h_yhat_given_y = 0.0

    for cls in range(latent.shape[1]):
        p_bar = np.mean(sm[target == cls], axis=0)
        h_cls: np.floating = -np.sum(p_bar * np.log2(p_bar))

        h_yhat_given_y += np.mean(target == cls) * h_cls

    mi_x: np.floating = h_yhat - h_yhat_given_x
    mi_y: np.floating = h_yhat - h_yhat_given_y  # type: ignore

    return mi_x, mi_y


def _bit_array_to_integer(arr: np.ndarray) -> np.ndarray:
    if np.any(np.logical_or(arr < 0, arr > 1)):
        raise ValueError('Provided array is not bitwise valued')

    dim = arr.shape

    if len(dim) > 2:
        raise NotImplementedError('Conversion only implemented for arrays of shape (D,) or (N, D)')
    elif len(dim) == 0:
        return arr
    
    dim = dim[1] if len(dim) > 1 else dim[0]

    return arr.dot(1 << np.arange(dim - 1, -1, -1))
