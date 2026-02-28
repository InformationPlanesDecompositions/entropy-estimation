import argparse
import itertools

import os
from os import path
import pathlib

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
import evaluation.information_planes
import evaluation.compression

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
    data_dir: pathlib.Path = args.data

    if not data_dir.is_dir():
        parser.error(f'Please provide an existing directory, did not find {data_dir}')
    
    dir_name = data_dir.name

    output_dir = f'output/ee/'

    if args.save:
        os.makedirs(output_dir, exist_ok=True)

    activation_path = path.join(data_dir, 'activations.h5')

    if not (path.exists(activation_path) and path.isfile(activation_path)):
        parser.error(f'No <activations.h5> found in given directory')

    activation_file = h5py.File(activation_path, 'r')

    run_idx = int(args.run) if args.run is None else 0

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
    data_dir: pathlib.Path = args.data

    if data_dir.is_dir():
        parser.error(f'Please provide an existing directory, did not find {data_dir}')

    dir_name = path.join(data_dir.parent.name, data_dir.name)

    output_dir = path.join('output/mi/', dir_name)

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
        activation_path = data_dir.joinpath('activations.h5')
        data_path = data_dir.joinpath('data.h5')

        if not activation_path.is_file():
            parser.error(f'No <activations.h5> found in given directory')

        if not data_path.is_file():
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

    dir_mi: pathlib.Path = args.dir_mi
    dir_exp: pathlib.Path | None = args.dir_experiments

    if not dir_mi.is_dir():
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    run_idx = config.get('run_idx', 0) if args.run is None else args.run
    plot_accuracy: bool = config.get('accuracy_plot', True) if args.accuracy_plot is None else args.accuracy_plot
    plot_losses: bool= config.get('loss_plot', False) if args.loss_plot is None else args.loss_plot

    files = ['mi_data.csv']
    dirs = [dir_mi]

    if plot_losses or plot_accuracy:
        if dir_exp is None:
            parser.error('No data directory for experiments provided')

        if not dir_exp.is_dir():
            parser.error(f'Invalid data directory for experiments provided, could not find {dir_exp}')

        files.append('metrics.csv')
        dirs.append(dir_exp)

    [df_mis, *df_metrics] = _concat_experiment_files(
        experiments,
        files=files,
        dirs=dirs,  # type: ignore
        is_key_path=True
    )

    if plot_accuracy or plot_losses:
        df_metrics, *_ = df_metrics
    else:
        df_metrics = None

    figsize = (5 * w_ratio, 5 * h_ratio)

    evaluation.information_planes.compare_information_planes(
        df_mis, df_metrics,
        experiment_names=list(experiments.values()),
        run_idx=run_idx,
        n_rows=n_rows, n_cols=n_cols,
        figsize=figsize,
        plot_losses=plot_losses, plot_accuracy=plot_accuracy
    )


def _compare_compression(parser: argparse.ArgumentParser, args: argparse.Namespace):
    def err_low(col: pd.Series):
        err = col.mean() - col.min()

        return err if not np.isclose(err, 0) else 0

    def err_high(col: pd.Series):
        err = col.max() - col.mean()
    
        return err if not np.isclose(err, 0) else 0

    config = cli.configure.read_config(args.config)
    config: dict = config.get('comparison', config)

    experiments = config.get('experiments', {})

    if type(experiments) != dict or len(experiments) == 0:
        parser.error(f'Please provide a dict of experiments')

    n_exp = len(experiments)

    dir_exp: pathlib.Path = args.dir_experiments
    dir_mi: pathlib.Path = args.dir_mi

    if not dir_exp.is_dir():
        parser.error(f'Invalid data directory for experiments provided, could not find {dir_exp}')

    if not dir_mi.is_dir():
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    layer_revoffset_idx = int(args.layer_offset_idx)
    if layer_revoffset_idx > 0:
        layer_revoffset_idx *= -1

    n_epochs = int(args.n_epochs)
    agg_func: str = str(args.agg_func)

    as_cbar: bool = bool(args.as_cbar)
    is_discrete_cbar: bool = as_cbar and bool(args.is_discrete_cbar)
    cbar_minimum: int = int(args.discrete_cbar_minimum)

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

    if bool(args.save):
        output: pathlib.Path = args.output

        # TODO: add default file name
        if output.suffix == '':
            output = output.with_name('tmp.pdf')
        elif output.suffix != '.pdf':
            output = output.with_suffix('.pdf')

        os.makedirs(output.parent, exist_ok=True)
        plt.savefig(output, dpi=300, format='pdf')

    if bool(args.show_plots):
        plt.show(block=True)

    
def _quantify_compression(parser: argparse.ArgumentParser, args: argparse.Namespace):
    config = cli.configure.read_config(args.config)
    config: dict = config.get('comparison', config)

    compression_config: dict = config.get('compression', {})

    output_dir: pathlib.Path = args.output
    save: bool = args.save
    show_plt: bool = args.show_plots

    dataset_order: list[str] = compression_config.get('dataset_order', [])

    if args.use_existing:
        df_rho = pd.read_csv('output/tmp/rho_data.csv', decimal=',', sep=';', index_col=0)

        evaluation.compression.plot_compression(
            df_rho,
            dataset_order=dataset_order,
            save=save, output_dir=output_dir,
            show_plt=show_plt
        )

        return
    
    experiment_groups: dict[str, dict[str, list[str]]] = config.get('experiment_groups', {})

    included_layer_indices: dict[str, list[int]] = compression_config.get('include_layer_indices', {})
    included_groups: list[str] = compression_config.get('groups', [])

    n_epochs: int = args.n_epochs

    dir_mi: pathlib.Path = args.dir_mi

    if not dir_mi.is_dir():
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

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

    df, *_ = _concat_experiment_files(
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

    s_n_layers_per_group = df.groupby(by=['Dataset', 'Group'])['Layer'].nunique()

    ref_x = df.groupby(by=['Dataset', 'Group', 'Experiment', 'Run', 'Layer'])['MI_x'].max()
    end_x = df[df['Epoch'].ge(df['Epoch'].max() - n_epochs + 1)].groupby(by=['Dataset', 'Group', 'Experiment', 'Run', 'Layer']).mean()['MI_x']

    s_rho = (ref_x - end_x) / ref_x
    df_rho = s_rho.reset_index(name='Rho')

    df_rho['WD'] = df_rho['Experiment'].str.extract(r'-wd-(\d+(?:\.\d+)?e[-+]\d)')

    df_rho.dropna(subset=['WD'], inplace=True)
    df_rho['WD'] = df_rho['WD'].astype(float)

    df_rho['#X-Axis'] = df_rho[['Dataset', 'Group', 'Layer']].apply(
        lambda row: f'{row['Group']}\nLayer {row['Layer']}'
            if s_n_layers_per_group[(row['Dataset'], row['Group'])] > 1
            else row['Group'],
        axis=1
    )

    if save:
        df_rho.to_csv(output_dir.joinpath('compression_factor_data.csv'), decimal=',', sep=';')
    
    evaluation.compression.plot_compression(
        df_rho,
        dataset_order=dataset_order,
        save=save, output_dir=output_dir,
        show_plt=show_plt
    )


def _concat_experiment_files(
    experiments: dict,
    files: list[str],
    dirs: list[str | pathlib.Path],
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


def _compute_compression_rank_correlation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    config = cli.configure.read_config(args.config)

    config = config.get('comparison', config)

    dir_mi: pathlib.Path = args.dir_mi
    dir_exp: pathlib.Path = args.dir_experiments

    if not dir_mi.is_dir():
        parser.error(f'Invalid data directory for MI estimates provided, could not find {dir_mi}')

    if not dir_exp.is_dir():
        parser.error(f'Invalid data directory for experiments provided, could not find {dir_exp}')

    to_latex = bool(args.to_latex)
    output_dir: pathlib.Path = args.output
    n_epochs = int(args.n_epochs)

    experiment_groups: dict[str, dict[str, list[str]]] = config.get('experiment_groups', {})
    experiments = {exp: exp for groups in experiment_groups.values() for group in groups.values() for exp in group}

    df_groupings = pd.DataFrame(
        [
            (ds_name, grp_name, exp) for ds_name, ds in experiment_groups.items()
            for grp_name, grp in ds.items()
            for exp in grp
        ],
        columns=['Dataset', 'Group', 'Experiment'],
    )

    df_metrics, df_mis, *_ = _concat_experiment_files(
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

    evaluation.compression.compute_compression_rank_correlation(
        df,
        to_latex=to_latex,
        output_dir=output_dir
    )


def _get_fn_from_args(args: argparse.Namespace):
    cmd = str(args.command).lower()
    task = str(getattr(args, 'task', '')).lower()

    match cmd, task:
        case 'evaluate', 'plug-in':
            return _perform_evaluation
        case 'evaluate', _:
            return _evaluate_subadditivity
        case ('mi', _) | ('q1', 'mi'):
            return _perform_mi_estimation
        case 'q1', 'ips':
            return _compare_experiments
        case 'q1', 'compression':
            return _quantify_compression
        case 'q2', 'compare':
            return _compare_compression
        case 'q2', 'correlation':
            return _compute_compression_rank_correlation
        
    return None

def main():
    parser = cli.parser.build_parser()
    args = parser.parse_args()

    fn = _get_fn_from_args(args)

    if fn is None:
        parser.error(f'Invalid command and task provided')

    fn(parser, args)


if __name__ == '__main__':
    main()
