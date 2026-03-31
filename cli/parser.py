import argparse
import datetime
import pathlib


def _run_selection(value: str):
    if value is None or value.lower() == 'all':
        return None

    try:
        val = int(value)

        if val < 0:
            raise ValueError()

        return val
    except ValueError:
        raise argparse.ArgumentError(
            argument=None,
            message=f'Invalid argument {value}. Must be a positive integer or "best".',
        )
        

def _add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_experiment: bool = False,
    is_experiment_required: bool = False,
    include_mi: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration .yaml file for comparison',
        required=True,
    )

    if include_mi:
        parser.add_argument(
            '--dir-mi',
            type=pathlib.Path,
            help='Path to directory containing MI data subdirectories',
            default='output/mi',
        )

    if include_experiment:
        parser.add_argument(
            '--dir-experiments',
            type=pathlib.Path,
            help='Path to directory containing the experimental data',
            required=is_experiment_required,
        )

    return parser


def _add_aggregation_arguments(
    parser: argparse.ArgumentParser,
    *,
    add_legend: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument(
        '-n', '--n-epochs',
        type=int,
        default=50,
        help='How many of the last epochs should be considered for aggregation',
    )
    parser.add_argument(
        '-f', '--agg-func',
        type=str,
        choices=['mean', 'median'],
        default='mean',
        required=False,
        help='Which aggregation function to use for plotting a point per experiment run'
    )

    if not add_legend:
        return parser
    
    parser.add_argument(
        '--as-cbar',
        type=bool,
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Should the experiment column be displayed as a colour bar (otherwise as normal legend)',
    )
    parser.add_argument(
        '--is-discrete-cbar',
        type=bool,
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='If set and <as-cbar> is True, displays the experiment column as a discrete colour bar'
    )
    parser.add_argument(
        '--discrete-cbar-minimum',
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        '--legend-title',
        type=str,
        required=False,
        default='Experiment',
    )

    return parser


def _add_save_arguments(
    parser: argparse.ArgumentParser,
    *,
    is_output_file: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument(
        '-s', '--save',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    o = 'output/tmp/'

    if is_output_file:
        o += f'tmp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf'

    parser.add_argument(
        '-o', '--output',
        type=pathlib.Path,
        default=o,
        help=f'Target output {"file" if is_output_file else "directory"}'
    )

    return parser
        

def build_parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser(prog='Information Plane Analysis')

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    # ============================================================
    # Entropy Evaluation
    # ============================================================
    eval_parser = subparsers.add_parser('evaluate', description='Evaluate the plug-in entropy estimator')
    eval_parser_group = eval_parser.add_subparsers(dest='task', required=True)

    # ------------------------------------------------------------
    # Toy Examples
    # ------------------------------------------------------------
    ee_plugin_parser = eval_parser_group.add_parser('plug-in', description='Evaluate the entropy estimator on toy examples')
    ee_plugin_parser.add_argument(
        '-M', '--n-experiments',
        type=int,
        help='Number of experiments to conduct',
        default=20,
    )
    ee_plugin_parser.add_argument(
        '-D', '--max_dimensions',
        type=int,
        default=20,
        help='The maximum number of dimension of the Bernoulli RV vectors',
    )
    ee_plugin_parser.add_argument(
        '-N', '--n_samples',
        type=int,
        help='The number of samples per experiment',
        required=True,
    )
    ee_plugin_parser.add_argument(
        '--use-existing',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Use already existing generated data for plotting. If not existing but set to True, the data will be generated regardless',
        default=True
    )
    ee_plugin_parser = _add_save_arguments(
        ee_plugin_parser,
        is_output_file=False,
    )

    # ------------------------------------------------------------
    # Entropy Evaluation on Trained Models
    # ------------------------------------------------------------
    ee_model_parser = eval_parser_group.add_parser('model', description='Evaluate the entropy estimator on activations from a model')
    ee_model_parser.add_argument(
        '-d', '--data',
        type=str,
        help='Path to the data directory',
        required=True,
    )
    ee_model_parser.add_argument(
        '-r', '--run',
        type=_run_selection,
        help='Experiment run to use',
        default=0,
    )
    ee_model_parser = _add_save_arguments(
        ee_model_parser,
        is_output_file=False
    )
    ee_model_parser.set_defaults(
        output='output/ee',
        file_name='activations.h5',
    )

    # ------------------------------------------------------------
    # Evaluation of the data-to-dimensionality regime
    # ------------------------------------------------------------
    ee_regime_parser = eval_parser_group.add_parser('regime', description='Plot the data-to-dimensionality regime as an approximate curve')
    ee_regime_parser.add_argument(
        '--min-dim',
        type=int,
        help='Minimum number of dimensions',
        default=1,
    )
    ee_regime_parser.add_argument(
        '--max-dim',
        type=int,
        help='Maximum number of dimensions',
        default=20,
    )
    ee_regime_parser = _add_save_arguments(
        ee_regime_parser,
        is_output_file=False,
    )

    # ============================================================
    # Mutual Information Estimation
    # ============================================================
    mi_parser = subparsers.add_parser('mi')
    build_mi_parser(mi_parser)

    # ============================================================
    # RQ1 (Compression Phase)
    # ============================================================
    q1_parser = subparsers.add_parser('q1', description='Evaluate the experiments regarding RQ1')
    q1_parser_group = q1_parser.add_subparsers(dest='task', required=True)

    # ------------------------------------------------------------
    # Single Information Plane (alias for simply <mi>)
    # ------------------------------------------------------------
    q1_parser_group.add_parser('mi', parents=[mi_parser], add_help=False)

    # ------------------------------------------------------------
    # Compare multiple Information Planes (optionally accuracy and loss curves, too)
    # ------------------------------------------------------------
    q1_comparison_parser = q1_parser_group.add_parser(
        'ips',
        description='Compare the experiments on their information plane (optionally also in accuracy and loss)',
    )
    q1_comparison_parser = _add_config_arguments(
        q1_comparison_parser,
        include_mi=True,
        include_experiment=True,
        is_experiment_required=False,
    )
    q1_comparison_parser.add_argument(
        '-r', '--run',
        type=_run_selection,
        help='Run number (positive integer)',
        default=0,
        required=False,
    )
    q1_comparison_parser.add_argument(
        '--accuracy-plot',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Plot validation accuracy over epochs',
        required=False,
    )
    q1_comparison_parser.add_argument(
        '--loss-plot',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Plot training and validation loss over epochs',
        required=False,
    )
    q1_comparison_parser.add_argument(
        '--plot-layout',
        type=int,
        nargs=2,
        help='Layout of the plots as (n_rows, n_cols)',
        required=False,
        default=(3, 3),
        metavar=('n_rows', 'n_cols')
    )
    q1_comparison_parser.add_argument(
        '--name-as-wd',
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, the axes' titles will be labelled as \'$\\lambda = <exp>$\'",
    )
    q1_comparison_parser = _add_save_arguments(
        q1_comparison_parser,
        is_output_file=True,
    )
    q1_comparison_parser.add_argument(
        '--show-plots',
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )

    # ------------------------------------------------------------
    # Compare compression factors of experiments
    # ------------------------------------------------------------
    q1_compression_parser = q1_parser_group.add_parser(
        'compression',
        description='Compare the experiments on their computed compression factor',
    )
    q1_compression_parser = _add_config_arguments(
        q1_compression_parser,
        include_mi=True,
        include_experiment=False,
    )
    q1_compression_parser.add_argument(
        '--use-existing',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    q1_compression_parser.add_argument(
        '-n', '--n-epochs',
        type=int,
        default=50,
        help='How many of the last epochs should be considered for aggregation',
    )
    q1_compression_parser.add_argument(
        '--show-plots',
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    q1_compression_parser = _add_save_arguments(
        q1_compression_parser,
        is_output_file=False,
    )

    # ============================================================
    # RQ2 (Relation compression <-> generalisation)
    # ============================================================
    q2_parser = subparsers.add_parser('q2', description='Evaluate the experiments regarding RQ2')
    q2_parser_group = q2_parser.add_subparsers(dest='task', required=True)

    # ------------------------------------------------------------
    # Compare I(X;T_\ell) and val. accuracy for multiple experiments
    # ------------------------------------------------------------
    q2_comparison_parser = q2_parser_group.add_parser(
        'compare',
        description='Compare the experiments on their compression in one layer w.r.t. the achieved validation accuracy',
    )
    q2_comparison_parser = _add_config_arguments(
        q2_comparison_parser,
        include_mi=True,
        include_experiment=True,
        is_experiment_required=True,
    )
    q2_comparison_parser.add_argument(
        '-l', '--layer-offset-idx',
        type=int,
        required=True,
        help='The layer idx, offset from the output layer, to show the compression on. i.e., <-1> means the layer *before* the output layer',
    )
    q2_comparison_parser = _add_aggregation_arguments(
        q2_comparison_parser,
        add_legend=True,
    )
    q2_comparison_parser = _add_save_arguments(
        q2_comparison_parser,
        is_output_file=True,
    )
    q2_comparison_parser.add_argument(
        '--show-plots',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Display the plot',
        default=True,
    )

    # ------------------------------------------------------------
    # Compute Spearman's rank correlation between I(X;T_\ell) and Acc.
    # ------------------------------------------------------------
    q2_correlation_parser = q2_parser_group.add_parser(
        'correlation',
        description='Compute the Spearman rank correlation coefficients between the MI and validation accuracy',
    )
    q2_correlation_parser = _add_config_arguments(
        q2_correlation_parser,
        include_mi=True,
        include_experiment=True,
        is_experiment_required=True,
    )
    q2_correlation_parser = _add_aggregation_arguments(
        q2_correlation_parser,
        add_legend=False,
    )
    q2_correlation_parser.add_argument(
        '--to-latex',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Additionally output the correlation data as table body',
        type=bool,
    )
    q2_correlation_parser = _add_save_arguments(
        q2_correlation_parser,
        is_output_file=False,
    )

    return root_parser


def build_mi_parser(
    mi_parser: argparse.ArgumentParser | None,
) -> argparse.ArgumentParser:
    if mi_parser is None:
        mi_parser = argparse.ArgumentParser()

    mi_parser.add_argument(
        '-d', '--data',
        type=pathlib.Path,
        help='Path to the data directory',
        required=True,
    )
    mi_parser.add_argument(
        '-s', '--save',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Save the generated plots',
        default=True,
    )
    mi_parser.add_argument(
        '-r', '--run',
        type=_run_selection,
        help='Run number (positive integer), None or "all" for all runs.',
        default='all',
        required=False,
    )
    mi_parser.add_argument(
        '--show-plots',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Display the plots',
        default=True,
    )
    mi_parser.add_argument(
        '--compute-mi',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='(Re-)compute the mutual information of the experiment',
        default=True,
    )
    mi_parser.add_argument(
        '--plot-as-pdf',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Export the generated plot as .pdf file. Otherwise, .png is used',
        default=False,
    )

    return mi_parser

