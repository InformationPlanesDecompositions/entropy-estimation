import argparse


def build_parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser(prog='EntropyEstimation')

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    # ====================
    # Entropy Evaluation
    # ====================
    eval_parser = subparsers.add_parser('evaluate', description='Evaluate the plug-in entropy estimator')
    eval_parser_group = eval_parser.add_subparsers(dest='eval_target', required=True)

    # --------------------
    # Toy Examples
    # --------------------
    ee_toy_parser = eval_parser_group.add_parser('toy', description='Evaluate the entropy estimator on toy examples')
    build_evaluation_parsers(ee_toy_parser)

    # --------------------
    # Entropy Evaluation on Trained Models
    # --------------------
    ee_model_parser = eval_parser_group.add_parser('model', description='Evaluate the entropy estimator on activations from a model')
    ee_model_parser.add_argument(
        '-d', '--data',
        type=str,
        help='Path to the data directory',
        required=True,
    )
    ee_model_parser.add_argument(
        '-r', '--run',
        type=int,
        help='Experiment run to use',
        default=0,
    )
    ee_model_parser.add_argument(
        '-s', '--save',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Save the generated plots',
        default=True,
    )

    # ====================
    # Mutual Information Estimation
    # ====================
    mi_parser = subparsers.add_parser('mi', aliases=['mutual-information'])
    build_mi_parser(mi_parser)

    # ====================
    # Compare experiments
    # ====================
    comparison_parser = subparsers.add_parser('compare', description='Compare experiments based on their MI estimates')
    comparison_parser_group = comparison_parser.add_subparsers(dest='comparison_target', required=True)

    comparison_parent_parser = argparse.ArgumentParser(add_help=False)
    comparison_parent_parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration .yaml file for comparison',
        required=True,
    )

    comparison_parent_parser.add_argument(
        '--dir-mi',
        type=str,
        help='Path to directory containing MI data subdirectories',
        default='./output/mi'
    )
    comparison_parent_parser.add_argument(
        '--dir-experiments',
        type=str,
        help='Path to directory containing the experimental data',
        required=True,
    )

    # --------------------
    # Information Plane
    # --------------------
    comparison_ip_parser = comparison_parser_group.add_parser(
        name='ip',
        aliases=['information-plane'],
        description='Compare the experiments on their information plane (optionally also in accuracy and loss)',
        parents=[comparison_parent_parser],
    )
    comparison_ip_parser.add_argument(
        '-r', '--run',
        type=int,
        help='Run number (positive integer)',
        default=0,
        required=False,
    )
    comparison_ip_parser.add_argument(
        '--accuracy-plot',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Plot validation accuracy over epochs',
        required=False,
    )
    comparison_ip_parser.add_argument(
        '--loss-plot',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Plot training and validation loss over epochs',
        required=False,
    )
    comparison_ip_parser.add_argument(
        '--plot-layout',
        type=int,
        nargs=2,
        help='Layout of the plots as (n_rows, n_cols)',
        required=False,
        default=(3, 3),
        metavar=('n_rows', 'n_cols')
    )

    # --------------------
    # Q1 (Compression)
    # --------------------
    comparison_q1_parser = comparison_parser_group.add_parser(
        name='q1',
        description='Compare the experiments on their compression factor',
    )
    comparison_q1_parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration .yaml file for comparison',
        required=True,
    )
    comparison_q1_parser.add_argument(
        '--dir-mi',
        type=str,
        help='Path to directory containing MI data subdirectories',
        default='./output/mi'
    )
    comparison_q1_parser.add_argument(
        '-r', '--reference_func',
        type=str,
        choices=['max', 'start'],
        required=False,
        default='start',
        help='How to determine the reference value for the compression factor',
    )
    comparison_q1_parser.add_argument(
        '-n', '--n-epochs',
        type=int,
        default=50,
        help='How many of the last epochs should be considered for aggregation',
    )
    comparison_q1_parser.add_argument(
        '--exp-as-cbar',
        type=bool,
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Should the experiment column be displayed as a cbar (alternatively: as legend)',
    )
    comparison_q1_parser.add_argument(
        '--legend-title',
        type=str,
        required=False,
        default='Experiment',
    )

    # --------------------
    # Q2 (Compression vs. Accuracy)
    # --------------------
    comparison_q2_parser = comparison_parser_group.add_parser(
        name='q2',
        description='Compare the experiments on their compression in one layer w.r.t. validation accuracy',
        parents=[comparison_parent_parser],
    )
    comparison_q2_parser.add_argument(
        '-l', '--layer-offset-idx',
        type=int,
        required=True,
        help='The layer idx, offset from the output layer, to show the compression on. i.e., <-1> means the layer *before* the output layer',
    )
    comparison_q2_parser.add_argument(
        '-n', '--n-epochs',
        type=int,
        default=50,
        help='How many of the last epochs should be considered for aggregation',
    )
    comparison_q2_parser.add_argument(
        '--agg-func',
        type=str,
        choices=['mean', 'median'],
        default='mean',
        required=False,
        help='Which aggregation function to use for plotting a point per experiment run'
    )
    comparison_q2_parser.add_argument(
        '--exp-as-cbar',
        type=bool,
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Should the experiment column be displayed as a cbar (alternatively: as legend)',
    )
    comparison_q2_parser.add_argument(
        '--legend-title',
        type=str,
        required=False,
        default='Experiment',
    )
    comparison_q2_parser.add_argument(
        '-s', '--save',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Save the generated plot',
        default=True,
    )
    comparison_q2_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path/name of the target file for the generated plot',
        required=False,
        default='output/mi/compression/tmp.pdf'
    )
    comparison_q2_parser.add_argument(
        '--show-plots',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Display the plot',
        default=True,
    )

    return root_parser


def build_mi_parser(
    mi_parser: argparse.ArgumentParser | None,
) -> argparse.ArgumentParser:
    def run_selection(value: str):
        if value is None or value.lower() == 'all':
            return None

        try:
            val = int(value)

            if val < 0:
                raise ValueError()

            return val
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Invalid argument {value}. Must be a positive integer or "best".'
            )

    if mi_parser is None:
        mi_parser = argparse.ArgumentParser()

    mi_parser.add_argument(
        '-d', '--data',
        type=str,
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
        type=run_selection,
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


def build_evaluation_parsers(
    eval_parser: argparse.ArgumentParser | None
) -> argparse.ArgumentParser:
    if eval_parser is None:
        eval_parser = argparse.ArgumentParser()

    eval_subparsers = eval_parser.add_subparsers(dest='evaluation_type', required=True)

    # ====================
    # Plug-In Estimate evaluation
    # ====================
    plugin_subparser = eval_subparsers.add_parser(
        'plug-in'
    )

    plugin_subparser.add_argument(
        '-M', '--n-experiments',
        type=int,
        help='Number of experiments to conduct',
        default=20,
    )
    plugin_subparser.add_argument(
        '-D', '--max_dimensions',
        type=int,
        default=20,
        help='The maximum number of dimension of the Bernoulli RV vectors',
    )
    plugin_subparser.add_argument(
        '-N', '--n_samples',
        type=int,
        help='The number of samples per experiment',
        required=True,
    )
    plugin_subparser.add_argument(
        '--use-existing',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Use already existing generated data for plotting. If not existing but set to True, the data will be generated regardless',
        default=True
    )
    plugin_subparser.add_argument(
        '-s', '--save',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Save the generated plots',
        default=True,
    )
    plugin_subparser.add_argument(
        '-o', '--output',
        type=str,
        help='Target directory for generated output',
        default='output/ee',
    )

    return eval_parser
