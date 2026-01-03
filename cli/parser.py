import argparse


def build_parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser(prog='EntropyEstimation')

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    # ====================
    # Entropy Evaluation
    # ====================
    eval_parser = subparsers.add_parser('evaluate')
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

    # ====================
    # Mutual Information Estimation
    # ====================
    mi_parser = subparsers.add_parser('mi', aliases=['mutual-information'])
    build_mi_parser(mi_parser)

    # ====================
    # Compare experiments
    # ====================
    comparison_parser = subparsers.add_parser('compare', description='Compare experiments based on their information plane, losses, and accuracy')
    comparison_parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration .yaml file for comparison',
        required=True,
    )

    comparison_parser.add_argument(
        '--dir-mi',
        type=str,
        help='Path to directory containing MI data subdirectories',
        default='./output/mi'
    )
    comparison_parser.add_argument(
        '--dir-experiments',
        type=str,
        help='Path to directory containing the experimental data',
        required=True,
    )

    comparison_parser.add_argument(
        '-r', '--run',
        type=int,
        help='Run number (positive integer)',
        default=0,
        required=False,
    )
    
    comparison_parser.add_argument(
        '--accuracy-plot',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Plot validation accuracy over epochs',
        required=False,
    )
    comparison_parser.add_argument(
        '--loss-plot',
        type=bool,
        action=argparse.BooleanOptionalAction,
        help='Plot training and validation loss over epochs',
        required=False,
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
