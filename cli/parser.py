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

    # ====================
    # Entropy Evaluation on Trained Models
    # ====================
    ee_model_parser = eval_parser_group.add_parser('model', description='Evaluate the entropy estimator on activations from a model')
    ee_model_parser.add_argument(
        '-d', '--data',
        type=str,
        help='Path to the data directory',
        required=True,
    )

    # ====================
    # Mutual Information Estimation
    # ====================
    mi_parser = subparsers.add_parser('mi', aliases=['mutual-information'])
    build_mi_parser(mi_parser)

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
        required=True,
    )
    plugin_subparser.add_argument(
        '-p', '--success-prob',
        type=float,
        help='Success probability per trial',
        required=True,
    )
    plugin_subparser.add_argument(
        '-D', '--dimensions',
        type=int,
        nargs=3,
        help='The dimensions of the Bernoulli RV vector. If none is provided, assume 1D. Otherwise,'
        + ' use arguments as (start, stop, n_steps)',
        metavar=('START', 'STOP', 'N_STEPS'),
        required=False,
    )
    plugin_subparser.add_argument(
        '-N', '--n_samples',
        type=int,
        nargs='+',
        help='The number of samples per experiment. If -D is provided, the first integer will be'
        + ' used for all experiments. Otherwise, use arguments as (start, stop, n_steps)',
        metavar=('N'),
        required=True,
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
        default='./output',
    )

    return eval_parser
