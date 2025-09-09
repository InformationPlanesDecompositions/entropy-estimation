import argparse


def build_parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser(prog='EntropyEstimation')

    subparsers = root_parser.add_subparsers(dest='command', required=True)

    # ====================
    # Evaluation
    # ====================
    eval_parser = subparsers.add_parser('evaluate')
    build_evaluation_parsers(eval_parser)

    return root_parser


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
