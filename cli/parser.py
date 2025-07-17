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

    eval_subparsers = eval_parser.add_subparsers(dest='distribution', required=True)

    eval_parent_parser = argparse.ArgumentParser(add_help=False)
    eval_parent_parser.add_argument(
        '-p', '--success-prob',
        type=float,
        help='Success probability per trial',
        required=True,
    )
    eval_parent_parser.add_argument(
        '--n-experiments',
        type=int,
        help='Number of experiments',
        required=True,
    )
    eval_parent_parser.add_argument(
        '--spacing-method',
        choices=['linear', 'logarithmic'],
        help='How to space the varying number of trials',
        required=True
    )
    eval_parent_parser.add_argument(
        '--n-min',
        type=int,
        help='Minimum number of trials',
        required=True
    )
    eval_parent_parser.add_argument(
        '--n-max',
        type=int,
        help='Maximum number of trials',
        required=True
    )

    # ====================
    # Binomial
    # ====================
    binomial_parser = eval_subparsers.add_parser(
        'binomial',
        parents=[eval_parent_parser]
    )
    binomial_parser.add_argument(
        '-s', '--n-samples',
        type=int,
        help='Number of samples per trial',
        required=True
    )

    # ====================
    # Bernoulli
    # ====================
    eval_subparsers.add_parser(
        'bernoulli',
        parents=[eval_parent_parser],
    )

    return eval_parser
