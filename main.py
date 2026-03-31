import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import cli.parser


def _get_fn_from_args(args: argparse.Namespace):
    from cli import handlers

    cmd = str(args.command).lower()
    task = str(getattr(args, 'task', '')).lower()

    match cmd, task:
        case 'evaluate', 'plug-in':
            return handlers.run_synthetic_plug_in_evaluation
        case 'evaluate', _:
            return handlers.run_practical_plug_in_evaluation
        case ('mi', _) | ('q1', 'mi'):
            return handlers.run_information_plane_generation
        case 'q1', 'ips':
            return handlers.run_ip_comparison
        case 'q1', 'compression':
            return handlers.run_compression_quantisation
        case 'q2', 'compare':
            return handlers.run_compression_comparison
        case 'q2', 'correlation':
            return handlers.run_compression_rank_correlation
        
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
