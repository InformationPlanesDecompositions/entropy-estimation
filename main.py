import argparse

import cli.parser
import cli.handlers


def _get_fn_from_args(args: argparse.Namespace):
    cmd = str(args.command).lower()
    task = str(getattr(args, 'task', '')).lower()

    match cmd, task:
        case 'evaluate', 'plug-in':
            return cli.handlers.run_synthetic_plug_in_evaluation
        case 'evaluate', 'regime':
            return cli.handlers.run_data_dim_regime_plotting
        case 'evaluate', _:
            return cli.handlers.run_practical_plug_in_evaluation
        case ('mi', _) | ('q1', 'mi'):
            return cli.handlers.run_information_plane_generation
        case 'q1', 'ips':
            return cli.handlers.run_ip_comparison
        case 'q1', 'compression':
            return cli.handlers.run_compression_quantisation
        case 'q2', 'compare':
            return cli.handlers.run_compression_comparison
        case 'q2', 'correlation':
            return cli.handlers.run_compression_rank_correlation
        
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
