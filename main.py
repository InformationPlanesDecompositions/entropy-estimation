import numpy as np

from estimators import plug_in
from corrections import mm

import distributions.bernoulli
import distributions.binomial

import cli.parser


def main():
    parser = cli.parser.build_parser()
    args = parser.parse_args()

    if args.command == 'evaluate':
        p: float = args.success_prob

        if p < 0 or p > 1:
            parser.error(f'Invalid success probability outside [0, 1], was {p}')

        n_exp: int = args.n_experiments

        spacing_method = args.spacing_method
        n_min, n_max = args.n_min, args.n_max

        if spacing_method == 'linear':
            sample_sizes = np.linspace(n_min, n_max, n_exp).astype(int)
        elif spacing_method == 'logarithmic':
            sample_sizes = np.geomspace(n_min, n_max, n_exp).astype(int)
        else:
            # Should be unreachable
            parser.error(f'Unknown spacing method provided, was {spacing_method}')

        if args.distribution == 'binomial':
            n_samples = args.n_samples

            for n_trials in sample_sizes:
                h_true = distributions.binomial.compute_true_entropy(p, n_trials=n_trials)

                samples = distributions.binomial.generate_samples(
                    p,
                    n_trials=n_trials,
                    size=n_samples
                )

                # TODO: Extract into method
                h_hat = plug_in.estimate_entropy(samples)
                var_hat = plug_in.estimate_entropy_variance(samples, h_hat)
                corr = mm.first_order(samples)

                # TODO: Extract into method
                print(
                    f'{n_trials} trials ({n_samples} samples):'
                    + f'\t H: {h_true:.5f}, H^: {h_hat:.5f} (+{corr:.5f}), Var^: {var_hat:.5f}'
                )

        if args.distribution == 'bernoulli':
            h_true = distributions.bernoulli.compute_true_entropy(p)

            for n_samples in sample_sizes:
                samples = distributions.bernoulli.generate_samples(p, n_samples)
                
                # TODO: Extract into method
                h_hat = plug_in.estimate_entropy(samples)
                var_hat = plug_in.estimate_entropy_variance(samples, h_hat)
                corr = mm.first_order(samples, n_classes=2)

                # TODO: Extract into method
                print(
                    f'{n_samples} samples:'
                    + f'\t H: {h_true:.5f}, H^: {h_hat:.5f} (+{corr:.5f}), Var^: {var_hat:.5f}'
                )


if __name__ == '__main__':
    main()
