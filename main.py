import numpy as np

import cli.parser

import evaluation
import evaluation.plug_in


def main():
    parser = cli.parser.build_parser()
    args = parser.parse_args()

    if args.command == 'evaluate':

        if args.evaluation_type == 'plug-in':
            p: float = args.success_prob
            n_experiments: int = args.n_experiments

            n_sample_steps: list[int] = args.n_samples

            if (dimensions := args.dimensions) is None:
                if len(n_sample_steps) < 3:
                    parser.error(f'Please provide 3 parameters für argument -N if -D is not set')

                n_start, n_stop, n_steps = n_sample_steps[:3]
                n_sample_space = np.linspace(n_start, n_stop, n_steps).astype(np.int64)

                evaluation.plug_in.onedimensional_bernoulli(n_experiments, p, n_sample_space)

                return
            
            n_samples = n_sample_steps[0]
            d_steps = np.linspace(*dimensions)

            evaluation.plug_in.multidimensional_bernoulli(
                n_experiments, p, n_samples,
                d_steps=d_steps,
                save=args.save, output_dir=args.output
            )


if __name__ == '__main__':
    main()

    pass
