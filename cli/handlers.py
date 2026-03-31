import argparse

import cli.configure


def run_synthetic_plug_in_evaluation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.plug_in import evaluate_plugin_estimate

    try:
        evaluate_plugin_estimate(
            n_experiments=args.n_experiments,
            n_samples=args.n_samples,
            max_d=args.max_dimensions,
            use_existing_data=args.use_existing,
            save=args.save,
            output_dir=args.output,
        )
    except Exception as e:
        parser.error(str(e))


def run_practical_plug_in_evaluation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.plug_in import evaluate_entropy_subadditivity

    try:
        evaluate_entropy_subadditivity(
            data_dir=args.data,
            activation_file_name=args.file_name,
            run_idx=args.run,
            save=args.save,
            output_dir=args.output,
        )
    except Exception as e:
        parser.error(str(e))

    
def run_data_dim_regime_plotting(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.plug_in import plot_approximate_data_to_dims_regime

    try:
        plot_approximate_data_to_dims_regime(
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            save=args.save,
            output_dir=args.output,
            show_plt=args.show_plots
        )
    except Exception as e:
        parser.error(str(e))


def run_information_plane_generation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.information_planes import generate_information_planes

    try:
        generate_information_planes(
            data_dir=args.data,
            run_idx=args.run,
            show_plots=args.show_plots,
            compute_mi=args.compute_mi,
            save=args.save,
            as_pdf=args.plot_as_pdf
        )
    except Exception as e:
        parser.error(str(e))


def run_ip_comparison(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.information_planes import compare_information_planes

    config = cli.configure.read_config(args.config)
    comparison_config = config.get('comparison', config)

    experiments: dict[str, str] = config.get('experiments', {})

    try:
        compare_information_planes(
            experiments=experiments,
            run_idx=args.run,
            dir_mi=args.dir_mi,
            dir_exp=args.dir_experiments,
            show_plots=args.show_plots,
            plot_layout=args.plot_layout,
            name_as_wd=args.name_as_wd,
            plot_accuracy=comparison_config.get('accuracy_plot', True) if args.accuracy_plot is None else args.accuracy_plot,
            plot_losses=config.get('loss_plot', False) if args.loss_plot is None else args.loss_plot,
            save=args.save,
            output=args.output
        )
    except Exception as e:
        parser.error(str(e))


def run_compression_quantisation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.compression import quantify_compression

    config = cli.configure.read_config(args.config)
    comparison_config: dict = config.get('comparison', config)
    compression_config: dict = comparison_config.get('compression', {})

    try:
        quantify_compression(
            experiment_groups=comparison_config.get('experiment_groups', {}),
            included_groups=compression_config.get('groups', []),
            included_layer_indices=compression_config.get('include_layer_indices', {}),
            dataset_order=compression_config.get('dataset_order', []),
            n_epochs=args.n_epochs,
            use_existing=args.use_existing,
            dir_mi=args.dir_mi,
            save=args.save,
            output_dir=args.output,
            show_plt=args.show_plots,
        )
    except Exception as e:
        parser.error(str(e))


def run_compression_comparison(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.compression import compare_compressions

    config = cli.configure.read_config(args.config)
    comparison_config: dict = config.get('comparison', config)

    experiments = comparison_config.get('experiments', {})

    try:
        compare_compressions(
            experiments=experiments,
            dir_exp=args.dir_experiments,
            dir_mi=args.dir_mi,
            layer_offset_idx=args.layer_offset_idx,
            n_epochs=args.n_epochs,
            agg_func=args.agg_func,
            legend_title=args.legend_title,
            as_cbar=args.as_cbar,
            is_discrete_cbar=args.as_cbar and args.is_discrete_cbar,
            cbar_minimum=args.discrete_cbar_minimum,
            save=args.save,
            output_dir=args.output,
            show_plt=args.show_plots,
        )
    except Exception as e:
        parser.error(str(e))


def run_compression_rank_correlation(parser: argparse.ArgumentParser, args: argparse.Namespace):
    from evaluation.compression import compute_compression_rank_correlation

    config = cli.configure.read_config(args.config)
    comparison_config: dict = config.get('comparison', config)

    experiment_groups = comparison_config.get('experiment_groups', {})    

    try:
        compute_compression_rank_correlation(
            experiment_groups=experiment_groups,
            dir_exp=args.dir_experiments,
            dir_mi=args.dir_mi,
            n_epochs=args.n_epochs,
            to_latex=args.to_latex,
            output_dir=args.output,
        )
    except Exception as e:
        parser.error(str(e))
