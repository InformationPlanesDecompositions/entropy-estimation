# Entropy Estimation

This project aim to compute the information-theoretic quantities of entropy and mutual information of trained binary
neural networks to plot and analyse information planes and information-theoretic compression during learning. See the
[BNN-Training Repository](https://github.com/InformationPlanesDecompositions/bnn-training) on GitHub for details on the
training procedure and extraction of activations.

Python version ~= 3.13

## How-to

This project is controlled via the CLI by calling `python main.py` with the respective arguments.
You can find and look up all commands and argument groups with the `-h` argument. The most important
commands are the following.

### Estimate MI and Create Information Plane (IP)

`python main.py mi -d DATA`

+ `DATA` is a folder containting an `activations.h5` and `data.h5` file (see the [BNN-Training Repository](https://github.com/InformationPlanesDecompositions/bnn-training) for details)
+ If `--save` is provided, a PDF of the IP per run as well as a CSV containing the MI data per layer and run is created in the provided `--output`

### Compare IPs of different experiments

`python main.py q1 ips -c CONFIG`

+ `CONFIG` is a YAML file, see the [config](config.yml) for details
+ The IPs of the provided experiments under `experiments` are generated and compared
+ Optionally, the `--loss-plot` and `--accuracy-plot` can be shown, too

### Quantify Compression and Compare

`python main.py q1 compression -c CONFIG`

+ This will generate a swarmplot for the `experiment_groups` (one per dataset, separated by the experiment groups)
+ Each dot represents the compression factor $\varrho$ computed as $\frac{I(X;T_\ell)_{\max} - \bar{I}(X;T_\ell)_{50}}{I(X;T_\ell)_{\max}}$

### Compare Compression versus Generalisation

`python main.py q2 compare -c CONFIG`

+ This will generate a scatter-plot with $I(X;T_\ell)$ as x-axis and validation accuracy as y-axis
+ Each dot is the mean or median (`--agg-func`) over the last `--n-epochs` training epochs
+ Each dot contains error bars going to the min/max values in the epoch period

### Compute Rank Correlation Compression-Generalisation

`python main.py q2 correlation -c CONFIG --dir-experiments DIR_EXPERIMENTS`

+ Compute the Spearman rank correlation coefficient and its $p$-value
+ If `--to-latex`, the result will be a LaTeX booktabs table
