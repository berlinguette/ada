# Bayesian Optimization Module using BoTorch

This folder contains the implementation of a **Bayesian optimization framework** built on top of the [BoTorch](https://botorch.org/) library. The optimizer is tailored for experimental design and parameter space exploration tasks, such as optimizing performance in electrochemical systems.

## Contents

### `multi_botorch_optimizer.py`
Defines a `Botorch` class for running multi-objective Bayesian optimization with support for:
- An acquisition function (`qNEHVI`)
- Gaussian process modeling (`FixedNoiseGP`, `SingleTaskGP`)
- Normalization of input data
- Torch-based optimization and sampling

This script also includes a `_random` function for generating initial random parameter sets from a dataset (`experiment_data.csv`).

### `opt_template.py`
Provides the `OptTemplate` abstract base class, which standardizes how optimizers are implemented and interacted with. It includes:
- Parameter and objective management
- Tracking of optimizer progress and sampled results
- Data serialization and reloading for checkpointing
- Abstract method `get_x()` to be implemented by child classes
