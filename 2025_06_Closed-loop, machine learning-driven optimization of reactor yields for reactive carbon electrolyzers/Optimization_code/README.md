# Bayesian Optimization Module using BoTorch

This folder contains the implementation of a **Bayesian optimization framework** built on top of the [BoTorch](https://botorch.org/) library. The optimizer is tailored for experimental design and parameter space exploration tasks, such as optimizing reactor performance in electrochemical systems.

## Contents

### `botorch_optimizer.py`
Defines a `Botorch` class for running Bayesian optimization with support for:
- Multiple acquisition functions (e.g., `qUCB`, `qEI`, `qNEI`, etc.)
- Gaussian process modeling (`FixedNoiseGP`, `SingleTaskGP`)
- Normalization of input data
- Torch-based optimization and sampling

This script also includes a `_random` function for generating initial random parameter sets from a dataset (`data.csv`).

### `opt_template.py`
Provides the `OptTemplate` abstract base class, which standardizes how optimizers are implemented and interacted with. It includes:
- Parameter and objective management
- Tracking of optimizer progress and sampled results
- Data serialization and reloading for checkpointing or transfer learning
- Abstract method `get_x()` to be implemented by child classes

## Features

- **Acquisition Function Flexibility**: Easily switch between different acquisition strategies (UCB, EI, PI, etc.)
- **Torch-based Parallel Optimization**: Accelerated candidate selection using `optimize_acqf`
- **Noise-aware Modeling**: Supports fixed and heteroskedastic noise models
- **State Tracking**: Encodes optimization state using `Progress` class for reproducibility
- **Extensibility**: Inherit from `OptTemplate` to implement new optimization strategies
