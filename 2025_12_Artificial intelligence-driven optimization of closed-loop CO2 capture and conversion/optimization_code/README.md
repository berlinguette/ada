# Bayesian Optimization Module using BoTorch

This folder contains the implementation of a **Bayesian optimization framework** built on top of the [BoTorch](https://botorch.org/) library. The optimizer is tailored for experimental design and parameter space exploration tasks, such as optimizing performance in electrochemical systems.

## Contents

### `multi_botorch_optimizer.py`
Defines functions for running multi-objective Bayesian optimization with support for:
- An acquisition function (`qNEHVI`)
- Gaussian process modeling (`FixedNoiseGP`)
- Normalization of input data
- Torch-based optimization and sampling

### `requirements.txt`
Defines requirements for running multi_botorch_optimizer.py python code:
`pip install -r requirements.txt`
