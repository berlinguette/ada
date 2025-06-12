# Simulation Benchmarking for Optimizer Comparison

This script benchmarks **Bayesian optimization** (via BoTorch) against **random sampling** for maximizing a target performance metric in a simulated environment.

It is used to compare optimization efficiency on a model trained to predict:
- **CO partial current density** (J<sub>CO</sub>) from input process parameters.

## Key Components

- **Input**: `data.csv` containing input features like current density, flow rates, temperatures, and concentrations.
- **Model**: `BotorchRegressor` (Gaussian Process-based surrogate model).
- **Optimizers**:  
  - `Random` – uniform sampling  
  - `Botorch` – Bayesian optimization using acquisition functions

## Workflow

1. Preprocess the dataset and normalize inputs.
2. Train a regression model to emulate the experimental process.
3. Run 50 simulations each for:
   - Random Sampling
   - Bayesian Optimization (BoTorch)
4. Compare performance over 100 iterations.
5. Plot:
   - Best observed performance vs. iteration
   - Uncertainty bands (standard deviation)
   - Start of optimization phase (after initialization)
