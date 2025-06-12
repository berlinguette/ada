import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tools.simulations import run_simulations
from tools.regressors.reg_botorch import BotorchRegressor
from tools.processors import Proc
from tools.engine import c
from botorch.utils.transforms import normalize
from opt_suite.optimizers.opt_random import Random
from opt_suite.optimizers.opt_botorch import Botorch


def run(standardize=0):
    dataset_name = "data.csv"
    df = pd.read_csv(dataset_name)

    # Define input and output columns
    input_cols = ["Input Current density", "Bicarbonate flow", "KOH flow", "Bicarb temp", "KOH conc", "Bicarb conc"]
    # output_cols = ["Reactor yield"]
    output_cols = ["CO partial current density"]

    # Apply optional standardization to the output column
    if standardize == 1:
        std_temp, mean_temp = df[output_cols].std().iloc[0], df[output_cols].mean().iloc[0]
        df[output_cols] = (df[output_cols] - mean_temp) / std_temp
    elif standardize == 2:
        scaler = StandardScaler().fit(df[output_cols])
        df[output_cols] = scaler.transform(df[output_cols])
    else:
        std_temp, mean_temp = 1, 0

    # Convert data to tensor and normalize inputs based on specified bounds
    x = torch.tensor(df[input_cols].to_numpy(), dtype=torch.double)
    y = df[output_cols].to_numpy()
    bounds = torch.tensor([[50, 40, 6, 25, 0.05, 0.5], [350, 200, 30, 80, 3, 3.05]], dtype=torch.double)
    x_normalized = normalize(x, bounds).numpy()

    # Initialize regression model
    model = BotorchRegressor(x=x_normalized, y=y)

    # Simulation configuration
    run_repeats, run_length = 50, 100

    # Run simulations with Random optimizer
    uid_random = run_simulations(
        model=model,
        x_dim=x.shape[1],
        y_dim=y.shape[1],
        x_min=x.min(axis=0),
        x_max=x.max(axis=0),
        optimizer=Random,
        run_repeats=run_repeats,
        run_length=run_length,
        multiprocessing=False,
    )

    # Run simulations with Botorch optimizer
    uid_botorch = run_simulations(
        model=model,
        x_dim=x.shape[1],
        y_dim=y.shape[1],
        x_min=x.min(axis=0),
        x_max=x.max(axis=0),
        optimizer=Botorch,
        run_repeats=run_repeats,
        run_length=run_length,
        multiprocessing=False,
    )

    # Process simulation results
    proc_random = Proc(uid_random)
    proc_botorch = Proc(uid_botorch)

    # Plot results
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(proc_botorch.domain, std_temp * proc_botorch.best_mean + mean_temp, c=c.blue.i800, label="Optimization")
    ax.plot(proc_random.domain, std_temp * proc_random.best_mean + mean_temp, c=c.green.i800, label="Random")

    # Mark the start of the optimization phase
    x_start_opt = 53
    y_opt_max = std_temp * float(proc_botorch.best_mean.max()) + mean_temp
    y_rand_max = std_temp * float(proc_random.best_mean.max()) + mean_temp

    ax.axvline(x=x_start_opt, color='r', linestyle='--')
    ax.text(x_start_opt + 0.15, 10, "Optimization starts", fontsize=16, color="r")
    ax.text(run_length - 0.9, y_opt_max, round(y_opt_max, 2), fontsize=16, color=c.blue.i800)
    ax.text(run_length - 0.9, 0.97 * y_rand_max, round(y_rand_max, 2), fontsize=16, color=c.green.i800)

    # Plot uncertainty bands for both optimizers
    ax.fill_between(proc_botorch.domain,
                    std_temp * (proc_botorch.best_mean - proc_botorch.best_std) + mean_temp,
                    std_temp * (proc_botorch.best_mean + proc_botorch.best_std) + mean_temp,
                    alpha=0.3, facecolor=c.blue.i800)

    ax.fill_between(proc_random.domain,
                    std_temp * (proc_random.best_mean - proc_random.best_std) + mean_temp,
                    std_temp * (proc_random.best_mean + proc_random.best_std) + mean_temp,
                    alpha=0.3, facecolor=c.green.i800)

    # Configure legend and axis labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='center left',
              bbox_to_anchor=(0.75, 0.1), fontsize=16, title_fontsize=16)

    # Format and save plot
    ax.set_xlim(1, max(proc_botorch.domain))
    ax.set_xlabel('Samples', fontsize=22)
    ax.set_ylabel('Max performance', fontsize=22)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_title(f'Max eval. obj. ({output_cols[0]}) - simulation run \n', fontsize=22)

    fig.set_dpi(500)
    fig.savefig(f"plot/JCO_" + time.strftime("%m%d_%H%M%S") + ".png")
    plt.close(fig)


if __name__ == '__main__':
    run(standardize=0)
