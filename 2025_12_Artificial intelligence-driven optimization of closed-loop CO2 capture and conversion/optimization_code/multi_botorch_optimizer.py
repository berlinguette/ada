#!/usr/bin/env python3
# coding: utf-8

import os
import time
import torch
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import gpytorch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox

from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Create a subfolder for the plots
image_folder = "visualization_results"
os.makedirs(image_folder, exist_ok=True)

# Define the names for the input variables and objectives
input_names = ['Temperature', 'Liquid flow rate', 'Flue gas flow rate', 'Current density']
objective_names = ['FE_CO', 'CO2 cap eff']


# Model initialization
# Noise std dev
_NOISE_SE = torch.tensor([7, 3], **tkwargs)
NOISE_SE = torch.tensor([0, 0], **tkwargs)

bounds = torch.tensor([[25, 10, 35, 100],
                       [80, 100, 235, 300]], **tkwargs)

_dim = bounds.shape[1]
_ref_point = torch.tensor([0,  0], **tkwargs)

initial_points = 8

# Global variable to hold the filename across the application's lifecycle
current_csv_file = None


def get_current_csv_file():
    global current_csv_file
    if current_csv_file is None or not os.path.exists(current_csv_file):
        # Create a new file with a timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        current_csv_file = f"New_param_{timestamp}.csv"
    return current_csv_file


def save_tensor_values_to_csv(tensor_values):
    # Ensure tensor_values is in a suitable format (list of lists) for DataFrame
    df = pd.DataFrame(tensor_values)
    csv_file = get_current_csv_file()

    if not os.path.exists(csv_file):
        # If the file does not exist, write header is True
        df.to_csv(csv_file, mode='w', header=True, index=False)
    else:
        # If the file exists, append without writing the header
        df.to_csv(csv_file, mode='a', header=False, index=False)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=tkwargs["device"])
    checkpoint['train_x'] = torch.tensor(checkpoint['train_x'], **tkwargs)
    checkpoint['train_obj'] = torch.tensor(checkpoint['train_obj'], **tkwargs)
    checkpoint['train_obj_true'] = torch.tensor(checkpoint['train_obj_true'], **tkwargs)

    # Initialize your model and mll here as you do in your script
    mll, model = initialize_model(checkpoint['train_x'], checkpoint['train_obj'])

    # Load the state dict with strict=False to ignore unexpected keys
    mll.load_state_dict(checkpoint['mll_state_dict'], strict=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return checkpoint, mll, model


def save_checkpoint(iteration, train_x, train_obj, train_obj_true, hvs, mll, model, base_filepath):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filepath = f"{base_filepath}_{timestamp}.pth"
    checkpoint = {
        "iteration": iteration,
        "train_x": train_x.cpu().numpy(),
        "train_obj": train_obj.cpu().numpy(),
        "train_obj_true": train_obj_true.cpu().numpy(),
        "hvs": hvs,
        "mll_state_dict": mll.state_dict(),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, filepath)


def convert_percentage(percentage_str):
    return float(percentage_str.strip('%'))


def generate_initial_data():
    # Read specific columns from the CSV file for the first `initial_points` rows
    df_in = pd.read_csv('Data.csv',
                        usecols=input_names,
                        nrows=initial_points)

    df_out = pd.read_csv('Data.csv',
                         usecols=objective_names,
                         nrows=initial_points)

    data_in = df_in.values.tolist()
    train_x = torch.tensor(data_in, **tkwargs)

    _data_out = df_out.values.tolist()
    data_out = [[convert_percentage(value) for value in sublist] for sublist in _data_out]
    train_obj_true = torch.tensor(data_out, **tkwargs)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE

    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    train_x = normalize(train_x, bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i: i + 1]
        train_yvar = torch.full_like(train_y, _NOISE_SE[i] ** 2)
        models.append(
            FixedNoiseGP(
                train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
            )
        )

    viz_corner(models)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def viz(models):
    # Define the grid for visualization in a 4D input space
    # Fixing the 3rd and 4th dimensions to some constant values (e.g., 0.5)
    fixed_values = [0.5, 0.5]  # Example fixed values for the 3rd and 4th dimensions
    max_mesh = 500
    x1_grid, x2_grid = torch.meshgrid(torch.linspace(0, 1, max_mesh),
                                      torch.linspace(0, 1, max_mesh))
    x_grid = torch.cat([
        x1_grid.flatten().unsqueeze(-1),
        x2_grid.flatten().unsqueeze(-1),
        torch.full_like(x1_grid.flatten().unsqueeze(-1), fixed_values[0]),
        torch.full_like(x2_grid.flatten().unsqueeze(-1), fixed_values[1])
    ], dim=-1)
    # x_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=-1)

    # Unnormalize the grid for plotting and saving
    x_grid_unnorm = unnormalize(x_grid, bounds[:, :4])
    x1_grid_unnorm = x_grid_unnorm[:, 0].view(max_mesh, max_mesh)
    x2_grid_unnorm = x_grid_unnorm[:, 1].view(max_mesh, max_mesh)

    # Visualize the posterior mean and variance for each objective
    for i, model in enumerate(models):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(x_grid)
        mean = posterior.mean.view(max_mesh, max_mesh).cpu().numpy()
        variance = posterior.variance.view(max_mesh, max_mesh).cpu().numpy()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save the data
        data = {
            "x1_grid": x1_grid_unnorm.numpy(),
            "x2_grid": x2_grid_unnorm.numpy(),
            "mean": mean,
            "variance": variance
        }
        data_filename = f"Objective_{i + 1}_Data_{timestamp}.pkl"
        with open(os.path.join("visualization_results", data_filename), "wb") as f:
            pickle.dump(data, f)

        # Save the plots with timestamp
        plt.figure(figsize=(12, 5))

        # Plot and save the posterior mean
        plt.subplot(1, 2, 1)
        plt.contourf(x1_grid_unnorm, x2_grid_unnorm, mean)
        plt.xlabel("Temperature")
        plt.ylabel("Liquid flow rate")
        plt.title(f"Objective {i + 1} - Posterior Mean")
        plt.colorbar()

        # Plot and save the posterior variance
        plt.subplot(1, 2, 2)
        plt.contourf(x1_grid_unnorm, x2_grid_unnorm, variance)
        plt.title(f"Objective {i + 1} - Posterior Variance")
        plt.xlabel("Temperature")
        plt.colorbar()
        mean_variance_filename = f"Objective_{i + 1}_Posterior_Mean_Variance_{timestamp}.png"
        plt.savefig(os.path.join(image_folder, mean_variance_filename))
        plt.close()


def viz_pairwise(models):
    max_mesh = 100

    # Create a unique subfolder for this particular call
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_path = os.path.join("visualization_results", f"viz_{timestamp}")
    os.makedirs(subfolder_path, exist_ok=True)

    # Iterate through all combinations of two parameters
    for dim1 in range(4):
        for dim2 in range(dim1 + 1, 4):
            # Prepare the grid
            x1_grid, x2_grid = torch.meshgrid(torch.linspace(0, 1, max_mesh), torch.linspace(0, 1, max_mesh))
            x_grid = torch.full((max_mesh ** 2, 4), 0.5)  # Initialize with fixed values
            x_grid[:, dim1] = x1_grid.flatten()
            x_grid[:, dim2] = x2_grid.flatten()

            # Unnormalize the grid
            x_grid_unnorm = unnormalize(x_grid, bounds)
            x1_grid_unnorm = x_grid_unnorm[:, dim1].view(max_mesh, max_mesh)
            x2_grid_unnorm = x_grid_unnorm[:, dim2].view(max_mesh, max_mesh)

            for i, model in enumerate(models):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior = model.posterior(x_grid)
                mean = posterior.mean.view(max_mesh, max_mesh).cpu().numpy()
                variance = posterior.variance.view(max_mesh, max_mesh).cpu().numpy()

                # Plotting
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.contourf(x1_grid_unnorm, x2_grid_unnorm, mean)
                plt.xlabel(f"Parameter {dim1 + 1}")
                plt.ylabel(f"Parameter {dim2 + 1}")
                plt.title(f"Objective {i + 1} - Posterior Mean")
                plt.colorbar()

                plt.subplot(1, 2, 2)
                plt.contourf(x1_grid_unnorm, x2_grid_unnorm, variance)
                plt.title(f"Objective {i + 1} - Posterior Variance")
                plt.xlabel(f"Parameter {dim1 + 1}")
                plt.colorbar()

                # Save the figure in the unique subfolder
                plt.tight_layout()
                plt.savefig(os.path.join(subfolder_path, f"Objective_{i + 1}_Params_{dim1 + 1}_{dim2 + 1}.png"))
                plt.close()


def collect_data_for_plots(models, bounds, max_mesh=100):
    data = {'mean': [[], []], 'variance': [[], []]}  # Structure to hold means and variances for both objectives

    for dim1 in range(4):
        for dim2 in range(dim1 + 1, 4):
            x1_grid, x2_grid = torch.meshgrid(torch.linspace(0, 1, max_mesh), torch.linspace(0, 1, max_mesh))
            x_grid = torch.full((max_mesh ** 2, 4), 0.5)  # Initialize with fixed values for other parameters
            x_grid[:, dim1] = x1_grid.flatten()
            x_grid[:, dim2] = x2_grid.flatten()

            x_grid_unnorm = unnormalize(x_grid, bounds)  # Assume unnormalize function is defined
            x1_grid_unnorm = x_grid_unnorm[:, dim1].view(max_mesh, max_mesh)
            x2_grid_unnorm = x_grid_unnorm[:, dim2].view(max_mesh, max_mesh)

            for i, model in enumerate(models):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior = model.posterior(x_grid)
                mean = posterior.mean.view(max_mesh, max_mesh).cpu().numpy()
                variance = posterior.variance.view(max_mesh, max_mesh).cpu().numpy()

                # Store data for plotting
                data['mean'][i].append((x1_grid_unnorm, x2_grid_unnorm, mean))
                data['variance'][i].append((x1_grid_unnorm, x2_grid_unnorm, variance))

    return data


def plot_corner(data, metric, objective_idx, subfolder_path):
    param_names = input_names
    objective_names = ['FE_CO', 'CO2 capture efficiency']

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust size as needed
    axs = axs.flatten()

    # Define parameter combinations for titles
    param_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for idx, ax in enumerate(axs):
        if idx < len(data[metric][objective_idx]):
            x1, x2, values = data[metric][objective_idx][idx]
            contour = ax.contourf(x1, x2, values)
            fig.colorbar(contour, ax=ax)

            # Update title and axis labels with parameter names
            param1, param2 = param_combinations[idx]
            ax.set_title(f'{param_names[param1]} vs {param_names[param2]}')
            ax.set_xlabel(param_names[param1])
            ax.set_ylabel(param_names[param2])
        else:
            ax.set_visible(False)  # Hide unused subplots

    # Update the suptitle to use objective names
    plt.suptitle(f'{objective_names[objective_idx]} - Posterior {metric.capitalize()}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(subfolder_path, f'CornerPlot_{objective_names[objective_idx].replace(" ", "_")}_{metric}.png'))
    plt.close()


def viz_corner(models):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_path = os.path.join("visualization_results", f"viz_{timestamp}")
    os.makedirs(subfolder_path, exist_ok=True)

    data = collect_data_for_plots(models, bounds)  # Assume `models` and `bounds` are defined

    for metric in ['mean', 'variance']:
        for objective_idx in range(len(models)):
            plot_corner(data, metric, objective_idx, subfolder_path)


# Define a helper functions that performs the essential BO step for $q$EHVI and $q$NEHVI
BATCH_SIZE = 1
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

standard_bounds = torch.zeros(2, _dim, **tkwargs)
standard_bounds[1] = 1


class InputDialog(QDialog):
    def __init__(self, parameter_name, num_exp):
        super().__init__()
        self.parameter_name = parameter_name
        self.num_exp = num_exp
        self.initUI()
        self.value = None
        self.cancelled = False

    def initUI(self):
        self.setWindowTitle(f'Exp: {self.num_exp+1-initial_points} - Input for {self.parameter_name}')
        layout = QVBoxLayout()

        self.label = QLabel(f"Enter the value for {self.parameter_name} (0-100 %):")
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        layout.addWidget(self.lineEdit)

        submitBtn = QPushButton('Submit', self)
        submitBtn.clicked.connect(self.submitClicked)
        layout.addWidget(submitBtn)

        cancelBtn = QPushButton('Cancel', self)
        cancelBtn.clicked.connect(self.cancelClicked)
        layout.addWidget(cancelBtn)

        self.setLayout(layout)

    def submitClicked(self):
        user_input = self.lineEdit.text()
        try:
            value = float(user_input)
            if 0 <= value <= 100:
                self.value = value
                self.close()
            else:
                QMessageBox.warning(self, 'Error', 'Value must be between 0 and 100.')
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid real number.')

    def cancelClicked(self):
        self.cancelled = True
        self.close()


def get_user_input_pyqt5(parameter_name, num_exp):
    app = QApplication.instance()  # Use the existing instance if available
    if app is None:  # Create a new instance if it does not exist
        app = QApplication([])
    value = None
    while value is None:
        dialog = InputDialog(parameter_name, num_exp)
        dialog.exec_()
        if dialog.cancelled:
            continue
        value = dialog.value
    return value


def show_results_ui(tensor_values):
    app = QApplication.instance()  # Use the existing instance if available
    if app is None:  # Create a new instance if it does not exist
        app = QApplication([])
    while True:
        dialog = ResultsDialog(tensor_values)
        dialog.exec_()
        if dialog.isHidden():  # Check if the dialog was closed (OK clicked or window closed)
            break


class ResultsDialog(QDialog):
    def __init__(self, tensor_values):
        super().__init__()
        self.label = None
        self.tensor_values = tensor_values
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Results')
        layout = QVBoxLayout()

        # Assuming tensor_values is a list of lists or similar iterable of iterables
        results_text = ", ".join([f"{value}" for value in self.tensor_values[0]])
        # self.label = QLabel(f"Temperature | Liq flow rate | Gas flow rate | Current density: {results_text}")
        self.label = QLabel(f"{input_names}: {results_text}")
        layout.addWidget(self.label)

        okBtn = QPushButton('OK', self)
        okBtn.clicked.connect(self.close)
        layout.addWidget(okBtn)

        self.setLayout(layout)


def round_tensor_params(params):
    """
    Round the parameters of a tensor based on specified rules.
    First two values are rounded to the nearest integer ending with 0 or 5.
    Last two values are rounded to the nearest integer.

    :param params: A tensor of parameters.
    :return: A tensor of rounded parameters.
    """
    # Round the first two parameters to the nearest integer ending with 0 or 5
    first_two = params[..., :2] / 5
    first_two_rounded = torch.round(first_two) * 5

    # Round the last two parameters to the nearest integer
    last_two_rounded = torch.round(params[..., 2:])

    # Combine the rounded values
    rounded_params = torch.cat([first_two_rounded, last_two_rounded], dim=-1)

    return rounded_params


# Integrating over function values at in-sample designs
def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=_ref_point.tolist(),  # use known reference point
        X_baseline=normalize(train_x, bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        # options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    new_x = round_tensor_params(new_x)
    print(" ")
    print(f"New parameters {input_names}: ", new_x)
    tensor_values = new_x.tolist()  # Convert tensor to list for displaying
    show_results_ui(tensor_values)
    save_tensor_values_to_csv(tensor_values)
    print("--------------")

    num_exp = len(train_x)

    # Get user input
    fe_co = get_user_input_pyqt5("FE_CO", num_exp)
    co2_cap_eff = get_user_input_pyqt5("CO2 cap eff", num_exp)
    new_obj_true = torch.tensor([[fe_co, co2_cap_eff]], **tkwargs)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true


N_BATCH = 30 if not SMOKE_TEST else 10
MC_SAMPLES = 1024 if not SMOKE_TEST else 16

verbose = True

hvs_qnehvi = []

# call helper functions to generate initial training data and initialize model
train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = generate_initial_data()
mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

# replace generate_initial_data with load_checkpoint if the run is stopped
# _checkpoint, mll_qnehvi, model_qnehvi = load_checkpoint('/CHECKPOINT_FILEPATH_<date>.pth')
# train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = _checkpoint['train_x'], _checkpoint['train_obj'], _checkpoint['train_obj_true']

# compute hypervolume
bd = DominatedPartitioning(ref_point=_ref_point, Y=train_obj_true_qnehvi)
volume = bd.compute_hypervolume().item()

hvs_qnehvi.append(volume)

# run N_BATCH rounds of BayesOpt after the initial random batch
for iteration in range(1, N_BATCH + 1):

    t0 = time.monotonic()

    # fit the models
    fit_gpytorch_mll(mll_qnehvi)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # optimize acquisition functions and get new observations
    (
        new_x_qnehvi,
        new_obj_qnehvi,
        new_obj_true_qnehvi,
    ) = optimize_qnehvi_and_get_observation(
        model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
    )

    # update training points
    train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
    train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
    train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

    # update progress
    bd = DominatedPartitioning(ref_point=_ref_point, Y=train_obj_true_qnehvi)
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)

    # reinitialize the models so they are ready for fitting on next iteration
    # Note: we find improved performance from not warm starting the model hyperparameters
    # using the hyperparameters from the previous iteration
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

    t1 = time.monotonic()

    if verbose:
        print(
            f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
            f"({hvs_qnehvi[-1]:>4.2f}), "
            f"time = {t1 - t0:>4.2f}.",
            end="",
        )
    else:
        print(".", end="")

    save_checkpoint(iteration, train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi, hvs_qnehvi, mll_qnehvi,
                    model_qnehvi, "CHECKPOINT_FILEPATH")

# 1: Plot the results
iters = np.arange(N_BATCH + 1) * BATCH_SIZE
_max_hv = 10000
log_hv_difference_qnehvi = np.log10(_max_hv - np.asarray(hvs_qnehvi))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.errorbar(
    iters,
    log_hv_difference_qnehvi,
    label="qNEHVI",
    linewidth=1.5,
)
ax.set(
    xlabel="number of observations (beyond initial points)",
    ylabel="Log Hypervolume Difference",
)
ax.legend(loc="lower left")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
fig.savefig(f"hypervolume_plot_{timestamp}.png")
plt.close(fig)

# 2: Plot the true objectives at the evaluated designs colored by iteration
fig, axes = plt.subplots(1, 1, figsize=(23, 7))
algos = "qNEHVI"
cm = plt.colormaps["viridis"]

batch_number = torch.cat(
    [
        torch.zeros(initial_points),
        torch.arange(1, N_BATCH + 1).repeat(BATCH_SIZE, 1).t().reshape(-1),
    ]
).numpy()

sc = axes.scatter(
    train_obj_true_qnehvi[:, 0].cpu().numpy(),
    train_obj_true_qnehvi[:, 1].cpu().numpy(),
    c=batch_number,
    alpha=0.8,
)
axes.set_title(algos)
axes.set_xlabel("FE_CO")
axes.set_ylabel("CO2 Capture Efficiency")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")
fig.savefig(f"objective_plot_{timestamp}.png")
plt.close(fig)
