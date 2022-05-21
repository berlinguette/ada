"""
This script creates figure S4 from the manuscript

"A self-driving laboratory designed to accelerate the discovery
of adhesive materials"

by

Michael B. Rooney,Benjamin P. MacLeod, Ryan Oldford, Zachary J. Thompson
Kolby L. White, Justin Tungjunyatham, Brian J. Stankiewicz,
Curtis P. Berlinguette

dependencies for this script are listed in "requirements.txt"
"""

# imports
from ax import optimize
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import warnings
import dill


def gaussian(x, mu, sigma):
    """
    Returns the value of a gaussian of mean mu and variance sigma**2
    evaluated at x; note the gaussian is scaled to have a maximum value of 1
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def noisy_gaussian(R):
    """
    A noisy gaussian function to emulate the adhesive strength experiments
    described in figure 6 of the manuscript.
    :param R: resin fraction of the adhesive formulation
    :return S: strength of the adhesive formulation in [Mpa]
    """
    S = 0.2 + \
        8 * gaussian(x=R, mu=0.4, sigma=0.05) \
        * (0.7 + 0.30 * np.random.rand())
    return S


def run_simulations(n_replicate_optimizations, n_optimization_rounds,
                    n_formulations_per_round, experiment_emulator_func):
    """
    runs simulated adhesive strength optimization campaigns (in replicate)
    using the ax Bayesian optimization package
    :param n_replicate_optimizations: number of replicate simulations to run
    :param n_optimization_rounds: number of rounds within each simulation
    :param n_formulations_per_round: number of formulations to test per round
    :param experiment_emulator_func: function to emulate the real experiment
    :return:
    """
    # create empty lists to store simulation results
    best_parameters_list = []
    best_values_list = []
    experiments_list = []
    models_list = []

    # simulate perform Bayesian optimizations on the experiment emulator
    for optimization_idx in range(0, n_replicate_optimizations):

        try:
            print(f'starting iteration #{optimization_idx}')
            best_parameters, best_values, experiment, model = optimize(
                total_trials=n_optimization_rounds,
                arms_per_trial=n_formulations_per_round,
                parameters=[
                    {
                        "name": "resin_fraction",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                    },
                ],
                evaluation_function=lambda p: experiment_emulator_func(
                    p["resin_fraction"]),
                minimize=False,
            )

            # append the results to the results lists
            best_parameters_list.append(best_parameters)
            best_values_list.append(best_values)
            experiments_list.append(experiment)
            models_list.append(model)

        except BaseException as err:
            warnings.warn(
                f"An error occurred during iteration {optimization_idx}")
            warnings.warn(f"The error was: \n {err}")

    # save the results along with the configuration
    config_dict = {'n_replicate_optimizations': n_replicate_optimizations,
                   'n_optimization_rounds': n_optimization_rounds,
                   'n_formulations_per_round': n_formulations_per_round,
                   'experiment_emulator_func': experiment_emulator_func}
    simulation_results = {"configuration": config_dict,
                          "best_params": best_parameters_list,
                          "best_values": best_values_list,
                          "experiments": experiments_list,
                          "models": models_list}
    time_stamp = datetime.now()
    results_filename = time_stamp.strftime('%Y%m%d_%H%M_%S') + "_sim_results"
    with open(results_filename, "wb") as dill_file:
        dill.dump(simulation_results, dill_file)
    print(f'Simulations complete - saved results to {results_filename}')

    return results_filename


def load_results(results_file):
    with open(sim_results_file, 'rb') as dill_file:
        simulation_results = dill.load(dill_file)

    return simulation_results


def analyze_results(results_file):
    # load results from file
    results = load_results(sim_results_file)

    # create an empty array to store max strength vs round for each simulation
    n_simulations = results['configuration']['n_replicate_optimizations']
    n_rounds = results['configuration']['n_optimization_rounds']
    rounds = range(n_rounds)
    max_strength_vs_round = np.empty([n_simulations, n_rounds])

    # loop over all simulation results to extract running maximum strength
    for result_idx, result in enumerate(results['experiments']):

        # loop over optimization rounds within each simulation
        for round_idx in rounds:
            df_temp = result.lookup_data().df  # simulation data from all rounds
            # data from all rounds up to round_idx (to get a *running* maximum)
            df_temp = df_temp[df_temp['trial_index'] <= round_idx]
            # add running max strength from this round to the array
            max_strength_vs_round[result_idx, round_idx] = df_temp[
                'mean'].max()

    # make figure S4 (plot of max strength distribution vs round)
    fig, ax = plt.subplots()
    colors = ['purple', 'tab:blue', 'mediumseagreen', 'orange']
    for (round, color) in zip(rounds, colors):
        x = (round + 1) * np.ones(
            n_simulations)  # round+1 so x-axis starts at 1
        y = max_strength_vs_round[:, round]
        ax.scatter(x, y, color=color)

    violin_parts = ax.violinplot(max_strength_vs_round, np.array(rounds) + 1,
                                 showmeans=True, showextrema=False)
    for part in violin_parts['bodies']:
        part.set_facecolor('grey')
    violin_parts['cmeans'].set_edgecolor('grey')
    ax.set_xlabel('Optimization round #')
    ax.set_ylabel('running maximum tensile stress (Mpa)')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim([0, 9])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_title(
        f"Convergence of simulated optimization campaigns \n from {results_file}")
    fig_path = results_file + "_convergence plot.png"
    plt.savefig(fig_path)

    return fig_path


if __name__ == "__main__":

    # configure the simulated optimization campaigns
    configuration_dict = {}
    configuration_dict['n_replicate_optimizations'] = 100
    configuration_dict['n_optimization_rounds'] = 4
    configuration_dict['n_formulations_per_round'] = 5
    configuration_dict['experiment_emulator_func'] = noisy_gaussian

    # optionally define a file containing existing simulation results
    # to run new simulations, leave blank;
    # to use existing results, specify a file path
    sim_results_file = '' #paper uses '20220520_1521_56_sim_results'

    # if no existing results file defined, run new simulations
    if sim_results_file == '':
        sim_results_file = run_simulations(**configuration_dict)

    # analyze and plot the results
    figure_path = analyze_results(sim_results_file)
    print(f"Script complete - see resulting figure at {figure_path}")
