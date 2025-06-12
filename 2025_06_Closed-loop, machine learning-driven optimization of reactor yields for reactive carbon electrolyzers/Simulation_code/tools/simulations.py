from tools.engine import dict_to_json_safe, generate_timestamp
from typing import Optional, Callable, Tuple, Iterable, Union
from tools.engine import get_google_drive_authentication
from .templates.reg_template import RegressorTemplate
from tools.engine import upload_file_to_google_drive
from opt_suite.optimizers.opt_random import Random
import pathos.multiprocessing as mp
import multiprocess.context as ctx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import psutil
import copy
import json
import time
import os


def restart(func):
    """
    A function decorator for restarting simulations.
    Restarts simulation runs that have thrown an exception. The _run_simulation
    function must be passed an 'auto_restart' argument.
    """
    def inner_function(*args, **kwargs):

        # If auto_restart is not being used, just return the function
        if not kwargs['auto_restart']:
            return func(*args, **kwargs)

        # If auto_restart is being used, track how many failures have been
        # caught and runs restarted.
        else:
            fail_count = 0
            while True:
                try:
                    return func(
                        *copy.deepcopy(args),
                        **copy.deepcopy(kwargs),
                    )
                except Exception as e:
                    fail_count += 1
                    print(f'{fail_count} failures')
    return inner_function


def convert_dimensions(
        n_bound: Union[float, Iterable],
        n_dim: int,
) -> np.ndarray:
    """
    Convert a x bound to an array of x_dim.
    :param n_bound: Either a float or an iterable.
    :param n_dim: The dimension of the x dimension.
    :return:
    """
    if not isinstance(n_bound, Iterable):
        return np.repeat(n_bound, n_dim)
    else:
        return np.array(n_bound)


def run_simulations(
        surface: Optional[Callable] = None,
        model: Optional[RegressorTemplate] = None,
        x_dim: int = None,
        y_dim: int = None,
        x_min: Union[int, Iterable] = None,
        x_max: Union[int, Iterable] = None,
        surface_params: Optional[dict] = None,
        x_noise_func: Optional[Callable] = None,
        x_noise_params: Optional[dict] = None,
        y_noise_func: Optional[Callable] = None,
        y_noise_params: Optional[dict] = None,
        optimizer=Random,
        optimizer_params: Optional[dict] = None,
        run_repeats: int = 100,
        run_length: Optional[int] = None,
        run_ftol: Optional[float] = None,
        run_ftol_length: Optional[float] = None,
        multiprocessing: bool = True,
        n_processes: Optional[int] = None,
        label: Optional[str] = None,
        judge_ynoise: bool = True,
        auto_restart: bool = False,
        print_iter: bool = False,
        drive_id: Optional[str] = None,
) -> str:
    """
    Perform a simulation based on a configuration and save the results to file.

    :param surface: The name of the surface function to use. See
    tools/surfaces.py for examples.
    :param model: The model object to optimize. See tools/models.py for
    examples.
    :param x_dim: The number of input dimensions.
    :param y_dim: The number of output dimensions.
    :param x_min: The minimum bound of all the parameters for the surface.
    :param x_max: The maximum bound of all the parameters for the surface.
    :param surface_params: Any additional kwargs to pass to the surface
    function.
    :param x_noise_func: The type of noise to add to the input parameters before
    surface evaluation. Choice of 'gaussian'.
    :param x_noise_params: The params of the input noise.
    :param y_noise_func: The type of noise to add to the surface after
    evaluation. Choice of 'gaussian'.
    :param y_noise_params: The params of the noise to add to the surface.
    :param optimizer: The optimizer to use. Choice of Random,
    :param optimizer_params: Additional parameters to pass to the optimizer.
    :param run_repeats: The number of repeats of the simulation to perform.
    :param run_length: The number of samples to include in one run.
    :param run_ftol: Perform the optimization until run_ftol is observed.
    :param run_ftol_length: If ftol and opt is deterministic, batch size.
    :param multiprocessing: Should multiprocessing be used.
    :param n_processes: If using multiprocessing, how many processes should be
    used. If this is not specified, the number of physical cores is used.
    :param label: Text that should be included in the name of the saved data.
    :param judge_ynoise: When adjudicating the performance of the simulation,
    should the ynoise be considered? If false, the y value without noise is
    used.
    :param auto_restart: If true, simulation runs that throw an exception are
    automatically restarted.
    :param print_iter: Should each iteration print the values. This is good for
    debugging, but messes up the progress bar.
    :param drive_id: If a Google Drive folder ID is passed, the results will be
    uploaded to it.
    :return: Filename.
    """

    # To ensure that runs get a unique ID, wait 1 second.
    time.sleep(1)

    # Either a surface or a model must be passed.
    if model is None and surface is None:
        raise ValueError('Either a model or a surface must be defined.')
    if model is not None and surface is not None:
        raise ValueError('Only a model or a surface must be defined.')

    # If neither run_length or run_ftol_length
    if run_length is None and run_ftol_length is None:
        raise ValueError('Must specify either run_length or run_ftol_length.')

    # If a model has been passed,
    if model is not None:

        # Ensure that the model is initiated
        if not model.initialized:
            model.initialize()

        # Get the callable surface
        surface = model.evaluate

        # Determine the number of params and bounds
        x_dim = model.x_dim
        y_dim = model.y_dim

        # # If external scaling is being used
        # if x_min is None and x_max is None:
        #     if not model.x_internal_scaling:
        #         x_min = np.min(model.x_scaled, axis=0)
        #         x_max = np.max(model.x_scaled, axis=0)
        #     else:
        #         x_min = np.min(model.x, axis=0)
        #         x_max = np.max(model.x, axis=0)

        # TODO remove conditions on scaling
        x_min = np.min(model.x, axis=0)
        x_max = np.max(model.x, axis=0)

    # Convert param nones to dicts
    optimizer_params = dict() if optimizer_params is None else optimizer_params
    x_noise_params = dict() if x_noise_params is None else x_noise_params
    y_noise_params = dict() if y_noise_params is None else y_noise_params
    surface_params = dict() if surface_params is None else surface_params

    # Get names for logging
    x_noise_name = x_noise_func.__name__ if x_noise_func is not None else None
    y_noise_name = y_noise_func.__name__ if y_noise_func is not None else None

    # If a single value has been passed for the bounds, convert to an array.
    x_min = 0 if x_min is None else x_min
    x_max = 1 if x_max is None else x_max
    x_min = convert_dimensions(x_min, x_dim)
    x_max = convert_dimensions(x_max, x_dim)
    assert len(x_min) == x_dim
    assert len(x_max) == x_dim

    # If y_dim is not 1, add to surface params
    y_dim = 1 if y_dim is None else y_dim
    if y_dim != 1:
        surface_params['y_dim'] = y_dim

    # Create a params dict that will be saved along side the data
    params = dict(
        time_start=generate_timestamp(),
        surface=surface,
        model=model,
        x_dim=x_dim,
        y_dim=y_dim,
        x_min=x_min,
        x_max=x_max,
        surface_params=surface_params,
        x_noise_func=x_noise_name,
        x_noise_params=x_noise_params,
        y_noise_func=y_noise_name,
        y_noise_params=y_noise_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        run_repeats=run_repeats,
        run_length=run_length,
        run_ftol=run_ftol,
        run_ftol_length=run_ftol_length,
        multiprocessing=multiprocessing,
        judge_ynoise=judge_ynoise,
        auto_restart=auto_restart,
    )

    # Configure the optimizer
    optimizer = optimizer(**optimizer_params)

    # Define the parameters for the optimizer
    for i in range(x_dim):
        optimizer.add_parameter(name=f'x{i}', pmin=x_min[i], pmax=x_max[i])

    # If the y_dim is not 1, add dimensions
    if y_dim != 1:
        for i in range(y_dim):
            optimizer.add_objective(
                name=f'y{i}',
                reference_point=0,
            )

    # Create a place to store results
    results = pd.DataFrame()

    # Define the params to pass into the simulation
    sim_params = dict(
        surface=surface,
        surface_params=surface_params,
        x_noise_func=x_noise_func,
        x_noise_params=x_noise_params,
        y_noise_func=y_noise_func,
        y_noise_params=y_noise_params,
        run_length=run_length,
        run_ftol=run_ftol,
        run_ftol_length=run_ftol_length,
        judge_ynoise=judge_ynoise,
        auto_restart=auto_restart,
        print_iter=print_iter,
    )

    # If performing the optimizations in parallel
    if multiprocessing:

        # If Botorch is being used, spawn instead of fork. See:
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        # https://stackoverflow.com/questions/40615795/pathos-enforce-spawning-on-linux
        spawn = False
        if optimizer.name == 'Botorch':
            spawn = True
        if model is not None:
            if model.__class__.__name__ == 'BotorchRegressor':
                spawn = True
        if spawn:
            ctx._force_start_method('spawn')

        # If number of processes not defined, determine from the number of
        # physical cores.
        if n_processes is None:
            n_processes = psutil.cpu_count(logical=False)

            # Log
            print(f'Detected {n_processes} CPUs.')

        # Create pool
        pool = mp.Pool(processes=n_processes)

        # Create a place to store the results
        results_list = list()

        # Create the jobs
        for i in range(run_repeats):
            results_list.append(
                pool.apply_async(
                    _run_simulation,
                    kwds=dict(
                        optimizer=copy.deepcopy(optimizer),
                        run_iter=i,
                        seed=np.random.randint(1E8),
                        **sim_params,
                    ),
                )
            )

        # Wait for daemon processes to complete
        for result in tqdm(results_list):
            result.get()

        # Close pool
        pool.close()
        pool.join()

        # Convert into a single dataframe
        results = pd.concat([result.get() for result in results_list])

    # If performing serially, repeat through for loop
    else:
        for i in tqdm(range(run_repeats)):

            # Create a new copy of the optimizer, iterate
            sim_params['optimizer'] = copy.deepcopy(optimizer)
            sim_params['run_iter'] = i

            # Perform simulation
            df = _run_simulation(**sim_params)

            # Store result
            results = pd.concat([results, df])

    # Record end time
    params['time_stop'] = generate_timestamp()

    # If no label passed, remove
    label = '' if label is None else '_' + label

    # Construct the name of the files
    file_name = f'{params["time_start"]}{label}'

    # Define the paths for the two files
    data_path = os.path.join(os.getcwd(), 'results', file_name + '.feather')
    meta_path = os.path.join(os.getcwd(), 'results', file_name + '.json')

    # Save results to file
    if not os.path.exists('results'):
        os.mkdir('results')
    results.reset_index().to_feather(data_path)

    # Save metadata to file
    jparams = dict_to_json_safe(params)
    with open(meta_path, 'w') as f:
        f.write(json.dumps(jparams, indent=2, sort_keys=True))

    # Log path
    print(file_name)

    # If a Drive ID has been passed, upload
    if drive_id is not None:
        print('Uploading.')
        for path, rtype in zip([meta_path, data_path], ['r', 'rb']):
            upload_file_to_google_drive(
                file_path=path,
                dest_id=drive_id,
                read_type=rtype,
            )
        print('Upload complete.')

    # Return filename
    return file_name


@restart
def _run_simulation(
        surface: Callable,
        surface_params: Optional[dict],
        optimizer,
        x_noise_func: Optional[Callable],
        x_noise_params: Optional[dict],
        y_noise_func: Optional[Callable],
        y_noise_params: Optional[dict],
        run_iter: int,
        seed: int = np.random.randint(1E8),
        run_length: Optional[int] = None,
        run_ftol: Optional[float] = None,
        run_ftol_length: Optional[float] = None,
        judge_ynoise: bool = True,
        auto_restart: bool = False,
        print_iter: bool = False,

):
    """
    Perform a single simulation run.
    :param surface: The name of the surface function to use. See the surfaces.py
    file for examples.
    :param surface_params: Any additional kwargs to pass to the surface
    function.
    :param x_noise_func: The type of noise to add to the input parameters before
    surface evaluation. Choice of 'gaussian'.
    :param x_noise_params: The size of the input noise.
    :param y_noise_func: The type of noise to add to the surface after
    evaluation. Choice of 'gaussian'.
    :param y_noise_params: The size of the noise to add to the surface.
    :param optimizer: The optimizer to use. Choice of Random,
    :param run_iter: The iteration of this run.
    :param run_length: The number of samples to include in one run.
    :param run_ftol: Perform the optimization until run_ftol is observed.
    :param run_ftol_length: If ftol and opt is deterministic, batch size.
    :param judge_ynoise: When adjudicating the performance of the simulation,
    should the ynoise be considered? If false, the y value without noise is
    used.
    :param auto_restart: If true, simulation runs that throw an exception are
    automatically restarted.
    :param seed: a seed for this instance
    :param print_iter: Print on each iteration.
    :return: The simulation result as a dataframe
    """

    # Create flags to stop the simulation
    halt = False
    iters = 0

    # Determine the y_dim
    if 'y_dim' in surface_params:
        y_dim = surface_params['y_dim']
    else:
        y_dim = 1

    # Create a place to store results
    xs_ideal = np.zeros(shape=(0, optimizer.get_xdim()))
    xs_real = np.zeros(shape=(0, optimizer.get_xdim()))
    ys_ideal = np.zeros(shape=(0, y_dim))
    ys_real = np.zeros(shape=(0, y_dim))

    # If the optimizer technique is deterministic, then sample the optimizer and
    # function in one large batch. This improves performance for large runs.
    deterministic = ['random', 'sobol', 'grid']

    # If deterministic, set limit
    if optimizer.name in deterministic:
        n = max(i for i in [run_length, run_ftol_length] if i is not None)
        halt = True
    else:
        n = 1

    # If optimizer should have a new seed, generate
    seeders = ['qEHVI', 'qNEHVI']
    if optimizer.name in seeders:
        optimizer.seed = seed

    # Continuously
    while True:

        x_ideal, x_real, y_ideal, y_real = _run_iteration(
            surface=surface,
            surface_params=surface_params,
            optimizer=optimizer,
            run_length=run_length,
            x_noise_func=x_noise_func,
            x_noise_params=x_noise_params,
            y_noise_func=y_noise_func,
            y_noise_params=y_noise_params,
            n=n,
        )

        # if using ftol
        if run_ftol is not None:

            # Determine how the end of the experiment should be triggered
            y_trigger = y_real if judge_ynoise else y_ideal

            # Determine if the simulation has finished
            if np.any(y_trigger >= run_ftol):

                # Mark the simulation as finished
                halt = True

                # Determine the index to crop to and crop
                idx = np.argmax(y_real >= run_ftol) + 1
                x_ideal = x_ideal[:idx]
                x_real = x_real[:idx]
                y_ideal = y_ideal[:idx]
                y_real = y_real[:idx]

        # If not using ftol
        else:

            # Increment the iteration and check for finish
            if iters + 1 == run_length:
                halt = True

        # Store
        xs_ideal = np.concatenate((xs_ideal, x_ideal))
        xs_real = np.concatenate((xs_real, x_real))
        ys_ideal = np.concatenate((ys_ideal, y_ideal))
        ys_real = np.concatenate((ys_real, y_real))

        # Increment the iteration
        iters += 1

        # Print state
        if print_iter:
            print(f'ID: {run_iter} iter: {iters} x: {x_ideal} yi: y{y_ideal} yr: {y_real}')

        # Stop if terminated
        if halt:
            break

    # Prepare data for dataframe
    data = np.concatenate((xs_ideal, xs_real, ys_ideal, ys_real), axis=1)
    x_dim = optimizer.get_xdim()
    names = []
    for p, n in zip(['x', 'y'], [x_dim, y_dim]):
        for t in ['ideal', 'real']:
            for i in range(n):
                names.append(f'{p}{i}_{t}')

    # Prepare dataframe
    df = pd.DataFrame(columns=names, data=data)
    df = df.reset_index()
    df = df.rename(columns=dict(index='sample'))
    df.insert(loc=0, column='run', value=run_iter)

    # Return the data
    return df


def _run_iteration(
        surface: Callable,
        surface_params: Optional[dict],
        optimizer,
        run_length: Optional[int],
        x_noise_func: Optional[Callable],
        x_noise_params: Optional[dict],
        y_noise_func: Optional[Callable],
        y_noise_params: Optional[dict],
        n: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a single simulation iteration

    :param surface: The name of the surface function to use. See the surfaces.py
    file for examples.
    :param surface_params: Any additional kwargs to pass to the surface
    function.
    :param run_length: The number of samples to include in one run.
    :param x_noise_func: The type of noise to add to the input parameters before
    surface evaluation. Choice of 'gaussian'.
    :param x_noise_params: The size of the input noise.
    :param y_noise_func: The type of noise to add to the surface after
    evaluation. Choice of 'gaussian'.
    :param y_noise_params: The size of the noise to add to the surface.
    :param optimizer: The optimizer to use. Choice of Random,
    :return: x_ideal, x_real, y_ideal, y_real
    """

    # Get the params from the optimizer as an array
    x_ideal = optimizer.get_x(n=n)

    # If x noise, apply
    if x_noise_func is not None:
        x_real = x_noise_func(x=x_ideal, **x_noise_params)
    else:
        x_real = x_ideal

    # Evaluate the surface. If it's a model, expect to receive meta data.
    y_response = surface(
        x=x_real,
        **surface_params,
    )

    # If metadata was returned, store
    if isinstance(y_response, tuple):
        y_ideal = y_response[0]
        y_meta = y_response[1]
    else:
        y_ideal = y_response
        y_meta = dict()

    # If 1D, make 2D
    if len(y_ideal.shape) == 1:
        y_ideal = y_ideal[:, None]

    # If y noise, apply
    if y_noise_func is not None:
        y_real = y_noise_func(
            x=y_ideal,
            **y_noise_params,
            **y_meta,
        )
    else:
        y_real = y_ideal

    # Return the result to the optimizer.
    optimizer.set_xy(
        x=x_real,
        y=y_real,
    )

    # Return
    return x_ideal, x_real, y_ideal, y_real


if __name__ == '__main__':
    pass
