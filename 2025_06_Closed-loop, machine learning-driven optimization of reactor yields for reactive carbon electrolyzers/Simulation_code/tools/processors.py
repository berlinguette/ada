from tools.engine import calc_enhancement, calc_acceleration,\
    calc_running_best, c, colorset, geometric_mean, geometric_std
from typing import Optional, List, Callable, Union, Tuple
from importlib import import_module
import pathos.multiprocessing as mp
from pygmo.core import hypervolume
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import pickle
import psutil
import json
import copy
import os


def data_reader(
        paths: Union[str, List[str]],
        combine_check: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    For paths or paths, read in the experimental data. If multiple paths have
    been passed, combine the datasets. Confirm that it is appropriate to combine
    them
    :param paths: The paths or paths to import
    :param combine_check: Should the datasets be checked if they are similar
    optimizations before they are combined
    :return: Data, metadata
    """

    # Homogenize the datatype of paths
    paths = np.array([paths] if isinstance(paths, str) else paths)

    # Create a place to store the data
    datas = list()
    metas = list()

    # Read in the optimization data
    for path in paths:

        # Load the data
        data = pd.read_feather(
            os.path.join('results', f'{path}.feather')
            # os.path.join('D:\\GitLab_ada\\simulations\\scripts\\MM\\spraycoater_modelling\\results', f'{path}.feather')
        )
        with open(os.path.join('results', f'{path}.json'), 'rb') as f:
        # with open(os.path.join('D:\\GitLab_ada\\simulations\\scripts\\MM\\spraycoater_modelling\\results',
        #                        f'{path}.json'), 'rb') as f:
            meta: dict = json.load(f)

        # Store
        datas.append(data)
        metas.append(meta)

    # If just one, return
    if len(paths) == 1:
        return datas[0], metas[0]

    # If multiple
    else:

        # If checking for consistency
        if combine_check:

            # Define the keys to not compare
            not_compare = [
                'auto_restart',
                'multiprocessing',
                'run_repeats',
                'time_start',
                'time_stop',
                'x_max',
                'x_min',
            ]

            # Confirm that the meta data is the same
            for meta in metas:
                for key in meta.keys():
                    if key not in not_compare:
                        assert meta[key] == metas[0][key]

        # Combine the data
        data_combined = pd.DataFrame()
        offset = 0

        # For each data, add
        for data in datas:

            # Offset the run count
            data['run'] += offset

            # Combine
            data_combined = pd.concat(
                [data_combined, data],
                ignore_index=True,
            )

            # Update the offset
            offset = max(data_combined['run']) + 1

        # Update the meta data
        meta_combined = metas[0]
        meta_combined['run_repeats'] = max(data_combined['run']) + 1

        # Return data
        return data_combined, meta_combined


class Proc(object):
    """
    A data class describing one simulation.
    """
    def __init__(
            self,
            paths: Union[str, List[str]],
            noise: str = 'real',
            combine_check: bool = True,
            mp: bool = True,
    ):
        """
        Load in a data class and process it.
        :param paths: The path to the file to load, excluding the .csv /.json
        :param noise: Either 'real' or 'ideal'. Determines if the noisy (real)
        or noiseless (ideal) observations should be used in processing.
        :param combine_check: Should the datasets be checked if they are similar
        optimizations before they are combined
        :param mp: should multiprocessing be used to calculate the hypervolume.
        """

        # Store noise mode
        noise_modes = ['real', 'ideal']
        if noise not in noise_modes:
            raise ValueError(f'noise must be in {noise_modes}')
        self.noise = noise

        # Define all metadata
        self.multiprocessing: Optional[bool] = None
        self.optimizer: Optional[str] = None
        self.optimizer_params: Optional[dict] = None
        self.x_dim: Optional[int] = None
        self.y_dim: Optional[int] = None
        self.x_max: Optional[float] = None
        self.x_min: Optional[float] = None
        self.run_length: Optional[int] = None
        self.run_repeats: Optional[int] = None
        self.run_ftol: Optional[float] = None
        self.surface: Optional[str] = None
        self.surface_params: Optional[dict] = None
        self.time_start: Optional[str] = None
        self.time_stop: Optional[str] = None
        self.x_noise_func: Optional[str] = None
        self.x_noise_params: Optional[dict] = None
        self.y_noise_func: Optional[str] = None
        self.y_noise_params: Optional[dict] = None
        self.mp = mp

        # Define other attributes
        self.performance: Optional[np.ndarray] = None

        # Read in the data
        self.data, self.meta = data_reader(
            paths=paths,
            combine_check=combine_check,
        )

        # Make metadata available as attributes
        for key, value in self.meta.items():
            setattr(self, key, value)

        # Convert parameters to integers
        self.run_repeats = int(self.run_repeats)
        self.run_ftol = float(self.run_ftol) if self.run_ftol != 'None' else None
        self.x_dim = int(self.x_dim)
        self.y_dim = int(self.y_dim)
        if self.x_min is not None:
            self.x_min = np.fromstring(self.x_min[1:-1], dtype=float, sep=' ')
        if self.x_max is not None:
            self.x_max = np.fromstring(self.x_max[1:-1], dtype=float, sep=' ')

        # Set the attributes from dictionaries as accessible
        for dstring in ['x_noise_params', 'y_noise_params', 'surface_params']:
            obj = getattr(self, dstring)
            if type(obj) is dict:
                for key, value in obj.items():
                    setattr(self, f'{dstring}_{key}', value)

        # If the simulation was a multiobjective optimization, then the metric
        # of interest is the hypervolume. Otherwise use y value.
        if self.y_dim == 1:
            self.performance = self.data[f'y0_{self.noise}'].to_numpy().reshape(
                self.run_repeats,
                self.run_length,
            )
        else:
            print('Calculating hypervolumes')
            if self.mp:
                self.performance = self._calc_hypervolume_mp()
            else:
                self.performance = self._calc_hypervolume().reshape(
                    self.run_repeats,
                    self.run_length,
                )

        # Get mean, std performance
        self.performance_mean = self.performance.mean(axis=0)
        self.performance_std = self.performance.std(axis=0)

        # If the dataset is square and complete, perform relevant statistics
        if not self.run_ftol:

            # Convert parameters to integers
            self.run_length = int(self.run_length)

            # Calculate running best
            self.best, self.best_mean, self.best_std = calc_running_best(self.performance)
            self.best_median = np.median(self.best, axis=0)

            # Calculate the interquartile range for best
            self.best_iqr_min = np.percentile(self.best, 25, axis=0)
            self.best_iqr_max = np.percentile(self.best, 75, axis=0)

            # Define domain for plotting
            self.domain = np.arange(self.run_length)

        # If using ftol
        else:

            # Store the sample counts required to get to ftol
            self.ftol_count = list()
            for i in self.data['run'].unique():
                run = self.data[self.data['run'] == i]
                self.ftol_count.append(run['sample'].iloc[-1])
            self.ftol_count = np.array(self.ftol_count)

            # Calc stats
            self.ftol_count_mean = self.ftol_count.mean()
            self.ftol_count_std = self.ftol_count.std()
            self.ftol_count_iqr_min, self.ftol_count_iqr_max = np.percentile(self.ftol_count, q=[25, 75])

        # Calculate the hardness of the surface. If a model is used, don't
        # evaluate. Models use the evaluate method.
        if self.surface != 'evaluate':
            surfaces = import_module('tools.surfaces')
            surface = getattr(surfaces, self.surface)
            # self.hardness = hardness(
            #     surface=surface,
            #     param_count=self.x_dim,
            #     param_min=self.x_min,
            #     param_max=self.x_max,
            #     surface_params=self.surface_params,
            # )

    def plot_best(
            self
    ):
        """
        Make a quick plot of the progress.
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Plot running best
        ax.plot(
            self.domain,
            self.best_mean,
        )
        ax.fill_between(
            self.domain,
            self.best_mean - self.best_std,
            self.best_mean + self.best_std,
            alpha=0.3,
        )

        # Format bounds based on mean data
        mean_min = np.min(self.best_mean)
        mean_max = np.max(self.best_mean)
        mean_span = mean_max - mean_min
        pbuff = 0.1
        pmin = mean_min - pbuff * mean_span
        pmax = mean_max + pbuff * mean_span
        ax.set_ylim(pmin, pmax)

        # Show
        plt.show()

    def plot_hardness(
            self
    ):
        """
        Plot the hardness of the surface used in this simulation.
        :return: None.
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Plot hardness
        ax.plot(
            self.hardness,
            c=c.blue.i800,
        )

        # Format
        length = len(self.hardness)
        ax.set_xlim(0, length)
        ax.set_xticks([0, length])
        ax.set_xticklabels(['worst', 'best'])
        ax.set_xlabel('surface samplings')
        ax.set_ylim(self.hardness[0], self.hardness[-1])
        ax.set_ylabel('surface value')

        # Show
        plt.show()

    def _calc_hypervolume(self):
        """
        Calculate the hypervolume of the dataset. All simulations assume a
        reference point of 0.
        :return: numpy array
        """

        # Get the data
        cols = np.array([f'y{i}_{self.noise}' for i in range(self.y_dim)])

        # Create a place to store the hypervolumes
        hypervolumes = []

        # For each run
        for run in tqdm(self.data['run'].unique()):
            rdata = self.data[self.data['run'] == run]

            # For each sample
            for sample in rdata['sample'].unique():
                sdata = rdata[rdata['sample'] <= sample]

                # Prepare data for calculating the hypervolume
                data = sdata[cols].to_numpy()
                ref_point = np.array([0] * data.shape[1])

                # If a point is outside or equal to the reference point, remove it.
                valid = ~np.all(data <= 0, axis=1)
                data = data[valid]

                # If any valid data points, calculate hv.
                if data.shape[0] > 0:

                    # If points are below zero, crop to zero
                    data = data.clip(min=0)

                    # Calculate hypervolume
                    hv = hypervolume(-data)
                    vol = hv.compute(ref_point=ref_point)

                # If no valid data points, volume is zero.
                else:
                    vol = 0

                # Store
                hypervolumes.append(vol)

        # Return
        return np.array(hypervolumes)

    def _calc_hypervolume_mp(self):
        """
        Same as _calc_hypervolume, but uses multiprocessing.
        :return: hv
        """

        # Get the data columns
        columns = np.array([f'y{i}_{self.noise}' for i in range(self.y_dim)])

        def worker(
                d: pd.DataFrame,
                cols=columns,
        ):
            """
            The worker called in multiprocessing.
            :param d: That run data
            :param cols: columns to use.
            :return: hvs
            """

            # Place to store hypervolumes
            hypervolumes = np.empty(len(d))

            # For each sample
            for i, sample in enumerate(d['sample'].unique()):
                sdata = d[d['sample'] <= sample]

                # Prepare data for calculating the hypervolume
                data = sdata[cols].to_numpy()
                ref_point = np.array([0] * data.shape[1])

                # If a point is outside or equal to the reference point, remove it.
                valid = ~np.all(data <= 0, axis=1)
                data = data[valid]

                # If any valid data points, calculate hv.
                if data.shape[0] > 0:

                    # If points are below zero, crop to zero
                    data = data.clip(min=0)

                    # Calculate hypervolume
                    hv = hypervolume(-data)
                    vol = hv.compute(ref_point=ref_point)

                # If no valid data points, volume is zero.
                else:
                    vol = 0

                # Store
                hypervolumes[i] = vol

            # Return
            return hypervolumes

        # Create the pool
        pool = mp.Pool(
            processes=psutil.cpu_count(
                logical=False,
            ) * 2
        )

        # Create a place to store results
        results = list()

        # For each run
        for run in self.data['run'].unique():

            # Select the data
            rdata = self.data[self.data['run'] == run]

            # Create the jobs
            results.append(
                pool.apply_async(
                    worker,
                    kwds=dict(
                        d=rdata,
                    )
                )
            )

        # Wait for results
        hv = np.empty((self.run_repeats, self.run_length))
        for i, result in enumerate(tqdm(results)):
            hv[i] = result.get()

        # Close pool
        pool.close()
        pool.join()

        # Return
        return hv


def cache_proc(
        use_cache: bool = True,
        **kwargs,
) -> Proc:
    """
    Use caching to load Proc class
    :rtype: Proc
    :param use_cache: Should the cache be used.
    :param kwargs: args for Proc
    :return:
    """
    # Create a uid
    paths = kwargs['paths']
    if type(paths) is str:
        uid = paths
    else:
        uid = '*'.join(paths)

    # If the processing should be regenerated
    if not use_cache or not os.path.exists(f'results/{uid}.pickle'):

        # Regenerate
        proc = Proc(**kwargs)

        # Save
        with open(f'results/{uid}.pickle', 'wb') as f:
            pickle.dump(proc, f)

    # Otherwise, load from disk
    else:

        # Load from disk
        with open(f'results/{uid}.pickle', 'rb') as f:
            proc = pickle.load(f)

    # Return
    return proc


def combine_procs(
        procs: List[Proc],
) -> Proc:

    # Make a base proc to update
    proc = copy.copy(procs[0])

    # For each proc, save the first one
    for iproc in procs[1:]:

        # Attributes to stack vertically
        vstack_attr = ['performance']

        # Attributes to add
        add_attr = ['run_repeats']

        # For each attribute to add
        for attr in add_attr:

            # For each attribute to be added
            total_val = getattr(proc, attr)
            new_val = getattr(iproc, attr)
            total_val += new_val
            setattr(
                proc,
                attr,
                total_val,
            )

        # For each attribute to stack vertically
        for attr in vstack_attr:

            # New value
            total_val = getattr(proc, attr)
            new_val = getattr(iproc, attr)
            total_val = np.vstack([total_val, new_val])

            # Store
            setattr(
                proc,
                attr,
                total_val,
            )

    # If square and complete
    if not proc.run_ftol:

        # Recalculate the running best
        proc.best, proc.best_mean, proc.best_std = calc_running_best(proc.performance)
        proc.best_median = np.median(proc.best, axis=0)

        # Calculate the interquartile range for best
        proc.best_iqr_min = np.percentile(proc.best, 25, axis=0)
        proc.best_iqr_max = np.percentile(proc.best, 75, axis=0)

    # Recalculate the mean, std
    proc.performance_mean = proc.performance.mean(axis=0)
    proc.performance_std = proc.performance.std(axis=0)

    # Return
    return proc


def load_paths(
        paths: Union[str, List[str]],
        **kwargs,
) -> Proc:

    # If just one path, return
    if isinstance(paths, str):
        proc = cache_proc(
            use_cache=True,
            paths=paths,
            **kwargs,
        )

    # If a list
    else:

        # Check all the same experiments
        combine_check(paths)

        # Create a place to store the procs
        procs = []

        # For each path
        for path in paths:

            # Cache load
            procs.append(cache_proc(paths=path, **kwargs))

        # Combine
        proc = combine_procs(procs)

    # Return
    return proc



class ProcPair(object):
    """
    Compare two runs.
    """

    def __init__(
            self,
            path_slow: str,
            path_fast: str,
            use_cache: bool = True,
    ):
        """
        Create a comparison of two runs
        :param path_slow: Path to the slower run, excluding the .csv / .json
        :param path_fast: Path to the faster run, excluding the .csv / .json
        :param use_cache: Should the acceleration cache be used.
        """

        # Constants
        self.acceleration = Optional[np.ndarray]

        # Store runs
        self.sim_slow = Proc(paths=path_slow)
        self.sim_fast = Proc(paths=path_fast)
        self.sims = [self.sim_slow, self.sim_fast]

        # If symmetric, perform statistics
        if not self.sim_slow.run_ftol and not self.sim_fast.run_ftol:

            # Calc acceleration, and look for a cache
            cache_path = f'cache/{path_slow}_{path_fast}.pkl'
            if use_cache and os.path.exists(cache_path):
                print('FAST')
                with open(cache_path, 'rb') as f:
                    self.acceleration, self.acceleration_gmean = pickle.load(f)

            else:
                print('SLOW')
                self.acceleration, self.acceleration_gmean = \
                    calc_acceleration(
                        a=self.sim_fast.best,
                        b=self.sim_slow.best,
                        multiprocessing=True,
                    )
                with open(cache_path, 'wb') as f:
                    pickle.dump((self.acceleration, self.acceleration_gmean), f)

            # Calculate enhancement
            self.enhancement, self.enhancement_gmean, self.enhancement_gstd = \
                calc_enhancement(
                    a=self.sim_fast.best,
                    b=self.sim_slow.best,
                )

            # Calculate the interquartile range for the accelerations and
            # enhancements. Suppress all nan warnings, as they are well handled.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)

                self.acceleration_iqr_min, self.acceleration_iqr_max = \
                    np.nanpercentile(self.acceleration, q=[25, 75], axis=0)
                self.enhancement_iqr_min, self.enhancement_iqr_max = \
                    np.nanpercentile(self.enhancement, q=[25, 75], axis=0)

            # Calculate medians
            self.acceleration_median = np.median(self.acceleration, axis=0)
            self.enhancement_median = np.median(self.enhancement, axis=0)

            # Store the acceleration nan fraction
            valid_sum = np.sum(~np.isnan(self.acceleration), axis=0)
            self.valid_frac = valid_sum / self.acceleration.shape[0]

            # The domain of a ProcPair is the minimum domain of it's contents
            self.run_length = min([sim.run_length for sim in self.sims])
            self.domain = np.arange(self.run_length)

        # If ftol
        else:

            # Calculate the acceleration
            slow_count = self.sim_slow.ftol_count + 1
            fast_count = self.sim_fast.ftol_count + 1
            accelerations = (slow_count / fast_count[:, None])
            self.ftol_accel = accelerations.flatten()

            # Calculate the statistics
            self.ftol_accel_gmean = geometric_mean(self.ftol_accel)
            self.ftol_accel_gstd = geometric_std(self.ftol_accel)
            self.ftol_accel_iqr_min, self.ftol_accel_iqr_max = np.percentile(self.ftol_accel, q=[25, 75])

    def plot_acceleration_dist(self):
        """
        Plot the distribution of accelerations
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Plot maxima
        ax.plot(
            self.domain,
            self.sim_slow.run_length / (self.domain + 1),
            label='observable max',
            c=c.gray.i800
        )

        # Plot points
        scat = None
        for i in range(self.run_length):
            scat = ax.scatter(
                np.repeat(i, self.acceleration.shape[0]),
                self.acceleration[:, i],
                c=c.gray.i800,
                s=2,
                alpha=0.02,
            )
        scat.set_label('observed')

        # Plot valid frac
        ax_frac: plt.Axes = ax.twinx()
        ax_frac.plot(
            self.domain,
            self.valid_frac,
            c=c.red.i800,
        )

        # Format valid
        ax_frac.set_ylim(0, 1)
        ax_frac.set_yticks([])
        ax_frac.set_ylabel('valid fraction')
        ax_frac.yaxis.label.set_color(c.red.i800)

        ax.set_ylim(
            float(np.nanmin(self.acceleration)),
            float(np.nanmax(self.acceleration)),
        )
        ax.set_xlabel('samples')
        ax.set_ylabel('acceleration')
        ax.set_xlim(0, self.run_length)
        ax.legend()
        plt.show()

    def plot_acceleration_slice(
            self,
            n: int
    ):
        """
        Plot the distributions of the accelerations at a given point.
        :param n: Sample int to slice at
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        ax.hist(
            self.acceleration[:, n],
            label=f'Sample {str(n)}',
            bins=100,
            edgecolor=c.blue.i800,
            facecolor=c.blue.i300,
            # range=(0, 5),
        )

        # Format
        ax.set_xlabel('acceleration')
        ax.set_ylabel('frequency')
        ax.legend()

        plt.show()

    def plot_acceleration(self):
        """
        Make a quick plot of the acceleration
        :return: None
        """

        # Make plot objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Plot mean
        ax.plot(
            self.domain,
            self.acceleration_gmean,
            c=c.blue.i800,
            linestyle=':',
            label='mean'
        )

        # Plot iqr
        ax.fill_between(
            self.domain,
            self.acceleration_iqr_min,
            self.acceleration_iqr_max,
            color=c.blue.i800,
            alpha=0.3,
        )

        # Indicate which fraction is not nan
        ax_frac: plt.Axes = ax.twinx()
        ax_frac.plot(
            self.domain,
            self.valid_frac,
            c=c.red.i800,
        )

        # Add a horizontal line for 1 accel
        ax.axhline(
            1,
            c=c.gray.i800,
            alpha=0.3,
            linestyle=':',
        )

        # Format
        ax_frac.set_ylim(0, 1)
        ax_frac.set_yticks([])
        ax_frac.set_ylabel('valid fraction')
        ax_frac.yaxis.label.set_color(c.red.i800)

        ax.legend()
        ax.set_xlabel('samples')
        ax.set_ylabel('acceleration')
        ax.set_xlim(self.domain[0], self.domain[-1])

        # Show
        plt.show()

    def plot_enhancement(self):
        """
        Make a quick plot of the enhancements
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Plot mean
        ax.plot(
            self.domain,
            self.enhancement_gmean,
            c=c.blue.i800,
            linestyle=':',
            label='mean',
        )

        # Plot iqr
        ax.fill_between(
            self.domain,
            self.enhancement_iqr_min,
            self.enhancement_iqr_max,
            color=c.blue.i800,
            alpha=0.3
        )

        # Add a horizontal line for 1 accel
        ax.axhline(
            1,
            c=c.gray.i800,
            alpha=0.3,
            linestyle=':',
        )

        # Format
        ax.legend()
        ax.set_xlabel('samples')
        ax.set_ylabel('enhancement')
        ax.set_xlim(self.domain[0], self.domain[-1])

        # Show
        plt.show()

    def plot_bests(self):
        """
        Make a quick plot of the running bests.
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Iter formatting
        color = [c.red, c.blue]
        label = ['slow', 'fast']

        # For both runs
        for i, run in enumerate([self.sim_slow, self.sim_fast]):

            # Plot median
            ax.plot(
                run.domain,
                run.best_median,
                c=color[i].i800,
                label=f'{label[i]} median',
            )

            # Plot mean
            ax.plot(
                run.domain,
                run.best_mean,
                c=color[i].i800,
                linestyle=':',
                label=f'{label[i]} mean',
            )

            # Plot iqr
            ax.fill_between(
                run.domain,
                run.best_iqr_min,
                run.best_iqr_max,
                color=color[i].i800,
                alpha=0.3,
            )

            # Format x axis
            ax.set_xlim(0, run.run_length)

        # Format
        ax.legend(loc='lower right')
        ax.set_xlabel('samples')
        ax.set_ylabel('best observed')

        # Show
        plt.show()

    def plot_hardness(self):
        self.sim_slow.plot_hardness()

    def get_attr(
            self,
            attr,
    ):
        """
        Get an attribute from the pair
        :param attr: attribute
        :return: attribute
        """
        if hasattr(self, attr):
            return getattr(self, attr)

        # check if asymmetric
        if attr[:5] in ['slow_', 'fast_']:
            return getattr(getattr(self, f'sim_{attr[:4]}'), attr[5:])

        else:
            slow_attr = getattr(self.sim_slow, attr)
            fast_attr = getattr(self.sim_fast, attr)
            if slow_attr != fast_attr:
                raise ValueError('Attributes are not equal')
            else:
                return slow_attr


class ProcSet(object):
    """
    A series of SimPairs
    """
    def __init__(
            self,
            var: str,
            easy_load: Optional[int] = None,
            paths: Optional[np.ndarray] = None,
    ):
        """
        Create a set of simulation comparisons for simple trend plotting. Either
        easy_load or paths argument must be passed.
        :param easy_load: Loads the last n experiments from the results folder.
        Note that these experiments must be performed in the order of slow_a,
        fast_a, slow_b, fast_b, etc. This number must be even.
        :param paths: An array of strings of IDs of shape n_var x 2, where the
        paths[:, 0] are the slow IDs, and paths[:, 1] are the fast IDs.
        :param var: The Proc attribute that the SimPairs are varied by. Must be
        an attribute of Proc.
        """

        # Store the variable of interest
        self.var: str = var

        # If easy-loading, generate the paths array as if paths were passed as
        # an argument.
        if easy_load is not None:

            # Get directory list
            dirs = np.sort(os.listdir('results'))

            # Get just csv names
            dirs = [d for d in dirs if 'feather' in d]

            # Remove .csv
            dirs = [d.split('.')[0] for d in dirs]

            # Get path names of interest
            paths = np.array(dirs)[-easy_load:].reshape(int(easy_load/2), 2)

        # Save the paths as pairs
        self.simpairs: List[ProcPair] = []
        for pair in paths:
            self.simpairs.append(ProcPair(pair[0], pair[1]))

        # Store the variables
        self.var_iter = list()
        self.ftol_accel_gmean = list()
        self.ftol_accel_gstd = list()

        for simpair in self.simpairs:
            if getattr(simpair.sim_slow, var) != getattr(simpair.sim_fast, var):
                raise ValueError(f'Variable {var} varies within simpair.')
            self.var_iter.append(getattr(simpair.sim_slow, var))

            # Store the ftol accelerations
            if simpair.sim_slow.run_ftol:
                self.ftol_accel_gmean.append(simpair.ftol_accel_gmean)
                self.ftol_accel_gstd.append(simpair.ftol_accel_gstd)

        # Define count
        self.count = len(self.simpairs)

    def plot_hardnesses(self):
        """
        Plot the hardness of the surfaces.
        :return: None.
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # Store running min, max
        rmin = np.inf
        rmax = -np.inf

        # For each of the surfaces
        for i, pair in enumerate(self.simpairs):

            # Plot
            ax.plot(
                pair.sim_slow.hardness,
                c=colorset(i, self.count),
                label=self.var_iter[i],
            )

            # Update pmin, pmax
            rmin = min(rmin, np.min(pair.sim_slow.hardness))
            rmax = max(rmax, np.min(pair.sim_slow.hardness))

        # Format
        ax.legend()
        length = len(self.simpairs[0].sim_slow.hardness)
        ax.set_xlim(0, length)
        ax.set_xticks([0, length])
        ax.set_xticklabels(['worst', 'best'])
        ax.set_xlabel('surface samplings')
        ax.set_ylim(rmin, rmax)
        ax.set_ylabel('surface value')

        # Show
        plt.show()

    def plot_bests(self):
        """
        Plot the running bests
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure(figsize=(10, 4.8))
        ax_0: plt.Axes = figure.add_subplot(1, 2, 1)
        ax_1: plt.Axes = figure.add_subplot(1, 2, 2)
        axs = [ax_0, ax_1]

        # For both the slow and fast simulations
        for i in range(2):

            # For each variable:
            for j in range(self.count):

                # Get the simulation to plot
                sim = self.simpairs[j].sims[i]
                color = colorset(j, self.count)

                # Plot median
                axs[i].plot(
                    sim.domain,
                    sim.best_median,
                    c=color,
                    label=self.var_iter[j]
                )

                # Plot mean
                axs[i].plot(
                    sim.domain,
                    sim.best_mean,
                    c=color,
                    linestyle=':',
                )

                # Plot iqr
                axs[i].fill_between(
                    sim.domain,
                    sim.best_iqr_min,
                    sim.best_iqr_max,
                    color=color,
                    alpha=0.3,
                )

        # Format
        ax_labels = ['slow', 'fast']
        for i, ax in enumerate(axs):
            ax.legend()
            ax.set_title(ax_labels[i])
            ax.set_xlabel('samples')
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
        axs[0].set_ylabel('performance')
        figure.subplots_adjust(
            left=0.1,
            right=0.95,
        )

        # Show
        plt.show()

    def plot_accelerations(self):
        """
        Plot the accelerations
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        # For each variable
        for i in range(self.count):

            # Get iter constants
            simpair = self.simpairs[i]
            color = colorset(i, self.count)
            length = len(simpair.acceleration_median)
            domain = np.arange(length)

            # Plot the median
            ax.plot(
                domain,
                simpair.acceleration_median,
                c=color,
                label=self.var_iter[i],
            )

            # Plot the mean
            ax.plot(
                domain,
                simpair.acceleration_gmean,
                c=color,
                linestyle=':',
            )

            # Plot iqr
            ax.fill_between(
                domain,
                simpair.acceleration_iqr_min,
                simpair.acceleration_iqr_max,
                color=color,
                alpha=0.3,
            )

        # Format
        ax.legend()
        ax.set_xlabel('sample')
        ax.set_xlim(0, max([pair.run_length for pair in self.simpairs]))
        ax.set_ylabel('acceleration')

        # Show
        plt.show()

    def plot_ftol_accelerations(self):
        """
        Plot the accelerations calculated from ftol
        :return: None
        """

        # Make figure objects
        figure: plt.Figure = plt.figure()
        ax: plt.Axes = figure.add_subplot()

        ax.scatter(
            self.var_iter,
            self.ftol_accel_gmean,
            c=c.blue.i800,
        )

        # Plot std
        for i in range(self.count):
            mean = self.ftol_accel_gmean[i]
            std = self.ftol_accel_gstd[i]
            ax.plot(
                [self.var_iter[i]] * 2,
                [mean - std, mean + std],
                color=c.blue.i800,
            )

        # Format
        ax.set_xticks(self.var_iter)
        ax.set_xlabel(self.var)
        ax.set_ylabel('acceleration')

        # Show
        plt.show()


class ProcGrid(object):
    """
    A set if SimPairs where multiple variables change.
    """
    def __init__(
            self,
            ids: Optional[np.ndarray] = None,
            min_date: Optional[str] = None,
            max_date: Optional[str] = None,
    ):
        """
        A set if SimPairs where multiple variables change.

        :param ids: an array of ids of size n x 2.
        :param min_date: min date for import
        :param max_date: max date for import
        """

        # Create a place to store pairs
        self.pairs: List[ProcPair] = list()

        # If bounds passed, generate ids
        if min_date is not None or max_date is not None:

            # Generate time bounds
            stamp_format = '%Y-%m-%d_%H-%M-%S'
            stamp_min = datetime.strptime(min_date, stamp_format)
            stamp_max = datetime.strptime(max_date, stamp_format)

            # Test each file
            ids = list()
            for file in sorted(os.listdir('results')):
                if file.endswith('.json'):
                    str_test = file.split('.')[0]
                    stamp_test = datetime.strptime(str_test, stamp_format)
                    if stamp_min < stamp_test < stamp_max:
                        ids.append(str_test)

            # Format
            ids = np.array(ids).reshape(-1, 2)
            print(f'{len(ids.flatten())} detected')

        # Create pairs
        for i, id_pair in enumerate(ids):
            print(f'{i}/{len(ids)}')
            self.pairs.append(
                ProcPair(path_slow=id_pair[0], path_fast=id_pair[1])
            )

    def get_attr(
            self,
            attr,
    ) -> np.ndarray:
        """
        Get a attribute from the SimPairs
        :param attr: Attribute to select
        :return: array of attributes
        """

        return np.array([p.get_attr(attr) for p in self.pairs])

    def attr_table(
            self,
            attrs: List[str],
    ) -> pd.DataFrame:
        """
        Get a table of attributes
        :param attrs: List of
        :return:
        """

        return pd.DataFrame({attr: self.get_attr(attr) for attr in attrs})


def show_results(
        print_df: bool = True,
        save_df: bool = True,
        return_df: bool = True,
):
    """
    Visualize the data currently sitting in the results folder as a table.
    :param print_df: Print the df
    :param save_df: Save df to csv
    :param return_df: Return the df
    :return: Pandas dataframe of results
    """

    # Get list of json files
    files = [f for f in os.listdir('results') if f.endswith('.json')]

    # Read in jsons
    jsons = []
    for file in files:
        with open(f'results/{file}') as f:
            jsons.append(json.load(f))

    # Make dataframe
    df = pd.DataFrame(jsons)

    # Define the column order
    columns = [
        'time_start',
        'optimizer',
        'x_dim',
        'run_length',
        'run_repeats',
        'surface',
        'x_min',
        'x_max',
        'optimizer_params',
        'surface_params',
        'multiprocessing',
        'x_noise_func',
        'x_noise_params',
        'y_noise_func',
        'y_noise_params',
        'time_stop',
    ]

    # Select / order
    df = df[columns]

    # Sort
    df.sort_values(by=['time_start', 'optimizer', 'x_dim'], inplace=True)

    # Filter

    # Print
    if print_df:
        print(df)

    if save_df:
        df.to_csv('results.csv')

    # Return
    if return_df:
        return df


def hardness(
        surface: Callable,
        param_min: np.ndarray,
        param_max: np.ndarray,
        param_count: int = 2,
        surface_params: Optional[dict] = None,
        n_samples: int = int(1E6),
        normalize: bool = False,
        x_noise_func: Optional[Callable] = None,
        x_noise_params: Optional[dict] = None,
        y_noise_func: Optional[Callable] = None,
        y_noise_params: Optional[dict] = None,
):
    """
    Calculate the hardness of a surface.

    :param surface: A function to evaluate. Must take an argument x of shape
    n_samples x n_dimensions.
    :param param_count: The number of parameters to sample, or n_dimensions.
    :param param_min: The min bound of the parameter.
    :param param_max: The max bound of the parameter.
    :param surface_params: Other kwargs to pass to the surface.
    :param n_samples: The number of samplings to perform.
    :param normalize: Should the performance be normalized to 1.
    :param x_noise_func: The type of noise to add to the input parameters before
    surface evaluation. Choice of 'gaussian'.
    :param x_noise_params: The size of the input noise.
    :param y_noise_func: The type of noise to add to the surface after
    evaluation. Choice of 'gaussian'.
    :param y_noise_params: The size of the noise to add to the surface.
    :return: An array of the hardness.
    """

    # If no surface args, make empty dict
    surface_params = dict() if surface_params is None else surface_params

    # Generate the random samplings
    x = np.random.rand(n_samples, param_count)

    # Scale
    param_span = param_max - param_min
    x_ideal = x * param_span + param_min

    # If x noise, apply
    if x_noise_func is not None:
        x_real = x_noise_func(x=x_ideal, **x_noise_params)
    else:
        x_real = x_ideal

    # Evaluate the surface
    y_ideal = surface(x=x_real, **surface_params)[:, None]

    # If y noise, apply
    if y_noise_func is not None:
        y_real = y_noise_func(x=y_ideal, **y_noise_params)
    else:
        y_real = y_ideal

    # Sample
    # y = surface(x, **surface_params)

    # Sort
    y = np.sort(y_real.flatten())

    # Normalize
    if normalize:
        y = y / y[-1]

    # Return
    return y


def plot_hardness(
        surface: Callable,
        param_min: np.ndarray,
        param_max: np.ndarray,
        param_count: int = 2,
        surface_params: Optional[dict] = None,
        n_samples: int = int(1E6),
        normalize: bool = False,
):
    """
    Plot the hardness of a surface.

    :param surface: A function to evaluate. Must take an argument x of shape
    n_samples x n_dimensions.
    :param param_count: The number of parameters to sample, or n_dimensions.
    :param param_min: The min bound of the parameter.
    :param param_max: The max bound of the parameter.
    :param surface_params: Other kwargs to pass to the surface.
    :param n_samples: The number of samplings to perform.
    :param normalize: Should the performance be normalized to 1.
    :return: An array of the hardness.
    """

    # Sample the hardness
    h = hardness(
        surface=surface,
        param_count=param_count,
        param_min=param_min,
        param_max=param_max,
        surface_params=surface_params,
        n_samples=n_samples,
        normalize=normalize,
    )

    # Make figure objects
    figure: plt.Figure = plt.figure()
    ax: plt.Axes = figure.add_subplot()

    ax.plot(
        np.linspace(0, 1, n_samples),
        h,
        c=c.blue.i800,
    )

    # Format
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(h))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['min\nvalue', 'max\nvalue'])
    ax.set_xlabel('surface samplings', labelpad=-10)
    ax.set_ylabel('surface value')

    # Show
    plt.show()


def make_large_table(
        min_date: Optional[str],
        max_date: Optional[str],
        attrs: List[str],
        df_path: str,
):

    # Generate time bounds
    stamp_format = '%Y-%m-%d_%H-%M-%S'
    stamp_min = datetime.strptime(min_date, stamp_format)
    stamp_max = datetime.strptime(max_date, stamp_format)

    # Test each file
    ids = list()
    for file in sorted(os.listdir('results')):
        if file.endswith('.json'):
            str_test = file.split('.')[0]
            stamp_test = datetime.strptime(str_test, stamp_format)
            if stamp_min < stamp_test < stamp_max:
                ids.append(str_test)

    # Format
    ids = np.array(ids).reshape(-1, 2)
    print(f'{len(ids.flatten())} detected')

    # Get attrs
    data = list()
    for id_pair in tqdm(ids):
        pair = ProcPair(path_slow=id_pair[0], path_fast=id_pair[1])
        d = {attr_id: copy.deepcopy(pair.get_attr(attr_id)) for attr_id in attrs}
        data.append(d)
        del pair

        # Save
        df = pd.DataFrame(data)
        df.to_csv(df_path)


def plot_hypervolumes(
        procs: np.ndarray,
):
    """
    Plot the hypervolumes.
    :param procs: an array of dicts. The dicts have the keys "paths" (either str
    or list or str), and label (str).
    :return: plt.Figure
    """

    # For each proc
    for p in procs:

        # Process the data
        p['proc'] = cache_proc(
            use_cache=True,
            paths=p['paths'],
            noise='ideal',
        )

        # Get the last data point
        p['last'] = p['proc'].best_median[-1]

    # Sort the last based on the last data point
    sort = np.argsort([p['last'] for p in procs])
    procs = np.flip(procs[sort])

    # Add the colors
    for i, p in enumerate(procs):
        p['color'] = plt.get_cmap('magma')((i+1)/(len(procs) + 2))

    # Make the figure
    figure: plt.Figure = plt.figure()
    ax: plt.Axes = figure.add_subplot()

    # Define some constants for generating the leader labels
    last = np.inf
    buf = 0.03

    # Fractions
    y_buff = 1.04
    yt_buff = 1.05
    y0 = procs[0]['proc'].domain[-1] + 1
    y1 = y0 * y_buff
    yt = y0 * yt_buff

    # For each dataset
    max_len = 0
    for i, p in enumerate(procs):

        # Get processed data
        proc = p['proc']
        color = p['color']

        # Plot
        ax.plot(
            proc.domain,
            proc.best_median,
            c=color,
        )

        # IQR
        ax.fill_between(
            proc.domain,
            proc.best_iqr_min,
            proc.best_iqr_max,
            facecolor=color,
            edgecolor='none',
            alpha=0.2,
        )

        # Store max length
        max_len = max(max_len, proc.run_length)

        # Begin plotting the leader lines.  Calculate the left and right points.
        left_point = right_point = proc.best_median[-1]
        if last - buf < left_point:
            right_point = last - buf

        # Plot the leader line
        ax.plot(
            [y0, y1],
            [left_point, right_point],
            clip_on=False,
            color=color,
        )

        # Plot text
        ax.text(
            yt,
            right_point - 0.003,
            p['label'],
            horizontalalignment='left',
            verticalalignment='center',
            color=color,
            clip_on=False,
        )

        # Store last point
        last = right_point

    # Format
    ax.axvline(x=9, lw=1, linestyle=':', color=c.gray.i500, zorder=-1)
    ax.set_xlim(0, max_len)
    ax.set_ylim(bottom=0)
    for pos in ['top', 'right']:
        ax.spines[pos].set_visible(False)
    ax.set_xlabel('samples')
    ax.set_ylabel('hypervolume')
    figure.subplots_adjust(
        left=0.10,
        top=0.95,
        bottom=0.1,
        right=0.75,
    )
    figure.set_dpi(300)

    # Return
    return figure


def combine_check(paths: List[str]):

    metas = []

    # Define the keys to not compare
    not_compare = [
        'auto_restart',
        'multiprocessing',
        'run_repeats',
        'time_start',
        'time_stop',
        'x_max',
        'x_min',
    ]

    # Read meta data
    for path in paths:
        with open(os.path.join('results', f'{path}.json'), 'rb') as f:
        # with open(os.path.join('D:\\GitLab_ada\\simulations\\scripts\\MM\\spraycoater_modelling\\results',
        #                        f'{path}.json'), 'rb') as f:
            meta: dict = json.load(f)
        metas.append(meta)

    for i, meta in enumerate(metas):
        for key in meta.keys():
            if key not in not_compare:
                if meta[key] != metas[0][key]:
                    raise ValueError(f'{paths[i]} does not match {paths[0]} for {key}.')


if __name__ == '__main__':
    pass
