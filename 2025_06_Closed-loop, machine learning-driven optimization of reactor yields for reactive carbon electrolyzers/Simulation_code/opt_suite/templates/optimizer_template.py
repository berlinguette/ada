import numpy as np
from typing import Union, Optional, List
import logging
import json
import os
from abc import ABC, abstractmethod
from opt_suite.parameters import Parameter, DiscreteParameter
from opt_suite.helper_functions import convert_np_native
import warnings


logger = logging.getLogger()


class Progress:
    def __init__(self):
        """
        A data class that conveys the progress of the optimization at a given iteration.
        """
        # save the seed value of the np.random function
        self.seed = None

        # A set of points that are sampled on the emulated surface
        self.x_observed = []

        # The evaluated response of the set
        self.y_observed = []

        self.x_best = []
        self.y_best = []

        self.x_loss_real_best = []
        self.x_loss_norm_best = []
        self.y_loss_real_best = []

        self.x_loss_real = []
        self.x_loss_norm = []
        self.y_loss_real = []

        # A place to store statistics about the surface
        self.x_loss_worst: Optional[float] = None


class OptTemplate(ABC):
    """
    This class is a template of the requirements for each optimizer that would be
    made available to the OptimizerOrganizer.  For example, an instance of this
    class would be written for the grid search method.
    """

    def __init__(self, batch_size=1):

        # Create a place to store the x and y values from the optimization.
        # These are instantiated as numpy arrays on the first call to return()
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        # Define lists where parameters and objectives can be defined.
        self.params = []
        self.objectives = []
        self.dim = 0
        self.batch_size = batch_size
        self.bounds = []
        self.init_randoms = 53

        # Setup where to store the state of the optimizer
        self.progress = Progress()
        self.seed = None

        # Some optimisers already utilise multiprocessing so there is limited
        # speedup to running simulations in parallel use this to limit the number of cores
        # used by Runner, i.e: 1 = 1 * os.cpu_count(), 0.5 = 0.5 * os.cpu_count
        self.cpu_ratio_limit = 1

        # these are attributes that will be excluded when used with Runner()
        # generally used for Queues, loggers, random seeds and other generally not pickle-able objects
        self.exclude_attributes = ['logger']

    def add_parameter(
            self,
            name: str,
            pmin: float,
            pmax: float,
            step: float = None,
            domain: List[str] = None,
            **kwargs,
    ) -> None:

        # Create the parameter object
        if step or domain:
            param = DiscreteParameter(name, pmin, pmax, step, domain)
        else:
            param = Parameter(name, pmin, pmax)

        if param in self.params:
            raise ValueError(f'Parameter by the name {name} already exists')

        # Store the parameter
        self.params.append(param)

        # Calculate the bounds for the acq optimization.
        self.bounds = [
            (p.pmin, p.pmax)
            for p in self.params
        ]

    def add_objective(
            self,
            name: str,
            reference_point: float,
            minimize=False,
            dynamic=False,
    ) -> None:
        """
        Define a new objective.
        """

        # Confirm that the objective name does not exist
        assert name not in [obj['name'] for obj in self.objectives]

        # Create objective
        objective = dict(
            name=name,
            reference_point=reference_point,
            minimize=minimize,
            index=len(self.objectives),
            dynamic=dynamic,
        )

        # Store objective
        self.objectives.append(objective)

    def get_xdim(self) -> int:
        return len(self.params)

    def get_ydim(self) -> int:
        ob = len(self.objectives)
        return 1 if ob == 0 else ob

    def get_num_param(
            self,
    ) -> int:
        warnings.warn('get_num_param() is depreciated. Use get_xdim(). '
                      'This feature will be removed after Oct 31, 2021.')
        return self.get_xdim()

    def set_xy(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ):
        """
        Return an evaluated x value to the optimizer with the observed y value.
        :param x: of shape (n_samples, x_dim)
        :param y: of shape (n_samples, y_dim)
        :return: None
        """

        # Confirm that the data passed has the same number of samples
        assert x.shape[0] == y.shape[0]

        # If the arrays to store results are not yet instantiated, create them.
        if self.x is None:
            self.x = np.empty((0, self.get_xdim()))
        if self.y is None:
            self.y = np.empty((0, self.get_ydim()))

        # Store the data
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))

    def get_param_names(self):
        return [param.name for param in self.params]

    @abstractmethod
    def get_x(
            self,
            n: int = 1,
            **kwargs,
    ) -> np.ndarray:
        """
        Request from the optimizer the next n parameter sets to complete.
        :param n: If n = 1, then a single float value is returned for each
        parameter.  If n > 1, then a numpy array of n values is returned for
        each parameter.  if n = 0, then all relevant parameter sets will be
        returned. For example, if a grid search has been setup for 100 samples,
        requesting n = 0 will return 100 parameter sets to evaluate.
        """
        pass

    def return_result(self,
                      realized_params: Union[np.ndarray, np.ndarray],
                      result: Union[np.ndarray, np.ndarray]) -> None:
        """
        Update the optimizer with results from sampling the experimental
        surface.
        :param realized_params: A numpy array including the sampled parameter values. Either a single float or a numpy
        array.
        :param result: The result of sampling the surface.  Either a single
        float or a numpy array.
        realized_params and result
        """
        warnings.warn('return_result() is now depreciated. Use set_xy() '
                      'instead. This feature will be removed after Oct 31, 2021.')
        self.set_xy(
            x=realized_params,
            y=result,
        )

    def request_params(
            self,
            n: int = 1,
            **kwargs,
    ):
        warnings.warn('request_params() is now depreciated. Use get_x() instead.'
                      'This feature will be removed after Oct 31, 2021.')
        return self.get_x(
            n=n,
            **kwargs,
        )

    def get_count(self):
        return len(self.values)

    def get_results(self):
        return self.y

    def get_candidates(self):
        return self.x

    @property
    def max_result(self):
        return max(self.get_results(), default=None)

    def get_realized(self):
        return self.samples_real

    def get_requested(self):
        return self.samples_ideal

    def get_progress(
            self
    ) -> Progress:
        return self.progress

    def _get_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if k not in self.exclude_attributes:
                d[k] = v
        return d

    def add_prior_data(self,
                       emulator,
                       json_filename: Optional[str] = None,
                       data: Optional[dict] = None):

        # TODO: Add test for checking variable names and domain are the same

        if json_filename is not None and data is not None:
            raise logging.exception(f'OPT TEMPLATE: Must only specify either json file or provide data dictionary in '
                                    f'the same format as the json file.')

        elif json_filename is not None:
            try:
                # Load in the data
                with open(json_filename + '.json') as f:
                    data = json.load(f)
            except Exception as e:
                raise logging.exception(f'OPT TEMPLATE: {e}')

        prior_parameters_names = [prior['name'] for prior in data.get('parameters')]
        posterior_parameters_names = [post.name for post in emulator.parameters]

        assert len(prior_parameters_names) == len(posterior_parameters_names) and sorted(
            prior_parameters_names) == sorted(posterior_parameters_names)

        # Loads in prior data from across iterations into a single
        for iteration in data.get('result'):

            for sample_number in range(len(iteration.get('x_observed'))):
                x_observed = iteration['x_observed'][sample_number]
                x_observed = np.array(x_observed).reshape(1, len(x_observed))

                y_observed = iteration['y_observed'][sample_number]
                y_observed = np.array(y_observed).reshape(1)

                self.samples_real = np.append(self.samples_real, x_observed, axis=0)
                self.values = np.vstack((self.values, y_observed))

                # Recalculate progress using prior data. Combines data across all runs.
                self.calc_progress(emulator=emulator)

    def prepare_save_dict(self, output_folder):
        """saves the current progress of the optimiser in a format that can be used with the emulator"""
        d = dict()
        d['results'] = {}
        for i, p in enumerate(self.params):
            d['results'][p.name] = self.samples_real[:, i]

        d['requested'] = {}
        for i, p in enumerate(self.params):
            d['requested'][p.name] = self.samples_ideal[:, i]

        d['results']['y'] = self.values

        d['parameters'] = {}
        d['parameters'] = [param.__dict__ for param in self.params]

        d = {**d, **self._save(output_folder)}

        return d

    def save(self, output_folder, name='opt_state.json'):
        d = self.prepare_save_dict(output_folder)

        try:
            with open(os.path.join(output_folder, name), 'w') as fp:
                json.dump(d, fp, indent=2, default=convert_np_native)
        except:
            logging.exception('OPT TEMPLATE: Failed to save opt_state.json')

    def _save(self, output_folder) -> dict:
        pass

    def load(self, path_to_prior, name='opt_state.json'):
        try:
            with open(os.path.join(path_to_prior, name)) as f:
                d = json.load(f)

                for param in d['parameters']:
                    self.add_parameter(**param)

                # x
                x = [val for k, val in d['results'].items() if k in [p.name for p in self.params]]
                self.x = np.atleast_2d(x).T

                # y
                self.y = np.atleast_2d(d['results'].get('y', [])).T


        except:
            logging.exception(f'OPT TEMPLATE: Failed to load opt_state.json')

        self._load(path_to_prior)

    def _load(self, path_to_prior):
        pass
