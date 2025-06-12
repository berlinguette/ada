import numpy as np
import os
import sys
import json
from typing import Optional


class Processor:
    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """
        This class processes the raw data from Runner into usable, plotable statistics.
        :param name: The name of the Runner to ingest.
        """

        # If no name passed, use the name of the script.
        if name is None:
            name = os.path.basename(sys.argv[0]).split('.')[0]
        self.name = name

        # Load in the raw data
        with open(name + '.json') as f:
            self.raw = json.load(f)

        # Store meta
        self.name = self.raw['name']
        self.desc = self.raw['desc']
        self.function = self.raw['function']
        self.runs = self.raw['runs']
        self.budget = self.raw['budget']

        # Calculated data, y
        self.y_best: Optional[np.ndarray] = None
        self.y_best_avg: Optional[np.ndarray] = None
        self.y_best_std: Optional[np.ndarray] = None
        self.y_best_med: Optional[np.ndarray] = None

        self.y_observed: Optional[np.ndarray] = None

        # Calculated data, x
        self.x_loss_real_best: Optional[np.ndarray] = None
        self.x_loss_norm_best: Optional[np.ndarray] = None

        self.x_loss_real_best_avg: Optional[np.ndarray] = None
        self.x_loss_norm_best_avg: Optional[np.ndarray] = None

        self.x_loss_real_best_std: Optional[np.ndarray] = None
        self.x_loss_norm_best_std: Optional[np.ndarray] = None

        # Other
        self.mean_run_index = None
        self.indicies_of_improvement: Optional[np.ndarray] = None
        self.values_of_improvement: Optional[np.ndarray] = None

        # X values
        self.x_observed: Optional[np.ndarray] = None


    def get_name(self):
        return self.name

    def get_desc(self):
        return self.desc

    def get_function(self):
        return self.function

    def get_runs(self):
        return self.runs

    def get_budget(self):
        return self.budget

    def get_domain(self) -> np.ndarray:
        """
        Get a domain of the data to plot against
        :return: np.ndarray
        """
        # return np.arange(self.budget)
        return np.arange(len(self.raw['result'][0]['x_observed']))

    # Get y-based metrics.
    def get_y_best(self) -> np.ndarray:
        if self.y_best is None:
            self.y_best = np.array([result['y_best'] for result in self.raw['result']])
        return self.y_best

    def get_y_observed(self):
        if self.y_observed is None:
            self.y_observed = np.array([result['y_observed'] for result in self.raw['result']])
        return self.y_observed

    def get_y_best_median(self) -> np.ndarray:
        if self.y_best_med is None:
            self.y_best_med = np.median(self.get_y_best(), axis=0)
        return self.y_best_med

    def get_y_best_avg(self) -> np.ndarray:
        if self.y_best_avg is None:
            self.y_best_avg = np.mean(self.get_y_best(), axis=0)
        return self.y_best_avg

    def get_y_best_quantile(self, q) -> np.ndarray:
        return np.quantile(self.get_y_best(), q, axis=0)

    def get_y_best_std(self) -> np.ndarray:
        if self.y_best_std is None:
            self.y_best_std = np.std(self.get_y_best(), axis=0)
        return self.y_best_std

    # Get x-based metrics.

    def get_x_loss_real_best(self) -> np.ndarray:
        if self.x_loss_real_best is None:
            self.x_loss_real_best = np.array([result['x_loss_real_best'] for result in self.raw['result']])
        return self.x_loss_real_best

    def get_x_loss_real_best_avg(self) -> np.ndarray:
        if self.x_loss_real_best_avg is None:
            self.x_loss_real_best_avg = np.mean(self.get_x_loss_real_best(), axis=0)
        return self.x_loss_real_best_avg

    def get_x_loss_real_best_std(self) -> np.ndarray:
        if self.x_loss_real_best_std is None:
            self.x_loss_real_best_std = np.std(self.get_x_loss_real_best(), axis=0)
        return self.x_loss_real_best_std

    def get_x_loss_norm_best(self) -> np.ndarray:
        if self.x_loss_norm_best is None:
            self.x_loss_norm_best = np.array([result['x_loss_norm_best'] for result in self.raw['result']])
        return self.x_loss_norm_best

    def get_x_loss_norm_best_avg(self) -> np.ndarray:
        if self.x_loss_norm_best_avg is None:
            self.x_loss_norm_best_avg = np.mean(self.get_x_loss_norm_best(), axis=0)
        return self.x_loss_norm_best_avg

    def get_x_loss_norm_best_std(self) -> np.ndarray:
        if self.x_loss_norm_best_std is None:
            self.x_loss_norm_best_std = np.std(self.get_x_loss_norm_best(), axis=0)
        return self.x_loss_norm_best_std

    # Additional functions
    def get_mean_run_index(self):
        """
        Get the index of the run which has a y_best closest to the mean. This is useful for
        visualization.
        :return: int
        """
        if self.mean_run_index is None:
            self.mean_run_index = np.argmin(np.sum(np.absolute(self.get_y_best() - self.get_y_best_avg())**2, axis=1))
        return self.mean_run_index

    def get_indicies_of_improvement(self):
        if self.indicies_of_improvement is None:
            self.indicies_of_improvement = np.array([np.where(np.roll(self.get_y_best()[i],1)!=self.get_y_best()[i])[0] for i in range(self.get_runs())])
        return self.indicies_of_improvement

    def get_values_of_improvement(self):
        if self.values_of_improvement is None:
            self.values_of_improvement = np.array([self.get_y_best()[i][self.get_indicies_of_improvement()[i]] for i in range(self.get_runs())])
        return self.values_of_improvement

    # Get x values
    def get_x_observed(self):
        if self.x_observed is None:
            self.x_observed = np.array([result['x_observed'] for result in self.raw['result']])
        return self.x_observed