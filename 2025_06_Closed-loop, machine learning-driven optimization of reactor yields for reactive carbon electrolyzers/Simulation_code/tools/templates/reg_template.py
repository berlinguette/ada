import matplotlib.transforms as transforms
from typing import Optional, List, Tuple
from tools.engine import get_cpu_count
from abc import ABC, abstractmethod
import pathos.multiprocessing as mp
from tools.engine import calc_rmse
import multiprocess.context as ctx
import matplotlib.pyplot as plt
from tools.engine import c
from tqdm import tqdm
import numpy as np
import copy


def norm_forward(
        x: np.ndarray,
        span: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data. If scaling info is passed, this will be used. Else the data
    will be scaled to itself.
    :param x: Of shape n_samples x n_dimensions
    :param span: Span of data, n_dimensions
    :param offset: Offset of data, n_dimensions
    :return: scaled, span, offset
    """
    if span is None:
        span = np.max(x, axis=0) - np.min(x, axis=0)
    if offset is None:
        offset = np.min(x, axis=0)
    scaled = (x - offset) / span
    return scaled, span, offset


def norm_backward(
        x: np.ndarray,
        span: np.ndarray,
        offset: np.ndarray,
) -> np.ndarray:
    """
    Unnormalize the data.
    :param x: Of shape n_samples x n_dimensions
    :param span: Span of data
    :param offset: Offset of data
    :return: Unnormalized data
    """
    un = (x * span) + offset
    return un


def stand_forward(
        x: np.ndarray,
        mean: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data. If scaling info is passed, this will be used. Else the
    data will be scaled to itself.
    :param x: Of shape n_samples x n_dimensions
    :param mean: Mean of data, n_dimensions
    :param sigma: std of data, n_dimensions
    :return: scaled, mean, sigma
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    if sigma is None:
        sigma = np.std(x, axis=0)
    scaled = (x - mean) / sigma
    return scaled, mean, sigma


def stand_backward(
        x: np.ndarray,
        mean: np.ndarray,
        sigma: np.ndarray,
) -> np.ndarray:
    """
    Standardize data
    :param x: Of shape n_samples x n_dimensions
    :param mean: mean
    :param sigma: std
    :return: scaled, mean, std
    """
    un = (x * sigma) + mean
    return un


class RegressorTemplate(ABC):

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_labels: Optional[List[str]] = None,
            y_labels: Optional[List[str]] = None,
            x_normalize: bool = False,
            x_standardize: bool = False,
            x_internal_scaling: bool = True,
            y_normalize: bool = False,
            y_standardize: bool = False,
            y_internal_scaling: bool = True,
    ):
        """
        All regressors must be passed training data.
        :param x: The x data to train the model of shape n_samples x x_dim
        :param y: The y data to train the model of shape n_samples x y_dim
        :param x_labels: Labels for x data.
        :param y_labels: Labels for y data.
        :param x_normalize: Should the x training data be normalized between 0
        and 1 prior to training the model.
        :param x_standardize: Should the x data be standardized to a mean of 0
        and a standard deviation of 1 prior to training the model.
        :param x_internal_scaling: Should the x data scaling be completed
        internal to the model, or should scaled data be expected during
        model evaluation.
        :param y_normalize: Should the y training data be normalized between 0
        and 1 prior to training the model.
        :param y_standardize: Should the y data be standardized to a mean of 0
        and a standard deviation of 1 prior to training the model.
        :param y_internal_scaling: Should the y data scaling be completed
        internal to the model, or should scaled data be returned during
        model evaluation.
        """

        # Calculate dimensionality
        self.x_dim = x.shape[1]
        self.y_dim = y.shape[1]

        # Confirm the labels
        if x_labels is None:
            x_labels = np.array([f'x{i}' for i in range(self.x_dim)])
        else:
            assert len(x_labels) == self.x_dim
        if y_labels is None:
            y_labels = np.array([f'y{i}' for i in range(self.y_dim)])
        else:
            assert len(y_labels) == self.y_dim

        # Confirm that an appropriate scaling has been asked for
        if x_normalize and x_standardize:
            raise ValueError('Cannot normalize and standardize x data.')
        if y_normalize and y_standardize:
            raise ValueError('Cannot normalize and standardize y data.')

        # Store the scaling preference
        self.x_normalize: bool = x_normalize
        self.x_standardize: bool = x_standardize
        self.x_internal_scaling: bool = x_internal_scaling
        self.y_normalize: bool = y_normalize
        self.y_standardize: bool = y_standardize
        self.y_internal_scaling: bool = y_internal_scaling

        # Store other data
        self.x = x
        self.y = y
        self.x_labels: np.ndarray = x_labels
        self.y_labels: np.ndarray = y_labels
        self.initialized = False
        self.count = None

        # Create a place to store CV results
        self.cv_x: Optional[np.ndarray] = None
        self.cv_y: Optional[np.ndarray] = None
        self.cv_meta: Optional[dict] = None

        # Create a place to store the scaled data
        self.x_scaled: Optional[np.array] = None
        self.y_scaled: Optional[np.array] = None

        # Scalars for normalization
        self.x_span: Optional[np.array] = None
        self.x_offset: Optional[np.array] = None
        self.y_span: Optional[np.array] = None
        self.y_offset: Optional[np.array] = None

        # Scalars for standardization
        self.x_mean: Optional[np.array] = None
        self.x_sigma: Optional[np.array] = None
        self.y_mean: Optional[np.array] = None
        self.y_sigma: Optional[np.array] = None

        # Create a place to store CV stat results
        self.cv_rmse: Optional[float] = None

        # Create a place for any additional values that should be returned in
        # the y_meta of each evaluation.
        self.y_meta_const = dict()

    def _scale(self):
        """
        Scale the training data according to the arguments on instantiation.
        https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
        :return: None
        """

        # If normalizing x
        if self.x_normalize:
            self.x_scaled, self.x_span, self.x_offset = norm_forward(self.x)

        # If normalizing y
        if self.y_normalize:
            self.y_scaled, self.y_span, self.y_offset = norm_forward(self.y)

        # If standardizing x
        if self.x_standardize:
            self.x_scaled, self.x_mean, self.x_sigma = stand_forward(self.x)

        # If standardizing y
        if self.y_standardize:
            self.y_scaled, self.y_mean, self.y_sigma = stand_forward(self.y)

        # If neither, store raw data
        if not self.x_normalize and not self.x_standardize:
            self.x_scaled = self.x
        if not self.y_normalize and not self.y_standardize:
            self.y_scaled = self.y

    def initialize(self):
        """
        Scale the data and initialize the model.
        :return: None
        """
        # TODO have my own scaling method
        # self._scale()
        self._initialize()
        self.initialized = True

    @abstractmethod
    def _initialize(self):
        """
        All regressors must be initialized. This function must be overwritten
        with the initialization of the model. Note that the model must be able
        to be instantiated multiple times.
        """
        pass

    @abstractmethod
    def _evaluate(
            self,
            x: np.ndarray,
            **kwargs,
    ):
        """
        Return an observation from the model
        """
        pass

    def evaluate(
            self,
            x: np.ndarray,
            **kwargs,
    ):
        """
        Evaluate the model.
        :param x: where to evaluate the model.
        """

        # If the model has not been initiated, initiate it and scale the data.
        if not self.initialized:
            self.initialize()

        # If the data should be scaled before evaluation, do so
        if self.x_internal_scaling:
            if self.x_normalize:
                x, span, offset = norm_forward(
                    x=x,
                    span=self.x_span,
                    offset=self.x_offset,
                )
            if self.x_standardize:
                x, mean, sigma = stand_forward(
                    x=x,
                    mean=self.x_mean,
                    sigma=self.x_sigma,
                )

        # Evaluate
        y, y_meta = self._evaluate(
            x=x,
            **kwargs,
        )

        # The following values in y_meta should also be scaled
        norm_abs_keys = ['median', 'iqr_l', 'iqr_u']
        norm_rel_keys = ['var']

        # If the data should be rescaled before returning, do so
        if self.y_internal_scaling:
            if self.y_normalize:
                y = norm_backward(
                    x=y,
                    span=self.y_span,
                    offset=self.y_offset,
                )
            if self.y_standardize:
                y = stand_backward(
                    x=y,
                    mean=self.y_mean,
                    sigma=self.y_sigma,
                )

            # Scale the relevant meta data as well
            if isinstance(y_meta, dict):

                # If the value is absolute
                for key in norm_abs_keys:
                    if key in y_meta.keys():
                        if self.y_normalize:
                            y_meta[key] = norm_backward(
                                x=y_meta[key],
                                span=self.y_span,
                                offset=self.y_offset,
                            )
                        if self.y_standardize:
                            y_meta[key] = stand_backward(
                                x=y_meta[key],
                                mean=self.y_mean,
                                sigma=self.y_sigma,
                            )

                # If the value is relative
                for key in norm_rel_keys:
                    if key in y_meta.keys():
                        if self.y_normalize:
                            y_meta[key] = y_meta[key] * self.y_span
                        if self.y_standardize:
                            y_meta[key] = y_meta[key] * self.y_sigma

        # Add any constant y_meta values to the dict, if they have been defined
        y_meta = dict() if y_meta is None else y_meta
        y_meta = dict(
            **y_meta,
            **self.y_meta_const,
        )

        # Revert to none if no args
        y_meta = None if y_meta == dict() else y_meta

        # Return values
        return y, y_meta

    def get_cv(
            self,
            multiprocessing: bool = True,
    ):
        """
        Calculate the CV pairs
        :param multiprocessing: should MP be used.
        :return: CV pairs, variance.
        """

        def eval_point(
                x: np.ndarray,
                y: np.ndarray,
                i: int,
                model: RegressorTemplate,
        ):
            """
            Evaluate an iteration of the CV
            :param x: The base x dataset
            :param y: The base y dataset
            :param i: The point to evaluate
            :param model: A copy of the model to use
            :return: CV pairs
            """

            x_train = np.delete(x, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            x_test = x[i][None, :]
            y_true = y[i][None]

            # Train model on train data
            model.x = x_train
            model.y = y_train

            # Force re initialize and evaluate
            model._initialize()
            y_test, y_meta = model.evaluate(x_test)

            # The test point has been properly scaled. However, the y_true point
            # has not. Forward scale the data.
            if not model.y_internal_scaling:
                if model.y_normalize:
                    y_true, y_span, y_offset = norm_forward(
                        x=y_true,
                        span=model.y_span,
                        offset=model.y_offset,
                    )
                if model.y_standardize:
                    y_true, y_mean, y_sigma = stand_forward(
                        x=y_true,
                        mean=model.y_mean,
                        sigma=model.y_sigma,
                    )

            # Return the CV pair
            return y_true, y_test

        # If the CV has not been calculated, calculate it
        if self.cv_x is None or self.cv_y is None:

            # Ensure that the model is initialized
            if not self.initialized:
                self.initialize()

            # Create a place to store the results
            y_train = list()
            y_test = list()
            count = len(self.x)

            # If multiprocessing
            if multiprocessing:

                # Certain models require spawning
                spawn_models = [
                    'GPFlowRegressor',
                    'BotorchRegressor',
                ]
                if self.__class__.__name__ in spawn_models:
                    ctx._force_start_method('spawn')

                # Create pool
                mp_results = list()
                pool = mp.Pool(
                    processes=get_cpu_count()
                )

                # Iteratively add the evaluations
                for i in range(count):
                    mp_results.append(
                        pool.apply_async(
                            eval_point,
                            kwds=dict(
                                x=self.x,
                                y=self.y,
                                i=i,
                                model=copy.deepcopy(self),
                            )
                        )
                    )

                # Wait for daemon processes to complete
                for result in tqdm(mp_results):
                    result.get()

                # Close pool
                pool.close()
                pool.join()

                # Get results
                for mp_result in mp_results:
                    iy_train, iy_test = mp_result.get()
                    y_train.append(iy_train)
                    y_test.append(iy_test)

            # If not multiprocessing
            else:

                # Evaluate each point in the model
                for i in tqdm(range(count)):
                    iy_train, iy_test = eval_point(
                        x=self.x,
                        y=self.y,
                        i=i,
                        model=copy.deepcopy(self)
                    )

                    # Store
                    y_train.append(iy_train)
                    y_test.append(iy_test)

            # store
            self.cv_x = np.vstack(y_train)
            self.cv_y = np.vstack(y_test)

        # Return
        return self.cv_x, self.cv_y

    def calc_rmse(self):
        """
        Calculate the statistic metrics from the CV.
        :return: None
        """

        # If stats not yet calculated, calc.
        if self.cv_rmse is None:
            self.get_cv()
            self.cv_rmse = calc_rmse(
                y0=self.cv_x,
                y1=self.cv_y
            )

        # Return
        return self.cv_rmse

    def plot_cv(
            self,
            save: bool = False,
            name: str = 'cv.png',
            multiprocessing: bool = False,
    ):
        """
        Plot the CV.
        :param save: Should the plot be saved.
        :param name: The name of the plot when saving.
        :param multiprocessing: Should mp be used to calc CV.
        """

        # Plotting constants
        color = c.blue

        # Get the cv data
        train, test = self.get_cv(
            multiprocessing=multiprocessing,
        )
        rmse = self.calc_rmse()
        mag = np.where(rmse != 0, np.floor(np.log10(rmse)), 0).astype(int)
        dim = train.shape[1]

        # Make figure objects
        figure: plt.Figure = plt.figure(figsize=(3.5 * dim, 4))
        axes: List[plt.Axes] = [figure.add_subplot(1, dim, i + 1) for i in range(dim)]

        # For each dimension
        for i in range(dim):

            # Get the axis
            ax = axes[i]

            # Get data
            x = train[:, i]
            y = test[:, i]

            # Get plotting bounds
            buff = 0.1
            dmin = min(x.min(), y.min())
            dmax = max(x.max(), y.max())
            domain = dmax - dmin
            pmin = dmin - buff * domain
            pmax = dmax + buff * domain

            # Plot the truth line
            ax.plot(
                [pmin, pmax],
                [pmin, pmax],
                lw=1,
                c=c.gray.i300,
                zorder=0,
            )
            ax.scatter(
                x,
                y,
                edgecolor=color.i800,
                facecolor=color.i200,
                lw=1,
                s=20,
                zorder=1,
            )

            # Add RMSE
            rmse_str = str(round(rmse[i]/(10**float(mag[i])), 3)) + f'×$10^{{{mag[i]}}}$'
            ax.text(
                0.03,
                0.92,
                f'RMSE: {rmse_str}',
                transform=ax.transAxes,
            )

            # Format
            ax.set_aspect('equal')
            ax.set_xlabel(f'{self.y_labels[i]} true', labelpad=-0.2)
            ax.set_ylabel(f'{self.y_labels[i]} predicted', labelpad=-0.2)
            ax.set_xticks([dmin, dmax])
            ax.set_yticks([dmin, dmax])
            ax.set_xlim(pmin, pmax)
            ax.set_ylim(pmin, pmax)
            figure.subplots_adjust(
                left=0.1,
                right=0.95,
                top=0.95,
                bottom=0.15,
            )

        # Save or show the plot
        if save:
            figure.set_dpi(300)
            figure.savefig(name)
        else:
            plt.show()

    def plot_model_frame(
            self,
            x_idx: int,
            y_idx: int,
            point: Optional[np.ndarray] = None,
            ax: Optional[plt.Axes] = None,
            save: bool = False,
            show: bool = True,
            name: str = 'model_frame.png',
            return_pbounds: bool = False,
    ) -> Optional[Tuple[float, float]]:
        """
        Plot a slice of the model for a given x and y axes, and a given position
        in the space.
        :param x_idx: Index of the x axis parameter.
        :param y_idx: Index of the y axis objective.
        :param point: The point at which to slice the model. If no point is
        passed, then the maximum experimental point at y_idx is used.
        :param ax: Optionally an axis can be passed.
        :param save: Should the plot be saved.
        :param show: Should the plot be shown.
        :param name: If it should be saved, the name of the file.
        :param return_pbounds: Should the y plotting bounds be returned.
        """

        # Plotting constants
        res = 1000
        color_main = c.blue

        # If an ax object was not passed, create one.
        figure: Optional[plt.Figure] = None
        if ax is None:
            figure = plt.figure()
            ax: plt.Axes = figure.add_subplot()

        # If no point is passed, then use max of y_idx to identify a point
        if point is None:
            p_idx = np.argmax(self.y[:, y_idx])
            point = self.x[p_idx]

        # Generate the values with which to sample the model
        x_eval = np.tile(point, res).reshape(res, len(point))

        # Modify the x parameter of interest.
        x_min = self.x[:, x_idx].min()
        x_max = self.x[:, x_idx].max()
        x = np.linspace(x_min, x_max, res)
        x_eval[:, x_idx] = x

        # Sample the model
        y_all, y_meta = self.evaluate(x=x_eval)
        y = y_all[:, y_idx]

        # Begin plotting
        ax.plot(
            x,
            y,
            lw=1,
            c=color_main.i800,
            label='mean',
            zorder=10,
        )

        # If the variance has been returned, plot it.
        var_min = np.inf
        var_max = -np.inf
        if 'var' in y_meta.keys():
            colors = [color_main.i100, color_main.i50]
            for i in range(2):
                var_min = y - (i+1) * y_meta['var'][:, y_idx]
                var_max = y + (i+1) * y_meta['var'][:, y_idx]
                ax.fill_between(
                    x,
                    var_min,
                    var_max,
                    # alpha=0.15,
                    zorder=-5-i,
                    color=colors[i],
                    label=f'{i+1} sigma'
                )

        # If Median passed, plot
        if 'median' in y_meta.keys():
            ax.plot(
                x,
                y_meta['median'][:, y_idx],
                color=c.bluegray.i500,
                lw=1,
                alpha=0.75,
                zorder=1,
                label='median',
            )

        # If IQR passed, plot
        for iqr in ['iqr_l', 'iqr_u']:
            if iqr in y_meta.keys():
                l = dict(label='IQR') if iqr == 'iqr_l' else dict()
                ax.plot(
                    x,
                    y_meta[iqr][:, y_idx],
                    color=c.bluegray.i500,
                    lw=1,
                    alpha=0.25,
                    zorder=1,
                    **l,
                )

        # Before plotting the grey line, evaluate the model
        point_y, point_meta = self.evaluate(x=point[None, :])
        point_idy = point_y[0][y_idx]

        # Plot the vertical line for the point where sliced
        point_idx = point[x_idx]
        ax.axvline(
            point_idx,
            lw=1,
            c=c.gray.i500,
            linestyle=':',
            label='slice point'
        )

        # Plot a point such that it is visible when it is on the edge of the plot
        ax.scatter(
            point_idx,
            point_idy,
            lw=1,
            s=20,
            edgecolor=c.gray.i500,
            facecolor='white',
            zorder=20,
        )

        # Add text label for the vertical line
        mag = np.where(point_idx != 0, np.floor(np.log10(point_idx)), 0).astype(int)
        point_idx_str = str(round(point_idx / (10 ** float(mag)), 3)) + f'×$10^{{{mag}}}$'
        ax.text(
            point_idx,
            1,
            point_idx_str,
            transform=transforms.blended_transform_factory(
                ax.transData,
                ax.transAxes,
            ),
            horizontalalignment='center',
            color=c.gray.i500,
        )

        # Format x
        ax.set_xlim(x_min, x_max)
        ax.set_xticks([x_min, x_max, point_idx])
        plt.xticks(rotation=45, ha="right")
        ax.set_xlabel(self.x_labels[x_idx])

        # Format y
        d_min = y.min()
        d_max = y.max()
        d_domain = d_max - d_min
        buf = 0.1

        # Calculate the view window
        v_min = min(d_min, var_min.min())
        v_max = max(d_max, var_max.max())
        v_domain = v_max - v_min

        # Calculate the plotting window
        p_min = v_min - buf * v_domain
        p_max = v_max + buf * v_domain
        ax.set_ylim(p_min, p_max)
        ax.set_ylabel(self.y_labels[y_idx])

        # Figure formatting
        if figure is not None:
            figure.subplots_adjust(
                left=0.05,
                right=0.98,
                bottom=0.15,
                top=0.95,
            )

        # Save or show the plot
        if save:
            figure.set_dpi(300)
            figure.savefig(name)
        elif show:
            plt.show()

        # If the pbounds should be returned, return them
        if return_pbounds:
            return p_min, p_max

    def plot_model_frames(
            self,
            point: Optional[np.ndarray] = None,
            save: bool = True,
            name: str = 'model_frames.png',
            sync_y: bool = True,
    ):
        """
        Plot the model sliced at the point.
        :param point: The point at which to slice the model. If no point is
        passed, then the maximum experimental point at y_idx is used.
        :param save: Should the plot be shown, or saved.
        :param name: If it should be saved, the name of the file.
        :param sync_y: Should the y axes be common
        """

        # Generate the figure
        figure: plt.Figure = plt.figure(
            figsize=(
                4 * self.x_dim,
                4 * self.y_dim,
            )
        )

        # # Create a place to store the axes
        axes = np.empty((self.y_dim, self.x_dim), dtype=plt.Axes)

        # Create a place to store the p_min, p_max.
        p_mins = np.repeat(np.inf, self.y_dim)
        p_maxs = np.repeat(-np.inf, self.y_dim)

        # For each x and y dimension
        for i in range(self.y_dim):
            for j in range(self.x_dim):

                # Create an axis
                pos = (j + 1) + (i * self.x_dim)
                ax: plt.Axes = figure.add_subplot(
                    self.y_dim,
                    self.x_dim,
                    pos,
                )

                # Store the axis for later formatting
                axes[i, j] = ax

                # Plot the slice
                p_min, p_max = self.plot_model_frame(
                    x_idx=j,
                    y_idx=i,
                    ax=ax,
                    save=False,
                    show=False,
                    return_pbounds=True,
                    point=point,
                )

                # If only one x axis, plot the actual training data
                if self.x_dim == 1:
                    ax.scatter(
                        self.x,
                        self.y,
                        lw=1,
                        s=5,
                        edgecolor=c.blue.i800,
                        facecolor='white',
                        zorder=10,
                        label='train'
                    )

                # Remove the labels
                if i != self.y_dim - 1:
                    ax.set_xlabel('')
                if j != 0:
                    ax.set_ylabel('')

                # If a common y axis is being used, store
                if sync_y:
                    p_mins[i] = min(p_mins[i], p_min)
                    p_maxs[i] = max(p_maxs[i], p_max)

                # Turn off unneeded axes
                for pos in ['top', 'right']:
                    ax.spines[pos].set_visible(False)
                if sync_y and j != 0:
                    ax.set_yticks([])
                    ax.spines['left'].set_visible(False)
                if i != self.y_dim - 1:
                    ax.set_xticks([])
                    ax.spines['bottom'].set_visible(False)

        # If synchronizing y axes
        if sync_y:

            # For each x and y dimension
            for i in range(self.y_dim):
                for j in range(self.x_dim):

                    axes[i, j].set_ylim(
                        p_mins[i],
                        p_maxs[i],
                    )

        # Add legend
        legend_args = dict(
            fontsize='small'
        )
        if self.x_dim == 1:
            axes[0, 0].legend(**legend_args)
        else:
            axes[-1, -1].legend(**legend_args)

        # Format
        figure.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.95,
        )

        # Save or show
        if save:
            figure.set_dpi(300)
            figure.savefig(name)
        else:
            plt.show()
