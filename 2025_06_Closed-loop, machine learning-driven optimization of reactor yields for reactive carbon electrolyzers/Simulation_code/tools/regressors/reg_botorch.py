from tools.templates.reg_template import RegressorTemplate
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize, Log10
# from botorch.utils.transforms import unnormalize, normalize
# from gpytorch.kernels import RBFKernel
from gpytorch.kernels import MaternKernel, RBFKernel
from typing import Optional
import numpy as np
import warnings
import torch


class BotorchRegressor(RegressorTemplate):
    """
    Build a GP model based on data
    """
    def __init__(
            self,
            model_kwargs: Optional[dict] = None,
            **kwargs,
    ):
        """
        Generate a function that is based off of data.
        :param model_kwargs: Kwargs for the GP
        """

        # Initialize
        super().__init__(**kwargs)

        # Convert, store kwargs
        model_kwargs = dict() if model_kwargs is None else model_kwargs
        self.model_kwargs = model_kwargs

    def _initialize(self):
        """
        Train the model
        """

        # Make training tensors
        self.train_x = torch.tensor(self.x, dtype=torch.double)
        self.train_y = torch.tensor(self.y, dtype=torch.double)

        # Note that an InputDataWarning wil be silenced from Botorch, requesting
        # that training data be scaled to zero mean and unit variance. Data will
        # instead be scaled to [0, 1] since this is more useful for later
        # hypervolume calculations.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=InputDataWarning)

            std_dev_input_data = 3

            # Make model
            self.gp = FixedNoiseGP(
                self.train_x,
                self.train_y,
                torch.full_like(self.train_y, std_dev_input_data ** 2, dtype=torch.double),
                outcome_transform=Standardize(m=1),
            )

            # Train
            self.mll = ExactMarginalLogLikelihood(
                self.gp.likelihood,
                self.gp,
            )

            self.mll.train()
            fit_gpytorch_mll(self.mll)

    def _evaluate(
            self,
            x: np.ndarray,
            noise_level: float = 0,
            **kwargs,
    ):
        """
        Evaluate the surface
        :param x: An array of shape n_samples x n_dimensions
        :param kwargs: All arguments passed to the Surface.
        :return: surface evaluation.
        """

        # Test the surface
        x_test = torch.tensor(x, dtype=torch.double)
        y_eval = self.gp.posterior(x_test)

        y_mean = y_eval.mean + torch.randn(y_eval.mean.size(), dtype=torch.double) * noise_level
        y_mean = y_mean.detach().numpy()
        y_var = y_eval.variance.detach().numpy()

        y_meta = dict(
            var=y_var,
        )

        # Return
        return y_mean, y_meta
