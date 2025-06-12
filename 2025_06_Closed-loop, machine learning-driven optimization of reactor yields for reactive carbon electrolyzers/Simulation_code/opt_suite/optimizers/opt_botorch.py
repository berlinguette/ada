from opt_suite.templates.optimizer_template import OptTemplate
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Log, Standardize, Power
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from gpytorch.kernels import MaternKernel, RBFKernel
from typing import Optional, Callable
from botorch.utils.transforms import unnormalize, normalize
import numpy as np
import warnings
import logging
import torch
import pandas as pd
from botorch.exceptions import BadInitialCandidatesWarning

# Set up logger and suppress warnings for cleaner output
logger = logging.getLogger()
warnings.filterwarnings(action='ignore', module='gpytorch')
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _random(_i, return_meta=False):
    """
    Generate a random point based on predefined bounds.
    :param _i: Row index for data slicing
    :param return_meta: Whether to return metadata
    :return: Normalized input parameters, optionally with metadata
    """
    param_type = "init part"
    dataset_name = "data.csv"

    # Load and slice dataset for a single random row
    df = pd.read_csv(dataset_name).iloc[_i:_i+1].reset_index(drop=True)
    input_cols = ["Input Current density", "Bicarbonate flow", "KOH flow", "Bicarb temp", "KOH conc", "Bicarb conc"]
    x = torch.tensor(df[input_cols].to_numpy(), dtype=torch.double)

    # Normalize inputs to predefined bounds
    bounds = torch.tensor([[50, 40, 6, 25, 0.05, 0.5], [350, 200, 30, 80, 3, 3.05]], dtype=torch.double)
    params = normalize(x, bounds).numpy()

    # Format and return normalized parameters
    params = np.array([np.hstack(params)])
    return (params, dict(type=param_type)) if return_meta else params


class Botorch(OptTemplate):
    """
    Botorch-based Bayesian Optimizer.
    """

    def __init__(self, init_randoms=53, acq_function=qUpperConfidenceBound, acq_params=None,
                 acq_restarts=50, acq_samples=256):
        """
        Initialize Botorch optimizer.
        :param init_randoms: Initial random samples
        :param acq_function: Acquisition function (default: qUpperConfidenceBound)
        :param acq_params: Parameters for acquisition function
        :param acq_restarts: Number of acquisition function restarts
        :param acq_samples: Number of samples for acquisition optimization
        """
        # Set description and default acquisition parameters
        self.name = 'Botorch'
        self.desc = 'Efficient Monte-Carlo Bayesian Optimization (https://botorch.org/)'
        if acq_params is None:
            acq_params = dict(beta=None, sampler=None)

        # Initialize parent class and store parameters
        super(Botorch, self).__init__()
        self.init_randoms = init_randoms
        self.remaining_randoms = init_randoms
        self._i = 0
        self.acq_function = acq_function
        self.acq_params = acq_params
        self.acq_restarts = acq_restarts
        self.acq_samples = acq_samples

        # Detect device for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_x(self, return_meta=False, params=None):
        """
        Generate or optimize next set of parameters.
        :param return_meta: Whether to return metadata
        :return: Optimal parameters, optionally with metadata
        """
        param_type = "botorch"
        beta_list = [0.25, 25, 400, None]

        # Initial random sampling
        if self.remaining_randoms != 0:
            self.remaining_randoms -= 1
            self._i += 1
            return self._random(self._i-1, return_meta=return_meta)
        else:
            # Prepare data for model training
            train_xt = torch.tensor(self.x, device=self.device, dtype=torch.double)
            train_yt = torch.tensor(self.y.astype(float), dtype=torch.double)

            # Define model bounds and noise level
            std_dev_input_data = 3
            bounds = torch.tensor(np.array([[p.pmin, p.pmax] for p in self.params]).astype(float).T,
                                  dtype=torch.double)

            # Initialize GP model with standardization
            gp = FixedNoiseGP(train_X=train_xt, train_Y=train_yt,
                              train_Yvar=torch.full_like(train_yt, std_dev_input_data ** 2, dtype=torch.double),
                              outcome_transform=Standardize(m=1))

            # Fit the model to maximize the log-likelihood
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            mll.train()
            fit_gpytorch_mll(mll)

            # Configure acquisition function based on type
            if self.acq_function.__name__ in {'ExpectedImprovement', 'ProbabilityOfImprovement'}:
                self.acq_params['best_f'] = train_yt.max()
            if self.acq_function.__name__ in {'qNoisyExpectedImprovement', 'qExpectedImprovement'}:
                self.acq_params['best_f'] = train_yt.max()
                self.acq_params['sampler'] = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
            if self.acq_function.__name__ in {'UpperConfidenceBound', 'qUpperConfidenceBound'}:
                self.acq_params['beta'] = 0.25

            # Create acquisition function and optimize it
            acq = self.acq_function(gp, **self.acq_params)
            _candidate, acq_value = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=self.acq_restarts,
                                                  raw_samples=self.acq_samples)

            # Extract optimized parameters
            candidate = _candidate.numpy().T
            result = [candidate[jx] for jx, p in enumerate(self.params)]
            params = np.array([np.hstack(result)])

            return (params, dict(type=param_type)) if return_meta else params


if __name__ == '__main__':
    # Instantiate optimizer and define parameters
    opt = Botorch()
    for i in range(3):
        opt.add_parameter(name=f'x{i}', pmin=0, pmax=1)

    # Perform optimization steps
    for i in range(15):
        param = opt.request_params()

        # Randomly evaluate results for demonstration purposes
        opt.return_result(realized_params=param, result=np.random.random(1))
        print(param)
