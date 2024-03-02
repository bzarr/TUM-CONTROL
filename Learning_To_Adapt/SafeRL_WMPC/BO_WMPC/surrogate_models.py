import torch

from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.transforms import standardize
from botorch import fit_gpytorch_mll

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood
)
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import DirichletClassificationLikelihood


class ObjectiveModel:
    """ Wrapper class to define the surrogate model for the objective data.
    Accepts (7 x n)-d input that is mapped to (2 x n)-d output.

    Args:
        X (Tensor): tensor of input data samples
        Y (Tensor): tensor of output data samples
    """

    def __init__(self, X: torch.tensor, Y: torch.tensor):

        # standardize objective data
        Y = standardize(Y)

        # set up a Gaussian process regression model
        self.gp = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            mean_module=ConstantMean(),
            covar_module=ScaleKernel(RBFKernel(ard_num_dims=7))
        ).to(X)

        # generate marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        
        #  fit the model
        fit_gpytorch_mll(self.mll)

    def evaluate(self, X: torch.tensor):
        """ Evaluate the model at a given location X and return the mean and
        standard deviation of the estimated result.
        
        Args:
            X (Tensor): location in input space to evaluate
            
        Returns:
            Tuple[float, float]: mean and standard deviation of the result
        """
        # evaluate the model at the given location(s)
        self.gp.eval()
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            distribution = self.gp(X)
        # obtain mean and confidence region from the prediction
        means = distribution.loc
        stds = distribution.stddev

        return means, stds


class DirichletGPModel(ExactGP):
    """ Implements a Dirichlet Gaussian Process Classifier as described in
    https://papers.nips.cc/paper/2018/file/b6617980ce90f637e68c3ebe8b9be745-Paper.pdf,
    an implementation can also be found in
    https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html.
    """

    def __init__(self, train_x, train_y, likelihood, num_classes):
        super().__init__(train_x, train_y, likelihood)
        _batch_shape = torch.Size((num_classes,))
        self.mean_module = ConstantMean(batch_shape=_batch_shape)
        self.mean_module.initialize(constant=1.)
        self.covar_module = (
            ScaleKernel(
                RBFKernel(
                    batch_shape=_batch_shape,
                    ard_num_dims=7
                ),
                batch_shape=_batch_shape
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ConstraintModel:
    """ Wrapper class to define the surrogate model for the constraint data.
    Accepts (7 x n)-d input that is mapped to (1 x n)-d output. The model works
    as a classifier that assigns the output a probability of feasibility in the
    range of [0, 1].

    Args:
        X (Tensor): tensor of input data samples
        Y (Tensor): tensor of output data samples
    """

    def __init__(self, X: torch.tensor, Y: torch.tensor):
        # use a Dirichlet likelihood for the classification targets
        likelihood = DirichletClassificationLikelihood(
            Y, alpha_epsilon=0.001, learn_additional_noise=True)
        
        # generate a Gaussian process classifier model
        self.gp = DirichletGPModel(
            train_x=X,
            train_y=likelihood.transformed_targets.to(dtype=torch.float64),
            likelihood=likelihood,
            num_classes=likelihood.num_classes,
        ).to(X)

        # set initial lengthscale
        # (NOTE: can lead to better representation but also to fitting failures)
        #init_lengthscale = 0.04
        #self.gp.covar_module.base_kernel.lengthscale = init_lengthscale

        # generate marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        
        # fit the model
        fit_gpytorch_mll(self.mll)

    def evaluate(self, X: torch.tensor):
        """ Evaluate the constraint model at a given location. Queries for the
        classified logits and then performs Monte Carlo sampling to obtain a
        probability value.
        
        Args:
            X (Tensor): location in input space to evaluate
        
        Returns:
            Tuple[float, float]: mean and standard deviation of the estimated
                                 probability of feasibility
        """

        # cast model outputs to double
        self.gp.double()
        self.gp.eval()
        # evaluate the model at the given location(s)
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            # predict constraint violations
            distribution = self.gp(X)

        # sample from constraint distribution
        pred_samples = distribution.sample(torch.Size((2**14,))).exp()
        # calculate the constraint satisfaction probability
        mean = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
        std = (pred_samples / pred_samples.sum(-2, keepdim=True)).std(0)

        # extract relevant dimensions
        mean = 1 - mean[0]
        std = std[0]

        return mean, std