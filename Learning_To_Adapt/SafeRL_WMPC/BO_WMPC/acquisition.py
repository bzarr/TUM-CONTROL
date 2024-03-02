import torch

from typing import List, Optional, Union
from torch import Tensor
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement
)
from botorch.sampling.base import MCSampler
from botorch.utils.objective import apply_constraints_nonnegative_soft
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_fully_bayesian,
    match_batch_shape,
    t_batch_mode_transform,
)

from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.surrogate_models import (
    ObjectiveModel, ConstraintModel
)


class FeasibilityWeightedEHVI(qNoisyExpectedHypervolumeImprovement):
    """ Extension to standard NoisyExpectedHypervolumeImprovement that uses a
    constraint model to obtain a probability of feasibility with which the
    acquisition value for the respective sample is weighted. Used to include a
    binary constraint model into the acquisition process.

    Args:
        model_obj (ObjectiveModel): model for the objective values
        model_con (ConstraintModel): binary classifier model to predict
                                     constraint violations
        ref_point (Tensor): reference point in objective space, used for hyper-
                            volume calculation
        X_baseline (Tensor): already observed points that are considered as the
                             potential best solutions
        epsilon (float): parameter to compromise between exploration and
                         exploitation
        prune_baseline (bool): remove points from X_baseline that are highly
                               unlikely to be Pareto-optimal
        sample: (MCSampler): sampler used to obtain MC samples
    """

    def __init__(
        self,
        model_obj: ObjectiveModel,
        model_con: ConstraintModel,
        ref_point: Union[List[float], Tensor],
        X_baseline: Tensor,
        epsilon: float = 0.8,
        prune_baseline: Optional[bool] = True,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        
        super().__init__(
            model=model_obj.gp,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=prune_baseline
        )

        self.model_con = model_con
        self.epsilon = epsilon
    
    def _compute_qehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).

        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = torch.ones(
                obj.shape[:-1], device=obj.device, dtype=obj.dtype
            )
            feas_weights = apply_constraints_nonnegative_soft(
                obj=feas_weights,
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
            )
        self._cache_q_subset_indices(q_out=q)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=obj.device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = self.q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # substract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i

        # NOTE: the solution feasibility is included here
        # this is the bottleneck regarding computation time!
        # -> should be vectorized to speed things up a little
        mu, sigma = self.model_con.evaluate(X)
        feasibility_factor = self.epsilon * mu + (1 - self.epsilon) * sigma
        areas_per_segment = areas_per_segment * feasibility_factor

        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1).mean(dim=0)
    
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # Note: it is important to compute the full posterior over `(X_baseline, X)`
        # to ensure that we properly sample `f(X)` from the joint distribution `
        # `f(X_baseline, X) ~ P(f | D)` given that we can already fixed the sampled
        # function values for `f(X_baseline)`.
        # TODO: improve efficiency by not recomputing baseline-baseline
        # covariance matrix.
        posterior = self.model.posterior(X_full)
        # Account for possible one-to-many transform and the MCMC batch dimension in
        # `SaasFullyBayesianSingleTaskGP`
        event_shape_lag = 1 if is_fully_bayesian(self.model) else 2
        n_w = (
            posterior._extended_shape()[X_full.dim() - event_shape_lag]
            // X_full.shape[-2]
        )
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)

        # Add previous nehvi from pending points.
        return self._compute_qehvi(samples=samples, X=X) + self._prev_nehvi

    
# NOTE: this is an alternative optimizer that can be used to find candidate
# points; however does not typically perform better than the native BoTorch
# implementation
"""def optimize_acq_function_cmaes(
        acq_function: FeasibilityWeightedEHVI,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
    """""" Optimize a given acquisition function using the Covariance Matrix
    Adaptation Evolution Strategy (CMA-ES) optimizer.

    Args:
        acq_function: acquisition function to be optimized
        device (torch.device): device to work on
        dtype (torch.dtype): datatype to use

    Returns:
        Tensor: maximum of the given acquisition function
    """"""

    population_size = 100

    # create the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(
        x0=np.random.rand(7),
        sigma0=0.4,
        inopts={"bounds": [0, 1], "popsize": population_size},
    )

    # speed up things by telling pytorch not to generate a compute graph
    with torch.no_grad():
        while not es.stop():
            # query new evaluation points
            xs = es.ask()
            X = torch.tensor(xs, device=device, dtype=dtype)
            Y = np.zeros(population_size)
            for i, x in enumerate(X):
                x = x.unsqueeze(-2)
                y = -acq_function(x)
                Y[i] = y
            # return the result to the optimizer
            es.tell(xs, Y)

    # convert result back to a torch tensor
    best_x = torch.from_numpy(es.best.x).to(X)

    return best_x"""