import torch
import numpy as np
import itertools
import time
import multiprocessing

from typing import List
from torch import Tensor
from copy import copy

from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.exceptions import ModelFittingError

from Learning_To_Adapt.SafeRL_WMPC.helpers import load_config, load_trials_from_csv
from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import (
    Nonlinear_Model_Predictive_Controller
)
from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.surrogate_models import (
    ObjectiveModel, ConstraintModel
)
from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.acquisition import FeasibilityWeightedEHVI
from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.track_segmentation import get_train_segments
from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.dataclasses import Trial, Dataset
from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.objective_function import objective_function


CONFIG_PATH = 'Config/'
TRAJECTORY_PATH = 'Trajectories/'
LOGS_PATH = 'Logs/'
SIM_MAIN_PARAMS_FILE = "EDGAR/sim_main_params.yaml"
MPC_PARAMS_FILE = "EDGAR/MPC_params.yaml"


########################### MULTIPROCESSING HELPERS ############################
""" Helper function to make the preinitialized controllers available globally,
so that they can be used from the worker pool. """
def _init_pool(optimizer):
    global global_controllers
    global_controllers = optimizer.controllers


""" Worker function for parallel trial evaluation, is called from each sub-
process of a worker pool. Takes a list of args and then performs an evalua-
tion run using the given parameters. """
def _evaluate_trial(args):

    # extract list of packed arguments
    (
        params,
        parameter_bounds,
        n_segment_groups,
        n_segments,
        track_segments,
        config,
        tkwargs,
    ) = args

    # obtain unique identifier (0, 1, ...) that is used to assign a controller
    identifier = (multiprocessing.current_process()._identity[0] - 1) % len(global_controllers)

    print("[Worker {}]: Evaluating parameter set [{}]".format(
        identifier,
        ', '.join([str(round(p.item(), 2)) for p in params.squeeze()])
    ))

    # unnormalize the parameters to complete the trial information
    params_nor = params
    params_nat = unnormalize(params_nor, parameter_bounds)

    # assign controller and update parameters
    MPC = global_controllers[identifier]
    
    # evaluate the objective function, i.e., simulate the vehicle behavior
    objectives, feasible = objective_function(
        MPC=MPC,
        parameterization=params_nat,
        n_segment_groups=n_segment_groups,
        n_segments=n_segments, track_segments=track_segments,
        tkwargs=tkwargs,
        config=config
    )

    # for unfeasible runs, assign NaN to all objectives
    if not feasible:
        objectives = torch.ones_like(objectives) * np.nan
    
    # generate a trial object that bundles the information
    trial = Trial(
        id=0,
        params_nat=params_nat,
        params_nor=params_nor,
        objectives=objectives,
        feasible=feasible
    )

    # notify the user over the outcome
    if trial.feasible:
        print(trial)
    else:
        print(f"[Worker {identifier}]: Trial failed.")

    return trial


########################### OPTIMIZATION MAIN CLASS ############################
class BayesianOptimization:
    """ Base class to conduct multiobjective Bayesian optimization, performs all
    necessary steps and manages data.
    
    Args:
        parameter_bounds (torch.tensor): lower and upper bounds of the parameter
                                         space for the optimization, i.e. the
                                         possible values of decision variables
        config (dict): configuration file containing relevant settings for the
                       optimization framework
        tkwargs (dict): torch keyword arguments, specify where to locate the
                        tensors and which data types to use
        dtype_con (torch.dtype): datatype representing the feasibility, must be
                                 integer
        trial_datafile (str): relative path to a file containing a set of trials
                              that are loaded initially

    Returns:
        None
    """

    def __init__(
            self,
            parameter_bounds: torch.tensor,
            config: dict,
            tkwargs: dict,
            dtype_con: torch.dtype = torch.int32,
            trial_datafile: str = None
        ) -> None:

        self.tkwargs = tkwargs
        self.device = tkwargs['device']
        self.dtype_obj = tkwargs['dtype']
        self.dtype_con = dtype_con
        print(f"BoTorch is using device: {self.device}")

        # size of decision space and objective space
        self.n_params = parameter_bounds.size(dim=1)
        self.n_objectives = 2

        # obtain track segments from reference trajectories
        self.track_segments = get_train_segments()
        self.n_segment_groups = 2
        self.n_segments = [len(group) for group in self.track_segments]

        # initialize datasets that store the evaluation data
        self.objective_data = []
        for _ in range(self.n_segment_groups):
            self.objective_data.append(Dataset(
                X=torch.empty((0, self.n_params), **self.tkwargs),
                Y=torch.empty((0, self.n_objectives), **self.tkwargs))
            )
        self.constraint_data = Dataset(
            X=torch.empty((0, self.n_params), **self.tkwargs),
            Y=torch.empty((0, 1), dtype=dtype_con, device=self.device))

        # indicator to select a segment group in turns
        self.active_group = 0

        # define parameter bounds and the normalized bounds used for the models
        self.parameter_bounds = parameter_bounds
        self.normalized_bounds = torch.stack(
            [torch.zeros(self.n_params, **self.tkwargs),
             torch.ones(self.n_params, **self.tkwargs)]
        )
        
        # initialize the sampler used to sample from the surrogate models
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([config['n_sobol_samples']])
        )

        # initialize the empty surrogate models
        self.objective_model = None
        self.constraint_model = None

        # reference point for hypervolume calculation
        self.reference_points = torch.tensor([
            config['reference_point_0'], config['reference_point_1']
        ], **self.tkwargs)
        
        # initialize hypervolume as a metric for multiobjective optimality
        self.hypervolumes = [0.] * self.n_segment_groups
        self.hv_traces = []

        # initialize objects for the Pareto front
        self.dominated_data     = [None] * self.n_segment_groups
        self.nondominated_data  = [None] * self.n_segment_groups
        self.pareto_trials      = [None] * self.n_segment_groups

        # initialize a list of conducted trials
        self.trials = []
        self.n_trials = 0

        # load existing trials if a file is specified
        if trial_datafile is not None:
            self._load_trials(filepath=trial_datafile)

        # load configuration dictionary
        self.config = config

        # set number of processes/CPU cores to occupy
        self.n_processes = config['n_processes']

        # preinitialize a separate controller for each process
        sim_main_params = load_config(CONFIG_PATH + SIM_MAIN_PARAMS_FILE)
        dummy_x0 = np.zeros(8)
        self.controllers = [
            Nonlinear_Model_Predictive_Controller(
                CONFIG_PATH, MPC_PARAMS_FILE, sim_main_params, dummy_x0
            ) for _ in range(self.n_processes)]
    
    def perform_bayesian_optimization_step(self) -> Tensor:
        """ Performs a single iteration of Bayesian optimization. Fits the
        surrogate models to the available data and optimizes an acquisition
        function to obtain a promising sampling location. Then evaluates the
        objective function at the selected location.

        Args:
            None
        
        Returns:
            Tensor: computation times for model fitting, candidate selection and
                    simulation; only returned if the iteration went as expected
        
        """

        iteration_start_time = time.time()

        # if no successful data exists, do random sampling
        if self.objective_data[0].X.shape[0] == 0:
            print("No successful data, choosing random samples instead.")
            parameterizations = draw_sobol_samples(
                bounds=self.normalized_bounds, n=self.config['batch_size'], q=1
            ).squeeze()
            
            # evaluate the random trials
            results = self._pooled_evaluation(parameterizations)
            for trial in results:
                self._add_trial(trial)
            return None

        # fit the models to the active group
        print(f"Using group {self.active_group} for candidate selection.")
        success = self.fit_surrogate_models(segment_group_id=self.active_group)
        t_fit = time.time() - iteration_start_time

        if success:
            print(f"Model fitting took {t_fit:.1f} seconds.")

        # choose random samples in case model fitting failed
        else:
            print(f"Failed to fit the surrogate model to the objective data.")
            print(f"Using random samples instead.")
            parameterizations = draw_sobol_samples(
                bounds=self.normalized_bounds, n=self.config['batch_size'], q=1
            ).squeeze()
            
            # evaluate the random trials
            results = self._pooled_evaluation(parameterizations)
            for trial in results:
                self._add_trial(trial)
            return None
        
        # define the acquisition function
        acquisition_function = FeasibilityWeightedEHVI(
            model_obj=self.objective_model,
            model_con=self.constraint_model,
            ref_point=self.reference_points[self.active_group],
            X_baseline=self.objective_data[self.active_group].X,
            epsilon=self.config['epsilon'],
            sampler=self.sampler,
            prune_baseline=True
        )

        # obtain a candidate point to sample next
        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.normalized_bounds,
            q=self.config['batch_size'],
            num_restarts=self.config['n_restarts'],
            raw_samples=self.config['n_raw_samples'],
            options={"batch_limit": 1, "maxiter": self.config['max_iter']},
            sequential=True,
        )
        parameterizations = candidates.detach()
        t_select = time.time() - iteration_start_time - t_fit
        print(f"Candidate selection took {(t_select):.1f} seconds.")

        results = self._pooled_evaluation(parameterizations)
        for trial in results:
            self._add_trial(trial)
        t_sim = time.time() - iteration_start_time - t_fit - t_select
        print(f"Batch simulation took {(t_sim):.1f} seconds.")

        # update active group for next iteration
        self.active_group += 1
        if self.active_group >= self.n_segment_groups:
            self.active_group = 0

        # assemble computation time vector
        times = torch.tensor([t_fit, t_select, t_sim])
        
        return times

    def _add_trial(self, trial: Trial, restore: bool = True) -> None:
        """ Add the data of a specified trial object to the optimizer's
        database, depending on whether the trial was feasible or not. Update all
        internal objects that contain information on the performed trials, i.e.
        the current hypervolume, optimization trace, and optimal parameters.

        Args:
            trial (Trial): an existing trial that is to be included in the model
        
        Returns:
            None
        """
        
        # update trial counter and add trial to list
        self.n_trials += 1
        self.trials.append(trial)

        # add the trial to the objective dataset only if it was successful
        if trial.feasible:
            for id in range(self.n_segment_groups):
                self.objective_data[id].add_data(
                    x=trial.params_nor, y=trial.objectives[id]
                )
        
        # add the trial to the constraint dataset
        feasibility = torch.tensor(
            trial.feasible, dtype=self.dtype_con, device=self.device
        )
        self.constraint_data.add_data(x=trial.params_nor, y=feasibility)
        
        for id in range(self.n_segment_groups):
            # compute hypervolume and update optimization trace
            if restore:
                bd = DominatedPartitioning(
                    ref_point=self.reference_points[id],
                    Y=self.objective_data[id].Y
                )
                self.hypervolumes[id] = bd.compute_hypervolume().item()

            # obtain the mask that marks the non-dominated objective data
            mask = is_non_dominated(Y=self.objective_data[id].Y)

            # update the stored Pareto data
            self.dominated_data[id] = self.objective_data[id].Y[~mask]
            self.nondominated_data[id] = self.objective_data[id].Y[mask]
            
            # obtain the Pareto-optimal trials
            successful_trials = np.array(
                [trial for trial in self.trials if trial.feasible]
            )
            mask = [e.item() for e in mask]
            self.pareto_trials[id] = successful_trials[mask]
            
        self.hv_traces.append(copy(self.hypervolumes))
    
    def _pooled_evaluation(self, parameterizations: Tensor) -> Tensor:
        """ Use a pool of subprocess workers to parallelize the CPU-intensive
        simulation runs. Invokes a subprocess for each worker. NOTE: it is
        important to hand a separate, existing controller object to each worker
        in order to avoid conflicts when generating the solvers at runtime. Such
        conflicts arise when trying to access the compilation code from multiple
        threads simultaneously.
        
        Args:
            parameterizations (Tensor): a number of parameterizations that are
                                        to be evaluated by the worker pool
        
        Returns:
            Tensor: results of the evaluation
        """

        # initialize a worker pool
        pool = multiprocessing.Pool(
            initializer=_init_pool,
            initargs=(self,),
            processes=self.n_processes
        )
        
        # assemble the argument list
        args = [[
            params,
            self.parameter_bounds,
            self.n_segment_groups,
            self.n_segments,
            self.track_segments,
            self.config,
            self.tkwargs
        ] for params in parameterizations]

        # use the pool workers to evaluate the given parameterizations
        results = pool.map(_evaluate_trial, args)

        # wait for all evaluations to finish
        pool.close()
        pool.join()

        return results

    def generate_initial_data(self, n: int) -> None:
        """ Generate an initial dataset from n random samples that are drawn
        uniformly from the parameter space.

        Args:
            n (int): the number of samples to generate
        
        Returns:
            None
        """

        # draw n random parameterizations
        parameterizations = draw_sobol_samples(
            bounds=self.normalized_bounds, n=n, q=1).squeeze()
        
        # evaluate the selected parameterizations using mutliple processes
        results = self._pooled_evaluation(parameterizations)
        for trial in results:
            self._add_trial(trial)
        
        return None
    
    def evaluate_at_boundaries(self) -> None:
        """ Evaluate the objective function at the boundaries of the decision
        space, i.e. at the extreme values of the parameters. This is likely to
        be done anyway during optimization but can save time when performed in
        the beginning. Will however lead to many failed iterations.

        Args:
            None
        
        Returns:
            None
        """

        # compute all permutations of the parameter extrema
        bounds = self.normalized_bounds.transpose(0, 1)
        permutations = itertools.product(*bounds)

        parameterizations = torch.tensor([
            perm for perm in permutations
        ])

        results = self._pooled_evaluation(parameterizations)
        for trial in results:
            self._add_trial(trial)

        return None

    def store_trials(self, filepath: str) -> None:
        """ Store all performed trials in a binary file to make them accessible
        for later use or for restoring an existing state.

        Args:
            filepath (str): relative filepath to the file that is generated
        
        Returns:
            None
        """

        if filepath is None:
            return None
        
        lines = []
        for trial in self.trials:
            lines.append(trial.to_csv())

        with open(filepath, 'w+') as file:
            file.writelines(lines)
        
        print(f"Stored {self.n_trials} trials at {filepath}.")

        return None

    def _load_trials(self, filepath: str) -> None:
        """ Restore trials from an existing file. This function is called
        automatically when initializing the BayesianOptimization object with a
        file location.

        Args:
            filepath (str): relative path to the file containing the trials
        
        Returns:
            None
        """

        trials = load_trials_from_csv(filepath=filepath)

        for trial in trials:
            self._add_trial(trial, restore=True)

        print(f"Loaded {self.n_trials} trials from {filepath}.")

        return None

    def fit_surrogate_models(self, segment_group_id: int) -> None:
        """ Fit the optimizer's surrogate models to the existing data.

        Args:
            segment_group_id (int): index of the segment group that is used to
                                    fit the models to
        
        Returns:
            bool: returns true if the model fitting was successful, false in the
                  contrary case
        """

        try:            
            # fit surrogate model to the objective function
            self.objective_model = ObjectiveModel(
                self.objective_data[segment_group_id].X,
                self.objective_data[segment_group_id].Y
            )
            
            # fit constraint model to the known feasibility values
            self.constraint_model = ConstraintModel(
                X=self.constraint_data.X,
                Y=self.constraint_data.Y.squeeze()
            )
            return True
        
        except ModelFittingError:
            return False