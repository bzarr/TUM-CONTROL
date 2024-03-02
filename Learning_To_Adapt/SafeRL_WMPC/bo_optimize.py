import torch
import os
import sys
import argparse
import yaml

# add parent directory to import path
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(current_script_directory))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.bayesian_optimization import (
    BayesianOptimization
)
from Learning_To_Adapt.SafeRL_WMPC.helpers import load_config, visualize_surrogate


# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-load',
    default=0,
    const=0,
    nargs='?'
)
args = parser.parse_args()

if args.load:
    identifier = args.load
    config = load_config(f'Learning_To_Adapt/SafeRL_WMPC/_logs/{identifier}/bo_config.yaml')
    base_path = f'Learning_To_Adapt/SafeRL_WMPC/_logs/{identifier}'
    load_trials_path = os.path.join(base_path, f'{identifier}.csv')

else:
    config = load_config('Learning_To_Adapt/SafeRL_WMPC/_config/bo_config.yaml')
    identifier = config['identifier']

    base_path = f'Learning_To_Adapt/SafeRL_WMPC/_logs/{identifier}'
    
    # save config in run directory
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, 'bo_config.yaml'), 'w+') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    load_trials_path = None

# NOTE: CUDA usage significantly accelerates candidate selection, but can lead
#       to difficulties in multiprocessing
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
tkwargs = {
    'dtype': torch.double,
    'device': device
}

if config['save_trials']:
    save_trials_path = os.path.join(base_path, f'{identifier}.csv')
else:
    save_trials_path = None

# parameter bounds
parameter_bounds = torch.tensor([
    config['lims_qxy'],
    config['lims_qyaw'],
    config['lims_qvel'],
    config['lims_rjerk'],
    config['lims_rsteering'],
    config['lims_L1'],
    config['lims_L2']],
    **tkwargs
).transpose(0, 1)

# instantiate optimization object
optimizer = BayesianOptimization(
    parameter_bounds=parameter_bounds,
    config=config,
    tkwargs=tkwargs,
    trial_datafile=load_trials_path
)

# NOTE: visualize the surrogate models for debugging purposes
#visualize_surrogate(optimizer, group_id=1)

# sample the boundaries of the decision space
if config['do_boundary_sampling']:
    optimizer.evaluate_at_boundaries()
    optimizer.store_trials(filepath=save_trials_path)

# generate random initial data
if config['do_random_sampling']:
    optimizer.generate_initial_data(n=config['n_initial'])
    optimizer.store_trials(filepath=save_trials_path)

computation_times = torch.zeros((config['n_bayesian_optimization'], 3))

# perform optimization loop
if config['do_bayesian_optimization']:
    for i in range(0, config['n_bayesian_optimization']):

        print(f"\nIteration {i+1} of {config['n_bayesian_optimization']}")
        times = optimizer.perform_bayesian_optimization_step()
        
        print(f"Current hypervolumes: {optimizer.hypervolumes}")

        # flatten the list of Pareto optimal trials
        pareto_trials = [i for sub in optimizer.pareto_trials for i in sub]
        _k = len(list(set(pareto_trials)))
        print(f"Found {_k} unique Pareto-optimal parametrizations.")

        optimizer.store_trials(filepath=save_trials_path)

        # update time tensor only if iteration went as expected
        if times is not None:
            print(f"Full iteration took {torch.sum(times):.1f} seconds.")
            computation_times[i] = times

        # NOTE: visualize the surrogate models for debugging purposes
        #visualize_surrogate(optimizer, group_id=0)