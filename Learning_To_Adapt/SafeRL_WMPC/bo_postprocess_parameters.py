import torch
import argparse
import os
import sys
import matplotlib.pyplot as plt

# add parent directory to import path
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(current_script_directory))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.postprocessing import (
    get_pareto_optimal_trials, reduce_point_cloud, export_params
)
from Learning_To_Adapt.SafeRL_WMPC.helpers import load_trials_from_csv
from Utils.colors import *


"""
Arguments:
<identifier>    name of the BO logs stored at 'Data/optimization/'
<num>           number of parameter sets to extract from each group
<save>          whether or not to save the extracted parameter sets
"""
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-identifier',
#     required=True,
#     type=str
# )
# parser.add_argument(
#     '-num',
#     required=True,
#     type=int
# )
# parser.add_argument(
#     '-save',
#     action='store_true'
# )
# args = parser.parse_args()

class MyObject:
    pass

args = MyObject()
args.identifier = 'G'
args.num = 13
args.save = True
# load stored trial objects
filepath = os.path.join(
    'Learning_To_Adapt', 'SafeRL_WMPC', '_logs',
    args.identifier, args.identifier + '.csv'
)
trials = load_trials_from_csv(filepath=filepath)

# extract feasible trials
feasible_trials = [trial for trial in trials if trial.feasible]

# visualize the Pareto front
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(8, 4)
fig.set_dpi(200)

extracted = set()


for i in range(2):
    # obtain Pareto optimal trials
    (
        nondominated_trials, dominated_trials
    ) = get_pareto_optimal_trials(feasible_trials, group=i)

    # extract objective data
    nondominated = torch.stack([
        trial.objectives[i] for trial in nondominated_trials
    ])
    dominated = torch.stack([
        trial.objectives[i] for trial in dominated_trials
    ])

    # extract reduced point cloud
    try:
        reduced, ids = reduce_point_cloud(nondominated, args.num)
    except ValueError:
        print("Not enough pareto front points found! Please reduce the number of the wished points or rerun the optimization with different configuration!")
        sys.exit(1)

    # extract trials and add to set
    reduced_trials = [nondominated_trials[i] for i in ids]
    [extracted.add(trial) for trial in reduced_trials]
    extracted_group= set()
    [extracted_group.add(trial) for trial in reduced_trials]
    # export selected parameters
    if args.save:
        filepath = os.path.join(
            'Learning_To_Adapt', 'SafeRL_WMPC', '_parameters', args.identifier + '_' + str(i) +'.csv'
        )
        extracted_trials = [trial for trial in extracted_group]
        export_params(trials=extracted_trials, filepath=filepath)

    # plot dominated points, Pareto front, and redcued point cloud
    axs[i].scatter(
        -dominated[:,0], -dominated[:,1],
        color='black', marker='.', alpha=0.2, label='dominated'
    )
    axs[i].scatter(
        -nondominated[:,0], -nondominated[:,1],
        color=TUM_BLUE_4, marker='o', label='nondominated'
    )
    axs[i].scatter(
        -reduced[:,0], -reduced[:,1],
        color=TUM_BLUE, marker='o', label='selected'
    )

    axs[i].grid()
    axs[i].set_xlabel('Lateral deviation RMS in m')
    axs[i].set_ylabel('Velocity deviation RMS in m/s')

axs[1].legend()

axs[0].set_title('high-curvature')
axs[1].set_title('low-curvature')

# export selected parameters
if args.save:
    filepath = os.path.join(
        'Learning_To_Adapt', 'SafeRL_WMPC', '_parameters', args.identifier + '.csv'
    )
    extracted_trials = [trial for trial in extracted]
    export_params(trials=extracted_trials, filepath=filepath)
plt.tight_layout()
plt.savefig(os.path.join(
    'Learning_To_Adapt', 'SafeRL_WMPC', '_parameters', args.identifier + '.pdf'))

# plt.show()