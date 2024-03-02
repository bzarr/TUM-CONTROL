import torch
import numpy as np
import yaml

from torch import Tensor
from typing import List, Tuple

from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator
from Utils.Logging_Plotting import Logger
from Utils.MPC_sim_utils import PlannerEmulator
from Utils.SimulationMode_main_class import MPC_Sim
from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import (
    Nonlinear_Model_Predictive_Controller
)
from Learning_To_Adapt.SafeRL_WMPC.helpers import get_root_mean_square


CONFIG_PATH = 'Config/'
TRAJECTORY_PATH = 'Trajectories/'
LOGS_PATH = 'Logs/'
SIM_MAIN_PARAMS_FILE = "EDGAR/sim_main_params.yaml"
MPC_PARAMS_FILE = "EDGAR/MPC_params.yaml"


def objective_function(
        MPC: Nonlinear_Model_Predictive_Controller,
        parameterization: Tensor,
        n_segment_groups: int,
        n_segments: List[int],
        track_segments: List[List[dict]],
        tkwargs: dict,
        config: dict
    ) -> Tuple[Tensor, bool]:

    """ Test the vehicle behavior for a given parameter set, return the
    combined performance metrics and a feasibility indicator.
    
    Args:
        MPC (MPC): controller to use for simulation (must be precomputed to
                   avoid conflicts when accessing the solver code from multiple
                   processes simultaneously)
        parameterization (Tensor): parameter set to test
        n_segment_groups (int): number of segment groups
        n_segments (list): number of segments per group
        track_segments (list): track segments as dicts containing start and end
                               indices
        tkwargs (dict): torch key args, required to assign device and datatype
        config (dict): BO configuration dictionary

    Returns:
        Tensor: objective values
        bool: feasibility indicator
    """

    n_objectives = 2

    MPC.update_cost_function_weights(parameterization)

    # preallocate space for the results of each segment group
    objectives = torch.zeros(
        n_segment_groups, n_objectives, **tkwargs)
    feasible = True

    # evaluate the parameterization for each segment group
    for group_id in range(n_segment_groups):

        # preallocate space for the results of each segment
        objs = torch.zeros(
            n_segments[group_id], n_objectives, **tkwargs
        )
        feas = True
        
        # evaluate the parametrization for each segment
        for i, segment in enumerate(track_segments[group_id]):
            trajectory  = segment['trajectory']

            ## Overwrite standard config file
            with open(CONFIG_PATH + SIM_MAIN_PARAMS_FILE, 'r') as file:
                sim_main_params = yaml.load(file, Loader=yaml.FullLoader)
            sim_main_params['track_file'] = f'track_{trajectory}.json'
            sim_main_params['ref_traj_file'] = f'reftraj_{trajectory}_edgar.json'
            sim_main_params['idx_ref_start'] = segment['start']
            
            # setup a simulation object
            simulation = MPC_Sim(
                sim_main_params,
                LOGS_PATH,
            )
            # setup a vehicle model
            vehicle = PassengerVehicleSimulator(
                CONFIG_PATH, simulation.sim_main_params, simulation.Ts
            )
            
            # reset MPC to start location
            X0_sim = simulation.set_Vehicle(vehicle)
            MPC.reset(simulation.X0_MPC)
            nx, nu = MPC.nx, MPC.model.u.size()[0]

            # initialize logger
            logger = Logger(
                simulation,  X0_sim, 
                vehicle.sim_constraints.alat,
                MPC
            )

            # simulate the behavior on the given segment
            done, crash = False, False
            while not done and not crash:
                # plan local trajectory
                current_ref_idx, current_ref_traj = PlannerEmulator(
                    simulation.ref_traj_set, simulation.current_pose,
                    simulation.N+1, simulation.Tp, loop_circuit=True
                )
                
                # solve optimal control problem
                u0, pred_X, MPC_stats = MPC.solve(current_ref_traj)
                
                # step vehicle model
                x = logger.CiLX[logger.current_step, :]
                (
                    x_next,
                    x_next_MPC,
                    x_next_sim,
                    x_next_sim_disturbed
                ) = simulation.sim_step(logger.current_step, x, u0, pred_X)

                # update MPC initial state
                x_next = simulation.StateEstimation(x_next)
                MPC.set_initial_state(x_next)

                # log system states
                logger.logging_step(
                    logger.current_step, u0, MPC_stats, current_ref_traj,
                    x_next_sim, x_next_sim_disturbed, x_next_MPC
                )
                logger.current_step += 1

                # check end condition
                done = current_ref_idx == segment['end']

                # check crash condition
                crash = _check_crash_condition(
                    logger=logger,
                    max_lat_dev=config['max_lat_dev'], 
                    max_a_comb=config['max_a_comb']
                )

                #if crash:
                #    logger.truncate()
                #    logger.evaluation(simulation.sim_main_params, LOGS_PATH, 0, simulation.tcomp_sum, simulation.tcomp_max, MPC.constraint, MPC.model, simulation.sim_disturbance_derivatives, simulation.sim_disturbance_state_estimation)


            # obtain the performance metrics
            # (with negative sign since BoTorch assumes maximization)
            objs[i,:] = _get_objectives(logger=logger)

            # if the simulation failed, stop iterating
            if crash:
                feas = False
                break

        # combine the results of the individual segments into one objective
        objectives[group_id] = torch.mean(objs, dim=0)

        # if the group was unfeasible, break
        if not feas:
            feasible = False
            break

    # if any run was unfeasible, assign NaN to all objectives
    if not feasible:
        objectives = torch.ones_like(objectives) * np.nan


    return objectives, feasible

""" Calculate objectives from a given logger object. """
def _get_objectives(logger: Logger) -> Tensor:
    return torch.tensor([
        # - get_root_mean_square(logger.lat_devs[:logger.current_step]),
        # - get_root_mean_square(logger.vel_devs[:logger.current_step])
        - np.max(np.abs(logger.lat_devs[:logger.current_step])),
        # - np.max(np.abs(logger.vel_devs[:logger.current_step]))
        - get_root_mean_square(logger.vel_devs[:logger.current_step])
    ])

""" Check if the vehicle violates crash constraints. """
def _check_crash_condition(
        logger: Logger, max_lat_dev: float, max_a_comb: float
    ) -> bool:
    
    max_dev_violated = logger.lat_devs[logger.current_step - 1] > max_lat_dev
    max_acomb_violated = logger.acomb > max_a_comb

    if max_acomb_violated:
        print(f"Violated max_a_comb")
    if max_dev_violated:
        print("Violated max_dev")

    return max_dev_violated or max_acomb_violated