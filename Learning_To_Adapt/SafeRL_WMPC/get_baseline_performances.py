import argparse
import os
import sys
import torch
import numpy as np
import time

# add parent directory to import path
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(current_script_directory))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import Nonlinear_Model_Predictive_Controller
from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator
from Utils.SimulationMode_main_class import MPC_Sim
from Utils.MPC_sim_utils import PlannerEmulator
from Utils.Logging_Plotting import Logger

""" Script to subsequently test all parameterizations contained in a set of
static weights. The results are required for performance comparison.

IMPORTANT:
To ensure functionality, make the following settings in MPC_params.yaml:
    solver_build: True
    solver_generate_C_code: True
    enable_WMPC: False

"""

CONFIG_PATH = 'Config/'
TRAJECTORY_PATH = 'Trajectories/'
LOGS_PATH       = 'Logs/'

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-parameters',
#     required=True,
#     type=str
# )
# args = parser.parse_args()

# identifier = args.parameters
identifier= 'F'

tracks = ['monteblanco']

# load parameter sets from saved location
parameters_file = os.path.join(
    'Learning_To_Adapt', 'SafeRL_WMPC', '_parameters', identifier
) + '.csv'
with open(parameters_file, 'r') as file:
    lines = file.readlines()
parameterizations = torch.empty((len(lines), 7))
for i, line in enumerate(lines):
    strs = line.strip().split(',')
    params = torch.tensor([float(p) for p in strs])
    parameterizations[i] = params

for track in tracks:

    print(f"Evaluating on track {track}.")

    # generate folder for the results
    baseline_dir = os.path.join(
        'Learning_To_Adapt', 'SafeRL_WMPC', '_baseline', identifier, track
    )
    os.makedirs(baseline_dir, exist_ok=True)

    # test each parameterization, run a simulation and save the results
    for params_id, params in enumerate(parameterizations):

        print(f'Running simulation with parameter set {params_id + 1} of {len(parameterizations)}.')

        sim_main_params_file = "EDGAR/sim_main_params.yaml"
        MPC_params_file = "EDGAR/MPC_params.yaml"

        sim = MPC_Sim(sim_main_params_file, CONFIG_PATH, TRAJECTORY_PATH, LOGS_PATH)

        ## -- Load Vehicle Simulator ---
        Passenger_Vehicle = PassengerVehicleSimulator(CONFIG_PATH, sim.sim_main_params, sim.Ts)
        X0_sim = sim.set_Vehicle(Passenger_Vehicle)

        ## --- Create MPC object ---
        MPC = Nonlinear_Model_Predictive_Controller(CONFIG_PATH, MPC_params_file, sim.sim_main_params, sim.X0_MPC)

        # update MPC parameters
        MPC.update_cost_function_weights(params)

        nx, nu     = MPC.nx, MPC.model.u.size()[0]
        ny      = nx + nu
        pred_X      = np.empty((0, nx)) # MPC Predictions

        ## --- Create Logger & Visualization Logger
        logger = Logger(sim.sim_main_params, sim.Nsim, nx, sim.nx_sim, nu, X0_sim, sim.X0_MPC, sim.current_pose, sim.track, sim.veh_length, sim.veh_width, Passenger_Vehicle.sim_constraints.alat, MPC)
        x_next = sim.X0_MPC
        t_start_sim = time.time()

        #for i in range(sim.Nsim):
        done = False
        i = 0
        for i in range(sim.Nsim):
            ## --- planner emulator ---: get reference trajectory based on current vehicle position
            current_ref_idx, current_ref_traj = PlannerEmulator(sim.ref_traj_set, sim.current_pose, sim.N+1, sim.Tp, loop_circuit=True)
            
            # end condition
            # if i >= 100 and current_ref_idx == 0:
            #     done = True
            
            ## --- solve MPC problem --- 
            t = time.time()
            u0, pred_X, MPC_stats = MPC.solve(current_ref_traj)
            if MPC_stats[-1] != 0:
                print("acados returned status {} in closed loop iteration {}.".format(MPC_stats[-1], i))
                MPC.reintialize_solver(x_next) 
            elapsed = time.time() - t
            # manage timings
            sim.tcomp_sum += elapsed
            if elapsed > sim.tcomp_max:
                sim.tcomp_max = elapsed
            
            ## --- Vehicle Simulation Step: step to next vehicle state ---
            x = logger.CiLX[i, :]
            x_next, x_next_MPC, x_next_sim, x_next_sim_disturbed = sim.sim_step(i, x, u0, pred_X)

            # update MPC initial state
            x_next = sim.StateEstimation(x_next)
            MPC.set_initial_state(x_next)

            ## --- Logging & Visualization ---
            logger.logging_step(i, u0, MPC_stats, current_ref_traj, x_next_sim, x_next_sim_disturbed, x_next_MPC)
            i += 1

        #logger.evaluation(sim.sim_main_params, LOGS_PATH, t_start_sim, sim.tcomp_sum, sim.tcomp_max, MPC.constraint, MPC.model, sim.sim_disturbance_derivatives, sim.sim_disturbance_state_estimation)

        logs_path = os.path.join(baseline_dir, f"{params_id}.npz")
        logger.save_logs(
            filepath=logs_path,
            sim_main_params=sim.sim_main_params,
            sim_disturbance_derivatives=sim.sim_disturbance_derivatives,
            sim_disturbance_state_estimation=sim.sim_disturbance_state_estimation
        )