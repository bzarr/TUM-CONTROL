# Created on Tue Dec 06 17:26 2022

# Author: Baha Zarrouki (baha.zarrouki@tum.de)

import time
import numpy as np
from casadi import *
from Utils.MPC_sim_utils import *
from Utils.Logging_Plotting import Logger
import yaml

from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator

""" Here several possible NMPCs: choose the one that suits you while creating the MPC object, 
-----just replace it with the controller wou wish ;) """
# from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import Nonlinear_Model_Predictive_Controller as Model_Predictive_Controller
# from Model_Predictive_Controller.Stochastic_NMPC.SNMPC_class import Stochastic_Nonlinear_Model_Predictive_Controller as Model_Predictive_Controller
from Model_Predictive_Controller.Reduced_Robustified_NMPC.Reduced_Robustified_NMPC_class import Reduced_Robustified_Nonlinear_Model_Predictive_Controller as Model_Predictive_Controller

from Utils.SimulationMode_main_class import MPC_Sim

config_path, logs_path  = 'Config/', 'Logs/'

## --- sim main params ---
sim_main_params_file    = "EDGAR/sim_main_params.yaml"
MPC_params_file         = "EDGAR/MPC_params.yaml" 
with open(config_path + sim_main_params_file, 'r') as file:
    sim_main_params = yaml.load(file, Loader=yaml.FullLoader)

## --- Create Simulation Object ---
sim = MPC_Sim(sim_main_params, logs_path)

## -- Load Vehicle Simulator ---
Passenger_Vehicle = PassengerVehicleSimulator(config_path, sim.sim_main_params, sim.Ts)
X0_sim = sim.set_Vehicle(Passenger_Vehicle)

## --- Create MPC object ---
MPC = Model_Predictive_Controller(config_path, MPC_params_file, sim.sim_main_params, sim.X0_MPC)
nx, nu  = MPC.nx, MPC.model.u.size()[0]
ny      = nx + nu
pred_X  = np.empty((0, nx)) # MPC Predictions

## --- Create Logger & Visualization Logger ---
Logger = Logger(sim, X0_sim, Passenger_Vehicle.sim_constraints.alat, MPC)
x_next = sim.X0_MPC
t_start_sim = time.time()
## --- Simulate ---
for i in range(sim.Nsim):
    
    ## --- planner emulator ---: get reference trajectory based on current vehicle position
    current_ref_idx, current_ref_traj = PlannerEmulator(sim.ref_traj_set, sim.current_pose, sim.N+1, sim.Tp, loop_circuit=True)
    if len(current_ref_traj['pos_x']) < sim.N:
        print(" --- MPC optimization is ending as ref traj is shorter than N ... ---")
        break
    
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
    x = Logger.CiLX[i, :]
    x_next, x_next_MPC, x_next_sim, x_next_sim_disturbed = sim.sim_step(i, x, u0, pred_X)

    # update MPC initial state
    x_next = sim.StateEstimation(x_next)
    MPC.set_initial_state(x_next)

    ## --- Logging & Visualization ---
    Logger.logging_step(i, u0, MPC_stats, current_ref_traj, x_next_sim, x_next_sim_disturbed, x_next_MPC)
    Logger.step_live_visualization(i, t_start_sim, current_ref_idx, current_ref_traj, pred_X)

## --- Evaluate ---
Logger.evaluation(sim.sim_main_params, logs_path, t_start_sim, sim.tcomp_sum, sim.tcomp_max, MPC.constraint, MPC.model, sim.sim_disturbance_derivatives, sim.sim_disturbance_state_estimation)