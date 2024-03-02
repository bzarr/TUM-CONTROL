"""
Created on Fri Jun 30 13:07:00 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
import os
import numpy as np
import casadi as cs
import yaml
import scipy
import torch
from torch import Tensor
from Model_Predictive_Controller.Stochastic_NMPC.SNMPC_acados_settings import acados_settings        
from Model_Predictive_Controller.Stochastic_NMPC.stochastic_mpc_utils import computeSamplesAndAmatrix, alphaGeneration, compute_x0dist
from Utils.MPC_sim_utils import LonLatDeviations

'''
7. MPC
    - initialisation: 
        + loading MPC params
        + loading Prediciton model params 
        + loading tire model Params
        + loading constraint, model, acados_solver, costfunction_type
        + setting the initial state / constraints
    - step:
        + set current x_next_sim_disturbed (initial state / constraints)
        + set current reference trajectory
        + solve
        + get MPC solution: x0,u0
        + extract current MPC predictions
        + get SolverDebug Stats
'''
"""
Source: [1] Zarrouki, Baha, Chenyang Wang, and Johannes Betz. 
        "A stochastic nonlinear model predictive control with an uncertainty propagation horizon 
        for autonomous vehicle motion control." arXiv preprint arXiv:2310.18753 (2023).
"""
class Stochastic_Nonlinear_Model_Predictive_Controller:
    def __init__(self,config_path, MPC_params_file, sim_main_params, X0_MPC):
        ## --- MPC cost function params ---
        with open(config_path + MPC_params_file, 'r') as file:
            self.MPC_params = yaml.load(file, Loader=yaml.FullLoader)
        self.Tp      = sim_main_params['Tp']     # prediction horizon [s]
        self.Ts      = sim_main_params['Ts']
        self.Ts_MPC  = sim_main_params['Ts_MPC'] # MPC prediction discretization step [s]
        self.N       = int(self.Tp / self.Ts_MPC)          # number of discretizaion steps MPC
        ## --- Model params ---     
        veh_params_file_MPC         = sim_main_params['veh_params_file_MPC']
        tire_params_file_MPC        = sim_main_params['tire_params_file_MPC']
        self.tire_params_full_path  = config_path+tire_params_file_MPC
        self.veh_params_full_path   = config_path+veh_params_file_MPC
        ## --- configure MPC cost function ---
        # scaling factors
        s_lon           = self.MPC_params['s_lon']
        s_lat           = self.MPC_params['s_lat']
        s_yaw           = self.MPC_params['s_yaw']
        s_vel           = self.MPC_params['s_vel']
        s_jerk          = self.MPC_params['s_jerk']
        s_steering_rate = self.MPC_params['s_steering_rate']
        # weights
        q_lon           = self.MPC_params['q_lon']
        q_lat           = self.MPC_params['q_lat']
        q_yaw           = self.MPC_params['q_yaw']
        q_vel           = self.MPC_params['q_vel']
        r_jerk          = self.MPC_params['r_jerk']
        r_steering_rate = self.MPC_params['r_steering_rate']
        
        self.L1_pen = self.MPC_params['L1_pen']
        self.L2_pen = self.MPC_params['L2_pen']
        self.Q   = np.diag([q_lon* (1/s_lon**2), q_lat* (1/s_lat**2), q_yaw * (1/s_yaw**2), q_vel * (1/s_vel**2)])   # pos_x, pos_y, yaw, v
        self.R   = np.diag([r_jerk * (1/s_jerk**2), r_steering_rate * (1/s_steering_rate**2)])
        self.Qe  = self.Q      # terminal weight

        solver_generate_C_code  = self.MPC_params['solver_generate_C_code']
        solver_build            = self.MPC_params['solver_build']
        ## --- load SNMPC parameters ---
        self.SNMPC_params = {}
        self.n_samples = self.MPC_params['n_samples']
        self.stds = self.MPC_params['stds']
        # disturbance_type = MPC_params['disturbance_type']
        self.expansion_degree = self.MPC_params['expansion_degree']
        uncertainty_propagation_horizon = self.MPC_params['uncertainty_propagation_horizon']
        self.SNMPC_params['n_samples'] = self.MPC_params['n_samples']
        self.SNMPC_params['gamma'] = self.MPC_params['gamma']
        self.SNMPC_params['expansion_degree'] = self.MPC_params['expansion_degree']
        self.SNMPC_params['disturbance_type'] = self.MPC_params['disturbance_type']
        self.n_vars = len(np.nonzero(self.stds)[0])
        self.SNMPC_params['n_vars'] = self.n_vars

        # generate alphas in Eq.2 [1]
        self.alphas = alphaGeneration(self.n_vars, self.expansion_degree) 
        # num_poly_terms is L in Eq.3 [1]
        self.num_poly_terms     = int(np.math.factorial(self.n_vars + self.expansion_degree) / (np.math.factorial(self.n_vars) * np.math.factorial(self.expansion_degree)))
        self.w_samples, self.A  = computeSamplesAndAmatrix(self.n_samples,self.alphas,self.num_poly_terms,self.stds)
        x0_samples = compute_x0dist(X0_MPC,self.w_samples,self.n_samples,self.stds)
        # generate stop flags for UPH
        self.stop_flags = np.zeros((self.N+1))
        self.stop_flags[uncertainty_propagation_horizon:] = 1
        self.risk_parameter = np.array(self.MPC_params['gamma']).reshape(1)

        ## --- configure MPC constraints ---
        # load acceleration limits 
        lookuptable_gg_limits_file  = self.MPC_params['lookuptable_gg_limits']
        self.combined_acc_limits    = self.MPC_params['combined_acc_limits']
        # generate interpoltaed local ax & ay limits
        self.ax_lim, self.ay_lim = calculate_velvar_latlon_acc(config_path + lookuptable_gg_limits_file)

        ## --- Load MPC formulation ---
        costfunction_type = self.MPC_params['costfunction_type']
        self.constraint, self.model, self.acados_solver, self.ocp = acados_settings(self.Tp, self.N, x0_samples.flatten(), 
            self.Q, self.R, self.Qe, self.L1_pen, self.L2_pen, self.ax_lim, self.ay_lim, self.combined_acc_limits, self.SNMPC_params,
            self.veh_params_full_path, self.tire_params_full_path, solver_generate_C_code = solver_generate_C_code, solver_build = solver_build)
        
        self.nx     = int(self.model.x.size()[0] / (self.SNMPC_params['n_samples']+1))
        self.x0     = X0_MPC
        self.costfunction_type = self.ocp.cost.cost_type
        # Get the number of non linear constraints in the stage- and terminal cost
        self.nh = self.ocp.model.con_h_expr.shape[0]
        self.nh_e = self.ocp.model.con_h_expr_e.shape[0]
        self.acados_solver.constraints_set(0,"lbx", x0_samples.flatten())
        self.acados_solver.constraints_set(0,"ubx", x0_samples.flatten())
        if costfunction_type == 'NONLINEAR_LS':
            for i in range(self.N+1):
                self.acados_solver.set(i, "p", np.concatenate((self.A.flatten(),self.risk_parameter,self.stop_flags[i].reshape(1))))
        else:
            for i in range(self.N+1):
                self.acados_solver.set(i, "p", np.concatenate((np.zeros(4),self.A.flatten(),self.risk_parameter,self.stop_flags[i].reshape(1))))
        count = 0
        for j in range(self.N+1):
            self.acados_solver.set(j, "x", x0_samples.flatten())

        self.stats = np.zeros(5)
        self.pred_X = np.empty((0, self.nx)) # initialize empty array with shape (0,6)

        ## --- Weights-varying MPC (WMPC)---
        self.WMPC = False
        # setup WMPC if the respective flag is set
        if self.MPC_params['enable_WMPC']:
            self.WMPC = True
            from stable_baselines3 import PPO
            from Learning_To_Adapt.SafeRL_WMPC.helpers import load_config
            from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.observation import ObservationGenerator

            # load the trained model
            base_path = os.path.join(self.MPC_params['WMPC_model'])
            model_path = os.path.join(base_path, 'best_model', 'best_model')
            self.WMPC_model = PPO.load(model_path)

            # load the model config file
            config = load_config(os.path.join(base_path, 'rl_config.yaml'))

            # load the available parameter sets from the specified file
            with open(config['actions_file'], 'r') as file:
                lines = file.readlines()
            n_actions = len(lines)
            self.WMPC_parameter_sets = torch.empty((n_actions, 7))
            for i, line in enumerate(lines):
                strs = line.strip().split(',')
                params = torch.tensor([float(p) for p in strs])
                self.WMPC_parameter_sets[i] = params

            # set up observation generator and tensor containing the observations
            self.observation_generator = ObservationGenerator(
                anticipation_horizon=config['obs_anticipation_horizon'],
                n_anticipation_points=config['obs_n_anticipation_points']
            )
            self.n_obs_stack = config['n_obs_stack']
            self.obs = torch.zeros(
                (self.observation_generator.n_observations * self.n_obs_stack)
            )

            # initialize weight update scheduler
            self.steps_since_weight_update = 0
            self.weight_update_period = self.MPC_params['weights_update_period']
            self.current_action = None

        return
    
    def solve(self, current_ref_traj):
        # update MPC reference
        for j in range(self.N):
            if self.costfunction_type == 'NONLINEAR_LS':
                yref = np.array([current_ref_traj['pos_x'][j],current_ref_traj['pos_y'][j],current_ref_traj['ref_yaw'][j],current_ref_traj['ref_v'][j], 0,0])
                self.acados_solver.set(j, "yref", yref)
                self.acados_solver.set(j, "p", np.concatenate((self.A.flatten(),self.risk_parameter,self.stop_flags[j].reshape(1))))
            if self.costfunction_type == 'EXTERNAL':
                yref = np.array([current_ref_traj['pos_x'][j],current_ref_traj['pos_y'][j],current_ref_traj['ref_yaw'][j],current_ref_traj['ref_v'][j]])
                self.acados_solver.set(j, "p", yref)
                self.acados_solver.set(j, "p", np.concatenate((yref,self.A.flatten(),self.risk_parameter,self.stop_flags[j].reshape(1))))
        yref_N = np.array([current_ref_traj['pos_x'][j+1],current_ref_traj['pos_y'][j+1],current_ref_traj['ref_yaw'][j+1],current_ref_traj['ref_v'][j+1]])
        if self.costfunction_type == 'NONLINEAR_LS':
            self.acados_solver.set(self.N, "yref", yref_N)
            self.acados_solver.set(self.N, "p", np.concatenate((self.A.flatten(),self.risk_parameter,self.stop_flags[self.N].reshape(1))))
        if self.costfunction_type == 'EXTERNAL':
            self.acados_solver.set(self.N, "p", yref_N)
            self.acados_solver.set(self.N, "p", np.concatenate((yref_N,self.A.flatten(),self.risk_parameter,self.stop_flags[self.N].reshape(1))))
        # solve MPC problem
        status = self.acados_solver.solve()
        # hpipm_status: 
            # 0: SUCCESS, // found solution satisfying accuracy tolerance
            # 1: MAX_ITER, // maximum iteration number reached
            # 2: MIN_STEP, // minimum step length reached
            # 3: NAN_SOL, // NaN in solution detected
            # 4: INCONS_EQ, // unconsistent equality constraints

        
        # get MPC solution
        # x0 = self.acados_solver.get(0, "x")
        u0 = self.acados_solver.get(0, "u")
        # u0 = acados_solver.solve_for_x0(x_next)
        if status == 0: 
            pred_X = np.empty((0, self.nx)) # initialize empty array with shape (0,6)
            for j in range(self.N):
                x       = self.acados_solver.get(j,"x")[0:8]
                x       = np.array(x) 
                pred_X  = np.concatenate((pred_X, x.reshape(1, -1)), axis=0)
            self.pred_X = pred_X
        self.stats[0] = self.acados_solver.get_cost()
        self.stats[1] = self.acados_solver.get_stats('time_tot')
        self.stats[2] = self.acados_solver.get_stats('sqp_iter')
        self.stats[3] = np.max(self.acados_solver.get_stats('qp_iter'))
        self.stats[4] = status

        ## --- Weights-varying MPC ---
        if self.WMPC:
            # weights update step
            if self.steps_since_weight_update >= self.weight_update_period:
                self.steps_since_weight_update = 0
                v = self.x0[3]
                _, lat_dev = LonLatDeviations(
                    self.x0[2], self.x0[0], self.x0[1],
                    current_ref_traj['pos_x'][0], current_ref_traj['pos_y'][0]
                )
                vel_dev = self.x0[3] - current_ref_traj['ref_v'][0]
                # obtain observation
                _obs = self.observation_generator.get_observation(
                    v, lat_dev, vel_dev, current_ref_traj, self.Ts
                )

                # if using observation stacking, shift the new observation into the stack
                if self.n_obs_stack > 1:
                    n_obs = self.observation_generator.n_observations
                    self.obs = torch.roll(self.obs, shifts=-n_obs)
                    self.obs[-n_obs:] = torch.tensor(_obs)
                else:
                    self.obs = _obs

                # select a parameter set as predicted by the model
                self.current_action, _ = self.WMPC_model.predict(self.obs, deterministic=True)
                params = self.WMPC_parameter_sets[self.current_action]

                # update the controller parameters
                self.update_cost_function_weights(params)

            self.steps_since_weight_update += 1

        return u0, self.pred_X, self.stats

    def set_initial_state(self, x0):
        self.x0 = x0
        # x0_samples, self.A = computeSamplesAndAmatrix(self.n_samples,self.alphas,self.num_poly_terms, self.x0, self.stds)
        x0_samples = compute_x0dist(x0,self.w_samples,self.n_samples,self.stds)
        self.acados_solver.constraints_set(0,"lbx", x0_samples.flatten())
        self.acados_solver.constraints_set(0,"ubx", x0_samples.flatten())

    def reset(self, x0):
        self.acados_solver.reset()
        x0_samples = compute_x0dist(x0,self.w_samples,self.n_samples,self.stds)
        self.acados_solver.constraints_set(0,"lbx", x0_samples.flatten())
        self.acados_solver.constraints_set(0,"ubx", x0_samples.flatten())
        for i in range(self.N+1):
            self.acados_solver.set(i, 'x', x0_samples.flatten())

    def reintialize_solver(self, X0_MPC, solver_generate_C_code = False, solver_build = False):
        ## --- Load MPC formulation ---
        costfunction_type = self.MPC_params['costfunction_type']
        # self.w_samples, self.A = computeSamplesAndAmatrix(self.n_samples,self.alphas,self.num_poly_terms,self.stds)
        x0_samples = compute_x0dist(X0_MPC,self.w_samples,self.n_samples,self.stds)
        self.constraint, self.model, self.acados_solver, self.costfunction_type = acados_settings(self.Tp, self.N, x0_samples.flatten(), 
            self.Q, self.R, self.Qe, self.L1_pen, self.L2_pen, self.ax_lim, self.ay_lim, self.combined_acc_limits, self.SNMPC_params,
            self.veh_params_full_path, self.tire_params_full_path, solver_generate_C_code = solver_generate_C_code, solver_build = solver_build)
        self.set_initial_state(X0_MPC)
        return
    
    def update_cost_function_weights(self, params: Tensor):

        if isinstance(params, Tensor):
            params = params.numpy()

        # assign cost function weights
        Q = np.diag([
            params[0],  # q_xy
            params[0],  # q_xy
            params[1],  # q_yaw
            params[2],  # q_vel
        ])
        R = np.diag([
            params[3],  # r_jerk
            params[4],  # r_steering_rate
        ])
        Qe  = Q
        # assign slack term penalties
        L1 = params[5]
        L2 = params[6]

        W = scipy.linalg.block_diag(Q, R)
        We = Qe

        # update cost function weights
        for i in range(self.N):
            self.acados_solver.cost_set(i, 'W', W)
        self.acados_solver.cost_set(self.N, 'W', We)

        # update slack terms
        z0 = np.ones((self.nh + 0,)) * L1
        Z0 = np.ones((self.nh + 0,)) * L2
        self.acados_solver.cost_set(0, 'zl', z0)
        self.acados_solver.cost_set(0, 'zu', z0)
        self.acados_solver.cost_set(0, 'Zl', Z0)
        self.acados_solver.cost_set(0, 'Zu', Z0)
        z_e = np.ones((self.nh_e + 1,)) * L1
        Z_e = np.ones((self.nh_e + 1,)) * L2
        self.acados_solver.cost_set(self.N, 'zl', z_e)
        self.acados_solver.cost_set(self.N, 'zu', z_e)
        self.acados_solver.cost_set(self.N, 'Zl', Z_e)
        self.acados_solver.cost_set(self.N, 'Zu', Z_e)
        z = np.ones((self.nh + 2,)) * L1
        Z = np.ones((self.nh + 2,)) * L2
        for i in range(1, self.N):
            self.acados_solver.cost_set(i, 'zl', z)
            self.acados_solver.cost_set(i, 'zu', z)
            self.acados_solver.cost_set(i, 'Zl', Z)
            self.acados_solver.cost_set(i, 'Zu', Z)

import pandas as pd
# calculate interpolated lateral and longitudinal acceleration limits based on current velocity 
def calculate_velvar_latlon_acc(lookuptable_gg_limits_file):
    # Read the lookup table from a CSV file
    # table = np.genfromtxt(lookuptable_gg_limits_file, delimiter=',', skip_header=1)
    data = pd.read_csv(lookuptable_gg_limits_file)
    # Extract the velocity, max lateral acceleration, and max longitudinal acceleration data
    velocity = np.array(data['vel_max_mps'])
    ax_max_mps2 = np.array(data['ax_max_mps2'])
    ay_max_mps2 = np.array(data['ay_max_mps2'])

    # Create interpolation functions for max lateral acceleration and max longitudinal acceleration using casadi.interpolant
    ax_max_interpolant = cs.interpolant('ax_max_interpolant', 'linear', [velocity], ax_max_mps2)
    ay_max_interpolant = cs.interpolant('ay_max_interpolant', 'linear', [velocity], ay_max_mps2)

    return ax_max_interpolant, ay_max_interpolant