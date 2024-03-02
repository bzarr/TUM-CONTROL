"""
Created on Wed Jun  7 13:07:00 2023

@author:    Baha Zarrouki (baha.zarrouki@tum.de)
            Joao Nunes    (joao.nunes@tum.de)
"""

import numpy as np
import casadi as cs
import yaml
import os
import scipy
import torch
from torch import Tensor
from Model_Predictive_Controller.Reduced_Robustified_NMPC.Reduced_Robustified_NMPC_acados_settings import acados_settings
from Model_Predictive_Controller.Reduced_Robustified_NMPC.Robust_NMPC_pred_model_utils import P_propagation
import time
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

class Reduced_Robustified_Nonlinear_Model_Predictive_Controller:
    def __init__(self,config_path, MPC_params_file, sim_main_params, X0_MPC):
        ## --- MPC cost function params ---
        with open(config_path + MPC_params_file, 'r') as file:
            self.MPC_params = yaml.load(file, Loader = yaml.FullLoader)
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
        self.Qe  = self.Q      # np.diag([10.0, 20.0, 10.0, 1.0])   # terminal weight
        
        solver_generate_C_code  = self.MPC_params['solver_generate_C_code']
        solver_build            = self.MPC_params['solver_build']
        ## --- configure MPC constraints ---
        # load acceleration limits 
        lookuptable_gg_limits_file  = self.MPC_params['lookuptable_gg_limits']
        self.combined_acc_limits    = self.MPC_params['combined_acc_limits']
        # generate interpoltaed local ax & ay limits
        self.ax_lim, self.ay_lim = calculate_velvar_latlon_acc(config_path + lookuptable_gg_limits_file)

        ## --- Load MPC formulation ---
        self.constraint, self.model, self.acados_solver, self.costfunction_type, self.ocp, self.acc_grad, self.steering_grad = acados_settings(
            self.Tp, self.N, X0_MPC, 
            self.Q, self.R, self.Qe, self.L1_pen, self.L2_pen, self.ax_lim, self.ay_lim, self.combined_acc_limits,
            self.veh_params_full_path, self.tire_params_full_path, solver_generate_C_code = solver_generate_C_code, solver_build = solver_build)
        
        self.nx     = self.model.x.size()[0]
        self.x0     = X0_MPC
        self.acados_solver.constraints_set(0, "lbx", self.x0)
        self.acados_solver.constraints_set(0, "ubx", self.x0)
        self.stats  = np.zeros(5)
        self.pred_X = np.empty((0, self.nx)) # initialize empty array with shape (0,6)
        # nparam  = self.ocp.model.p.size()[0]

        ## --- Configure the Zoro Robust MPC ---
        ## --- Load Robust MPC Formulation ---
        self.tol = 0.00001      # TODO: which symbol in Paper and which Equation?
        # this is a tolerace used here in this work to avoid exact zero multiplication. It was used when 
        # developing, as the code would not converge. There is no appearence in any other work and (very likely)
        # can be discarded

        self.ZoRo = False # robustified original NMPC or robustified ZoRo NMPC
        self.stds = self.MPC_params['stds']
        # --- Load Robust MPC Formulation ---
        w_yaw     = self.stds[2] 
        w_vlong   = self.stds[3]  
        w_vlat    = self.stds[4]  
        w_yawrate = self.stds[5] 

        uncertainty_matrix_w = np.diag([
                                        w_yaw,
                                        w_vlong,
                                        w_vlat,
                                        w_yawrate,
                                        ])
        #  W_dist is the matrix that defines the ellipsoid in the disturbance space
        #  The Ellipsoid is defined as: (x-x0)'*W_dist*(x-x0) = 1 with x0 = 0
        #  https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15859-f11/www/notes/lecture08.pdf

        W_dist = (uncertainty_matrix_w) ** 2
        uncertainty_matrix_Sigma = np.diag([0.00001,
                                            0.00001,
                                            # 0.00001,
                                            w_yaw,
                                            w_vlong,
                                            w_vlat,
                                            w_yawrate,
                                            0.00001,
                                            0.00001,
                                            ])
        # TODO: which symbol in Paper and which Equation?
        coeff_Sigma = 0.5 # Sigma defines the ellipsoid where the state is defined
        # TODO: which symbol in Paper and which Equation?
        
        # Sigma_0_mat is the initial uncertainty matrix, just like x0. It defined the ellipsoid where the x0
        # is. In Zanelli's paper, it is the Sigma_0 in equation 5. In this work, specifically, it defined as a
        # coeficient scaling the different disturbances in each component to test which combination led to a best performance
        # (reducing the number of degrees of freedom). Therefore, there is no coeff_sigma in any other paper

        Sigma_0_mat = (coeff_Sigma * uncertainty_matrix_Sigma) ** 2

        self.Sigma_mat_list = [None] * (self.N+1)
        self.Sigma_mat_list[0] = Sigma_0_mat

        # This comes from the implementation of Zanelli's paper
        # Line 254 - https://github.com/FreyJo/zoro-NMPC-2021/blob/main/run_tailored_robust_control.py
        self.W_dist_disc = self.Ts_MPC * W_dist

        # Get the nominal bounds for later on robustification
        # 1) get the nominal bounds for the steering angle
        self.delta_f_min = self.model.delta_f_min
        self.delta_f_max = self.model.delta_f_max

        # 2) get the nominal bounds for the acceleration
        self.acc_min = self.ocp.constraints.lh
        self.acc_max = self.ocp.constraints.uh
        self.n_acc = len(self.acc_grad)

        # Maximum value that the acceleration can be clipped
        # acc_tol = 1

        # Maximum shooting node that the disturbance is propagated
        # after that, the correction term is equal to the last one

        self.uncertainty_propagation_horizon = self.MPC_params['uncertainty_propagation_horizon'] # 6 for Diamond Shaped, 8 or 2 for Circle Shaped

        self.dSigma_mat_list = [None] * (self.N+1)

        if not self.ocp.solver_options.nlp_solver_type == 'SQP_RTI':
            print(f"Zero order model only works for SQP_RTI solver")
            exit()

        # Hardcode B for discrete time disturbance - comes from line 244 
        # It will be used to calculate equation 8 Zanelli's paper
        # https://github.com/FreyJo/zoro-NMPC-2021/blob/main/run_tailored_robust_control.py
        self.B = np.array([[0, 0, 0, 0], 
                        [0, 0, 0, 0], 
                        [1, 0, 0, 0], 
                        [0, 1, 0, 0],        
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 0]])
        # self.B = np.array([ [0, 0, 0], 
        #                     [0, 0, 0], 
        #                     [0, 0, 0], 
        #                     [1, 0, 0],        
        #                     [0, 1, 0], 
        #                     [0, 0, 1], 
        #                     [0, 0, 0], 
        #                     [0, 0, 0]])
        self.correction_term_steering = 0
        self.correction_terms_acc = np.zeros((self.n_acc))
        # Parameters to be (possibly) later on learned:
        # 1) uncertainty matrix Sigma_0
        # 2) uncertainty matrix W_dist
        # 3) limit node

        ## --- END of Robust MPC Formulation ---
        # Get the number of non linear constraints in the stage- and terminal cost
        self.nh = self.ocp.model.con_h_expr.shape[0]
        self.nh_e = self.ocp.model.con_h_expr_e.shape[0]   

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
            if self.costfunction_type == 'EXTERNAL':
                yref = np.array([current_ref_traj['pos_x'][j],current_ref_traj['pos_y'][j],current_ref_traj['ref_yaw'][j],current_ref_traj['ref_v'][j]])
                self.acados_solver.set(j, "p", yref)
        yref_N = np.array([current_ref_traj['pos_x'][j+1],current_ref_traj['pos_y'][j+1],current_ref_traj['ref_yaw'][j+1],current_ref_traj['ref_v'][j+1]])
        if self.costfunction_type == 'NONLINEAR_LS':
            self.acados_solver.set(self.N, "yref", yref_N)
        if self.costfunction_type == 'EXTERNAL':
            self.acados_solver.set(self.N, "p", yref_N)
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
            back_offs = []
            pred_X = np.empty((0, self.nx)) # initialize empty array with shape (0,6)
            for stage in range(self.N):
                x       = self.acados_solver.get(stage,"x")
                x       = np.array(x) 
                pred_X  = np.concatenate((pred_X, x.reshape(1, -1)), axis=0)
            self.pred_X = pred_X
            # extract current MPC predicitons   
            # update pred_X only if a solution was found, else keep last solution to simulate with
            time_begin_constraint_tightening = time.time()
            for stage in range(self.uncertainty_propagation_horizon):
                #
                # This section is inspired in 235 to 282 from 
                # https://github.com/FreyJo/zoro-NMPC-2021/blob/main/run_tailored_robust_control.py
                # but changed to improve computational time and performance
                #
                # Update the constraints regarding the disturbances 
                # Get A matrices
                A = self.acados_solver.get_from_qp_in(stage, "A")

                # Get X
                x_sqp_loop = self.acados_solver.get(stage, "x")

                # Propagate Sigma
                self.Sigma_mat_list[stage + 1] = P_propagation(self.Sigma_mat_list[stage], A, self.B, self.W_dist_disc)
                
                #TODO: Optimize the code below
                if self.ZoRo:
                    P_mat_old = self.Sigma_mat_list[stage + 1]
                    if isinstance(P_mat_old, type(None)):
                        # i == 0
                        dSigma_bar = np.zeros((self.nx,self.nx))
                    else:
                        dSigma_bar = self.Sigma_mat_list[stage+1] - P_mat_old

                # compute backoff using P
                Sigma = self.Sigma_mat_list[stage]

                if stage > 0:
                    
                    # These two equations are inspired by line 267 of 
                    # https://github.com/FreyJo/zoro-NMPC-2021/blob/main/run_tailored_robust_control.py
                    # but in our case the constraint is not linear, like in the code
                    # therefore, the equation 14 from Zanelli's paper
                    # for the steering angle constraint
                    if self.ZoRo:
                        self.correction_term_steering = np.sqrt(Sigma[6, 6] + self.tol) + .5 * dSigma_bar[6, 6] / np.sqrt(Sigma[6, 6] + self.tol)
                    else:
                        steering_grad_val = self.steering_grad(x_sqp_loop)
                        self.correction_term_steering = float(np.sqrt(steering_grad_val.T @ Sigma @ steering_grad_val))
                    # CODE BELOW FOR CALCULATING the self.correction_term_steering the usual way (without simplification)
                    # As the acceleration constraint is not a direct state, we need to calculate using the whole expression
                    # steering_grad_val = steering_grad(x_sqp_loop)
                    # self.correction_term_steering = np.sqrt(steering_grad_val.T @ Sigma @ steering_grad_val + self.tol) + \
                    #         .5 * (np.sum(np.multiply(steering_grad_val @ steering_grad_val.T, dSigma_bar)) ) / \
                    #         np.sqrt(steering_grad_val.T @ Sigma @ steering_grad_val + self.tol)
                
                    # set bounds with backoff (nabla h available since h linear)
                    self.acados_solver.constraints_set(stage, "lbx", self.delta_f_min + self.correction_term_steering)
                    self.acados_solver.constraints_set(stage, "ubx", self.delta_f_max - self.correction_term_steering)
                    
                    self.correction_terms_acc = np.zeros((self.n_acc))

                    # CODE BELOW IF THE ACCELERATION CLIPPING IS NECESSARY
                    # for i_acc in range(n_acc):
                    #     acc_grad_val = acc_grad[i_acc](x_sqp_loop)
                    #     aux_acc = acc_grad_val.T @ Sigma @ acc_grad_val + self.tol
                    #     self.correction_terms_acc[i_acc] = np.minimum(np.maximum(np.sqrt(aux_acc) + \
                    #         .5 * (np.sum(np.multiply(acc_grad_val @ acc_grad_val.T, dSigma_bar)) ) / \
                    #         np.sqrt(aux_acc + self.tol), -acc_tol), acc_tol)
                    # CODE BELOW IF THE ACCELERATION CLIPPING IS NOT NECESSARY
                    for i_acc in range(self.n_acc):
                        acc_grad_val = self.acc_grad[i_acc](x_sqp_loop)  
                        aux_acc = acc_grad_val.T @ Sigma @ acc_grad_val
                        back_off = np.sqrt(aux_acc)
                        if self.ZoRo:
                            self.correction_terms_acc[i_acc] = np.sqrt(aux_acc) + \
                                .5 * (np.sum(np.multiply(acc_grad_val @ acc_grad_val.T, dSigma_bar)) ) / \
                                np.sqrt(aux_acc + self.tol)
                        else:
                            self.correction_terms_acc[i_acc] =  back_off
                    # self.acados_solver.constraints_set(stage, "lh", self.acc_min + self.correction_terms_acc)
                    self.acados_solver.constraints_set(stage, "uh", self.acc_max - self.correction_terms_acc)
            
            for stage in range(self.uncertainty_propagation_horizon, self.N):
                self.acados_solver.constraints_set(stage, "lbx", self.delta_f_min + self.correction_term_steering)
                self.acados_solver.constraints_set(stage, "ubx", self.delta_f_max - self.correction_term_steering)
                # self.acados_solver.constraints_set(stage, "lh", self.acc_min + self.correction_terms_acc)
                self.acados_solver.constraints_set(stage, "uh", self.acc_max - self.correction_terms_acc)
            time_end_constraint_tightening = time.time()
            self.stats[0] = self.acados_solver.get_cost()
            self.stats[1] = self.acados_solver.get_stats('time_tot') + time_end_constraint_tightening - time_begin_constraint_tightening
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
        self.acados_solver.constraints_set(0,"lbx", self.x0)
        self.acados_solver.constraints_set(0,"ubx", self.x0)
        return
 
    def reset(self, x0):
        self.acados_solver.reset()
        self.set_initial_state(x0)
        for i in range(self.N+1):
            self.acados_solver.set(i, 'x', self.x0)
    def reintialize_solver(self, X0_MPC, solver_generate_C_code = False, solver_build = False):
        ## --- Load MPC formulation ---
        costfunction_type = self.MPC_params['costfunction_type']
        self.constraint, self.model, self.acados_solver, self.costfunction_type, self.ocp, self.acc_grad, self.steering_grad = acados_settings(
            self.Tp, self.N, X0_MPC, 
            self.Q, self.R, self.Qe, self.L1_pen, self.L2_pen, self.ax_lim, self.ay_lim, self.combined_acc_limits,
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