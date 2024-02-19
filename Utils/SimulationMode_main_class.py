# Created on Tue Dec 06 17:26 2022

# Author: Baha Zarrouki (baha.zarrouki@tum.de)

import json
from Utils.MPC_sim_utils import *
import numpy as np
from collections import deque

def moving_average_filter(data, window_size):
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        if i < window_size:
            # For the first values, use a truncated moving average
            filtered_data[i] = np.mean(data[:i+1])
        else:
            # For subsequent values, use a moving average of the last window_size values
            filtered_data[i] = np.mean(data[i-window_size+1:i+1])
    return filtered_data[-1]

class MPC_Sim:
    def __init__(self, sim_main_params, logs_path):
        self.logs_path = logs_path
        ## load main simulation parameters
        self.sim_main_params = sim_main_params
        trajectory_path     = self.sim_main_params['trajectory_path']
        self.simMode        = self.sim_main_params['simMode'] # 0 -> CiL (MPC + simulation model: separate dynamics model)     |   1 -> MPC Sim (MPC + simulation model: MPC predictions)
        ref_trajectory_type = self.sim_main_params['ref_trajectory_type']  # 0: smooth continuous reference trajectory, 1: discrete set of trajectories

        self.Tp      = self.sim_main_params['Tp']     # prediction horizon [s]
        self.Ts_MPC  = self.sim_main_params['Ts_MPC'] # MPC prediction discretization step [s]
        self.Ts      = self.sim_main_params['Ts']     # Simulation sampling period [s]
        self.N       = int(self.Tp / self.Ts_MPC)          # number of discretizaion steps MPC
        self.T       = self.sim_main_params['T']      # simulation time [s]

        ## --- Load reference trajectory file ---
        track_file      = self.sim_main_params['track_file']
        ref_traj_file   = self.sim_main_params['ref_traj_file']
        with open(trajectory_path + ref_traj_file, 'r') as f:
            self.ref_traj_set = json.load(f)
        with open(trajectory_path + track_file, 'r') as f:
            self.track = json.load(f)

        # init vehicle position
        idx_ref_start = self.sim_main_params['idx_ref_start'] # start index on the trajectory
        self.current_pose = np.zeros(4)
        if ref_trajectory_type == 0:
            self.current_pose[0]       = self.ref_traj_set['pos_x'][idx_ref_start]
            self.current_pose[1]       = self.ref_traj_set['pos_y'][idx_ref_start]
            self.current_pose[2]       = postprocess_yaw(self.ref_traj_set['ref_yaw'][idx_ref_start])
            self.current_pose[3]       = self.ref_traj_set['ref_v'][idx_ref_start]
            # dt = np.linalg.norm(np.array([self.ref_traj_set['pos_x'][idx_ref_start],self.ref_traj_set['pos_y'][idx_ref_start]]) - np.array([self.ref_traj_set['pos_x'][idx_ref_start-1],self.ref_traj_set['pos_y'][idx_ref_start-1]]))/self.ref_traj_set['ref_v'][idx_ref_start]
            # self.current_pose['a_lon']       = 0#(self.ref_traj_set['ref_v'][idx_ref_start] - self.ref_traj_set['ref_v'][idx_ref_start-1]) / dt
            # self.current_pose['yaw_rate']    = 0#(postprocess_yaw(self.ref_traj_set['ref_yaw'][idx_ref_start]) - postprocess_yaw(self.ref_traj_set['ref_yaw'][idx_ref_start-1]))/dt
        else:
            self.current_pose[0]       = self.ref_traj_set[str(idx_ref_start)]['pos_x'][0]
            self.current_pose[1]       = self.ref_traj_set[str(idx_ref_start)]['pos_y'][0]
            self.current_pose[2]       = postprocess_yaw(self.ref_traj_set['ref_yaw'][0])
            self.current_pose[3]       = self.ref_traj_set[str(idx_ref_start)]['ref_v'][0]

        # initial state MPC: [x,y,yaw,v_lon_v_lat,yaw_rate,delta_f,acceleration]
        # heading = postprocess_yaw(math.atan2(self.current_pose[0], self.current_pose[1]))
        # delta_yaw = self.current_pose[2] - heading
        v_lon0 = self.current_pose[3]# * np.abs(math.cos(delta_yaw))
        v_lat0 = 0 #self.current_pose[3] #* np.abs(math.sin(delta_yaw))
        self.X0_MPC = np.array([self.current_pose[0], self.current_pose[1], self.current_pose[2], v_lon0 , v_lat0 , 0.0, 0.0, 0.0]) 

        # Vehicle Object or Vehicle Simulator Object
        self.Vehicle = None 

        # dimensions
        if self.simMode == 1:
            self.Nsim    = int(self.T / self.Ts_MPC)
        elif self.simMode == 0:
            self.Nsim    = int(self.T / self.Ts)
        
        # x0 = model.x0
        self.x_next      = self.X0_MPC                            # MPC first step
        self.tcomp_sum   = 0
        self.tcomp_max   = 0
        
        buffer_size = 15
        self.x_buffer = []
        for i in range(len(self.x_next)):
            self.x_buffer.append(deque(maxlen=buffer_size))
        self.window_sizes = np.array([1,1,4,2,2,3,4,2])
        return

    def set_Vehicle(self, Vehicle):
        self.Vehicle = Vehicle
        self.nx_sim  = self.Vehicle.x.size()[0]
        self.veh_length, self.veh_width  = self.Vehicle.params.veh_length, self.Vehicle.params.veh_width
        
        # --- Simulator Disturbance setup
        self.disturbances_state_derivatives_sim        = self.sim_main_params['simulate_disturbances']
        self.state_estimation_sim    = self.sim_main_params['simulate_state_estimation']
        if self.disturbances_state_derivatives_sim or self.state_estimation_sim:
                self.playback_disturbance, self.disturbance_range_derivatives, self.disturbance_range_state_estimation, self.disturbance_bounds_derivatives, self.disturbance_bounds_state_estimation, self.disturbance_types,self.sim_disturbance_derivatives, self.sim_disturbance_state_estimation = initDisturbanceSim(
                self.sim_main_params, self.logs_path, self.Nsim, self.nx_sim)
        else:
            self.sim_disturbance_derivatives, self.sim_disturbance_state_estimation = np.zeros((self.Nsim, self.nx_sim)), np.zeros((self.Nsim, self.nx_sim))
            self.playback_disturbance = False
        X0_sim = np.array([self.current_pose[0], self.current_pose[1], self.current_pose[2], self.current_pose[3], 0.0 , 0, 0.0])
        return X0_sim

    def sim_step(self, i, x, u0, pred_X):
        # --- Simulation Step: step to next vehicle state ---
        if self.simMode == 1:
            x_next                  = pred_X[1,:]
            x_next_MPC              = x_next
            x_next_sim              = x_next
            x_next_sim_disturbed    = x_next
        elif self.simMode == 0:
            # MPC computes jerk and steering rate, simulation model takes acceleration and steering rate as input
            x_next_MPC = pred_X[1,:]
            acceleration_input  = x_next_MPC[7]
            steering_rate_input = u0[1]
            sim_model_control_input = np.array([acceleration_input,steering_rate_input])
            x_next_sim = self.Vehicle.simulator_step(x, sim_model_control_input)
            x_next_sim = np.array(x_next_sim).reshape(-1)
            if not self.disturbances_state_derivatives_sim:
                disturbances_derivatives = np.zeros(self.nx_sim)
                x_next_sim_disturbed = x_next_sim
            else:
                if self.playback_disturbance:
                    disturbances_derivatives = self.sim_disturbance_derivatives[i, :]
                else:
                    # generate disturbances on the state derivatives
                    disturbances_derivatives = generate_disturbances(self.disturbance_bounds_derivatives, self.disturbance_types[0])
                x_next_sim_disturbed = self.Vehicle.simulator_step_disturbed(x, np.concatenate((sim_model_control_input,np.array(disturbances_derivatives)), axis=0))
                x_next_sim_disturbed = np.array(x_next_sim_disturbed).reshape(-1)
                # log disturbance
                for j in range(self.nx_sim):
                    self.sim_disturbance_derivatives[i, j]  = disturbances_derivatives[j]
            if self.state_estimation_sim:
                if self.playback_disturbance:
                    disturbances_state_estimation = self.sim_disturbance_state_estimation[i, :]
                else:
                    disturbances_state_estimation = generate_disturbances(self.disturbance_bounds_state_estimation, self.disturbance_types[1])
                x_next_sim_disturbed = [x_next_sim_disturbed[j] + disturbances_state_estimation[j] for j in range(len(x_next_sim_disturbed))]
                # log disturbance
                for j in range(self.nx_sim):
                    self.sim_disturbance_state_estimation[i, j]  = disturbances_state_estimation[j]
            # add acceleration state 
            x_next = np.append(x_next_sim_disturbed, acceleration_input)
            self.current_pose[0] = x_next_sim[0]
            self.current_pose[1] = x_next_sim[1]
            self.current_pose[2] = x_next_sim[2]
            self.current_pose[3] = np.sqrt(x_next_sim[3]**2 + x_next_sim[4]**2)
        return x_next, x_next_MPC, x_next_sim, x_next_sim_disturbed 
    
    def StateEstimation(self, x_next):     
        for i in range(len(x_next)):
            self.x_buffer[i].append(x_next[i])
            x_next[i] = moving_average_filter(np.array(self.x_buffer[i]), self.window_sizes[i])
        return x_next