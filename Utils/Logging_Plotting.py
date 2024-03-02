"""
Created on Thu Jun 15 21:47:38 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from Utils.MPC_sim_utils import *
import time
import imageio
import datetime
from typing import Tuple

# TODO: 2 Logging modes:
    # MODE1: predefined simulation time length 
    # MODE2: online --> undefined simulation time length
# TODO: automatic create and save a log file and save plots in online Mode
# TODO: extend Logger with a possibility to load saved data and use the final evaluation method 
class Logger:
    def __init__(self, sim, X0_sim, alat, MPC): #acc_min, ax_lim, ay_lim):
        self.MPC = MPC
        nref    = 4                # save refs for velocity and yaw
        self.nu = MPC.model.u.size()[0]
        self.nx = MPC.nx
        self.nx_sim = sim.nx_sim
        self.Nsim = sim.Nsim
        sim_main_params = sim.sim_main_params
        X0_MPC = sim.X0_MPC
        current_pose = sim.current_pose
        # initialize data structs
        if self.Nsim is not None:
            self.MPC_SimX        = np.ndarray((self.Nsim + 1, self.nx))     # MPC states
            self.CiLX            = np.ndarray((self.Nsim + 1, self.nx_sim)) # vehicle simulation states
            self.DisturbedX      = np.ndarray((self.Nsim + 1, self.nx_sim)) # disturbed vehicle simulation states
            self.simU            = np.ndarray((self.Nsim, self.nu))         # MPC optimal vehicle control inputs
            self.simREF          = np.ndarray((self.Nsim, nref))       # save references
            self.simSolverDebug  = np.ndarray((self.Nsim, 5))          # [cost, time_tot, sqp_iter, max_qp_iter, status]
            self.lat_devs        = np.zeros((self.Nsim))               # lateral deviation
            self.vel_devs        = np.zeros((self.Nsim))               # velocity deviation
            self.current_step    = 0
            # log first state
            for j in range(self.nx_sim):
                self.CiLX[0, j]          = X0_sim[j]
            for j in range(self.nx_sim):
                self.DisturbedX[0, j]    = X0_sim[j]    
            for j in range(self.nx):
                self.MPC_SimX[0, j]      = X0_MPC[j]
        else: 
            self.CiLX           = X0_MPC
            self.DisturbedX     = X0_MPC
            self.MPC_SimX       = X0_MPC
            self.simU           = np.zeros(self.nu)
            self.simREF         = np.zeros(nref)
            self.simSolverDebug = np.zeros(5)
        ## --- COMING SOON: Weights-varying MPC ---
        if MPC.WMPC:
            self.n_actions = 1       # TODO: this is hard coded, refine it by extracting the size of the action space from the loaded model
            if self.Nsim is not None:
                self.RL_actions      = np.zeros((self.Nsim + 1, self.n_actions))     # RL actions
                for j in range(self.n_actions):
                    self.RL_actions[0, j]          = None
            else:
                self.RL_actions = np.zeros(self.n_actions)
                for j in range(self.n_actions):
                    self.RL_actions[j] = None
        ## -- Live Visualization --
        self.live_visualization, self.live_plot_freq = sim_main_params['live_visualization'], sim_main_params['live_plot_freq']
        self.a_lat, self.acc_min, self.ax_lim, self.ay_lim = alat, self.MPC.model.acc_min, self.MPC.ax_lim, self.MPC.ay_lim
        self.Ts      = sim_main_params['Ts']     # Simulation sampling period [s]
        self.xwidth, self.ywidth      = sim_main_params['xwidth'], sim_main_params['ywidth']# plot width in [m]
        self.veh_length, self.veh_width = sim.veh_length, sim.veh_width
        self.track = sim.track
        X_c = self.track["X"]
        Y_c = self.track["Y"]
        X_i = self.track["X_i"]
        X_o = self.track["X_o"]
        Y_i = self.track["Y_i"]
        Y_o = self.track["Y_o"]
        if self.live_visualization != 0:
            # fig, ax = plt.subplots(figsize=(11,10*xwidth/ywidth+ 0.25))
            if self.live_visualization == 1: 
                self.fig, self.ax = plt.subplots(figsize=(10,10))
                self.MPC_pred_line, self.Ref_traj_line, self.Disturbed_traj_line,  self.ax,  self.center_line, self.inner_line,  self.outer_line,  self.line,  self.dot,  self.text= initLiveVisuMode1(self.ax, self.CiLX, current_pose, self.xwidth, self.ywidth, X_c, X_i, X_o, Y_c, Y_i, Y_o, self.veh_length, self.veh_width)
                ani = animation.FuncAnimation(self.fig, updateLiveVisuMode1, frames=self.Nsim, repeat=False, blit=True)
            if self.live_visualization == 2: 
                self.fig = plt.figure(figsize=(15,8))
                # manager = plt.get_current_fig_manager()
                # manager.full_screen_toggle()
                sub1 = self.fig.add_subplot(2,2,(1,3))
                sub3 = self.fig.add_subplot(2,4,3)
                sub3 = self.fig.add_subplot(2,4,4)
                sub3 = self.fig.add_subplot(2,4,(7,8))
                ax_list = self.fig.axes
                self.ax = ax_list[0]
                self.ax_vel = ax_list[1]
                self.ax_devlat = ax_list[2]
                self.ax_gg = ax_list[3]
                live_plot_sim_time_vec = np.linspace(0.0, 5*self.Ts,5) # random numbers to initialize the plot
                self.MPC_pred_line,self.Ref_traj_line,self.Disturbed_traj_line, self.ax, self.ax_devlat, self.ax_vel,self.ax_gg, self.heatmap, self.cbar,self.center_line,self.inner_line, self.outer_line, self.line, self.dot, self.text , self.vel_ref_line, self.vel_sim_line,self.devlat_line = initLiveVisuMode2(
                    live_plot_sim_time_vec,self.ax,self.ax_vel,self.ax_devlat,self.ax_gg, self.CiLX, self.simREF, current_pose, self.xwidth, self.ywidth, X_c, X_i, X_o, Y_c, Y_i, Y_o, self.veh_length, self.veh_width)
                ani = animation.FuncAnimation(self.fig, updateLiveVisuMode2, frames=self.Nsim, repeat=False, blit=True)
            im = plt.imread('Utils/TUM-CONTROL_logo.png') # insert local path of the image.
            newax = self.fig.add_axes([0.06, 0.007, 0.2, 0.07], anchor='SE', zorder=1)
            newax.imshow(im)
            newax.axis('off')
            # im = plt.imread('EDGAR_pic.png') # insert local path of the image.
            # edgar_ax = self.fig.add_axes([0.2, 0.00007, 0.2, 0.12], anchor='SE', zorder=1)
            # edgar_ax.imshow(im)
            # edgar_ax.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            self.GIF_animation_generation, self.GIF_file_name =  sim_main_params['GIF_animation_generation'], sim_main_params['GIF_file_name']

            if self.GIF_animation_generation:
                # Initialize an empty list to store the frames of the GIF
                self.frames = []
        else:
            self.GIF_animation_generation = False
            
        self.current_step = 0

    def logging_step(self, step, u0, MPC_stats, current_ref_traj, x_next_sim, x_next_sim_disturbed, x_next_MPC):
        self.current_step = step
        # log solution
        if self.Nsim is not None: 
            for j in range(self.nu):
                self.simU[self.current_step, j]      = u0[j]    
            # Debugging
            for j in range(len(MPC_stats)):
                self.simSolverDebug[self.current_step, j] = MPC_stats[j]
            # log current reference yaw and velocities 
            for j in range(self.nx_sim):
                self.CiLX[self.current_step+1, j]      = x_next_sim[j]
            for j in range(self.nx_sim):
                self.DisturbedX[self.current_step+1, j]= x_next_sim_disturbed[j]
            for j in range(self.nx):
                self.MPC_SimX[self.current_step+1, j]  = x_next_MPC[j]
            self.simREF[self.current_step, 0] = current_ref_traj['pos_x'][0]
            self.simREF[self.current_step, 1] = current_ref_traj['pos_y'][0]
            self.simREF[self.current_step, 2] = current_ref_traj['ref_yaw'][0]
            self.simREF[self.current_step, 3] = current_ref_traj['ref_v'][0]
        else:
            self.CiLX           = np.c_[self.CiLX,x_next_sim]
            self.DisturbedX     = np.c_[self.DisturbedX,x_next_sim_disturbed]
            self.MPC_SimX       = np.c_[self.MPC_SimX,x_next_MPC]
            self.simU           = np.c_[self.simU,u0]
            self.simREF         = np.c_[self.simREF, np.array([current_ref_traj['pos_x'][0],current_ref_traj['pos_y'][0],current_ref_traj['ref_yaw'][0],current_ref_traj['ref_v'][0]])]
            self.simSolverDebug = np.c_[self.simSolverDebug,MPC_stats]

        # log lateral and velocity deviation
        _, lat_dev = LonLatDeviations(
            self.CiLX[self.current_step, 2], self.CiLX[self.current_step, 0], self.CiLX[self.current_step, 1],
            self.simREF[self.current_step, 0], self.simREF[self.current_step, 1]
        )
        vel_dev = self.CiLX[self.current_step, 3] - self.simREF[self.current_step, 3]
        self.lat_devs[self.current_step] = lat_dev
        self.vel_devs[self.current_step] = vel_dev

        # store current reference trajectory
        self.current_ref_traj = current_ref_traj

        # log current combined acceleration
        alat = np.array(self.a_lat(np.transpose(self.CiLX[self.current_step]))).reshape(-1)
        alon = x_next_MPC[7]

        alat_lim = np.array(self.ay_lim(self.CiLX[self.current_step, 3])).reshape(-1)

        if alon > 0:
            alon_lim = np.array(self.ax_lim(self.CiLX[self.current_step, 3])).reshape(-1)
        else:
            alon_lim = self.acc_min

        alat_nor = float(alat / alat_lim)
        alon_nor = float(alon / alon_lim if alon > 0 else abs(alon) / alon_lim)

        acomb = np.sqrt(alon_nor**2 + alat_nor**2)
        self.acomb = acomb

        ## --- COMING SOON: Weights-varying MPC ---
        # Handle RL-WMPC actions
        if self.MPC.WMPC:
            if self.Nsim is not None: 
                if self.n_actions > 1:
                    for j in range(self.n_actions):
                        self.RL_actions[self.current_step + 1, j]      = self.MPC.current_action[j]
                else:
                    self.RL_actions[self.current_step + 1, 0] = self.MPC.current_action
            else:
                self.RL_actions           = np.c_[self.RL_actions,self.MPC.current_action]

    def step_live_visualization(self,i, t_start_sim, current_ref_idx, current_ref_traj, pred_X):
        # live simulation visualization
        if self.live_visualization != 0:
            if np.mod(i,self.live_plot_freq) == 0:
                if self.Nsim is not None:
                    if self.live_visualization == 1: 
                        updateLiveVisuMode1(i,time.time() - t_start_sim, self.CiLX, self.simU,self.DisturbedX, 
                                            current_ref_traj, pred_X, self.Ref_traj_line, self.MPC_pred_line, self.Disturbed_traj_line, self.ax, self.xwidth, self.ywidth, self.line, self.dot, self.text, self.veh_length, self.veh_width, current_ref_idx=current_ref_idx)
                    if self.live_visualization == 2:
                        live_plot_sim_time_vec = np.linspace(0.0, i*self.Ts,i)
                        a_lat_sim           = np.array(self.a_lat(np.transpose(self.CiLX[:i,:]))).reshape(-1)
                        updateLiveVisuMode2(i,live_plot_sim_time_vec,time.time() - t_start_sim,self.simREF, self.CiLX, self.DisturbedX, self.simU, self.MPC_SimX[:i,7], a_lat_sim, current_ref_traj, pred_X, self.acc_min, self.ax_lim, 
                                            self.ay_lim, self.Ref_traj_line, self.MPC_pred_line, self.Disturbed_traj_line,self.ax,self.ax_devlat, self.ax_vel,self.ax_gg,self.xwidth, self.ywidth, self.line, self.dot, self.text, 
                                            self.veh_length, self.veh_width, current_ref_idx, self.vel_ref_line, self.vel_sim_line, self.devlat_line, self.heatmap, self.cbar)
                else: 
                    CiLX           = self.CiLX.T
                    DisturbedX     = self.DisturbedX.T
                    MPC_SimX       = self.MPC_SimX.T
                    simU           = self.simU.T
                    simREF         = self.simREF.T
                    # posprocess --> remove the zeros at the start index 
                    # simU = simU[1:,:]
                    simREF = simREF[1:,:]
                    if self.live_visualization == 1: 
                        updateLiveVisuMode1(i,time.time() - t_start_sim, CiLX, simU,DisturbedX, 
                                            current_ref_traj, pred_X, self.Ref_traj_line, self.MPC_pred_line, self.Disturbed_traj_line, self.ax, self.xwidth, self.ywidth, self.line, self.dot, self.text, self.veh_length, self.veh_width, current_ref_idx=current_ref_idx)
                    if self.live_visualization == 2:
                        live_plot_sim_time_vec = np.linspace(0.0, i*self.Ts,i)
                        a_lat_sim           = np.array(self.a_lat(np.transpose(CiLX[:i,:]))).reshape(-1)
                        updateLiveVisuMode2(i,live_plot_sim_time_vec,time.time() - t_start_sim,simREF, CiLX, DisturbedX, simU, MPC_SimX[:i,7], a_lat_sim, current_ref_traj, pred_X, self.acc_min, self.ax_lim, 
                                            self.ay_lim, self.Ref_traj_line, self.MPC_pred_line, self.Disturbed_traj_line,self.ax,self.ax_devlat, self.ax_vel,self.ax_gg,self.xwidth, self.ywidth, self.line, self.dot, self.text, 
                                            self.veh_length, self.veh_width, current_ref_idx, self.vel_ref_line, self.vel_sim_line, self.devlat_line, self.heatmap, self.cbar)
                plt.pause(0.00001) # pause for a while to let the animation update
                if self.GIF_animation_generation:
                    frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                    self.frames.append(frame)
    
    def evaluation(self, sim_main_params, logs_path, t_start_sim,tcomp_sum, tcomp_max, MPC_constraints, MPC_model, sim_disturbance_derivatives, sim_disturbance_state_estimation):
        if self.Nsim is None: 
            self.CiLX           = self.CiLX.T
            self.DisturbedX     = self.DisturbedX.T
            self.MPC_SimX       = self.MPC_SimX.T
            self.simU           = self.simU.T
            self.simREF         = self.simREF.T
            self.simSolverDebug = self.simSolverDebug.T
            # posprocess --> remove the zeros at the start index 
            self.simU = self.simU[1:,:]
            self.simREF = self.simREF[1:,:]
            self.simSolverDebug = self.simSolverDebug[1:,:]
            self.Nsim = self.simU.shape[0]
        # Print some statistics
        time_needed_for_simulation = time.time() - t_start_sim
        # vel = np.sqrt(self.CiLX[:,3]**2 + self.CiLX[:,4]**2)
        vel = self.CiLX[:,3]
        print("Time needed for simulation: {}".format(time_needed_for_simulation))
        print("Average Time needed per iteration (inkl. reference traj finding): {}".format(time_needed_for_simulation / self.Nsim))
        print("Average computation time: {}".format(tcomp_sum / self.Nsim))
        print("Maximum computation time: {}".format(tcomp_max))
        print("Average speed:{}m/s".format(np.average(vel)))

        # Performance computations & post-processing
        dev_vel             = np.abs(vel[1:] - self.simREF[:,3])
        self.CiLX[:,2]       = postprocess_yaw(self.CiLX[:,2])
        self.MPC_SimX[:,2]   = postprocess_yaw(self.MPC_SimX[:,2])
        self.DisturbedX[:,2]   = postprocess_yaw(self.DisturbedX[:,2])

        dev_yaw             = np.abs((self.CiLX[1:,2]) -(self.simREF[:,2]))
        dev_long, dev_lat   = LonLatDeviations(self.CiLX[1:, 2], self.CiLX[1:, 0], self.CiLX[1:, 1], self.simREF[:, 0], self.simREF[:, 1])
        a_lat_sim           = np.array(self.a_lat(np.transpose(self.CiLX))).reshape(-1)
        # a_lat_pred          = np.array(MPC_constraints.a_lat(np.transpose(self.MPC_SimX))).reshape(-1)
        a_lat_pred          = self.MPC_SimX[:,3]*self.MPC_SimX[:,5]
        T = sim_main_params['T']
        t = np.linspace(0.0, T, self.Nsim)

        # save GIF 
        if self.GIF_animation_generation:
            # Save the frames as a GIF using imageio
            imageio.mimsave(self.GIF_file_name, self.frames, fps=10)

        # save logs
        now = datetime.datetime.now()
        save_logs = sim_main_params['save_logs']
        file_logs_name = logs_path + sim_main_params['file_logs_name'] + now.strftime("%Y-%m-%d_%H-%M-%S")
        if save_logs: 
            # Check if the directory exists
            if not os.path.exists(file_logs_name):
                # If it doesn't exist, create it
                os.makedirs(file_logs_name)
            np.savez(file_logs_name + "/full_logs.npz", MPC_SimX= self.MPC_SimX, CiLX=self.CiLX, simU=self.simU, simREF=self.simREF, simSolverDebug=self.simSolverDebug, sim_disturbance_derivatives=sim_disturbance_derivatives,sim_disturbance_state_estimation=sim_disturbance_state_estimation, a_lat=a_lat_sim, dev_lat =dev_lat, dev_long = dev_long, dev_vel = dev_vel, dev_yaw=dev_yaw, t = t)
            
            ## --- COMING SOON: Weights-varying MPC ---
            # Handle WMPC logs
            if self.MPC.WMPC:
                np.savez(file_logs_name + "/RL_WMPC_logs.npz", t = t, RL_actions=self.RL_actions, WMPC_sets=self.MPC.WMPC_parameter_sets)
        disturbances_state_derivatives_sim        = sim_main_params['simulate_disturbances']
        state_estimation_sim    = sim_main_params['simulate_state_estimation']

        # Plot results
        # plotTrackSim(self.track, SimX_x = self.CiLX[1:,0], SimX_y = self.CiLX[1:,1], SimX_vel= vel[1:], dev_lat= dev_lat, save_logs=save_logs, log_file_ID= file_logs_name)
        plotRes(self.CiLX[1:,:],self.simREF, self.MPC_SimX[1:,:], dev_lat, dev_long, a_lat_pred[1:], a_lat_sim[1:], MPC_model,MPC_constraints,self.ax_lim, self.ay_lim,MPC_acceleration= self.MPC_SimX[1:,7], MPC_steering_angle= self.MPC_SimX[1:,6], MPC_jerk= self.simU[:,0], MPC_steering_rate=self.simU[:,1], t=t, save_logs=save_logs, log_file_ID= file_logs_name)
        BoxPlots(dev_vel, dev_yaw, dev_lat, save_logs=save_logs, log_file_ID= file_logs_name)
        plotMPCperf(self.simSolverDebug, t, self.track, SimX_x = self.CiLX[1:,0], SimX_y = self.CiLX[1:,1], save_logs=save_logs, log_file_ID= file_logs_name)
        if disturbances_state_derivatives_sim:
            # plot disturbance realisation
            plotDisturbancesRealization(t,sim_disturbance_derivatives, save_logs=save_logs, log_file_ID= file_logs_name)
        if state_estimation_sim:
            # plot simulated state error: real state - measured state
            plotSimulatedStateErrors(t,sim_disturbance_state_estimation, save_logs=save_logs, log_file_ID= file_logs_name)
        if os.environ.get("ACADOS_ON_CI") is None:
            plt.show()

    """ Get current velocity, current lateral deviation, and current velocity
    deviation as required by the RL observation generator. """
    def get_observation_states(self) -> Tuple[float, float, float]:
        v = self.CiLX[self.current_step, 3]
        lat_dev = self.lat_devs[self.current_step]
        vel_dev = self.vel_devs[self.current_step]

        return v, lat_dev, vel_dev
    
    def save_logs(
            self,
            filepath: str,
            sim_main_params,
            sim_disturbance_derivatives,
            sim_disturbance_state_estimation
        ) -> None:

        self.truncate()

        if self.Nsim is None: 
            self.CiLX           = self.CiLX.T
            self.DisturbedX     = self.DisturbedX.T
            self.MPC_SimX       = self.MPC_SimX.T
            self.simU           = self.simU.T
            self.simREF         = self.simREF.T
            self.simSolverDebug = self.simSolverDebug.T
            # posprocess --> remove the zeros at the start index 
            self.simU = self.simU[1:,:]
            self.simREF = self.simREF[1:,:]
            self.simSolverDebug = self.simSolverDebug[1:,:]
            self.Nsim = self.simU.shape[0]
        
        vel = self.CiLX[:,3]

        # Performance computations & post-processing
        dev_vel             = np.abs(vel[1:] - self.simREF[:,3])
        self.CiLX[:,2]       = postprocess_yaw(self.CiLX[:,2])
        self.MPC_SimX[:,2]   = postprocess_yaw(self.MPC_SimX[:,2])
        self.DisturbedX[:,2]   = postprocess_yaw(self.DisturbedX[:,2])

        dev_yaw             = np.abs((self.CiLX[1:,2]) -(self.simREF[:,2]))
        dev_long, dev_lat   = LonLatDeviations(self.CiLX[1:, 2], self.CiLX[1:, 0], self.CiLX[1:, 1], self.simREF[:, 0], self.simREF[:, 1])
        a_lat_sim           = np.array(self.a_lat(np.transpose(self.CiLX))).reshape(-1)
        T = sim_main_params['T']
        t = np.linspace(0.0, T, self.Nsim)

        # save GIF 
        if self.GIF_animation_generation:
            # Save the frames as a GIF using imageio
            imageio.mimsave(self.GIF_file_name, self.frames, fps=10)

        # save logs
        np.savez(
            filepath,
            MPC_SimX=self.MPC_SimX,
            CiLX=self.CiLX,
            simU=self.simU,
            simREF=self.simREF,
            simSolverDebug=self.simSolverDebug,
            sim_disturbance_derivatives=sim_disturbance_derivatives,
            sim_disturbance_state_estimation=sim_disturbance_state_estimation,
            a_lat=a_lat_sim,
            dev_lat=dev_lat,
            dev_long=dev_long,
            dev_vel=dev_vel,
            dev_yaw=dev_yaw,
            t=t
        )

    """ Remove zeros from preallocated logging arrays. """
    def truncate(self):
        length = self.current_step
        self.MPC_SimX = self.MPC_SimX[0:length + 1]
        self.CiLX = self.CiLX[0:length + 1]
        self.DisturbedX = self.DisturbedX[0:length + 1]
        self.simU = self.simU[0:length]
        self.simREF = self.simREF[0:length]
        self.simSolverDebug = self.simSolverDebug[0:length]
        self.lat_devs = self.lat_devs[0:length]
        self.vel_devs = self.vel_devs[0:length]
        self.Nsim = length