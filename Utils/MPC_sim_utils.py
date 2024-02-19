# Created on Tue Dec 06 17:32 2022

# Author: Baha Zarrouki (baha.zarrouki@tum.de)

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import math
import pandas as pd
import csv
from scipy.interpolate import interp1d
import casadi as cs
import os
import scipy

def initDisturbanceSim(sim_main_params, logs_path, Nsim, nx_sim):
    simulate_disturbances   = sim_main_params['simulate_disturbances']
    state_estimation        = sim_main_params['simulate_state_estimation']
    if simulate_disturbances or state_estimation:
        # create disturbances range vector (bounded disturbances)
        disturbance_types  = [sim_main_params['disturbance_type_derivatives'],sim_main_params['disturbance_type_state_estimation']]
        disturbance_range_derivatives = {'posx_dot': [-sim_main_params['w_posx_dot'], sim_main_params['w_posx_dot']],
                        'posy_dot': [-sim_main_params['w_posy_dot'], sim_main_params['w_posy_dot']],
                        'yaw_dot': [-sim_main_params['w_yaw_dot'], sim_main_params['w_yaw_dot']], 
                        'vlong_dot': [-sim_main_params['w_vlong_dot'], sim_main_params['w_vlong_dot']],
                        'vlat_dot': [-sim_main_params['w_vlat_dot'], sim_main_params['w_vlat_dot']],
                        'yawrate_dot': [-sim_main_params['w_yawrate_dot'], sim_main_params['w_yawrate_dot']], 
                        'delta_f_dot': [-sim_main_params['w_delta_f_dot'], sim_main_params['w_delta_f_dot']]}
        disturbance_range_state_estimation = {'posx': [-sim_main_params['w_posx'], sim_main_params['w_posx']],
                        'posy': [-sim_main_params['w_posy'], sim_main_params['w_posy']],
                        'yaw': [-sim_main_params['w_yaw'], sim_main_params['w_yaw']], 
                        'vlong': [-sim_main_params['w_vlong'], sim_main_params['w_vlong']],
                        'vlat': [-sim_main_params['w_vlat'], sim_main_params['w_vlat']],
                        'yawrate': [-sim_main_params['w_yawrate'], sim_main_params['w_yawrate']], 
                        'delta_f': [-sim_main_params['w_delta_f'], sim_main_params['w_delta_f']]}
        # Initialize the disturbance dictionary
        disturbance_bounds_derivatives = [disturbance_range_derivatives[key] for key in disturbance_range_derivatives.keys()]
        disturbance_bounds_state_estimation = [disturbance_range_state_estimation[key] for key in disturbance_range_state_estimation.keys()]
        # Check if we need to playback a certain disturbance realization
        playback_disturbance = sim_main_params['disturbance_playback']
        if playback_disturbance:
            disturbance_playback_log_file = sim_main_params['playback_log_file']
            loaded_data = np.load(os.path.join(logs_path, disturbance_playback_log_file))
            sim_disturbance_derivatives = loaded_data['sim_disturbance_derivatives']
            sim_disturbance_state_estimation = loaded_data['sim_disturbance_state_estimation']
        else:
            sim_disturbance_derivatives         = np.zeros((Nsim, nx_sim)) # vehicle simulation states
            sim_disturbance_state_estimation    = np.zeros((Nsim, nx_sim)) # vehicle simulation states

    return playback_disturbance,disturbance_range_derivatives,disturbance_range_state_estimation, disturbance_bounds_derivatives, disturbance_bounds_state_estimation, disturbance_types, sim_disturbance_derivatives, sim_disturbance_state_estimation


# Define a function for generating random disturbances
def generate_disturbances(disturbance_bounds, disturbance_type):
    if disturbance_type == 'uniform':
        # Sample disturbances from a uniform distribution within the specified range
        return sampleFromEllipsoid(np.zeros(len(disturbance_bounds)),np.diag(np.array(disturbance_bounds).T[1,:]))
        # return [np.random.uniform(disturbance_bounds[j][0], disturbance_bounds[j][1]) for j in range(len(disturbance_bounds))]
    elif disturbance_type == 'gaussian':
        # mean of the Gaussian disturbance: 0 and standard deviation: disturbance_bounds
        return [np.random.normal(0, disturbance_bounds[j][1]) for j in range(len(disturbance_bounds))]
    elif disturbance_type == 'absolute':
        return [disturbance_bounds[j][1] for j in range(len(disturbance_bounds))]
    # default: uniform
    return [np.random.uniform(disturbance_bounds[j][0], disturbance_bounds[j][1]) for j in range(len(disturbance_bounds))]


def sampleFromEllipsoid(w, Z):
    """
    draws uniform(?) sample from ellipsoid with center w and variability matrix Z
    """

    n = w.shape[0]                  # dimension
    lam, v = np.linalg.eig(Z)

    # sample in hypersphere
    r = np.random.rand()**(1/n)     # radial position of sample
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    x *= r
    # project to ellipsoid
    y = v @ (np.sqrt(lam) * x) + w

    return y

# Define a function for generating random disturbances
def simulate_state_estimation(states, error_bounds, disturbance_type):
    if disturbance_type == 'uniform':
        # Sample disturbances from a uniform distribution within the specified range
        return [states[j] + np.random.uniform(error_bounds[j][0], error_bounds[j][1]) 
                for j in range(len(states))]
    elif disturbance_type == 'gaussian':
        # mean of the Gaussian disturbance: 0 and standard deviation: disturbance_bounds
        return [states[j] + np.random.normal(0, error_bounds[j][1]) for j in range(len(states))]
    elif disturbance_type == 'absolute':
        return [states[j] + error_bounds[j][1] for j in range(len(states))]
    # default: uniform
    return [states[j] + np.random.uniform(error_bounds[j][0], error_bounds[j][1]) 
            for j in range(len(states))]


def LonLatDeviations(ego_yaw, ego_x, ego_y, ref_x,ref_y):
    '''
    This method is based on rotating the deviation vectors by the negative 
    of the yaw angle of the vehicle, which aligns the deviation vectors with 
    the longitudinal and lateral axes of the vehicle.
    '''
    rotcos      = np.cos(-ego_yaw)
    rotsin      = np.sin(-ego_yaw)
    dev_long    = rotcos * (ref_x - ego_x) - rotsin * (ref_y - ego_y)
    dev_lat     = rotsin * (ref_x - ego_x) + rotcos * (ref_y - ego_y)
    return dev_long, dev_lat

def LatLonDeviation(ego_x, ego_y, ref_x,ref_y):
    # calculate the deviation vectors
    deviation_x = ego_x - ref_x
    deviation_y = ego_y - ref_y

    # calculate the longitudinal deviations as the magnitude of the deviation vectors
    longitudinal_deviation = np.sqrt(deviation_x**2 + deviation_y**2)

    # calculate the lateral deviations as the magnitude of the cross product of the deviation vectors and the reference trajectory vectors, divided by the magnitude of the reference trajectory vectors.
    lateral_deviation = np.abs(np.cross(np.column_stack((deviation_x, deviation_y)), np.column_stack((ref_x, ref_y)))) / np.sqrt(ref_x**2 + ref_y**2)
    return longitudinal_deviation, lateral_deviation

def postprocess_yaw(yaw):
    if isinstance(yaw, (list, np.ndarray)):
        yaw = np.fmod(yaw, 2*np.pi)
        yaw[yaw < 0] += 2*np.pi
        return yaw
    else:
        yaw = math.fmod(yaw, 2*math.pi)
        if yaw < 0:
            yaw += 2*math.pi
        return yaw

def PlannerEmulator(ref_traj_set, current_pose, N, Tp, loop_circuit):
    
    # 1. Step: calculate the euclidean distance between the current pose and each point in the reference trajectory
    #euclidean_dist = [np.linalg.norm(np.array([ref_traj_set['pos_x'][i],ref_traj_set['pos_y'][i]]) - np.array([current_pose[0],current_pose[1]])) for i in range(len(ref_traj_set['pos_x']))]
    
    a = np.array([current_pose[0:2]])
    b = np.column_stack([ref_traj_set['pos_x'], ref_traj_set['pos_y']])

    dists = scipy.spatial.distance.cdist(a, b)
    closest_point_index = np.argmin(dists)
    
    # find the index of the point in the reference trajectory with the minimum euclidean distance
    #closest_point_index = euclidean_dist.index(min(euclidean_dist))
        # shift closest_point_index to extract trajectory 5 points behind the vehicle: not a good idea
        # closest_point_index -= 5
        # if closest_point_index < 0:
        #     closest_point_index = 0
        
    # 2. Step: extract trajectory indexes that are in the N*Ts temporal horizon 
    temporal_idx= list()
    T = 0
    temporal_idx.append(closest_point_index)
    while T <= Tp:
        # check if we are at the end of the trajectory:
        curr_idx = temporal_idx[-1]
        if curr_idx + 1 >= len(ref_traj_set['pos_x']):
            if loop_circuit:
                temporal_idx.append(0)
            else:
                print("trajectory extraction failed: END OF TRAJECTORY REACHED")
                # return
        else:    
            temporal_idx.append(curr_idx + 1)
        T += np.linalg.norm(np.array([ref_traj_set['pos_x'][temporal_idx[-1]],ref_traj_set['pos_y'][temporal_idx[-1]]]) - np.array([ref_traj_set['pos_x'][temporal_idx[-2]],ref_traj_set['pos_y'][temporal_idx[-2]]]))/ref_traj_set['ref_v'][temporal_idx[-1]]

    # 3. Step: extract the trajectory corresponding to the indexes 
    extracted_traj = {key: [value[i] for i in temporal_idx] for key, value in ref_traj_set.items()}

    # 4. Step: interpolate/extrapolate values to have N traj points with Ts distance
    # Create an array of linearly interpolated values
    if N != len(extracted_traj['pos_x']):
        final_extracted_traj = {}
        for key in extracted_traj.keys():
            interpolated_values = np.interp(np.linspace(0, len(extracted_traj[key])-1, N), np.arange(len(extracted_traj[key])), extracted_traj[key])
            # original reference yaw is defined in [0, 2pi] and jumps from 0 to 2pi and backwards if the interval is exceeded
            # interp generates values in between, which is not correct --> solution:  
            if key == "ref_yaw":
                if (np.abs(np.diff(extracted_traj[key])) > np.deg2rad(250)).any():
                    x = np.linspace(0, len(extracted_traj[key])-1, N)
                    xp = np.arange(len(extracted_traj[key]))
                    fp = extracted_traj[key]
                    period = 2*np.pi
                    interpolated_values = np.mod(np.interp(x, xp, np.unwrap(fp, period=period)), period)
            final_extracted_traj[key] = interpolated_values
    else: 
        final_extracted_traj = extracted_traj
    
    return closest_point_index, final_extracted_traj

# TODO: fix PlannerEmulator when no trajectory is given --> at the end of the trajectory file when not in circuit
def PlannerEmulator_old(ref_traj_set, current_pose, N, Tp, loop_circuit):
    
    # 1. Step: calculate the euclidean distance between the current pose and each point in the reference trajectory
    euclidean_dist = [np.linalg.norm(np.array([ref_traj_set['pos_x'][i],ref_traj_set['pos_y'][i]]) - np.array([current_pose[0],current_pose[1]])) for i in range(len(ref_traj_set['pos_x']))]
    # find the index of the point in the reference trajectory with the minimum euclidean distance
    closest_point_index = euclidean_dist.index(min(euclidean_dist))
        # shift closest_point_index to extract trajectory 5 points behind the vehicle: not a good idea
        # closest_point_index -= 5
        # if closest_point_index < 0:
        #     closest_point_index = 0
        
    # 2. Step: extract trajectory indexes that are in the N*Ts temporal horizon 
    temporal_idx= list()
    T = 0
    temporal_idx.append(closest_point_index)
    while T <= Tp:
        # check if we are at the end of the trajectory:
        curr_idx = temporal_idx[-1]
        if curr_idx + 1 >= len(ref_traj_set['pos_x']):
            if loop_circuit:
                temporal_idx.append(0)
            else:
                print("trajectory extraction failed: END OF TRAJECTORY REACHED")
                # return
        else:    
            temporal_idx.append(curr_idx + 1)
        T += np.linalg.norm(np.array([ref_traj_set['pos_x'][temporal_idx[-1]],ref_traj_set['pos_y'][temporal_idx[-1]]]) - np.array([ref_traj_set['pos_x'][temporal_idx[-2]],ref_traj_set['pos_y'][temporal_idx[-2]]]))/ref_traj_set['ref_v'][temporal_idx[-1]]

    # 3. Step: extract the trajectory corresponding to the indexes 
    extracted_traj = {key: [value[i] for i in temporal_idx] for key, value in ref_traj_set.items()}

    # 4. Step: interpolate/extrapolate values to have N traj points with Ts distance
    # Create an array of linearly interpolated values
    if N != len(extracted_traj['pos_x']):
        final_extracted_traj = {}
        for key in extracted_traj.keys():
            interpolated_values = np.interp(np.linspace(0, len(extracted_traj[key])-1, N), np.arange(len(extracted_traj[key])), extracted_traj[key])
            # original reference yaw is defined in [0, 2pi] and jumps from 0 to 2pi and backwards if the interval is exceeded
            # interp generates values in between, which is not correct --> solution:  
            if key == "ref_yaw":
                if (np.abs(np.diff(extracted_traj[key])) > np.deg2rad(250)).any():
                    x = np.linspace(0, len(extracted_traj[key])-1, N)
                    xp = np.arange(len(extracted_traj[key]))
                    fp = extracted_traj[key]
                    period = 2*np.pi
                    interpolated_values = np.mod(np.interp(x, xp, np.unwrap(fp, period=period)), period)
            final_extracted_traj[key] = interpolated_values
    else: 
        final_extracted_traj = extracted_traj
    return closest_point_index, final_extracted_traj


def gen_car_shape(x, y, yaw, length, width):
    """
        Generates 2 vectors containing the floats for the positions from the car
        x: float, x-coordenate of the vehicle center
        y: float, y-coordenate of the vehicle center
        yaw: float, yaw angle of the vehicle
        return x_shape, y_shape, arrays containing the desired plotting points
    """
    xl = 0.5 * np.cos(yaw) * length
    xw = 0.5 * np.sin(yaw) * width
    yl = 0.5 * np.sin(yaw) * length
    yw = 0.5 * np.cos(yaw) * width * (-1)

    shape_x = [x + xl + xw, x + xl - xw, x - xl - xw, x - xl + xw, x + xl + xw]
    shape_y = [y + yl + yw, y + yl - yw, y - yl - yw, y - yl + yw, y + yl + yw]

    return shape_x, shape_y


def initLiveVisuMode2(live_plot_sim_time_vec,ax, ax_vel,ax_devlat,ax_gg,CiLX,simREF, current_pose,xwidth, ywidth, X_c, X_i, X_o, Y_c, Y_i, Y_o, veh_length, veh_width):
    start_idx, end_idx = 0, len(X_c)
    center_line, = ax.plot(X_c[start_idx:end_idx], Y_c[start_idx:end_idx], "k--", alpha=1, linewidth=1)
    inner_line, = ax.plot(X_i[start_idx:end_idx], Y_i[start_idx:end_idx], "k-")
    outer_line, = ax.plot(X_o[start_idx:end_idx], Y_o[start_idx:end_idx], "k-")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")    
    ax.set_aspect('equal', 'box')
    # initialize the plot with the first point of the simulation
    MPC_pred_line, = ax.plot(0, 0, label='current MPC predicted trajectory', marker='d', markersize=2, color='b')
    Ref_traj_line, = ax.plot(0, 0, label='current reference trajectory',marker='d', markersize=2, color='r')
    Disturbed_traj_line, = ax.plot(0, 0, label='disturbed MPC initial states',marker='x', markersize=4, color='m')
    line, = ax.plot(0, 0, "g-", label='ego driven trajectory')
    cur_x, cur_y, cur_yaw = current_pose[0], current_pose[1], current_pose[2]
    shape_x, shape_y = gen_car_shape(cur_x, cur_y, cur_yaw, length= veh_length, width= veh_width)
    dot, = ax.plot(shape_x, shape_y, "k-")
    text = ax.text(0.05, 1., '', transform=ax.transAxes, fontsize='xx-large', color = 'g')
    ax.set_xlim(cur_x-0.25*xwidth, cur_x+0.75*xwidth)
    ax.set_ylim(cur_y-0.25*ywidth, cur_y+0.75*ywidth)
    ax.legend() # add legend

    ax_vel.set_title("velocity")
    vel_ref_line, = ax_vel.plot(live_plot_sim_time_vec, np.zeros(len(live_plot_sim_time_vec)), label='reference')
    vel_sim_line, = ax_vel.plot(live_plot_sim_time_vec, np.zeros(len(live_plot_sim_time_vec)), label='simulated')
    ax_vel.set_ylabel("v [m/s]")
    ax_vel.set_xlabel("t [s]")  
    ax_vel.legend() # add legend
    ax_vel.grid(True)


    ax_devlat.set_title("lateral deviation")
    # dev_long, dev_lat   = LonLatDeviations(CiLX[:5, 2], CiLX[:5, 0], CiLX[:5, 1], simREF[:5, 0], simREF[:5, 1])
    devlat_line, = ax_devlat.plot(live_plot_sim_time_vec, np.zeros(len(live_plot_sim_time_vec)))
    ax_devlat.set_ylabel("deviation [m]")
    ax_devlat.set_xlabel("t [s]")  
    ax_devlat.grid(True)

    acc_0 = np.zeros(5)
    ax_gg.set_aspect('equal', 'box')
    # heatmap, = ax_gg.plot(acc_0, acc_0, marker='o')
    heatmap = plt.scatter(acc_0, acc_0, c=acc_0, cmap=cm.rainbow, edgecolor='none', marker='o')

    cbar = plt.colorbar(heatmap, fraction=0.035, location = 'left', ax = ax_gg)
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label("velocity [m/s]", rotation=90)
    # Set the limits of the plot
    ax_gg.set_xlim([-1.1, 1.1])
    ax_gg.set_ylim([-1.1, 1.1])
    # plot Kammscher Kreis
    # Generate an array of angles
    theta = np.linspace(0, 2*np.pi, 100)
    # Generate the x and y coordinates of the circle
    xmax = np.cos(theta)
    ymax = np.sin(theta)
    # plot the circle
    ax_gg.plot(xmax, ymax)
    # plot a diamond
    # x_diamond = [0, 1, 0, -1, 0]
    # y_diamond = [1, 0, -1, 0, 1]
    # ax_gg.plot(x_diamond, y_diamond)
    ax_gg.set_title('accelerations')
    # Use the MathText renderer
    plt.rcParams['text.usetex'] = False
    ax_gg.set_xlabel('$a_y/a_{ymax}$', fontsize = 'large')
    ax_gg.set_ylabel('$a_x/a_{xmax}$', fontsize = 'large')
    ax_gg.yaxis.tick_right()
    ax_gg.yaxis.set_label_position("right")
    ax_gg.grid(True)
    return MPC_pred_line,Ref_traj_line,Disturbed_traj_line, ax, ax_devlat, ax_vel,ax_gg, heatmap,cbar, center_line,inner_line, outer_line, line, dot, text, vel_ref_line, vel_sim_line,devlat_line


def updateLiveVisuMode2(sim_step,live_plot_sim_time_vec, time,simREF, CiLX, DisturbedX, SimU, MPC_acceleration,
                   a_lat_sim,current_ref_traj, pred_X,acc_min, ax_lim, ay_lim,Ref_traj_line, 
                   MPC_pred_line,Disturbed_traj_line, ax, ax_devlat, ax_vel,ax_gg,xwidth, ywidth, line, dot, text, 
                   veh_length, veh_width, current_ref_idx, vel_ref_line, vel_sim_line,
                   devlat_line,heatmap,cbar):
    # start_idx, end_idx = 0, len(X_c)
    # center_line.set_data(X_c[start_idx:end_idx], Y_c[start_idx:end_idx])
    # inner_line.set_data(X_i[start_idx:end_idx], Y_i[start_idx:end_idx])
    # outer_line.set_data(X_o[start_idx:end_idx], Y_o[start_idx:end_idx])
    Ref_traj_line.set_data(current_ref_traj['pos_x'], current_ref_traj['pos_y'])
    MPC_pred_line.set_data(pred_X[:, 0], pred_X[:, 1])
    Disturbed_traj_line.set_data(DisturbedX[:sim_step, 0], DisturbedX[:sim_step, 1])
    line.set_data(CiLX[:sim_step, 0], CiLX[:sim_step, 1])
    cur_x, cur_y, cur_yaw = CiLX[sim_step, 0], CiLX[sim_step, 1], CiLX[sim_step, 2]  
    shape_x, shape_y = gen_car_shape(cur_x, cur_y, cur_yaw, length= veh_length, width= veh_width)
    dot.set_data(shape_x, shape_y)

    # Get the last point of the reference trajectory
    last_x, last_y = current_ref_traj['pos_x'][-1], current_ref_traj['pos_y'][-1]
    # Get the current x and y limits of the plot
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xmin, xmax = xlim
    ymin, ymax = ylim

    if not (last_x > xmin + 0.2*(xmax-xmin) and last_x < xmin + 0.8*(xmax-xmin) and last_y > ymin + 0.2*(ymax-ymin) and last_y < ymin + 0.8*(ymax-ymin)):
        yaw = math.fmod(current_ref_traj['ref_yaw'][0], 2*math.pi)
        if yaw < 0:
            yaw += 2*math.pi
        if 0 <= yaw < math.pi/2:            
            # The nearest corner is the top right corner
            ax.set_xlim(cur_x-0.2*xwidth, cur_x+1.0*xwidth)
            ax.set_ylim(cur_y-0.2*ywidth, cur_y+1.0*ywidth)
        elif math.pi/2 <= yaw < math.pi:
            # The nearest corner is the top left corner
            ax.set_xlim(cur_x-1.0*xwidth, cur_x+0.2*xwidth)
            ax.set_ylim(cur_y-0.2*ywidth, cur_y+1.0*ywidth)
        elif math.pi <= yaw < 3*math.pi/2:
            # The nearest corner is the bottom left corner
            ax.set_xlim(cur_x-1.0*xwidth, cur_x+0.2*xwidth)
            ax.set_ylim(cur_y-1.0*ywidth, cur_y+0.2*ywidth)
        else:
            # The nearest corner is the bottom right corner
            ax.set_xlim(cur_x-0.2*xwidth, cur_x+1.0*xwidth)
            ax.set_ylim(cur_y-1.0*ywidth, cur_y+0.2*ywidth)
    # delta_f = np.rad2deg(SimU[sim_step, 1])
    delta_f = np.rad2deg(CiLX[sim_step,6])
    # velocity = np.sqrt(CiLX[sim_step,3]**2 + CiLX[sim_step,4]**2) * 3.6
    velocity = CiLX[sim_step,3] * 3.6
    # text.set_text("steering angle: {:.2f}°, velocity: {:.2f}km/h, sim step: {}, sim time: {:.2f}s, trajectory index: {}".format(delta_f, velocity, sim_step, time, current_ref_idx))
    text.set_text("steering angle: {:.2f}°,      velocity: {:.2f}km/h".format(delta_f, velocity))
    if sim_step >= 5:
        vel_ref_line.set_data(live_plot_sim_time_vec, simREF[:sim_step,3])
        sim_vel = CiLX[:sim_step,3]
        vel_sim_line.set_data(live_plot_sim_time_vec,sim_vel)
        ax_vel.set_xlim(0, max(live_plot_sim_time_vec))
        ax_vel.set_ylim(0.95*min(min(sim_vel),min(simREF[:sim_step,3])), 1.05*max(max(sim_vel),max(simREF[:sim_step,3])))
        dev_long, dev_lat   = LonLatDeviations(CiLX[:sim_step, 2], CiLX[:sim_step, 0], CiLX[:sim_step, 1], simREF[:sim_step, 0], simREF[:sim_step, 1])
        devlat_line.set_data(live_plot_sim_time_vec,dev_lat)
        ax_devlat.set_xlim(0, max(live_plot_sim_time_vec))
        ax_devlat.set_ylim(0.95*min(dev_lat), 1.05*max(dev_lat))

        lon_acc_limit = np.array(ax_lim(sim_vel)).reshape(-1)
        lat_acc_limit = np.array(ay_lim(sim_vel)).reshape(-1)
        postprocessed_lon_acc = [lon_acc_limit[j] if MPC_acceleration[j] >=0 else -acc_min for j in range(len(lon_acc_limit))]
        # heatmap.set_data(a_lat_sim/lat_acc_limit, MPC_acceleration/postprocessed_lon_acc)
        heatmap.set_offsets(np.c_[a_lat_sim/lat_acc_limit,MPC_acceleration/postprocessed_lon_acc])
        minima = min(sim_vel)
        maxima = max(sim_vel)
        norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
        colorlist = [mapper.to_rgba(vel) for vel in sim_vel]
        heatmap.set(clim=(min(sim_vel),max(sim_vel)),cmap=cm.rainbow, edgecolor=colorlist)

    # return center_line, inner_line, outer_line, line, dot, text

def initLiveVisuMode1(ax, CiLX, current_pose,xwidth, ywidth, X_c, X_i, X_o, Y_c, Y_i, Y_o, veh_length, veh_width):
    start_idx, end_idx = 0, len(X_c)
    center_line, = ax.plot(X_c[start_idx:end_idx], Y_c[start_idx:end_idx], "k--", alpha=1, linewidth=1)
    inner_line, = ax.plot(X_i[start_idx:end_idx], Y_i[start_idx:end_idx], "k-")
    outer_line, = ax.plot(X_o[start_idx:end_idx], Y_o[start_idx:end_idx], "k-")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")    
    ax.set_aspect('equal', 'box')
    # initialize the plot with the first point of the simulation
    MPC_pred_line, = ax.plot(0, 0, label='MPC predicted trajectory', marker='d', markersize=2, color='b')
    Ref_traj_line, = ax.plot(0, 0, label='reference trajectory',marker='d', markersize=2, color='r')
    Disturbed_traj_line, = ax.plot(0, 0, label='disturbed MPC initial states',marker='x', markersize=4, color='m')
    line, = ax.plot(0, 0, "g-")
    cur_x, cur_y, cur_yaw = current_pose[0], current_pose[1], current_pose[2]
    shape_x, shape_y = gen_car_shape(cur_x, cur_y, cur_yaw, length= veh_length, width= veh_width)
    dot, = ax.plot(shape_x, shape_y, "k-")
    text = ax.text(0.05, 1., '', transform=ax.transAxes)
    ax.set_xlim(cur_x-0.25*xwidth, cur_x+0.75*xwidth)
    ax.set_ylim(cur_y-0.25*ywidth, cur_y+0.75*ywidth)
    ax.legend() # add legend
    return MPC_pred_line,Ref_traj_line, Disturbed_traj_line, ax, center_line,inner_line, outer_line, line, dot, text


def updateLiveVisuMode1(sim_step, time, CiLX, SimU,DisturbedX, current_ref_traj, pred_X, Ref_traj_line, MPC_pred_line, Disturbed_traj_line, ax,xwidth, ywidth, line, dot, text, veh_length, veh_width, current_ref_idx):
    # start_idx, end_idx = 0, len(X_c)
    # center_line.set_data(X_c[start_idx:end_idx], Y_c[start_idx:end_idx])
    # inner_line.set_data(X_i[start_idx:end_idx], Y_i[start_idx:end_idx])
    # outer_line.set_data(X_o[start_idx:end_idx], Y_o[start_idx:end_idx])
    Ref_traj_line.set_data(current_ref_traj['pos_x'], current_ref_traj['pos_y'])
    MPC_pred_line.set_data(pred_X[:, 0], pred_X[:, 1])
    Disturbed_traj_line.set_data(DisturbedX[:sim_step, 0], DisturbedX[:sim_step, 1])
    line.set_data(CiLX[:sim_step, 0], CiLX[:sim_step, 1])
    cur_x, cur_y, cur_yaw = CiLX[sim_step, 0], CiLX[sim_step, 1], CiLX[sim_step, 2]  
    shape_x, shape_y = gen_car_shape(cur_x, cur_y, cur_yaw, length= veh_length, width= veh_width)
    dot.set_data(shape_x, shape_y)

    # Get the last point of the reference trajectory
    last_x, last_y = current_ref_traj['pos_x'][-1], current_ref_traj['pos_y'][-1]
    # Get the current x and y limits of the plot
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xmin, xmax = xlim
    ymin, ymax = ylim

    if not (last_x > xmin + 0.2*(xmax-xmin) and last_x < xmin + 0.8*(xmax-xmin) and last_y > ymin + 0.2*(ymax-ymin) and last_y < ymin + 0.8*(ymax-ymin)):
        yaw = math.fmod(current_ref_traj['ref_yaw'][0], 2*math.pi)
        if yaw < 0:
            yaw += 2*math.pi
        if 0 <= yaw < math.pi/2:            
            # print("The nearest corner is the top right corner")
            # The nearest corner is the top right corner
            ax.set_xlim(cur_x-0.2*xwidth, cur_x+1.0*xwidth)
            ax.set_ylim(cur_y-0.2*ywidth, cur_y+1.0*ywidth)
        elif math.pi/2 <= yaw < math.pi:
            # print("The nearest corner is the top left corner")
            # The nearest corner is the top left corner
            ax.set_xlim(cur_x-1.0*xwidth, cur_x+0.2*xwidth)
            ax.set_ylim(cur_y-0.2*ywidth, cur_y+1.0*ywidth)
        elif math.pi <= yaw < 3*math.pi/2:
            # print("The nearest corner is the bottom left corner")
            # The nearest corner is the bottom left corner
            ax.set_xlim(cur_x-1.0*xwidth, cur_x+0.2*xwidth)
            ax.set_ylim(cur_y-1.0*ywidth, cur_y+0.2*ywidth)
        else:
            # print("The nearest corner is the bottom right corner")
            # The nearest corner is the bottom right corner
            ax.set_xlim(cur_x-0.2*xwidth, cur_x+1.0*xwidth)
            ax.set_ylim(cur_y-1.0*ywidth, cur_y+0.2*ywidth)
    # delta_f = np.rad2deg(SimU[sim_step, 1])
    delta_f = np.rad2deg(CiLX[sim_step,6])
    # velocity = np.sqrt(CiLX[sim_step,3]**2 + CiLX[sim_step,4]**2) * 3.6
    velocity = CiLX[sim_step,3] * 3.6
    text.set_text("steering angle: {:.2f}°, velocity: {:.2f}km/h, sim step: {}, sim time: {:.2f}s, trajectory index: {}".format(delta_f, velocity, sim_step, time, current_ref_idx))
    # text.set_text("steering angle: {:.2f}°,      velocity: {:.2f}km/h".format(delta_f, velocity))
    # return center_line, inner_line, outer_line, line, dot, text


def plotRes(simX, simREF, MPCX, dev_lat, dev_long, a_lat_pred, a_lat_sim, model, constraint, ax_lim, ay_lim, MPC_acceleration, MPC_steering_angle, MPC_jerk, MPC_steering_rate,t, save_logs, log_file_ID):
    # simulated_velocity = np.sqrt(simX[:,3]**2 + simX[:,4]**2)
    simulated_velocity = simX[:,3]
    lon_acc_limit = np.array(ax_lim(simulated_velocity)).reshape(-1)
    lat_acc_limit = np.array(ay_lim(simulated_velocity)).reshape(-1)
    # plot results
    fig = plt.figure(figsize=(15,8))
    plt.title('closed-loop simulation')
    plt.subplot(3, 3, 1)
    plt.step(t, MPC_acceleration, color='g')
    plt.title('MPC optimal acceleration output')
    plt.step(t, lon_acc_limit, 'g--')
    # plt.plot([t[0], t[-1]],[model.acc_max, model.acc_max],'g--')
    plt.plot([t[0], t[-1]],[model.acc_min, model.acc_min],'g--')
    plt.ylabel('acceleration [m/s²]')
    # plt.xlabel('t [s]')
    plt.grid(True)
    
    plt.subplot(3, 3, 3)
    plt.step(t, MPC_jerk, color='b')
    plt.title('MPC optimal jerk output')
    plt.plot([t[0], t[-1]],[model.jerk_max, model.jerk_max],'b--')
    plt.plot([t[0], t[-1]],[model.jerk_min, model.jerk_min],'b--')
    plt.ylabel('jerk [m/s³]')
    # plt.xlabel('t [s]')
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.step(t, np.rad2deg(MPC_steering_angle), color='g')
    plt.title('MPC optimal steering angle output')
    plt.plot([t[0], t[-1]],[np.rad2deg(model.delta_f_max), np.rad2deg(model.delta_f_max)],'g--')
    plt.plot([t[0], t[-1]],[np.rad2deg(model.delta_f_min), np.rad2deg(model.delta_f_min)],'g--')
    plt.ylabel('steering angle [°]')
    # plt.xlabel('t [s]')
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.step(t, np.rad2deg(MPC_steering_rate), color='b')
    plt.title('MPC optimal steering rate output')
    plt.plot([t[0], t[-1]],[np.rad2deg(model.delta_f_dot_max), np.rad2deg(model.delta_f_dot_max)],'b--')
    plt.plot([t[0], t[-1]],[np.rad2deg(model.delta_f_dot_min), np.rad2deg(model.delta_f_dot_min)],'b--')
    plt.ylabel('steering rate [°/s]')
    # plt.xlabel('t [s]')
    plt.grid(True)

    plt.subplot(3, 3, 4)
    plt.title("velocity")
    plt.plot(t, np.sqrt(MPCX[:,3]**2 + MPCX[:,4]**2))
    plt.plot(t, simulated_velocity)
    plt.plot(t, simREF[:,3])
    plt.ylabel('v [m/s]')
    # plt.xlabel('t [s]')
    plt.legend(['MPC predictions','simulated velocity','reference velocity'])
    plt.grid(True)

    plt.subplot(3, 3, 5)
    plt.title("orientation")
    plt.plot(t, (MPCX[:,2]), color='b')
    plt.plot(t, (simX[:,2]), color='g')
    plt.plot(t, (simREF[:,2]), color='r')
    plt.ylabel('yaw [rad]')
    # plt.xlabel('t [s]')
    plt.legend(['MPC predictions','simulated yaw','reference yaw'])
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.title("lateral acceleration")
    if a_lat_pred is not None:
        plt.plot(t, a_lat_pred)
    if a_lat_sim is not None:
        plt.plot(t, a_lat_sim)
    plt.step(t, lat_acc_limit, 'k--')
    plt.step(t, -lat_acc_limit, 'k--')
    # plt.plot([t[0], t[-1]],[constraint.lat_acc_max, constraint.lat_acc_max],'k--')
    # plt.plot([t[0], t[-1]],[constraint.lat_acc_min, constraint.lat_acc_min],'k--')
    plt.legend(['predicted','simulated','lat acc min/max'])
    plt.ylabel('a_lat [m/s²]')
    plt.xlabel('t [s]')
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(3, 3, 8)
    plt.step(t, dev_lat)
    plt.title('lateral deviation')
    plt.ylabel('deviation [m]')
    plt.xlabel('t [s]')
    plt.grid(True)

    # plt.subplot(3, 3, 9)
    # plt.step(t, dev_long)
    # plt.title('longitudinal deviation')
    # plt.ylabel('deviation [m]')
    # plt.xlabel('t [s]')
    # plt.grid(True)

    plt.subplot(3, 3, 9)
    plt.axis('equal')
    # add heat map:
    # compute normierte accelerations
    # heatmap = plt.scatter(a_lat_sim, MPC_acceleration, c=simulated_velocity, cmap=cm.rainbow, edgecolor='none', marker='o')
    postprocessed_lon_acc = [lon_acc_limit[j] if MPC_acceleration[j] >=0 else -model.acc_min for j in range(len(lon_acc_limit))]
    heatmap = plt.scatter(a_lat_sim/lat_acc_limit, MPC_acceleration/postprocessed_lon_acc, c=simulated_velocity, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035, location = 'left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label("velocity [m/s]", rotation=90)
    # Move the colorbar to the left
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    # Set the limits of the plot
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    # plot Kammscher Kreis
    # Generate an array of angles
    theta = np.linspace(0, 2*np.pi, 100)
    # Generate the x and y coordinates of the circle
    xmax = np.cos(theta)
    ymax = np.sin(theta)
    # plot the circle
    ax.plot(xmax, ymax)
    # plot a diamond
    x_diamond = [0, 1, 0, -1, 0]
    y_diamond = [1, 0, -1, 0, 1]
    ax.plot(x_diamond, y_diamond)
    # plt.plot(a_lat_sim, MPC_acceleration, marker='o', alpha=1, linewidth=2)
    # plt.step(a_lat_sim, MPC_acceleration)
    plt.title('accelerations')
    # Use the MathText renderer
    plt.rcParams['text.usetex'] = False
    plt.xlabel('$a_y/a_{ymax}$', fontsize = 'large')
    plt.ylabel('$a_x/a_{xmax}$', fontsize = 'large')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.grid(True)

    if save_logs:
        plt.savefig(log_file_ID+'/SimResults.png')

def BoxPlots(dev_vel, dev_yaw, dev_lat, save_logs, log_file_ID):
    # deviations box plots
    fig = plt.figure(figsize=(15,8))
    # velocity deviation
    plt.subplot(1, 4, 1)
    df = pd.DataFrame(dev_vel, columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("velocity absolute deviation")
    plt.ylabel('absolute deviation [m/s]')
    # yaw deviation 
    plt.subplot(1, 4, 2)
    dev_yaw = [dev_yaw[j] if dev_yaw[j] <5 else np.abs(dev_yaw[j]- 2*np.pi) for j in range(len(dev_yaw))]
    df = pd.DataFrame(dev_yaw, columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("yaw absolute deviation")
    plt.ylabel('absolute deviation [rad]')
    # lateral deviation
    plt.subplot(1, 4, 3)
    df = pd.DataFrame(dev_lat, columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("lateral deviation")
    plt.ylabel('deviation [m]')
    # absolute lateral deviation
    plt.subplot(1, 4, 4)
    df = pd.DataFrame(np.abs(dev_lat), columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("absolute lateral deviation")
    plt.ylabel('deviation [m]')

    # ax.get_figure().savefig(exp_tag+'_Vel_RMSE_Barplot.png',dpi=500)
    # yaw deviation
    # compute lateral error
    # compute lateral acceleration
    # simLatAcc = yawrate * simX[:, 3]   # lateral acceleration = vehicle speed * yaw rate
    plt.tight_layout()
    if save_logs:
        plt.savefig(log_file_ID+'/SimResBoxplots.png')


def plotMPCperf(simSolverDebug, t, track, SimX_x, SimX_y, save_logs, log_file_ID):
    # deviations box plots
    fig = plt.figure(figsize=(15,8))
    plt.subplot(3, 4, 1)
    df = pd.DataFrame(simSolverDebug[:, 0], columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("solution cost value")
    plt.ylabel('...')
    
    plt.subplot(3, 4, 2)
    df = pd.DataFrame(simSolverDebug[:, 1], columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("total CPU time previous call")
    plt.ylabel('solving time [s]')
    
    plt.subplot(3, 4, 3)
    df = pd.DataFrame(simSolverDebug[:, 2], columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("SQP iterations needed")
    plt.ylabel('Iterations')
    
    plt.subplot(3, 4, 4)
    df = pd.DataFrame(simSolverDebug[:, 3], columns=['NMPC'])
    boxplot = df.boxplot(column=['NMPC'])  
    plt.title("max QP iterations")
    plt.ylabel('Iterations')

    plt.subplot(3, 4, 5)
    plt.step(t, simSolverDebug[:, 0])
    plt.title('solution cost value')
    plt.ylabel('cost')
    plt.xlabel('t [s]')
    plt.grid(True)

    plt.subplot(3, 4, 6)
    plt.step(t, simSolverDebug[:, 1])
    plt.title('total CPU time previous call')
    plt.ylabel('total time [s]')
    plt.xlabel('t [s]')
    plt.grid(True)

    plt.subplot(3, 4, 7)
    plt.step(t, simSolverDebug[:, 3])
    plt.title('max QP iterations')
    plt.ylabel('max QP iterations')
    plt.xlabel('t [s]')
    plt.grid(True)

    plt.subplot(3, 4, 8)
    plt.step(t, simSolverDebug[:, 4])
    plt.title('ACADOS status')
    plt.ylabel('status')
    plt.xlabel('t [s]')
    plt.grid(True)

    if track is not None:
        X_c = track["X"]
        Y_c = track["Y"]
        X_i = track["X_i"]
        X_o = track["X_o"]
        Y_i = track["Y_i"]
        Y_o = track["Y_o"]
    plt.subplot(3, 4, 9)
    plt.axis('equal')
    if track is not None:
        plt.plot(X_c, Y_c, "k--", alpha=1, linewidth=1)
        plt.plot(X_i, Y_i, "k-")
        plt.plot(X_o, Y_o, "k-")
    # add heat map:
    heatmap = plt.scatter(SimX_x, SimX_y, c=simSolverDebug[:, 0], cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("solution cost value")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.plot(SimX_x, SimX_y, "k-", alpha=1, linewidth=2)

    plt.subplot(3, 4, 10)
    plt.axis('equal')
    if track is not None:
        plt.plot(X_c, Y_c, "k--", alpha=1, linewidth=1)
        plt.plot(X_i, Y_i, "k-")
        plt.plot(X_o, Y_o, "k-")
    # add heat map:
    heatmap = plt.scatter(SimX_x, SimX_y, c=simSolverDebug[:, 1], cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("acados computation time [s]")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.plot(SimX_x, SimX_y, "k-", alpha=1, linewidth=2)

    plt.subplot(3, 4, 11)
    plt.axis('equal')
    if track is not None:
        plt.plot(X_c, Y_c, "k--", alpha=1, linewidth=1)
        plt.plot(X_i, Y_i, "k-")
        plt.plot(X_o, Y_o, "k-")
    # add heat map:
    heatmap = plt.scatter(SimX_x, SimX_y, c=simSolverDebug[:, 3], cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("max QP iterations")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.plot(SimX_x, SimX_y, "k-", alpha=1, linewidth=2)
    # ax.get_figure().savefig(exp_tag+'_Vel_RMSE_Barplot.png',dpi=500)
    # yaw deviation
    # compute lateral error
    # compute lateral acceleration
    # simLatAcc = yawrate * simX[:, 3]   # lateral acceleration = vehicle speed * yaw rate
    plt.tight_layout()
    if save_logs:
        plt.savefig(log_file_ID+'/MPC_ACADOS_performance.png')


def plotTrackSim(track, SimX_x, SimX_y, SimX_vel, dev_lat, save_logs, log_file_ID):    
    # plot track + inner + outer lines
    X_c = track["X"]
    Y_c = track["Y"]
    X_i = track["X_i"]
    X_o = track["X_o"]
    Y_i = track["Y_i"]
    Y_o = track["Y_o"]
    fig = plt.figure(figsize=(15,8))
    plt.subplot(3,1,1)
    plt.axis('equal')
    plt.plot(X_c, Y_c, "k--", alpha=1, linewidth=1)
    plt.plot(X_i, Y_i, "k-")
    plt.plot(X_o, Y_o, "k-")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")    
    # Draw driven trajectory and velocity heatmap
    heatmap = plt.scatter(SimX_x, SimX_y, c=SimX_vel, cmap=cm.rainbow, edgecolor='none', marker='o')
    cbar = plt.colorbar(heatmap, fraction=0.035)
    cbar.set_label("velocity in [m/s]")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.plot(SimX_x, SimX_y, "r-", alpha=1, linewidth=2)
    
    plt.subplot(3,1,2)
    plt.plot(X_c, Y_c, "k--", alpha=1, linewidth=1)
    plt.plot(X_i, Y_i, "k-")
    plt.plot(X_o, Y_o, "k-")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")    
    # heatmap of lateral deviation
    heatmap = plt.scatter(SimX_x, SimX_y, c=np.array(dev_lat).reshape(-1), cmap=cm.rainbow, edgecolor='none', marker='.')
    cbar = plt.colorbar(heatmap, fraction=0.05)
    cbar.set_label("lateral deviation in [m]")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # Draw driven trajectory without heatmap
    plt.subplot(3,1,3)
    plt.plot(X_c, Y_c, "k--", alpha=1, linewidth=1, label = 'reference trajectory')
    plt.plot(X_i, Y_i, "k-")
    plt.plot(X_o, Y_o, "k-")
    plt.plot(SimX_x, SimX_y, "g-", label = 'driven trajectory')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")    
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    # if save_logs:
    #     plt.savefig(log_file_ID+'/TrackSim.png')

def plotDisturbancesRealization(t, sim_disturbance, save_logs, log_file_ID):
    Nsim = sim_disturbance.shape[0]  # number of simulation steps
    nx_sim = sim_disturbance.shape[1]    # number of states
    fig, axs = plt.subplots(nrows=nx_sim, sharex=True, figsize=(15,8))
    plt.suptitle('disturbances realization: derivatives')
    state_keys = ['posx','posy','yaw','vlong','vlat','yawrate','delta_f']
    for i in range(len(state_keys)):
        axs[i].plot(t, sim_disturbance[:,i], label='disturbance')
        # axs[i].plot(t, np.ones(Nsim)*disturbance_bounds[i][0], 'r--', label='lower bound')
        # axs[i].plot(t, np.ones(Nsim)*disturbance_bounds[i][1], 'g--', label='upper bound')
        if np.min(sim_disturbance[:,i]) != np.max(sim_disturbance[:,i]):
            axs[i].set_ylim(np.min(sim_disturbance[:,i]),np.max(sim_disturbance[:,i]))
        axs[i].set_ylabel(state_keys[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('time')
    plt.tight_layout()
    if save_logs:
        plt.savefig(log_file_ID+'/Disturbances.png')

def plotSimulatedStateErrors(t, sim_disturbance_state_estimation, save_logs, log_file_ID):
    Nsim = sim_disturbance_state_estimation.shape[0]        # number of simulation steps
    nx_sim = sim_disturbance_state_estimation.shape[1]      # number of states
    fig, axs = plt.subplots(nrows=nx_sim, sharex=True, figsize=(15,8))
    plt.suptitle('disturbances realization: state estimation')
    state_keys = ['posx','posy','yaw','vlong','vlat','yawrate','delta_f']
    for i in range(len(state_keys)):
        # state_errors = CiLX[1:,i]-DisturbedX[1:,i]
        # axs[i].plot(t, state_errors)
        axs[i].plot(t, sim_disturbance_state_estimation[:,i])
        # axs[i].plot(t, np.ones(Nsim)*disturbance_bounds[i][0], 'r--', label='lower bound')
        # axs[i].plot(t, np.ones(Nsim)*disturbance_bounds[i][1], 'g--', label='upper bound')
        if np.min(sim_disturbance_state_estimation[:,i]) != np.max(sim_disturbance_state_estimation[:,i]):
            axs[i].set_ylim(np.min(sim_disturbance_state_estimation[:,i]),np.max(sim_disturbance_state_estimation[:,i]))
        axs[i].set_ylabel(state_keys[i])
        # axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel('time')
    plt.tight_layout()
    if save_logs:
        plt.savefig(log_file_ID+'/SimulatedStateErrors.png')