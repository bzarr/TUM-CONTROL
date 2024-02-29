"""
Created on Thu Sep 14 22:57:40 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import casadi as cs
# Define the path to the directory containing the log files
log_dir = 'Papers_Plots/ACC24_SNMPC'
benchmark_runs_folders = [
    'NMPC_FdistSE_n15uph15y0.035v0.1vlt0.05yrt0.001p0.82023-09-17_15-50-35',
    'SNMPC_FdistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_15-55-13'
    ]
benchmark_names = [ 
    'disturbed NMPC',
    'disturbed SNMPC',
    ]
lookuptable_gg_limits_file = 'Config/EDGAR/ggv.csv'
acc_min = -3.5
log_file_ID = log_dir #'Python/Papers/SNMPC'
# Define your custom font size
custom_font_size = 20
import matplotlib as mpl

# Modify Matplotlib's rcParams to set font size for different elements
mpl.rcParams['font.size'] = custom_font_size  # Default font size
mpl.rcParams['axes.titlesize'] = custom_font_size  # Title font size
mpl.rcParams['axes.labelsize'] = custom_font_size  # Axis label font size
mpl.rcParams['xtick.labelsize'] = custom_font_size  # X-axis tick label font size
mpl.rcParams['ytick.labelsize'] = custom_font_size  # Y-axis tick label font size
mpl.rcParams['legend.fontsize'] = custom_font_size  # Legend font size
import os 
os.environ["PATH"] += os.pathsep + '/usr/bin/latex'
# Set LaTeX rendering for labels, titles, and tick labels
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}" #for \text command
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
ax_lim, ay_lim = calculate_velvar_latlon_acc(lookuptable_gg_limits_file)

# Initialize lists to store KPIs for each experiment
velocity_rmse_list = []
velocity_deviation_absolute_max_list = []
lateral_rmse_list = []
lateral_deviation_absolute_75th_percentile_list = []
lateral_deviation_absolute_max_list = []
lateral_deviation_absolute_median_list = []
# Initialize lists to store solver computational time KPIs for each experiment
solver_time_max_list = []
solver_time_median_list = []
solver_time_75th_percentile_list = []

fig = plt.figure(constrained_layout=False)
fig.set_size_inches(10, 5)
gs1 = fig.add_gridspec(nrows=2, ncols=3, wspace = 0.3, height_ratios=[1, 1], width_ratios=[1, 1, 0.75])#, wspace=1., hspace = 0.4)
f8_ax1 = fig.add_subplot(gs1[:-1, :-1])
f8_ax2 = fig.add_subplot(gs1[-1, :-1])
f8_ax3 = fig.add_subplot(gs1[:, 2])
axs = fig.axes

axs[0].set_ylabel(r'velocity [m/s]')
axs[1].set_ylabel(r'lateral deviation [m]')
axs[1].set_xlabel(r't [s]')
colors = ['coral','mediumseagreen']
# Iterate through the log files in the list
for file, trial_name, color in zip(benchmark_runs_folders, benchmark_names, colors):
    # Load the data from the log file
    data = np.load(os.path.join(log_dir, file) + '/full_logs.npz')
    simX = data['CiLX']
    simREF = data['simREF']
    dev_lat = data['dev_lat']
    dev_vel = data['dev_vel']
    dev_yaw = [data['dev_yaw'][j] if data['dev_yaw'][j] < 5 else np.abs(data['dev_yaw'][j] - 2 * np.pi) for j in
               range(len(data['dev_yaw']))]
    t = data['t']

    #TODO: check this!!!!!!
    # Calculate KPIs
    velocity_rmse = np.sqrt(np.mean(np.square(dev_vel)))
    velocity_deviation_absolute_max = np.max(np.abs(dev_vel))
    lateral_rmse = np.sqrt(np.mean(np.square(dev_lat)))
    lateral_deviation_absolute_75th_percentile = np.percentile(np.abs(dev_lat), 75)
    lateral_deviation_absolute_max = np.max(np.abs(dev_lat))
    lateral_deviation_absolute_median = np.median(np.abs(dev_lat))

    # Append KPIs to respective lists
    velocity_rmse_list.append(velocity_rmse)
    velocity_deviation_absolute_max_list.append(velocity_deviation_absolute_max)
    lateral_rmse_list.append(lateral_rmse)
    lateral_deviation_absolute_75th_percentile_list.append(lateral_deviation_absolute_75th_percentile)
    lateral_deviation_absolute_max_list.append(lateral_deviation_absolute_max)
    lateral_deviation_absolute_median_list.append(lateral_deviation_absolute_median)

    CPU_time = data['simSolverDebug'][:, 1]
    t = data['t']

    # Calculate solver computational time KPIs
    solver_time_max = np.max(CPU_time)
    solver_time_median = np.median(CPU_time)
    solver_time_75th_percentile = np.percentile(CPU_time, 75)

    # Append KPIs to respective lists
    solver_time_max_list.append(solver_time_max)
    solver_time_median_list.append(solver_time_median)
    solver_time_75th_percentile_list.append(solver_time_75th_percentile)

    # Extract the file name without extension
    # trial_name = os.path.splitext(file)[0]
    if trial_name.__contains__('dis'):
        axs[0].plot(t, simX[1:, 3], label=trial_name, color = color)
        axs[1].step(t, dev_lat, color = color)
        
        simX = data['CiLX']
        simulated_velocity = simX[0:,3]
        lon_acc_limit = np.array(ax_lim(simulated_velocity)).reshape(-1)
        lat_acc_limit = np.array(ay_lim(simulated_velocity)).reshape(-1)
        MPCX = data['MPC_SimX']
        MPC_acceleration= MPCX[0:,7]
        a_lat_sim = data['a_lat']
        postprocessed_lon_acc = np.array([lon_acc_limit[j] if MPC_acceleration[j] >=0 else -acc_min for j in range(len(lon_acc_limit))])
        
        ax = axs[2]
        # plt.axis('equal')
        # heatmap = ax.scatter(a_lat_sim/lat_acc_limit, MPC_acceleration/postprocessed_lon_acc, c=simulated_velocity, cmap=cm.rainbow, edgecolor='none', marker='x',label=trial_name)
        selected_points = np.arange(0, len(a_lat_sim), 5)
        ax.scatter(a_lat_sim[selected_points] / lat_acc_limit[selected_points] , MPC_acceleration[selected_points] / postprocessed_lon_acc[selected_points], marker='o',s = 5, color = color)

# Plot reference data on the first subplot
axs[0].plot(t, simREF[:, 3], label=r'reference')
# axs[0].legend(loc='best',ncol=3, bbox_to_anchor=(0.05, 1.15))#, fontsize=10)
axs[0].set_xticks([])
axs[0].set_yticks([10,20,30,38])
axs[0].yaxis.tick_right()
# Enable grid for subplots
axs[0].xaxis.set_ticks_position('both')  # Show both major and minor grid lines
axs[0].xaxis.grid(True)
axs[0].grid(True)
axs[1].grid(True)
# axs[1].set_yticks([0,1,-1])
axs[1].yaxis.tick_right()
ax=axs[2]
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
ax.plot(xmax, ymax, 'tab:gray', linestyle='--')
ax.grid(True)
ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
ax.set_xlabel(r'$a_y/a_{y_{max}}$')#, fontsize = 'large')
ax.set_ylabel(r'$a_x/a_{x_{max}}$')#, fontsize = 'large')
ax.set_xticks([0,1])
ax.set_yticks([0,1])

legend_ax = fig.add_axes([0.1, 0.2, 0.8, 0.8])
legend_ax.axis('off')
legend_ax.legend(*axs[0].get_legend_handles_labels(), ncol=3,loc='upper center')
# fig.legend(loc='upper center',ncol=3)#, fontsize=10)

# plt.legend()
# plt.tight_layout()
plt.savefig(log_file_ID+'/ACC24_SimRes.pdf', bbox_inches='tight')
# plt.show()