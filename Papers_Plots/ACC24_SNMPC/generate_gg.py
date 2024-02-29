"""
Created on Thu Sep 14 22:57:40 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl
# import subprocess
# subprocess.check_call(["latex"])
import os 
os.environ["PATH"] += os.pathsep + '/usr/bin/latex'
# Set LaTeX rendering for labels, titles, and tick labels
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}" #for \text command
# Define the path to the directory containing the log files
log_dir = 'Papers_Plots/ACC24_SNMPC'
benchmark_runs_folders = [
    'NMPC_FdistSE_n15uph15y0.035v0.1vlt0.05yrt0.001p0.82023-09-17_15-50-35',
    'SNMPC_FdistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_15-55-13'
]
benchmark_names = [
    # 'NMPC', 
    # 'SNMPC', 
    'NMPC disturbed',
    'SNMPC disturbed']
lookuptable_gg_limits_file = 'Config/EDGAR/ggv.csv'
acc_min = -3.5
log_file_ID = log_dir #'Python/Papers/SNMPC'


## GG Diag
import pandas as pd
import casadi as cs
from matplotlib import cm, colors
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
ax_lim, ay_lim = calculate_velvar_latlon_acc(lookuptable_gg_limits_file)
fig, axs = plt.subplots(1, 1, figsize=(4.0, 4.0))  # Adjust the width as needed
# plt.title('gg accelerations')
# Use the MathText renderer

i = 0
for file, trial_name in zip(benchmark_runs_folders, benchmark_names):
    if trial_name.__contains__('dist'):
        data = np.load(os.path.join(log_dir, file) + '/full_logs.npz')
        # add heat map:
        # compute normierte accelerations
        # heatmap = plt.scatter(a_lat_sim, MPC_acceleration, c=simulated_velocity, cmap=cm.rainbow, edgecolor='none', marker='o')
        simX = data['CiLX']
        simulated_velocity = simX[0:,3]
        lon_acc_limit = np.array(ax_lim(simulated_velocity)).reshape(-1)
        lat_acc_limit = np.array(ay_lim(simulated_velocity)).reshape(-1)
        MPCX = data['MPC_SimX']
        MPC_acceleration= MPCX[0:,7]
        a_lat_sim = data['a_lat']
        postprocessed_lon_acc = np.array([lon_acc_limit[j] if MPC_acceleration[j] >=0 else -acc_min for j in range(len(lon_acc_limit))])
        
        ax = axs
        # plt.axis('equal')
        # heatmap = ax.scatter(a_lat_sim/lat_acc_limit, MPC_acceleration/postprocessed_lon_acc, c=simulated_velocity, cmap=cm.rainbow, edgecolor='none', marker='x',label=trial_name)
        selected_points = np.arange(0, len(a_lat_sim), 5)
        ax.scatter(a_lat_sim[selected_points] / lat_acc_limit[selected_points] , MPC_acceleration[selected_points] / postprocessed_lon_acc[selected_points], marker='o',s = 5, label=trial_name)
        i = i + 1

ax=axs
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
ax.plot(xmax, ymax, 'tab:gray')
ax.grid(True)
axs.set_xlabel('$a_y/a_{ymax}$', fontsize = 'x-large')
axs.set_ylabel('$a_x/a_{xmax}$', fontsize = 'x-large')

plt.tight_layout()
plt.savefig(log_file_ID+'/ggv.pdf', bbox_inches='tight')
plt.show()