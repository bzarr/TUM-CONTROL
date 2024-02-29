"""
Created on Thu Sep 14 22:57:40 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import casadi as cs
import seaborn as sns

# Define the path to the directory containing the log files
log_dir = 'Papers_Plots/ACC24_SNMPC'
benchmark_runs = [
    'NMPC_FnodistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_16-06-27',
    'SNMPC_FnodistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_16-03-35',
    'NMPC_FdistSE_n15uph15y0.035v0.1vlt0.05yrt0.001p0.82023-09-17_15-50-35',
    "SNMPC_FdistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_15-55-13"
    ]
benchmark_names = [
    'NMPC', 
    'SNMPC', 
    'disturbed\nNMPC',
    'disturbed\nSNMPC',
    ]
lookuptable_gg_limits_file = 'Config/EDGAR/ggv.csv'
acc_min = -3.5
log_file_ID = log_dir# 'Python/Papers/SNMPC'
# Define your custom font size
custom_font_size = 16
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
# Initialize lists to store solver computational time KPIs for each experiment
solver_time_max_list = []
solver_time_mean_list = []
solver_time_75th_percentile_list = []

# Iterate through the log files in the list
for file, trial_name in zip(benchmark_runs, benchmark_names):
    # Load the data from the log file
    data = np.load(os.path.join(log_dir, file) + '/full_logs.npz')
    CPU_time = data['simSolverDebug'][:, 1]
    t = data['t']

    # Calculate solver computational time KPIs
    solver_time_max = np.max(CPU_time)*1000
    solver_time_mean = np.mean(CPU_time)*1000
    solver_time_75th_percentile = np.percentile(CPU_time, 75)*1000

    # Append KPIs to respective lists
    solver_time_max_list.append(solver_time_max)
    solver_time_mean_list.append(solver_time_mean)
    solver_time_75th_percentile_list.append(solver_time_75th_percentile)


# Create a DataFrame with solver computational time KPIs
solver_time_df = pd.DataFrame({
    'Experiment': benchmark_names,
    'Maximum [ms]': solver_time_max_list,
    'Mean [ms]': solver_time_mean_list,
    # '75th Perc. []': solver_time_75th_percentile_list
})

# Transpose the DataFrame to have rows as KPIs and columns as experiments
solver_time_df = solver_time_df.set_index('Experiment').T

# Change the formatting of numbers to scientific notation with 3 digits after the comma
# solver_time_df = solver_time_df.apply(lambda x: '{:.1f}'.format(x) if isinstance(x, (int, float)) else x)

# Save the transposed DataFrame as a CSV file with scientific notation formatting
solver_time_df.to_csv(log_file_ID+'/solver_time_experiments.csv')#, float_format='%.1f')
