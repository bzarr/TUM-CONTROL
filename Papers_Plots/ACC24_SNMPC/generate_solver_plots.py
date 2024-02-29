# """
# Created on Thu Sep 14 22:57:40 2023

# @author: Baha Zarrouki (baha.zarrouki@tum.de)
# """
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import casadi as cs
import seaborn as sns

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

## Fused Plot
# Define the path to the directory containing the log files
log_dir = 'Papers_Plots/ACC24_SNMPC'
log_file_ID = log_dir
benchmark_runs_right = [
      'SNMPC_FdistSE_n38uph10xy0.3y0.05vl0.8vlt0.8yr0.1d0.012023-09-18_19-33-29',
     'SNMPC_FdistSE_n10uph10xy0.3y0.05vl0.8vlt0.8yr0.1d0.012023-09-18_19-30-09'
    ]

benchmark_runs_wrong = [
    'SNMPC_FdistSE_n38uphvl0.8vlt0.8yr0.12023-09-18_18-43-32',
     'SNMPC_FdistSE_n10uphvl0.8vlt0.8yr0.12023-09-18_18-39-30',
    ]

fig, (ax1,ax2)= plt.subplots(2,1,sharex = True,sharey=True, figsize=(10,2))
fig.subplots_adjust(hspace=0.5)

# ax1.set_ylabel('solver status')
# ax1.set_xlabel('t [s]')
# Iterate through the log files in the list
# for file, trial_name in zip(benchmark_runs, benchmark_names):
    # Load the data from the log file
data = np.load(os.path.join(log_dir, benchmark_runs_right[0]) + '/full_logs.npz')
solver_status = data['simSolverDebug'][:, 4]
t = data['t']
for i in range(len(solver_status)):
    if solver_status[i] == 4:
        solver_status[i] = 0
    elif solver_status[i] == 0:
        solver_status[i] = 1
ax1.step(t, solver_status, label= r'UPH=$T_p$', color = 'coral')

data = np.load(os.path.join(log_dir, benchmark_runs_right[1]) + '/full_logs.npz')
solver_status = data['simSolverDebug'][:, 4]
t = data['t']
for i in range(len(solver_status)):
    if solver_status[i] == 4:
        solver_status[i] = 0
    elif solver_status[i] == 0:
        solver_status[i] = 1
ax1.step(t, solver_status, label= r'UPH=$0.8s$', color = 'mediumseagreen')

ax1.set_yticks([0,1])
ax1.set_ylim([0,1.1])
# ax1.legend(loc='center',ncol=2)#, fontsize=10)
# fig.tight_layout()
ax1.set_title("exact disturbance assumption")
ax1.grid(True)
# plt.savefig(log_file_ID+'/ACC24_SolverStatus_wrong_assumption.pdf', bbox_inches='tight') 


data = np.load(os.path.join(log_dir, benchmark_runs_wrong[0]) + '/full_logs.npz')
solver_status = data['simSolverDebug'][:, 4]
t = data['t']
for i in range(len(solver_status)):
    if solver_status[i] == 4:
        solver_status[i] = 0
    elif solver_status[i] == 0:
        solver_status[i] = 1
ax2.step(t, solver_status, label= r'UPH=$T_p$', color = 'coral')

data = np.load(os.path.join(log_dir, benchmark_runs_wrong[1]) + '/full_logs.npz')
solver_status = data['simSolverDebug'][:, 4]
t = data['t']
for i in range(len(solver_status)):
    if solver_status[i] == 4:
        solver_status[i] = 0
    elif solver_status[i] == 0:
        solver_status[i] = 1
ax2.step(t, solver_status, label= r'UPH=$0.8s$=$0.26T_p$', color = 'mediumseagreen')

ax2.set_yticks([0,1])
ax2.set_xticks([0,20,40,60,80,100])
ax2.legend(loc='center left',ncol=2)#, fontsize=10)
ax2.set_title("wrong disturbance assumption")
ax2.set_ylim([0,1.1])
fig.text(0.88, -0.02, 't[s]', ha='center')
fig.text(0.08, 0.5, 'solver status', va='center', rotation='vertical')
# fig.tight_layout()
# ax2.set_xlabel('t[s]',loc = 'right')
ax2.grid(True)
plt.savefig(log_file_ID+'/ACC24_SolverStatus.pdf', bbox_inches='tight') 
plt.show()