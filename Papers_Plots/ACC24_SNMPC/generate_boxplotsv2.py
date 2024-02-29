"""
Created on Sun Oct 15 15:21:40 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import casadi as cs
import seaborn as sns

# Define the path to the directory containing the log files
log_dir = 'Papers_Plots/ACC24_SNMPC/'
log_file_ID = 'Papers_Plots/ACC24_SNMPC'
benchmark_runs = [
    ####### Monteblanco overall figure ########
    'NMPC_FnodistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_16-06-27',
    'NMPC_FdistSE_n15uph15y0.035v0.1vlt0.05yrt0.001p0.82023-09-17_15-50-35',
]
benchmark_runs_2 = [
    # 'seed6_monteblanco_vlong0.5-1.0vlat0.5-1.0yawrate0.04-0.08/SNMPC_optimal_no_dist_monteblanco',
    'SNMPC_FnodistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_16-03-35',
    'SNMPC_FdistSE_n10uph15v0.8vlt0.35yrt0.035p0.82023-09-17_15-55-13'
]
benchmark_names = [
    'no\ndisturbance',
    'under\ndisturbance',
    ]
lookuptable_gg_limits_file = 'Config/EDGAR/ggv.csv'
acc_min = -3.5
# log_file_ID = 'Python/Papers/ECC24_RNMPC' 
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
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}" #for \text command
# Box performance plots
# initialize lists to store data
dev_vel_list = []
dev_yaw_list = []
dev_lat_list = []

# load data from files
for filename in benchmark_runs:
    data = np.load(os.path.join(log_dir, filename)+ '/full_logs.npz')
    # dev_vel_list.append(data['dev_vel'])
    # dev_yaw = [data['dev_yaw'][j] if data['dev_yaw'][j] <5 else np.abs(data['dev_yaw'][j]- 2*np.pi) for j in range(len(data['dev_yaw']))]
    # dev_yaw_list.append(dev_yaw)
    dev_lat_list.append(data['dev_lat'].reshape(-1))

# plot boxplots
fig= plt.figure(figsize=(10,4))
ax = plt.gca()
# title = 'absolute lateral deviation'
ylabel = r'absolute lateral deviation [m]'
data_list = [np.abs(data) for data in dev_lat_list]
# Reverse the order of columns
benchmark_names = benchmark_names[::-1]
data_list = data_list[::-1]
# Define colors for box plots (e.g., 'b' for blue, 'g' for green)
# colors = ['b'] * (len(benchmark_names) - 1) + ['g']  # Replace 'your_color' with the desired color

df = pd.DataFrame(dict(zip(benchmark_names, data_list)))
# df = df.iloc[:, ::-1]
# boxplot = df.boxplot(ax=ax, column=benchmark_names)
c = 'coral'
boxprops = dict(facecolor = c, linestyle='-', linewidth=1, color='black',hatch = '\\\\')
medianprops = dict(linestyle='--', linewidth=1, color='black')
whiskerprops = dict(linestyle='-', linewidth=1, color='black')
capprops = dict(linestyle='-', linewidth=1, color='black')
meanlineprops = dict(linestyle='-', linewidth=2.5, color='black')
# flierprops = dict(marker='o', markersize=1, markerfacecolor='black')
color_palette = {'boxes': 'black'}

# boxplot = df.boxplot(ax=ax, vert=False,column=benchmark_names, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)#, flierprops=flierprops)
bp1 = ax.boxplot(df,showmeans=True, meanline=True,notch=True,patch_artist=True, showfliers=True, widths = 0.2, 
                    vert = False,positions=np.arange(df.shape[1])+.15,
                    boxprops= boxprops, #dict(facecolor = c,color='k',hatch = '//'),                                                                                                                       capprops=dict(color=c),
                    whiskerprops=whiskerprops,#dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    meanprops = meanlineprops,
                    medianprops= medianprops#dict(color='k'),)
)

# Annotate the boxes with multiple metrics (median, mean, 25th, 75th, max)
metrics = ['median', 'mean', '25th', '75th', 'max']  # Include 'max'

# Define vertical positions for annotations
vertical_positions = np.arange(df.shape[1])+.01 #np.arange(1, len(benchmark_names) + 1)

# for metric in metrics:
for i, (name, values) in enumerate(zip(benchmark_names, data_list)):
    median = np.median(values)
    mean = np.mean(values)
    twe = np.percentile(values, 25)
    sev = np.percentile(values, 75)
    max = np.max(values)

    # Add a small vertical offset to each annotation
    offset, offset2 = 0.25, 0.23 
    ax.text(0, vertical_positions[i] + offset2, 
            rf"25th: {twe:.2f}m, \space mean: {mean:.2f}m, \space 75th: {sev:.2f}m", 
            verticalalignment='bottom', color='black', fontweight='bold')
    # ax.text(max, vertical_positions[i] + offset, "max: {:.3f}m".format(max), horizontalalignment='right', verticalalignment='bottom', color='black', fontweight='bold')
    # Highlighting the best values
    ax.text(
        max + 0.01,
        vertical_positions[i] + 0.05,
        rf"{max:.3f}m",
        verticalalignment='bottom',
        color='darkred',  # Set text color to green
        fontweight='bold'
    )


dev_lat_list = []
# load data from files
for filename in benchmark_runs_2:
    data = np.load(os.path.join(log_dir, filename)+ '/full_logs.npz')
    dev_lat_list.append(data['dev_lat'].reshape(-1))

data_list = [np.abs(data) for data in dev_lat_list]
data_list = data_list[::-1]
df2 = pd.DataFrame(dict(zip(benchmark_names, data_list)))
c= 'mediumseagreen'
boxprops = dict(facecolor = c, linestyle='-', linewidth=1, color='black',hatch = '//')
medianprops = dict(linestyle='--', linewidth=1, color='black')
whiskerprops = dict(linestyle='-', linewidth=1, color='black')
capprops = dict(linestyle='-', linewidth=1, color='black')
meanlineprops = dict(linestyle='-', linewidth=2.5, color='black')
bp2 = ax.boxplot(df2,showmeans=True, meanline=True,notch=True,patch_artist=True, showfliers=True, widths = 0.2, vert = False,positions=np.arange(df2.shape[1])-.15,
            boxprops= boxprops, #dict(facecolor = c,color='k',hatch = '//'),                                                                                                                       capprops=dict(color=c),
            whiskerprops=whiskerprops,#dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            meanprops = meanlineprops,
            medianprops= medianprops#dict(color='k'),)
            )

# Annotate the boxes with multiple metrics (median, mean, 25th, 75th, max)
metrics = ['median', 'mean', '25th', '75th', 'max']  # Include 'max'

# Define vertical positions for annotations
vertical_positions = np.arange(df2.shape[1]) #np.arange(1, len(benchmark_names) + 1)

# for metric in metrics:
for i, (name, values) in enumerate(zip(benchmark_names, data_list)):
    median = np.median(values)
    mean = np.mean(values)
    twe = np.percentile(values, 25)
    sev = np.percentile(values, 75)
    max = np.max(values)

    # Add a small vertical offset to each annotation
    offset, offset2 = 0.25, -0.45
    ax.text(0, vertical_positions[i] + offset2, 
            rf"25th: {twe:.2f}m, \space mean: {mean:.2f}m, \space 75th: {sev:.2f}m", 
            verticalalignment='bottom', color='black', fontweight='bold')
    # ax.text(max, vertical_positions[i] + offset, "max: {:.3f}m".format(max), horizontalalignment='right', verticalalignment='bottom', color='black', fontweight='bold')
    # Highlighting the best values
    ax.text(
        max + 0.01 ,
        vertical_positions[i] - offset,
        rf"{max:.3f}m",
        verticalalignment='bottom',
        color='darkred',  # Set text color to green
        fontweight='bold'
    )


# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# Customize the plot
# ax.set_yticks(vertical_positions)
ax.set_yticks(np.arange(df2.shape[1]))
ax.set_yticklabels(benchmark_names)
ax.set_xlabel(ylabel)

fig.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Nominal NMPC', 'Stochastic NMPC'], loc='upper center',ncol=2) 

# Show the plot
plt.grid(axis='x')

fig.savefig(log_file_ID+'/ACC24_Boxplots_NMPC_SNMPC.pdf', bbox_inches='tight')
# plt.show()
