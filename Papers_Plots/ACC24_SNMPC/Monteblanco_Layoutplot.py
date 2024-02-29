#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 18:20:00 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
import pandas as pd
import json
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
import os
custom_font_size = 18
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

path = 'Python/Trajectories'
# os.chdir(path)
input_traj_name = "traj_monteblanco_edgar.csv"
input_track_name =  "track_monteblanco.csv"
# Function to rotate a point (x, y) by an angle (in radians)
def rotate_point(x, y, angle):
    rotated_x = x * math.cos(angle) - y * math.sin(angle)
    rotated_y = x * math.sin(angle) + y * math.cos(angle)
    return rotated_x, rotated_y

def rotate_vector(yaw):
    yaw += math.pi / 2  # add 90 degrees to rotate from north to east
    if yaw < 0:
        yaw += 2 * math.pi  # adjust for negative angles
    return yaw

# Read the .csv file into a DataFrame
df = pd.read_csv(path+"/Raw_tracks/"+input_traj_name)

# Extract the desired columns and create a new DataFrame
data = df[["x_m", "y_m", "psi_rad", "vx_mps", "ax_mps2"]]


X = data["x_m"].tolist()
Y = data["y_m"].tolist()
ref_v = (data["vx_mps"]).tolist()
# ref_yaw = (-(data["psi_rad"]- np.pi)-np.pi/2).tolist()
ref_yaw = (data["psi_rad"])
ref_yaw = [rotate_vector(y) for y in ref_yaw]
# ref_yaw = np.clip(ref_yaw, 0, 2*np.pi)
# plt.plot(ref_yaw)
# ref_yaw = ref_yaw.tolist()
ref_a = data["ax_mps2"].tolist()


def sign(bool_var):
    return 1 if bool_var == True else -1

df2 = pd.read_csv(path+"/Raw_tracks/" + input_track_name)
# data1 = df[["psi_rad","kappa_radpm"]]
data2 = df2[["x_ref_m", "y_ref_m", "width_right_m", "width_left_m", "x_normvec", "y_normvec"]]
# for x, y, kappa, width_right, width_left in zip(X_c, Y_c, data1["kappa_radpm"], data2["width_right_m"], data2["width_left_m"]):
# psi_rad = data1["psi_rad"]
# kappa_radpm = data1["kappa_radpm"]
width_right_m, width_left_m = data2["width_right_m"], data2["width_left_m"]
x_normvec, y_normvec = data2["x_normvec"], data2["y_normvec"]
# Create variables to hold the optimal line data
X_c = data2["x_ref_m"].tolist()
Y_c = data2["y_ref_m"].tolist()
# Compute the normal vector of the track

# Compute the inner and outer line coordinates
X_i = []
Y_i = []
X_o = []
Y_o = []
angles = []

for x, y, width_right, width_left, x_norm, y_norm in zip(X_c, Y_c, width_right_m, width_left_m, x_normvec, y_normvec):
    # angle = math.atan2(y_norm, x_norm)
    # angles.append(angle)
    width = width_left #if angle < 0 else width_right
    X_i.append(x - width*x_norm)
    Y_i.append(y - width*y_norm)
    width = width_right #if angle < 0 else width_left
    X_o.append(x + width*x_norm)
    Y_o.append(y + width*y_norm)


# Rotate all coordinates by the specified yaw angle
X_rotated = []
Y_rotated = []
yaw_angle_radians = - ref_yaw[0] +np.pi
for x, y in zip(X, Y):
    rotated_x, rotated_y = rotate_point(x, y, yaw_angle_radians)
    X_rotated.append(rotated_x)
    Y_rotated.append(rotated_y)
X, Y = X_rotated, Y_rotated

X_rotated = []
Y_rotated = []
for x, y in zip(X_i, Y_i):
    rotated_x, rotated_y = rotate_point(x, y, yaw_angle_radians)
    X_rotated.append(rotated_x)
    Y_rotated.append(rotated_y)
X_i, Y_i = X_rotated, Y_rotated

X_rotated = []
Y_rotated = []
for x, y in zip(X_o, Y_o):
    rotated_x, rotated_y = rotate_point(x, y, yaw_angle_radians)
    X_rotated.append(rotated_x)
    Y_rotated.append(rotated_y)
X_o, Y_o = X_rotated, Y_rotated

X_rotated = []
Y_rotated = []
for x, y in zip(X_c, Y_c):
    rotated_x, rotated_y = rotate_point(x, y, yaw_angle_radians)
    X_rotated.append(rotated_x)
    Y_rotated.append(rotated_y)
X_c, Y_c = X_rotated, Y_rotated

fig= plt.figure(figsize=(10,5.5))
plt.axis('equal')
plt.plot(X, Y, "k--", alpha=1, linewidth=1, label = r'optimal raceline')
# plt.plot(X_i, Y_i, "k-")
# plt.plot(X_o, Y_o, "k-")
ax = plt.gca()
# ax.xaxis.set_ticks_position('top')  # Move x-axis to the top
# ax.yaxis.tick_right()
# ax.xaxis.set_label_position("top")
# plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)  # Move x-axis and labels to the top

plt.xlabel(r"X [m]")
plt.ylabel(r"Y [m]")   
# Draw driven trajectory and velocity heatmap
heatmap = plt.scatter(X, Y, c=ref_v, cmap=cm.rainbow, edgecolor='none', marker='o')
cbar = plt.colorbar(heatmap, fraction=0.1, orientation='horizontal', shrink=0.6)#, aspect=30,pad = -0.3)
cbar.set_label(r"velocity [m/s]")
# cbar.ax.set_position([0.8, 0.1, 0.5, 0.02])  # Adjust the values based on your preference
# cbar.set_ticks([0, 0.5 , 1 , 1.5, 2, 2.5, 3.0])
ax = plt.gca()
ax.set_aspect('equal', 'box')
ax.grid(True)

# ax.set_title("velocity")
plt.tight_layout()
fig.legend(loc='center',ncol=2, frameon= False)
# plt.show()
log_file_ID = 'Python/Papers/ACC24_SNMPC'
plt.savefig(log_file_ID+'/ECC24_Monteblanco.png')