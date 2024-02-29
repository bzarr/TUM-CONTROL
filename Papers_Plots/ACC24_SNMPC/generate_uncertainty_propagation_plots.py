import numpy as np
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import matplotlib
# import subprocess
# subprocess.check_call(["latex"])
import os 
os.environ["PATH"] += os.pathsep + '/usr/bin/latex'

log_dir = 'Papers_Plots/ACC24_SNMPC'

#######################################################
# PROPAGATION HORIZON
#######################################################

# Set LaTeX rendering for labels, titles, and tick labels
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.rc('font', family='serif')
plt.rc('font', size=20)
plt.rcParams['font.weight'] = 'bold'

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
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

Nd = 15
Propagation_horizon = 4
Nv = 500
disc = 0.001   
# plt.figure(figsize=(10,7))

base = Nv*disc+0.3
kw = [1, (4*Nv*disc+0.2 - 2*Nv*disc+ 0.2)/base, (3.2 - 5*Nv*disc+0.2)/base]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,gridspec_kw={'width_ratios': kw},figsize=(10,5))
fig.subplots_adjust(hspace=0.02)  # adjust space between axes

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([ 1], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([ 0], [0], transform=ax3.transAxes, **kwargs)

ax1.set_xlim(-0.3, Nv*disc+0.2) 
ax2.set_xlim(2*Nv*disc-0.2, 4*Nv*disc+0.2) 
ax3.set_xlim(5*Nv*disc-0.2, 3.2) 
ax1.set_ylim(0,1.4) 

x0 = np.arange(0, 3+disc, disc)
x1 = np.arange(0, 3+disc, disc)
x2 = np.arange(0, 3+disc, disc)
x3 = np.arange(0, 3+disc, disc)
x4 = np.arange(0, 3+disc, disc)
x5 = np.arange(0, 3+disc, disc)

y0 = np.arange(0, 3+disc, disc)
offset = 0.3
y1 = 0.1 * np.sin(x1[:(Propagation_horizon)*Nv]+2) + 1.8 - offset
y2 = 0.1 * np.sin(x2[:(Propagation_horizon)*Nv]+3) + 1.5 - offset
y3 = 0.1 * np.sin(x3[:(Propagation_horizon)*Nv]+4) + 0.9 - offset
y4 = 0.1 * np.sin(x4[:(Propagation_horizon)*Nv]+5) + 0.55 - offset
y5 = 0.1 * np.sin(x4[:(Propagation_horizon)*Nv]+6) + 0.2 - offset
y0[:(Propagation_horizon)*Nv] = (y1 + y2 + y3 + y4 ) / 4 - 0.03 * np.sin(x0[:(Propagation_horizon)*Nv]+1.5) - 0.07

y0[(Propagation_horizon-1)*Nv:] = 0.1 * np.cos(x0[(Propagation_horizon-1)*Nv:]-x0[(Propagation_horizon-1)*Nv]) + y0[(Propagation_horizon-1)*Nv] - 0.1

lb0 = [r"$\mathbb{{E}}(x_{})$".format(str(i)) for i in range(len(y0[::Nv]))]
lb1 = [r"$\tilde{x}"+"_{}".format(str(i))+"^{{(1)}}$" for i in range(len(y1[::Nv]))]
lb2 = [r"$\tilde{x}"+"_{}".format(str(i))+"^{{(1)}}$" for i in range(len(y2[::Nv]))]
lb3 = [r"$\tilde{x}"+"_{}".format(str(i))+"^{{(2)}}$" for i in range(len(y3[::Nv]))]
lb4 = [r"$\tilde{x}"+"_{}".format(str(i))+"^{{(3)}}$" for i in range(len(y4[::Nv]))]
lb5 = [r"$\tilde{x}"+"_{}".format(str(i))+"^{{(5)}}$" for i in range(len(y5[::Nv]))]
lb0[0] = r"$x_0="+r"\mathbb{{E}}(x_{0})$"

r1 = 0.15
r2 = 0.7
theta = np.arange(0, 2*np.pi, 0.01)
x = x0[1] + r1 * np.cos(theta)
y = y0[1] + r2 * np.sin(theta)
ax1.fill(x,y,'gray',alpha = 0.3)

ax1.plot(x0[:(Propagation_horizon-1)*Nv], y0[:(Propagation_horizon-1)*Nv], 'b',label='estimated state')
# ax1.plot(x1[:(Propagation_horizon-1)*Nv], y1[:(Propagation_horizon-1)*Nv], 'b--',label='Uncertain States Evolution')
ax1.plot(x2[:(Propagation_horizon-1)*Nv], y2[:(Propagation_horizon-1)*Nv], 'b--',label='Uncertain States Evolution')
ax1.plot(x3[:(Propagation_horizon-1)*Nv], y3[:(Propagation_horizon-1)*Nv], 'b--')
ax1.plot(x4[:(Propagation_horizon-1)*Nv], y4[:(Propagation_horizon-1)*Nv], 'b--')
# ax1.plot(x5[:(Propagation_horizon-1)*Nv], y5[:(Propagation_horizon-1)*Nv], 'b--')

ax2.plot(x0[:(Propagation_horizon-1)*Nv], y0[:(Propagation_horizon-1)*Nv], 'b',label='estimated state')
# ax2.plot(x1[:(Propagation_horizon-1)*Nv], y1[:(Propagation_horizon-1)*Nv], 'b--',label='Uncertain States Evolution')
ax2.plot(x2[:(Propagation_horizon-1)*Nv], y2[:(Propagation_horizon-1)*Nv], 'b--',label='Uncertain States Evolution')
ax2.plot(x3[:(Propagation_horizon-1)*Nv], y3[:(Propagation_horizon-1)*Nv], 'b--')
ax2.plot(x4[:(Propagation_horizon-1)*Nv], y4[:(Propagation_horizon-1)*Nv], 'b--')
# ax2.plot(x5[:(Propagation_horizon-1)*Nv], y5[:(Propagation_horizon-1)*Nv], 'b--')

ax3.plot(x0[:(Propagation_horizon-1)*Nv], y0[:(Propagation_horizon-1)*Nv], 'b',label='estimated state')
# ax3.plot(x1[:(Propagation_horizon-1)*Nv], y1[:(Propagation_horizon-1)*Nv], 'b--',label='Uncertain States Evolution')
ax3.plot(x2[:(Propagation_horizon-1)*Nv], y2[:(Propagation_horizon-1)*Nv], 'b--',label='uncertain states\' evolution')
ax3.plot(x3[:(Propagation_horizon-1)*Nv], y3[:(Propagation_horizon-1)*Nv], 'b--')
ax3.plot(x4[:(Propagation_horizon-1)*Nv], y4[:(Propagation_horizon-1)*Nv], 'b--')
# ax3.plot(x5[:(Propagation_horizon-1)*Nv], y5[:(Propagation_horizon-1)*Nv], 'b--')

ax1.vlines(x = x0[(Propagation_horizon-1)*Nv], ymin = 0, ymax = 3, color = 'y', linestyle = '--', alpha = 0.5)
ax1.vlines(x = x0[-1], ymin = 0, ymax = 3, color = 'orange', linestyle = '--', alpha = 0.5)
ax1.plot(x0[(Propagation_horizon-1)*Nv:],y0[(Propagation_horizon-1)*Nv:],'g',label='evolution via nominal \n dynamics')

# ax1.scatter(x1[:(Propagation_horizon)*Nv:Nv],y1[:(Propagation_horizon)*Nv:Nv],c='b')
ax1.scatter(x2[:(Propagation_horizon)*Nv:Nv],y2[:(Propagation_horizon)*Nv:Nv],c='b')
ax1.scatter(x3[:(Propagation_horizon)*Nv:Nv],y3[:(Propagation_horizon)*Nv:Nv],c='b')
ax1.scatter(x4[:(Propagation_horizon)*Nv:Nv],y4[:(Propagation_horizon)*Nv:Nv],c='b')
# ax1.scatter(x5[:(Propagation_horizon)*Nv:Nv],y5[:(Propagation_horizon)*Nv:Nv],c='b')
ax1.scatter(x0[::Nv],y0[::Nv],c='r')

ax2.vlines(x = x0[(Propagation_horizon-1)*Nv], ymin = 0, ymax = 3, color = 'y', linestyle = '--', alpha = 0.5)
ax2.vlines(x = x0[-1], ymin = 0, ymax = 3, color = 'orange', linestyle = '--', alpha = 0.5)
ax2.plot(x0[(Propagation_horizon-1)*Nv:],y0[(Propagation_horizon-1)*Nv:],'g',label='evolution via nominal \n dynamics')

# ax2.scatter(x1[:(Propagation_horizon)*Nv:Nv],y1[:(Propagation_horizon)*Nv:Nv],c='b')
ax2.scatter(x2[:(Propagation_horizon)*Nv:Nv],y2[:(Propagation_horizon)*Nv:Nv],c='b')
ax2.scatter(x3[:(Propagation_horizon)*Nv:Nv],y3[:(Propagation_horizon)*Nv:Nv],c='b')
ax2.scatter(x4[:(Propagation_horizon)*Nv:Nv],y4[:(Propagation_horizon)*Nv:Nv],c='b')
# ax2.scatter(x5[:(Propagation_horizon)*Nv:Nv],y5[:(Propagation_horizon)*Nv:Nv],c='b')
ax2.scatter(x0[::Nv],y0[::Nv],c='r')

ax3.vlines(x = x0[(Propagation_horizon-1)*Nv], ymin = 0, ymax = 3, color = 'y', linestyle = '--', alpha = 0.5)
ax3.vlines(x = x0[-1], ymin = 0, ymax = 3, color = 'orange', linestyle = '--', alpha = 0.5)
ax3.plot(x0[(Propagation_horizon-1)*Nv:],y0[(Propagation_horizon-1)*Nv:],'g',label='evolution via nominal \n dynamics')

# ax3.scatter(x1[:(Propagation_horizon)*Nv:Nv],y1[:(Propagation_horizon)*Nv:Nv],c='b')
ax3.scatter(x2[:(Propagation_horizon)*Nv:Nv],y2[:(Propagation_horizon)*Nv:Nv],c='b')
ax3.scatter(x3[:(Propagation_horizon)*Nv:Nv],y3[:(Propagation_horizon)*Nv:Nv],c='b')
ax3.scatter(x4[:(Propagation_horizon)*Nv:Nv],y4[:(Propagation_horizon)*Nv:Nv],c='b')
# ax3.scatter(x5[:(Propagation_horizon)*Nv:Nv],y5[:(Propagation_horizon)*Nv:Nv],c='b')
ax3.scatter(x0[::Nv],y0[::Nv],c='r')

for i in range(len(y0[:((Propagation_horizon)*Nv):Nv])):
    if i == 0:
        ax1.annotate(lb0[i], xy = (x0[:(Propagation_horizon)*Nv:Nv][i], y0[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x0[:(Propagation_horizon)*Nv:Nv][i]-0.2, y0[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    else:
        ax1.annotate(lb0[i], xy = (x0[:(Propagation_horizon)*Nv:Nv][i], y0[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x0[:(Propagation_horizon)*Nv:Nv][i]+0.05, y0[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    # ax1.annotate(lb1[i], xy = (x1[:(Propagation_horizon)*Nv:Nv][i], y1[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x1[:(Propagation_horizon)*Nv:Nv][i]+0.05, y1[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax1.annotate(lb2[i], xy = (x2[:(Propagation_horizon)*Nv:Nv][i], y2[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x2[:(Propagation_horizon)*Nv:Nv][i]-0.05, y2[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax1.annotate(lb3[i], xy = (x3[:(Propagation_horizon)*Nv:Nv][i], y3[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x3[:(Propagation_horizon)*Nv:Nv][i]+0.05, y3[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax1.annotate(lb4[i], xy = (x4[:(Propagation_horizon)*Nv:Nv][i], y4[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x4[:(Propagation_horizon)*Nv:Nv][i]+0.05, y4[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    # ax1.annotate(lb5[i], xy = (x5[:(Propagation_horizon)*Nv:Nv][i], y5[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x5[:(Propagation_horizon)*Nv:Nv][i]+0.05, y5[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax1.annotate(r'$T_{}$'.format(str(i)), xy = (x4[:(Propagation_horizon)*Nv:Nv][i], 0), xytext = (x4[:(Propagation_horizon)*Nv:Nv][i], -0.12),)

for i in range(len(y0[:((Propagation_horizon)*Nv):Nv])):
    tmp = f'u-{3-i}'
    if i == 3:
        tmp = 'u'
    lb1[i] = r"$\tilde{x}_"+'{'+tmp+"}^{{(1)}}$"
    lb2[i] = r"$\tilde{x}_"+'{'+tmp+"}^{{(1)}}$"
    lb3[i] = r"$\tilde{x}_"+'{'+tmp+"}^{{(2)}}$"
    lb4[i] = r"$\tilde{x}_"+'{'+tmp+"}^{{(3)}}$"
    lb5[i] = r"$\tilde{x}_"+'{'+tmp+"}^{{(5)}}$"
    lb0[i] = r"$\mathbb{{E}}(x_"+'{'+tmp+"})$"
    ax2.annotate(lb0[i], xy = (x0[:(Propagation_horizon)*Nv:Nv][i], y0[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x0[:(Propagation_horizon)*Nv:Nv][i]+0.05, y0[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    # ax2.annotate(lb1[i], xy = (x1[:(Propagation_horizon)*Nv:Nv][i], y1[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x1[:(Propagation_horizon)*Nv:Nv][i]+0.05, y1[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax2.annotate(lb2[i], xy = (x2[:(Propagation_horizon)*Nv:Nv][i], y2[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x2[:(Propagation_horizon)*Nv:Nv][i]-0.05, y2[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax2.annotate(lb3[i], xy = (x3[:(Propagation_horizon)*Nv:Nv][i], y3[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x3[:(Propagation_horizon)*Nv:Nv][i]+0.05, y3[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax2.annotate(lb4[i], xy = (x4[:(Propagation_horizon)*Nv:Nv][i], y4[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x4[:(Propagation_horizon)*Nv:Nv][i]+0.05, y4[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    # ax2.annotate(lb5[i], xy = (x5[:(Propagation_horizon)*Nv:Nv][i], y5[:(Propagation_horizon)*Nv:Nv][i]), xytext = (x5[:(Propagation_horizon)*Nv:Nv][i]+0.05, y5[:(Propagation_horizon)*Nv:Nv][i]+0.05))
    ax2.annotate(f'$T_' + '{' + tmp + '}$', xy = (x4[:(Propagation_horizon)*Nv:Nv][i], 0), xytext = (x4[:(Propagation_horizon)*Nv:Nv][i], -0.12),)

for i in range(len(y1[:((Propagation_horizon)*Nv):Nv]),len(y0[::Nv])):
    tmp = f'u-{6-i}'
    if i == 4:
        tmp = 'u+1'
    ax2.annotate(f'$x_' + '{' + tmp + '}$', xy = (x0[::Nv][i], y0[::Nv][i]), xytext = (x0[::Nv][i]+0.05, y0[::Nv][i]+0.05))
    ax2.annotate(f'$T_' + '{' + tmp + '}$', xy = (x0[::Nv][i], 0), xytext = (x4[::Nv][i], -0.12))
    
for i in range(len(y1[:((Propagation_horizon)*Nv):Nv]),len(y0[::Nv])):
    tmp = f'p-{6-i}'
    if i == 6:
        tmp = 'p'
    ax3.annotate(f'$x_' + '{' + tmp + '}$', xy = (x0[::Nv][i], y0[::Nv][i]), xytext = (x0[::Nv][i]+0.05, y0[::Nv][i]+0.05))
    ax3.annotate(f'$T_' + '{' + tmp + '}$', xy = (x0[::Nv][i], 0), xytext = (x4[::Nv][i], -0.12))


ax1.spines['right'].set_color('lightgray')
ax1.spines['top'].set_color('none')
plt.yticks([])
ax1.yaxis.set_major_formatter(plt.NullFormatter())
ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.annotate("Uncertainty \n Propagation \n Horizon\n (UPH)",xy=((Propagation_horizon-1) * Nv * disc,0.1),xytext=((Propagation_horizon) * Nv * disc-0.2,0.15),arrowprops=dict(arrowstyle="->"))
ax1.annotate("Prediction \n Horizon",xy=(x0[-1],0.5),xytext=(x0[-1]-0.8,0.5),arrowprops=dict(arrowstyle="->"))
ax1.text(Nv*disc+0.54, 0.5, '.....',
        verticalalignment='center', 
        transform=ax1.transAxes)

ax2.spines['right'].set_color('lightgray')
ax2.spines['top'].set_color('none')
ax2.spines['left'].set_color('lightgray')
plt.yticks([])
ax2.yaxis.set_major_formatter(plt.NullFormatter())
ax2.xaxis.set_major_formatter(plt.NullFormatter())
ax2.annotate("Uncertainty \n Propagation \n Horizon\n (UPH)",xy=((Propagation_horizon-1) * Nv * disc,0.1),xytext=((Propagation_horizon) * Nv * disc-0.2,0.05),arrowprops=dict(arrowstyle="->"))
ax2.annotate("Prediction \n Horizon",xy=(x0[-1],0.5),xytext=(x0[-1]-0.6,0.4),arrowprops=dict(arrowstyle="->"))
ax2.text(5*Nv*disc+0.55, 0.5, '.....',
        verticalalignment='center', 
        transform=ax1.transAxes)

ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
ax3.spines['left'].set_color('lightgray')
ax3.spines['left'].set_linestyle('-.')
plt.yticks([])
ax3.yaxis.set_major_formatter(plt.NullFormatter())
ax3.xaxis.set_major_formatter(plt.NullFormatter())

ax3.annotate("Uncertainty \n Propagation \n Horizon (UPH)",xy=((Propagation_horizon-1) * Nv * disc,0.1),xytext=((Propagation_horizon) * Nv * disc-0.2,0.05),arrowprops=dict(arrowstyle="->"))
ax3.annotate("Prediction \n Horizon",xy=(x0[-1],0.5),xytext=(x0[-1]-0.65,0.4),arrowprops=dict(arrowstyle="->"))


# plt.xlabel(r'Descrete Time $(s)$',labelpad=12)
ax3.legend(loc =(-0.8,0.65))
# plt.grid(True)
plt.savefig(log_dir+'/ACC24_propagation_horizon.pdf', bbox_inches='tight')
plt.show()

######################################################
# CONSTRAINT TIGHTENING - UNCERTAINTY PROPAGATION
######################################################


plt.xlim(0, 3)
plt.ylim(-1, 3)

propragation_horizon = int(1 / 0.001)

x_convex_upper = np.arange(0, 3, 0.001)
x_convex_lower = np.arange(0, 3, 0.001)
x_ineq = np.arange(0, 3, 0.001)

y_convex_upper = 0.5 * np.sin(x_convex_upper * 1.5 + np.pi / 2) + 2.5
y_convex_lower = 0.5 * np.cos(x_convex_lower - 3) - 0.3
y_ineq = 0.222 * x_ineq**2 - 1.333 * x_ineq + 3
y_ineq_tighten = 0.389 * x_ineq**2 - 2.333 * x_ineq + 3

xt = np.arange(0, 3, 0.001)

plt.plot(x_convex_upper,y_convex_upper,'b-')
plt.plot(x_convex_lower,y_convex_lower,'b-')
plt.fill_between(x_convex_lower,y1=y_convex_lower,y2=y_convex_upper,color='b',alpha = 0.1,label = 'convex space for $\mathbb{E}(x_t)$')
plt.fill_between(x_ineq,y1=-1,y2=y_convex_lower,color='gray',alpha = 0.3,label = 'prohibited space')
plt.fill_between(x_ineq,y1=y_convex_upper,y2=3.0,color='gray',alpha = 0.3)

plt.plot(x_ineq[:propragation_horizon],y_ineq[:propragation_horizon],label = 'nominal constraint $h(x_t, u)$')
plt.plot(x_ineq[propragation_horizon:],y_ineq_tighten[propragation_horizon:],'g--',label = 'tightened constraint '+ r'$h_u(x_t, u)$')
plt.plot(x_ineq[propragation_horizon:],y_ineq[propragation_horizon:],'r-')
plt.plot(x_ineq[:propragation_horizon],y_ineq_tighten[:propragation_horizon],'r-',label = 'final applied Constraint')

plt.vlines(x = x_ineq[propragation_horizon], ymin = -1, ymax = 3, color = 'y', linestyle = '--')
plt.vlines(x = x_ineq[propragation_horizon], ymin = y_ineq_tighten[propragation_horizon], ymax = y_ineq[propragation_horizon], color = 'r', linestyle = '-')

plt.fill_between(x_ineq[:propragation_horizon],y1=y_ineq_tighten[:propragation_horizon],y2=y_convex_upper[:propragation_horizon],color='gray',alpha = 0.2)
plt.fill_between(x_ineq[propragation_horizon:],y1=y_ineq[propragation_horizon:],y2=y_convex_upper[propragation_horizon:],color='gray',alpha = 0.2)

# plt.xlabel(r'Time',labelpad=10)
# plt.title(r'Space for solving the nonlinear inequality constraints becoming infeasible')
plt.ylabel(r'$\mathbb{E}(x_t)$')
plt.xticks([])
plt.yticks([])

plt.scatter(1.9349, 0.389 * 1.9349**2 - 2.333 * 1.9349 + 3, color='k')

ax = plt.gca()
ax.annotate('Infeasibility', xy=(1.9349+0.01, 0.389 * 1.9349**2 - 2.333 * 1.9349 + 3 - 0.01), xytext=(1.9349 + 0.1, -0.8),
            arrowprops=dict(facecolor='black', shrink=0.005))

ax.annotate('Uncertainty \nPropagation \nHorizon', xy=(x_ineq[propragation_horizon], 0), xytext=(x_ineq[propragation_horizon] - 0.8, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.005))
ax.annotate(f'$T_' + '{' + 'u' + '}$', xy = (x_ineq[propragation_horizon], 0), xytext = (x_ineq[propragation_horizon], -1.2))
ax.annotate(f'$T_' + '{' + '0' + '}$', xy = (0, 0), xytext = (0, -1.2))
ax.annotate(f'$T_' + '{' + 'p' + '}$', xy = (3, 0), xytext = (3, -1.2))

ax.annotate('', xy=(1.5, 0.222 * 1.5**2 - 1.333 * 1.5 + 3), xytext=(1.5, 0.389 * 1.5**2 - 2.333 * 1.5 + 3),
            arrowprops=dict(facecolor='black', arrowstyle="<->"))
plt.text(1.55,0.8 , r'$\kappa\sqrt{Var[h(x_t,u)]}$')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

# plt.grid(True)
plt.legend(loc = 'upper right',fontsize=10)
plt.savefig(log_dir+'/ACC24_Convex_Space.png', bbox_inches='tight')
plt.show()


# #######################################################
# # CONSTRAINT INFEASIBILITY
# #######################################################
# # Set LaTeX rendering for labels, titles, and tick labels
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
# plt.rc('font', family='serif')
# plt.rc('font', size=11)

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


fig = plt.figure(figsize=(10,3))
plt.xlim(0, 2)
plt.ylim(-1.1, 1.1)


propragation_horizon = int(0.5 / 0.001)

x1 = np.arange(0, 3, 0.001)
xt = np.arange(0, 3, 0.001)


yu = np.sin(x1 / 2 + 1)
yl = -np.sin(x1 / 2 + 1.5)
y_u_t = np.sin(xt / 2 + 1) - 0.5 * xt
y_l_t = -np.sin(xt / 2 + 1.5) + 0.5 * xt

plt.plot(x1[:propragation_horizon], yu[:propragation_horizon], 'k--',label='nominal bounds '+ r'$\overline{h}$'+r'$, \underline{h}$')
plt.plot(x1[:propragation_horizon], yl[:propragation_horizon], 'k--')
plt.plot(xt[propragation_horizon:],y_u_t[propragation_horizon:], 'r--',label='uncertainty-aware bounds '+r'$\overline{h}_u$'+ r', $\underline{h}_u$')
plt.plot(xt[propragation_horizon:],y_l_t[propragation_horizon:], 'r--')

plt.plot(x1[propragation_horizon:], yu[propragation_horizon:], 'g-')
plt.plot(x1[propragation_horizon:], yl[propragation_horizon:], 'g-')
plt.plot(xt[:propragation_horizon],y_u_t[:propragation_horizon], 'g-',label = 'effective bounds with UPH')
plt.plot(xt[:propragation_horizon],y_l_t[:propragation_horizon], 'g-')

plt.vlines(x = x1[propragation_horizon], ymin = y_u_t[propragation_horizon], ymax = yu[propragation_horizon], color = 'g', linestyle = '-')
plt.vlines(x = x1[propragation_horizon], ymin = y_l_t[propragation_horizon], ymax = yl[propragation_horizon], color = 'g', linestyle = '-')

plt.vlines(x = x1[propragation_horizon], ymin = -1.1, ymax = 1.1, color = 'k', linestyle = '--', alpha = 0.5)

plt.ylabel(r'constraint $h(x_t,u)$')

plt.xticks([])
plt.yticks([])
ax = plt.gca()
plt.scatter(1.681,0.1223)
ax.annotate('Infeasibility', xy=(1.681, 0.1233), xytext=(1.681-0.01, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.005))

ax.annotate('Uncertainty \nPropagation \nHorizon\n(UPH)', xy=(x1[propragation_horizon], 0), xytext=(x1[propragation_horizon] - 0.45, -0.45),
            arrowprops=dict(facecolor='black', shrink=0.005))
ax.annotate(f'$T_' + '{' + 'u' + '}$', xy = (x1[propragation_horizon], 0), xytext = (x1[propragation_horizon], -1.3))
ax.annotate(f'$T_' + '{' + '0' + '}$', xy = (0, 0), xytext = (0, -1.3))
ax.annotate(f'$T_' + '{' + 'p' + '}$', xy = (2, 0), xytext = (2, -1.3))

ax.annotate('', xy=(1, np.sin(1/2 + 1)), xytext=(1, np.sin(1/2 + 1)-0.5 * 1),
            arrowprops=dict(facecolor='black', arrowstyle="<->"))
plt.text(1.05,0.7 , r'$\kappa\sqrt{\text{Var}[h(x_t,u)]}$')
plt.grid(True)
# plt.legend(loc = 'lower right')
legend_ax = fig.add_axes([0.1, 0.45, 0.8, 0.8])
legend_ax.axis('off')
legend_ax.legend(*ax.get_legend_handles_labels(), ncol=2,loc='upper center')
ax.spines['top'].set_visible(False)
plt.savefig(log_dir+'/ACC24_constraint_infeasibility.pdf', bbox_inches='tight')
plt.show()
