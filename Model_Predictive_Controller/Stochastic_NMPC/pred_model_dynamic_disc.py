# Created on Tue Jan 10 11:07:38 2023

# Author: Baha Zarrouki (baha.zarrouki@tum.de)
#         Chenyang Wang (16chenyang.wang@tum.de)
from casadi import *
import numpy as np
from pylab import deg2rad
import matplotlib.pyplot as plt
import yaml

# TODO: longitudinal tire forces are missing   
# TODO: consider down force!
"""
Source: [1] Zarrouki, Baha, Chenyang Wang, and Johannes Betz. 
        "A stochastic nonlinear model predictive control with an uncertainty propagation horizon 
        for autonomous vehicle motion control." arXiv preprint arXiv:2310.18753 (2023).
"""
def pred_stm(Ts, num_poly_terms,n_samples, veh_params_file, tire_params_file = "pacejka_params_file.yaml"):   
    """
    This function implements a simplified dynamic bicycle model of a vehicle. 
    It uses the CasADi Model to define the states and control inputs, 
    and sets the dynamics of the vehicle using various tire and physical parameters. 
    The function can use two different models for tire behavior: Pacejka 'magic formula' 
    and AT-model. The default tire model is Pacejka 'magic formula'. 
    The function takes two required arguments: 'veh_params_file' and an optional argument
    'model_type' for the tire model. If the tire model is Pacejka, 
    it also takes an optional argument 'tire_params_file', which is set to "pacejka_params_file.yaml" by default.
    """

    # reference point: center of mass
    constraint  = types.SimpleNamespace()
    model       = types.SimpleNamespace()
    params      = types.SimpleNamespace()
    model_name  = "pred_dynamic_bicycle_model"
    # load vehicle params
    with open(veh_params_file, 'r') as file:
        veh_params = yaml.load(file, Loader=yaml.FullLoader)
    
    lf  = veh_params['lf']  # distance from spring mass center of gravity to front axle [m]  LENA
    lr  = veh_params['lr']  # distance from spring mass center of gravity to rear axle [m]  LENB
    m   = veh_params['m']   # vehicle mass [kg]  MASS
    Iz  = veh_params['Iz']  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    g = 9.81  #[m/s^2]
    banking = deg2rad(0)
    # Source: Dr. M. Gerdts, The single track model, Universität Bayreuth, SS 2003
    ro  = veh_params['ro']              # air density 
    S   = veh_params['S']            # frontal area
    Cd  = veh_params['Cd']           # drag coeff
    fr0 = 0.009         # friction coefficient factors
    fr1 = 0.002         # friction coefficient factors
    fr4 = 0.0003        # friction coefficient factors
    # cW_F = -1 * -0.522  # Aero lift coefficient at front axle
    # cW_R = -1 * -1.034  # Aero lift coefficient at rear axle

    params.lf = lf
    params.lr = lr
    params.Iz = Iz
    params.m  = m
    params.veh_length = veh_params['veh_length'] 
    params.veh_width = veh_params['veh_width'] 
    # load tire params Pacejka 'magic formula'
    with open(tire_params_file, 'r') as file:
        tire_params = yaml.load(file, Loader=yaml.FullLoader)
    Bf = tire_params['tire_params']['front']['Bf'] #stiffness factor
    Cf = tire_params['tire_params']['front']['Cf'] #shape factor
    Df = tire_params['tire_params']['front']['Df'] #peak value
    Ef = tire_params['tire_params']['front']['Ef'] #curvature factor
    Br = tire_params['tire_params']['rear']['Br'] #stiffness factor
    Cr = tire_params['tire_params']['rear']['Cr'] #shape factor
    Dr = tire_params['tire_params']['rear']['Dr'] #peak value
    Er = tire_params['tire_params']['rear']['Er'] #curvature factor
    mu = tire_params['mu']  #Lateral friction Muy

    params.Bf  = Bf
    params.Cf  = Cf
    params.Df  = Df
    params.Ef  = Ef
    params.Br  = Br
    params.Cr  = Cr
    params.Dr  = Dr
    params.Er  = Er

    ## CasADi Model
    # states & control inputs
    posx    = {}
    posy    = {}
    yaw     = {}
    vlong   = {}
    vlat    = {}
    yawrate = {}
    delta_f = {}
    a       = {}            
    x = vertcat([])

    # controls
    jerk                = MX.sym("jerk")
    steering_rate       = MX.sym("steering_rate")
    u                   = vertcat(jerk, steering_rate)

    ## constraints
    # state bounds
    model.jerk_min          = veh_params['jerk_min']       # [m/s³]
    model.jerk_max          = veh_params['jerk_max']       # [m/s³]
    constraint.lat_acc_min  = veh_params['lat_acc_min']    # [m/s²]
    constraint.lat_acc_max  = veh_params['lat_acc_max']    # [m/s²]
    # input bounds
    model.acc_min           = veh_params['acc_min']                 # [m/s²]
    model.acc_max           = veh_params['acc_max']                 # [m/s²]      
    model.delta_f_min       = veh_params['delta_f_min']             # steering angle [rad]
    model.delta_f_max       = veh_params['delta_f_max']             # steering angle [rad]         
    model.delta_f_dot_min   = veh_params['delta_f_dot_min']         # steering angular velocity [rad/s]
    model.delta_f_dot_max   = veh_params['delta_f_dot_max']         # steering angular velocity [rad/s]    

    # disturbances as parameters
    A_pce           = MX.sym('A_pce', num_poly_terms*n_samples)
    risk_parameter  = MX.sym('risk_parameter')
    stop_flag       = MX.sym('stop_flag', 1, 1)
    p               = vertcat(A_pce, risk_parameter, stop_flag)

    # creating symbols for n samples
    for i in range(n_samples+1):
        posx["posx_"        + str(i)] = MX.sym("posx_"+str(i))
        posy["posy_"        + str(i)] = MX.sym("posy_"+str(i))
        yaw["yaw_"          + str(i)] = MX.sym("yaw_"+str(i))
        vlong["vlong_"      + str(i)] = MX.sym("vlong_"+str(i))
        vlat["vlat_"        + str(i)] = MX.sym("vlat_"+str(i))
        yawrate["yawrate_"  + str(i)] = MX.sym("yawrate_"+str(i))
        delta_f["delta_f_"  + str(i)] = MX.sym("delta_f_"+str(i))
        a["a_"              + str(i)] = MX.sym("a_"+str(i))
        
        x = vertcat(x, 
                    posx["posx_" + str(i)], posy["posy_" + str(i)], yaw["yaw_" + str(i)] , 
                    vlong["vlong_" + str(i)], vlat["vlat_" + str(i)], yawrate["yawrate_" + str(i)], 
                    delta_f["delta_f_" + str(i)],a["a_" + str(i)])

    A = reshape(A_pce, n_samples, num_poly_terms).T
    f_disc = vertcat([])

    posx_       = x[0]
    posy_       = x[1]
    yaw_        = x[2]
    vlong_      = x[3]
    vlat_       = x[4]
    yawrate_    = x[5]
    delta_f_    = x[6]
    a_          = x[7]
    x_nominal   = vertcat(posx_, posy_, yaw_, vlong_, vlat_, yawrate_, delta_f_, a_)

    alpha_f     = MX.sym("alpha_f")            # front tire slip angle 
    alpha_r     = MX.sym("alpha_r")            # rear tire slip angle 
    v       = sqrt(vlong_**2 + vlat_**2) * 3.6        # v in [km/h] ...
    fr      = fr0 + fr1 * v/100 + fr4*(v/100)**4    # friction coefficient
    Fz_f    = m* lr * g / (lf + lr)                 # static tyre load at the front axle
    Fz_r    = m* lf * g / (lf + lr)                 # static tyre load at the rear axle
    Fr_f    = fr * Fz_f                             # rolling resistance at the front wheel
    Fr_r    = fr * Fz_r                             # rolling resistance at the rear wheel
    Fbanking_x      = m * g * sin(banking) * sin(mu)      # force on x-axis due to the road banking
    Fbanking_y      = m * g * sin(banking) * cos(mu)      # force on y-axis due to the road banking
    Faero           = 0.5 * ro * S * Cd * vlong_**2        # aerodynamic effects
    # F_braking  = 0                  # braking force
    # Fb_f       = 2/3 * F_braking    # braking force at the front wheel
    # Fb_r       = 1/3 * F_braking    # braking force at the rear wheel
    Fd          = m * a_              # driving force at the wheel
    Fx_f        = - Fr_f
    Fx_r        = Fd - Fr_r   # rear wheels drive powertrain
    alpha_f     = if_else(vlong_ > 0.001, delta_f_ - arctan((vlat_ + lf * yawrate_) / vlong_), 0.0)
    alpha_r     = if_else(vlong_ > 0.001, arctan((lr * yawrate_ - vlat_)/ vlong_), 0.0)
    Fy_f_lat    = Df * sin(Cf * arctan(Bf * alpha_f - Ef * (Bf * alpha_f - arctan(Bf * alpha_f))))
    Fy_r_lat    = Dr * sin(Cr * arctan(Br * alpha_r - Er * (Br * alpha_r - arctan(Br * alpha_r))))
    Fmax_f      = sqrt(Fz_f**2 + (Cf * Fz_f)**2)
    Fmax_r      = sqrt(Fz_r**2 + (Cr * Fz_r)**2)
    Gy_f        = fmax(fmin(Fx_f/Fmax_f, 0.98), -0.98)      # combined slip weighting factor: clip at 0.98 to avoid singularity issues
    Gy_r        = fmax(fmin(Fx_r/Fmax_r, 0.98), -0.98)      # combined slip weighting factor: clip at 0.98 to avoid singularity issues
    Fy_f        = Fy_f_lat * cos(arcsin(Gy_f)) # consider combined slip of lateral and long dynamics
    Fy_r        = Fy_r_lat * cos(arcsin(Gy_r)) # consider combined slip of lateral and long dynamics

    f_posx_dot      = vlong_ * cos(yaw_) - vlat_ * sin(yaw_)
    f_posy_dot      = vlong_ * sin(yaw_) + vlat_ * cos(yaw_)
    f_yaw_dot       = yawrate_
    f_vlong_dot     = 1 / m *(Fx_r - Faero - Fy_f * sin(delta_f_) + Fx_f * cos(delta_f_) - Fbanking_x + m * vlat_ * yawrate_) 
    f_vlat_dot      = 1 / m *(Fy_r + Fy_f * cos(delta_f_) + Fx_f * sin(delta_f_) - Fbanking_y - m * vlong_ * yawrate_) 
    f_yawrate_dot   = 1 / Iz * (lf * (Fy_f * cos(delta_f_) + Fx_f * sin(delta_f_)) - lr * Fy_r) 
    f_delta_f_dot   = steering_rate 
    f_a_dot         = jerk
    f_expl          = vertcat(f_posx_dot, f_posy_dot, f_yaw_dot, f_vlong_dot, f_vlat_dot, f_yawrate_dot, f_delta_f_dot, f_a_dot)
    
    itg = Function('f', [x_nominal,u], [f_expl], ['state','control'], ['x_dot'])
    k1 = itg(x_nominal, u)
    k2 = itg(x_nominal + Ts * k1 / 2, u)
    k3 = itg(x_nominal + Ts * k2 / 2, u)
    k4 = itg(x_nominal + Ts * k3 , u)
    x_next_nominal = x_nominal + (k1 + 2 * k2 + 2 * k3 + k4) * Ts / 6

    for i in range(1,n_samples+1):
        posx_       = x[8*i]
        posy_       = x[8*i + 1]
        yaw_        = x[8*i + 2]
        vlong_      = x[8*i + 3]
        vlat_       = x[8*i + 4]
        yawrate_    = x[8*i + 5]
        delta_f_    = x[8*i + 6]
        a_          = x[8*i + 7]
        x_current   = vertcat(posx_, posy_, yaw_, vlong_, vlat_, yawrate_, delta_f_, a_)
        k1 = itg(x_current, u)
        k2 = itg(x_current + Ts * k1 / 2, u)
        k3 = itg(x_current + Ts * k2 / 2, u)
        k4 = itg(x_current + Ts * k3 , u)
        x_next = x_current + (k1 + 2 * k2 + 2 * k3 + k4) * Ts / 6
        x_next = if_else( stop_flag == 1, x_current, x_next)
        f_disc = vertcat(f_disc,x_next)
    
    # A represents A in Eq.8 [1]
    # Eq.7-8 to calculate coeff_ineq
    # Eq.6 to calculate Expectation 
    coeff_x = A @ reshape(f_disc,8,n_samples).T
    mean_x  = coeff_x[0,:]
    # var_x = sum1(coeff_x[1:,:]**2)
    x0_next = if_else( stop_flag == 1, x_next_nominal, mean_x.T)
    
    f_disc  = vertcat(x0_next,f_disc)

    x0 = np.ndarray(((n_samples+1) * 8 , 1))
    
    ## CasADi model struct

    model.f_disc        = f_disc
    model.x             = x
    model.u             = u
    model.p             = p
    model.name          = model_name
    model.params        = params
    model.x0            = x0
    constraint.alat = Function("a_lat", [x], [vlong_ * yawrate_])    
    
    return model, constraint
