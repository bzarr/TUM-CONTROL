# Created on Tue Jan 10 11:07:38 2023

# Author: Baha Zarrouki (baha.zarrouki@tum.de)

from casadi import *
import numpy as np
from pylab import *
import yaml

# TODO: longitudinal tire forces are missing   
# TODO: consider down force!

def pred_stm(veh_params_file, tire_params_file = "pacejka_params_file.yaml"):   
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
    cW_F = -1 * -0.522  # Aero lift coefficient at front axle
    cW_R = -1 * -1.034  # Aero lift coefficient at rear axle

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
    posx    = MX.sym("posx")
    posy    = MX.sym("posy")
    yaw     = MX.sym("yaw")
    vlong   = MX.sym("vlong")
    vlat    = MX.sym("vlat")
    yawrate = MX.sym("yawrate")
    delta_f = MX.sym("delta_f")         # steering angle
    a       = MX.sym("a")               # acceleration
    x = vertcat(posx, posy, yaw, vlong, vlat, yawrate, delta_f, a)

    # controls
    jerk                = MX.sym("jerk")
    steering_rate       = MX.sym("steering_rate")
    u                   = vertcat(jerk, steering_rate)

    # xdot
    posx_dot    = MX.sym("posx_dot")
    posy_dot    = MX.sym("posy_dot")
    yaw_dot     = MX.sym("yaw_dot")
    vlong_dot   = MX.sym("vlong_dot")
    vlat_dot    = MX.sym("vlat_dot")
    yawrate_dot = MX.sym("yawrate_dot")
    delta_f_dot = MX.sym("delta_f_dot")
    a_dot       = MX.sym("a_dot")
    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat([])
    
    # Help variables 
    alpha_f     = MX.sym("alpha_f")            # front tire slip angle 
    alpha_r     = MX.sym("alpha_r")            # rear tire slip angle 
    
    ## Dynamics Definition

    # Rolling Resistance Forces
    # Source: Gerdts, M. "The single track model." (2003).
    v       = sqrt(vlong**2 + vlat**2) * 3.6        # v in [km/h] ...
    fr      = fr0 + fr1 * v/100 + fr4*(v/100)**4    # friction coefficient
    Fz_f    = m* lr * g / (lf + lr)                 # static tyre load at the front axle
    Fz_r    = m* lf * g / (lf + lr)                 # static tyre load at the rear axle
    Fr_f    = fr * Fz_f                             # rolling resistance at the front wheel
    Fr_r    = fr * Fz_r                             # rolling resistance at the rear wheel
    # Alternative Source Rolling resistance + "Optimal Vehicle Dynamics Control for Combined Longitudinal and Lateral Autonomous Vehicle Guidance" as a reference 
    # i_br = lr / (lr + lf)
    # Fx_f = m / (cos(delta_f) + (1 - i_br) / i_br) * a -  Fr_f
    # Fx_r = m / (1 + (i_br / (1 - i_br)) * cos(delta_f)) * a - Fr_r
    
    # Banking and Aerodynamik Forces. 
    # Source: Euroracing
    Fbanking_x      = m * g * sin(banking) * sin(mu)      # force on x-axis due to the road banking
    Fbanking_y      = m * g * sin(banking) * cos(mu)      # force on y-axis due to the road banking
    Faero           = 0.5 * ro * S * Cd * vlong**2        # aerodynamic effects

    # Longitudinal Tire Forces
    # Source: Euroracing
    F_braking  = 0                  # braking force
    Fb_f       = 2/3 * F_braking    # braking force at the front wheel
    Fb_r       = 1/3 * F_braking    # braking force at the rear wheel
    Fd         = m * a              # driving force at the wheel
    Fx_f       = -Fb_f - Fr_f
    Fx_r       = Fd - Fb_r - Fr_r   # rear wheels drive powertrain
    
    # Lateral Behaviour
    # Source:  Effects of Model Complexity on the Performance of Automated Vehicle Steering Controllers: Model Development, Validation and Comparison
    # adapted: formula from source hold only for vlong>0. Adapted with the assumption that 
    # side slip angle can be taken as 0 for negligeable longitudinal velocity
    alpha_f     = if_else(vlong > 0.001, delta_f - arctan((vlat + lf * yawrate) / vlong), 0.0)
    alpha_r     = if_else(vlong > 0.001, arctan((lr * yawrate - vlat)/ vlong), 0.0)
    
    # Lateral Tire Forces
    # Pacejka 'magic formula'(constant tyre load)
    # https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
    Fy_f_lat        = Df * sin(Cf * arctan(Bf * alpha_f - Ef * (Bf * alpha_f - arctan(Bf * alpha_f))))
    Fy_r_lat        = Dr * sin(Cr * arctan(Br * alpha_r - Er * (Br * alpha_r - arctan(Br * alpha_r))))

    # combined lateral tire forces (combined slip)
    Fmax_f      = sqrt(Fz_f**2 + (Cf * Fz_f)**2)
    Fmax_r      = sqrt(Fz_r**2 + (Cr * Fz_r)**2)
    Gy_f        = fmax(fmin(Fx_f/Fmax_f, 0.98), -0.98)      # combined slip weighting factor: clip at 0.98 to avoid singularity issues
    Gy_r        = fmax(fmin(Fx_r/Fmax_r, 0.98), -0.98)      # combined slip weighting factor: clip at 0.98 to avoid singularity issues
    Fy_f        = Fy_f_lat * cos(arcsin(Gy_f)) # consider combined slip of lateral and long dynamics
    Fy_r        = Fy_r_lat * cos(arcsin(Gy_r)) # consider combined slip of lateral and long dynamics
    
    # States Derivatives
    # Source: Ge, Qiang, et al. "Numerically stable dynamic bicycle model for discrete-time control." 2021 IEEE Intelligent Vehicles Symposium Workshops (IV Workshops). IEEE, 2021.
    posx_dot    = vlong * cos(yaw) - vlat * sin(yaw)
    posy_dot    = vlong * sin(yaw) + vlat * cos(yaw)
    yaw_dot     = yawrate
    # Source: EuroRacing
    vlong_dot   = 1 / m *(Fx_r - Faero - Fy_f * sin(delta_f) + Fx_f * cos(delta_f) - Fbanking_x + m * vlat * yawrate)
    vlat_dot    = 1 / m *(Fy_r + Fy_f * cos(delta_f) + Fx_f * sin(delta_f) - Fbanking_y - m * vlong * yawrate)
    yawrate_dot = 1 / Iz * (lf * (Fy_f * cos(delta_f) + Fx_f * sin(delta_f)) - lr * Fy_r)
    delta_f_dot = steering_rate 
    a_dot = jerk
    
    xdot = vertcat(posx_dot, posy_dot, yaw_dot, vlong_dot, vlat_dot, yawrate_dot, delta_f_dot, a_dot)

    f_expl = vertcat(
        posx_dot, 
        posy_dot, 
        yaw_dot, 
        vlong_dot,
        vlat_dot, 
        yawrate_dot,
        delta_f_dot,
        a_dot,
        jerk, 
        steering_rate,
    )

    ## Initial Conditions
    x0 = np.array([0, 0, 0, 0, 0.00, 0, 0.0, 0.0])

    ## constraints
    # state bounds
    model.jerk_min          = veh_params['jerk_min']       # [m/s³]
    model.jerk_max          = veh_params['jerk_max']       # [m/s³]
    constraint.lat_acc_min  = veh_params['lat_acc_min']    # [m/s²]
    constraint.lat_acc_max  = veh_params['lat_acc_max']    # [m/s²]
    constraint.alat = Function("a_lat", [x], [vlong * yawrate])
    # input bounds
    model.acc_min           = veh_params['acc_min']                 # [m/s²]
    model.acc_max           = veh_params['acc_max']                 # [m/s²]      
    model.delta_f_min       = veh_params['delta_f_min']             # steering angle [rad]
    model.delta_f_max       = veh_params['delta_f_max']             # steering angle [rad]         
    model.delta_f_dot_min   = veh_params['delta_f_dot_min']         # steering angular velocity [rad/s]
    model.delta_f_dot_max   = veh_params['delta_f_dot_max']         # steering angular velocity [rad/s]    
    
    ## CasADi model struct
    model.f_expl_expr   = f_expl
    model.x             = x
    model.xdot          = xdot
    model.u             = u
    model.z             = z
    model.p             = p
    model.name          = model_name
    model.params        = params
    model.x0            = x0
        
    
    return model, constraint
