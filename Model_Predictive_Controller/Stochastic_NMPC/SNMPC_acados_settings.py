# Created on Tue Dec 06 11:20 2022

# Author: Baha Zarrouki (baha.zarrouki@tum.de)
#         Chenyang Wang (16chenyang.wang@tum.de)

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from .pred_model_dynamic_disc import pred_stm
from .stochastic_mpc_utils import *
import scipy.linalg
import numpy as np
from casadi import vertcat, sqrt, fmod, pi, if_else, MX, Function, sum1

"""
Source: [1] Zarrouki, Baha, Chenyang Wang, and Johannes Betz. 
        "A stochastic nonlinear model predictive control with an uncertainty propagation horizon 
        for autonomous vehicle motion control." arXiv preprint arXiv:2310.18753 (2023).
"""

def acados_settings(Tf, N, x0, Q, R, Qe, L1_pen, L2_pen,ax_max_interpolant, ay_max_interpolant, combined_acc_limits, SNMPC_params, veh_params_file = "veh_parameters_bmw320i.yaml", tire_params_file= "pacejka_params_file.yaml", solver_generate_C_code = True, solver_build = True):

    ocp = AcadosOcp()
    Ts                  = Tf / N
    # --- load SNMPC parameters --- #
    n_samples           = SNMPC_params['n_samples'] # number of samples
    gamma               = SNMPC_params['gamma']     # gamma is p in Eq.13 [1]: desired probability of constraints violation 
    expansion_degree    = SNMPC_params['expansion_degree']
    n_w                 = SNMPC_params['n_vars']    # n_w: total number of uncertain system params in Eq.3 [1]  
    num_poly_terms      = int(np.math.factorial(n_w + expansion_degree) / (np.math.factorial(n_w) * np.math.factorial(expansion_degree)))
    # num_poly_terms is L in Eq.3 [1]

    # load prediction model
    pred_model, constraints = pred_stm(Ts, num_poly_terms,n_samples,veh_params_file, tire_params_file)

    # define acados ODE
    model               = AcadosModel()
    model.disc_dyn_expr = pred_model.f_disc
    model.x             = pred_model.x
    model.u             = pred_model.u
    model.p             = pred_model.p
    model.name          = 'SNMPC'
    ocp.model           = model

    ocp.dims.N = N
    unscale = 1 # N / Tf

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p

    # cost function weights
    ocp.cost.W      = 0.01 * unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e    = 0.01 * Qe / unscale

    ## --- Core part of SNMPC --- 
   
    # x_nominal represent the mean value of n samples
    x_nominal = x[0:8]
    a_lon_nominal = x_nominal[7]
    a_lat_nominal = x_nominal[3] * x_nominal[5]

    # compute mean ax_max and mean ay_max at every single shooting node
    vel_abs_nominal = sqrt(x_nominal[3]**2 + x_nominal[4] ** 2)
    ay_max_nominal = ay_max_interpolant(vel_abs_nominal)
    ax_max_nominal = ax_max_interpolant(vel_abs_nominal)
    ax_max_nominal = if_else( a_lon_nominal < 0, -pred_model.acc_min, ax_max_nominal)
    if combined_acc_limits == 0:
        a_lon_limits_nominal = a_lon_nominal / ax_max_nominal
        a_lat_limits_nominal = a_lat_nominal / ay_max_nominal
    elif combined_acc_limits == 1:
        acc_limits_ineq1_nominal = a_lon_nominal/ax_max_nominal + a_lat_nominal/ay_max_nominal
        acc_limits_ineq2_nominal = a_lon_nominal/ax_max_nominal - a_lat_nominal/ay_max_nominal
    else:
        acc_lim_ineq_nominal = (a_lon_nominal/ax_max_nominal)**2 + (a_lat_nominal/ay_max_nominal)**2

    # Weighting matrices initialization for close-form solution for l2-norm regression
    if combined_acc_limits == 0:
        a_lon_limits_ineq_simulation = MX(n_samples,1)
        a_lat_limits_ineq_simulation = MX(n_samples,1)
    elif combined_acc_limits == 1:
        acc_limits_ineq1_simulation = MX(n_samples,1)
        acc_limits_ineq2_simulation = MX(n_samples,1)
    else:
        acc_limits_ineq_simulation = MX(n_samples,1)

    # collect values of nonlinear constraints from every samples
    for i in range(1,n_samples+1):
        x_current = x[8*i:8*i+8]
        a_lon_current = x_current[7]
        a_lat_current = x_current[3] * x_current[5]

        # compute ax_max and ay_max at every single shooting node
        vel_abs_current = sqrt(x_current[3]**2 + x_current[4] ** 2)
        ay_max_current = ay_max_interpolant(vel_abs_current)
        ax_max_current = ax_max_interpolant(vel_abs_current)
        ax_max_current = if_else( a_lon_current < 0, -pred_model.acc_min, ax_max_current)
        if combined_acc_limits == 0:
            a_lon_limits_ineq = a_lon_current / ax_max_current
            a_lat_limits_ineq = a_lat_current / ay_max_current
            a_lon_limits_ineq_simulation[i-1] = a_lon_limits_ineq
            a_lat_limits_ineq_simulation[i-1] = a_lat_limits_ineq
        elif combined_acc_limits == 1:
            acc_limits_ineq1 = a_lon_current / ax_max_current + a_lat_current / ay_max_current
            acc_limits_ineq2 = a_lon_current / ax_max_current - a_lat_current / ay_max_current
            acc_limits_ineq1_simulation[i-1] = acc_limits_ineq1
            acc_limits_ineq2_simulation[i-1] = acc_limits_ineq2
        else:
            acc_limits_ineq = (a_lon_current/ax_max_current)**2 + (a_lat_current/ay_max_current)**2
            # acc_limits_ineq_simulation = vertcat(acc_limits_ineq_simulation,acc_limits_ineq)
            acc_limits_ineq_simulation[i-1] = acc_limits_ineq

    # A represent the PCE matrix A in Eq.8 [1]
    A = reshape(p[:-2],n_samples,num_poly_terms).T
    
    # Eq.8 [1] to calculate coeff_ineq
    # Eq.15,16,17 [1] to calculate Expectation and Variance
    if combined_acc_limits == 0:
        coeff_ineq_lon = A @ a_lon_limits_ineq_simulation
        coeff_ineq_lat = A @ a_lat_limits_ineq_simulation
        mean_ineq_lon = coeff_ineq_lon[0]
        mean_ineq_lat = coeff_ineq_lat[0]
        var_ineq_lon = sum1(coeff_ineq_lon[1:]**2)
        var_ineq_lat = sum1(coeff_ineq_lat[1:]**2)
    elif combined_acc_limits == 1:
        coeff_ineq_1 = A @ acc_limits_ineq1_simulation
        coeff_ineq_2 = A @ acc_limits_ineq2_simulation
        mean_ineq_1 = coeff_ineq_1[0]
        mean_ineq_2 = coeff_ineq_2[0]
        var_ineq_1 = sum1(coeff_ineq_1[1:]**2)
        var_ineq_2 = sum1(coeff_ineq_2[1:]**2)
    else:
        coeff_ineq = A @ acc_limits_ineq_simulation
        mean_ineq = coeff_ineq[0]
        var_ineq = sum1(coeff_ineq[1:]**2)
    
    # adjust yaw to [0..2*pi]
    yaw_mean = fmod(x_nominal[2], 2*pi)
    yaw_mean = if_else( yaw_mean < 0,yaw_mean + 2*pi , yaw_mean) # adjust for negative angles
    
    ocp.cost.cost_type      = "NONLINEAR_LS" # Cost type at intermediate shooting nodes (1 to N-1)
    ocp.cost.cost_type_e    = "NONLINEAR_LS" # Cost type at terminal shooting node (N)0101
    ocp.model.cost_y_expr   = vertcat(x_nominal[0],x_nominal[1], yaw_mean, vel_abs_nominal,  u) 
    ocp.model.cost_y_expr_e = vertcat(x_nominal[0],x_nominal[1], yaw_mean, vel_abs_nominal)

    # intial references
    ocp.cost.yref   = np.array([0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0])


    x_fun       = MX.sym('x_fun',8,1)
    a_lat       = x_fun[3] * x_fun[5] 
    constraints.a_lat = Function("a_lat", [x_fun], [a_lat]) # define function for evaluation

    # all chance constraints are transformed to deterministic constraints based on Eq.13-14 [1]
    if combined_acc_limits == 0:
        # separated limits
        a_lon_limits_ineq = if_else(p[-1] == 1, a_lon_limits_nominal, mean_ineq_lon + sqrt(var_ineq_lon) * sqrt((1 - gamma) / gamma))
        a_lat_limits_ineq = if_else(p[-1] == 1, a_lat_limits_nominal, mean_ineq_lat + sqrt(var_ineq_lat) * sqrt((1 - gamma) / gamma))
        # constraints: nonlinear inequalities
        ocp.model.con_h_expr = vertcat(a_lon_limits_ineq, a_lat_limits_ineq)
        ocp.model.con_h_expr_e = vertcat(a_lon_limits_ineq, a_lat_limits_ineq)
        # nonlinear constraints
        # stage bounds for nonlinear inequalities
        ocp.constraints.lh      = np.array([-1, -1])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
        ocp.constraints.uh      = np.array([1, 1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
        # terminal bounds for nonlinear inequalities
        ocp.constraints.lh_e    = np.array([-1, -1])  # lower bound for nonlinear inequalities at terminal shooting node N
        ocp.constraints.uh_e    = np.array([1, 1])  # upper bound for nonlinear inequalities at terminal shooting node N
    elif combined_acc_limits == 1:
        # Diamond shaped combined lat lon acceleration limits
        acc_limits_ineq1 = if_else(p[-1] == 1, acc_limits_ineq1_nominal, mean_ineq_1 + sqrt(var_ineq_1) * sqrt((1 - gamma) / gamma))
        acc_limits_ineq2 = if_else(p[-1] == 1, acc_limits_ineq2_nominal, mean_ineq_2 + sqrt(var_ineq_2) * sqrt((1 - gamma) / gamma))
        # constraints: nonlinear inequalities
        ocp.model.con_h_expr = vertcat(acc_limits_ineq1, acc_limits_ineq2)
        ocp.model.con_h_expr_e = vertcat(acc_limits_ineq1, acc_limits_ineq2)
        # nonlinear constraints
        # stage bounds for nonlinear inequalities
        ocp.constraints.lh      = np.array([-1, -1])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
        ocp.constraints.uh      = np.array([1, 1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
        # terminal bounds for nonlinear inequalities
        ocp.constraints.lh_e    = np.array([-1, -1])  # lower bound for nonlinear inequalities at terminal shooting node N
        ocp.constraints.uh_e    = np.array([1, 1])  # upper bound for nonlinear inequalities at terminal shooting node N
    else:
        # Circle shaped combined lat lon acceleration limits
        acc_limits_ineq = if_else(p[-1] == 1 , acc_lim_ineq_nominal, mean_ineq + sqrt(var_ineq) * sqrt((1 - gamma) / gamma))
        # constraints: nonlinear inequalities
        ocp.model.con_h_expr = vertcat(acc_limits_ineq)
        ocp.model.con_h_expr_e = vertcat(acc_limits_ineq)
        # nonlinear constraints
        # stage bounds for nonlinear inequalities
        ocp.constraints.lh      = np.array([0])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
        ocp.constraints.uh      = np.array([1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
        # terminal bounds for nonlinear inequalities
        ocp.constraints.lh_e    = np.array([0])  # lower bound for nonlinear inequalities at terminal shooting node N
        ocp.constraints.uh_e    = np.array([1])  # upper bound for nonlinear inequalities at terminal shooting node N

    # acc_limits_ineq = if_else(p[-1] == 1 , acc_lim_ineq_nominal, mean_ineq + sqrt(var_ineq) * sqrt((1 - gamma) / gamma))
    # model.con_h_expr = vertcat(acc_limits_ineq)
    # model.con_h_expr_e = vertcat(acc_limits_ineq)

    # # nonlinear constraints
    # # stage bounds for nonlinear inequalities
    # ocp.constraints.lh      = np.array([0])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
    # ocp.constraints.uh      = np.array([1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
    # # terminal bounds for nonlinear inequalities
    # ocp.constraints.lh_e    = np.array([0])  # lower bound for nonlinear inequalities at terminal shooting node N
    # ocp.constraints.uh_e    = np.array([1])  # upper bound for nonlinear inequalities at terminal shooting node N

    # stage bounds on x
    ocp.constraints.lbx     = np.array([pred_model.delta_f_min]) # lower bounds on x at intermediate shooting nodes (1 to N-1)
    ocp.constraints.ubx     = np.array([pred_model.delta_f_max]) # upper bounds on x at intermediate shooting nodes (1 to N-1)
    ocp.constraints.idxbx   = np.array([6])   # indices of bounds on x (defines Jbx) at intermediate shooting nodes (1 to N-1)
    # terminal bounds on x
    ocp.constraints.lbx_e   = np.array([pred_model.delta_f_min])   # lower bounds on x at terminal shooting node N
    ocp.constraints.ubx_e   = np.array([pred_model.delta_f_max])   # upper bounds on x at terminal shooting node N
    ocp.constraints.idxbx_e = np.array([6])     # Indices for bounds on x at terminal shooting node N (defines Jebx)

    ocp.constraints.lbu     = np.array([pred_model.delta_f_dot_min])
    ocp.constraints.ubu     = np.array([pred_model.delta_f_dot_max])
    ocp.constraints.idxbu   = np.array([1])

    # soft constraints --> help the solver to find a solution (QP solver sometimes does not find a solution with hard constraints)
    # source: https://discourse.acados.org/t/infeasibility-issues-when-using-hard-nonlinear-constraints/1021 
    # Slack variables are useful to relax the constraints and help the solver to find a solution
    # Define parameters for the slack variables in the cost function
    # L1 is the linear term and L2 is the quadratic term
    # Tuning: if it is too small, the constraint will not be satisfied, if it is too big, it will become a hard constraint
    # and the solver will not find a solution --> tune it to find the right balance.

    # Slack variables for the stage cost constraints
    # The main idea is to add a cost term in the cost function that penalizes the violation of the constraints
    # The slack variables are added to the cost function and the constraints are relaxed (check eq 1 from link below)
    # https://raw.githubusercontent.com/acados/acados/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Get the number of non linear constraints in the stage cost
    nh = ocp.model.con_h_expr.shape[0]
    # Define which one of the input constraints will be written as slack variables in the stage cost function
    # In our case, only one input constraint is defined, so we will use the one
    ocp.constraints.Jsbu = np.eye(1) 
    # Define which one of the state constraints will be written as slack variables in the stage cost function
    # In our case, only one state constraint is defined, so we will use the one
    ocp.constraints.Jsbx = np.eye(1) 
    # Define which one of the nonlinear constraints will be written as slack variables in the stage cost function
    # In our case, we will use all of them
    ocp.constraints.Jsh = np.eye(nh) 

    # Define the penalty for the slack variables
    # z_1 is the linear penalty for the slack variables in the stage cost function
    z_1 = np.ones((nh + 2,)) * L1_pen
    # z_2 is the quadratic penalty for the slack variables in the stage cost function
    z_2 = np.ones((nh + 2,)) * L2_pen
    
    # Add the penalty to the cost function
    # Obs: the vectors Zl, Zu, zl, zu, Zl_e, Zu_e, zl_e, zu_e are defined as the weight of the constraints violation
    # When we have slack variables in input, state and nonlinear constraints, the order that these variables are stacked is
    # [input_slack, state_slack, nonlinear_slack], according to the link above, last line of page 1.
    # Quadratic penalty to when the constraint is violated in lower bound in the stage cost
    ocp.cost.Zl = z_2
    # Quadratic penalty to when the constraint is violated in upper bound in the stage cost
    ocp.cost.Zu = z_2
    # Linear penalty to when the constraint is violated in lower bound in the stage cost
    ocp.cost.zl = z_1
    # Linear penalty to when the constraint is violated in upper bound in the stage cost    
    ocp.cost.zu = z_1

    # Quadratic penalty to when the constraint is violated in lower bound in the stage cost
    ocp.cost.Zl_0 = np.ones((nh + 0,)) * L2_pen
    # Quadratic penalty to when the constraint is violated in upper bound in the stage cost
    ocp.cost.Zu_0 = np.ones((nh + 0,)) * L2_pen
    # Linear penalty to when the constraint is violated in lower bound in the stage cost
    ocp.cost.zl_0 = np.ones((nh + 0,)) * L1_pen
    # Linear penalty to when the constraint is violated in upper bound in the stage cost    
    ocp.cost.zu_0 = np.ones((nh + 0,)) * L1_pen

    # Slack variables for the terminal cost constraints
    # Get the number of non linear constraints in the terminal cost
    nh_e = ocp.model.con_h_expr_e.shape[0]    
    # Define which one of the state constraints will be written as slack variables in the terminal cost function
    # In our case, only one state constraint is defined, so we will use the one
    ocp.constraints.Jsbx_e = np.eye(1) 
    # Define which one of the nonlinear constraints will be written as slack variables in the terminal cost function
    # In our case, we will use all of them
    ocp.constraints.Jsh_e = np.eye(nh_e) 

    # Define the penalty for the slack variables
    # z_1 is the linear penalty for the slack variables in the terminal cost function
    z_1_e = np.ones((nh_e + 1,)) * L1_pen
    # z_2 is the quadratic penalty for the slack variables in the terminal cost function
    z_2_e = np.ones((nh_e + 1,)) * L2_pen

    # Add the penalty to the cost function  
    # Quadratic penalty to when the constraint is violated in lower bound in the terminal cost
    ocp.cost.Zl_e = z_2_e
    # Quadratic penalty to when the constraint is violated in upper bound in the terminal cost
    ocp.cost.Zu_e = z_2_e
    # Linear penalty to when the constraint is violated in lower bound in the terminal cost
    ocp.cost.zl_e = z_1_e
    # Linear penalty to when the constraint is violated in upper bound in the terminal cost    
    ocp.cost.zu_e = z_1_e

    # set initial condition
    ocp.constraints.x0 = x0
    
    # set QP solver and integration0.9
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max = 50 #  Default: 50
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.sim_method_num_stages = 4        # Runge-Kutta int. stages: (1) RK1, (2) RK2, (4) RK4
    ocp.solver_options.sim_method_num_steps = 3

    ocp.parameter_values = np.ones((num_poly_terms*(n_samples)+2))
    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_SNMPC.json", generate = solver_generate_C_code, build=solver_build)
    # acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_test.json",generate=False,build=False)

    return constraints, pred_model, acados_solver,ocp



# %% Backlog: cost function with external cost: take y_ref as parameter to vary at each shooting node
    posx_ref    = MX.sym("posx_ref")
    posy_ref    = MX.sym("posy_ref")
    yaw_ref     = MX.sym("yaw_ref")
    v_ref   = MX.sym("v_ref")
    p = vertcat(posx_ref, posy_ref, yaw_ref, v_ref)    
    ocp.model.p   = p
    ocp.parameter_values = np.array([0, 0, 0, 0]) 
    # set p in main function at each shooting node: 
    # for j in range(N):
    #     yref = np.array([current_ref_traj['pos_x'][j],current_ref_traj['pos_y'][j],current_ref_traj['ref_yaw'][j],current_ref_traj['ref_v'][j]])
    #     acados_solver.set(j, "p", yref)
    #     # for i in range(len(yref)):
    #     #     acados_solver.set_params_sparse(j, i, yref[i])
    # yref_N = np.array([current_ref_traj['pos_x'][j+1],current_ref_traj['pos_y'][j+1],current_ref_traj['ref_yaw'][j+1],current_ref_traj['ref_v'][j+1]])
    # acados_solver.set(N, "p", yref_N)
    cost_x_expr = vertcat(x[0]-posx_ref, x[1]-posy_ref, yaw - yaw_ref, vel_abs - v_ref)

    ocp.cost.cost_type      = 'EXTERNAL'    # Cost type at intermediate shooting nodes (1 to N-1)
    ocp.cost.cost_type_e    = 'EXTERNAL'    # Cost type at terminal shooting node (N)
    # cost_x_expr = = vertcat(x[:2], yaw, vel_abs,  u) 
    # ocp.model.cost_expr_ext_cost = (cost_x_expr - ocp.model.p).T @ Q @ (cost_x_expr - ocp.model.p) + u.T @ R @ u
    # ocp.model.cost_expr_ext_cost_e = (cost_x_expr - ocp.model.p).T @ Qe @ (cost_x_expr - ocp.model.p)
    ocp.model.cost_expr_ext_cost = 0.5 * cost_x_expr.T @ cs.mtimes(Q, cost_x_expr) + 0.5 * u.T @ cs.mtimes(R, u)
    ocp.model.cost_expr_ext_cost_e = 0.5*cost_x_expr.T @ cs.mtimes(Q, cost_x_expr)
   
