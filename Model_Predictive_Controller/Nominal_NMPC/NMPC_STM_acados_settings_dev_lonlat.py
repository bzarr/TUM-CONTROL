# Created on Tue Dec 06 11:20 2022

# Author: Baha Zarrouki (baha.zarrouki@tum.de)
# based on acados python documentation: https://docs.acados.org/python_interface/index.html

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from Prediction_Models.pred_model_dynamic_stm_pacejka import pred_stm
import scipy.linalg
import numpy as np
from casadi import vertcat, sqrt, mod, pi, if_else, MX, interp1d, Function, cos, sin, mtimes

def LonLatDeviations(ego_yaw, ego_x, ego_y, ref_x,ref_y):
    '''
    This method is based on rotating the deviation vectors by the negative 
    of the yaw angle of the vehicle, which aligns the deviation vectors with 
    the longitudinal and lateral axes of the vehicle.
    '''
    rotcos      = cos(-ego_yaw)
    rotsin      = sin(-ego_yaw)
    dev_long    = rotcos * (ref_x - ego_x) - rotsin * (ref_y - ego_y)
    dev_lat     = rotsin * (ref_x - ego_x) + rotcos * (ref_y - ego_y)
    return dev_long, dev_lat


def acados_settings(Tf, N, x0, Q, R, Qe, L1_pen, L2_pen, ax_max_interpolant, ay_max_interpolant, combined_acc_limits, 
                    veh_params_file = "veh_parameters_bmw320i.yaml", tire_params_file= "pacejka_params_file.yaml", 
                    xy_deviations_cost = False , solver_generate_C_code = True, solver_build = True):

    ocp = AcadosOcp()

    # import prediction model
    pred_model, constraints = pred_stm(veh_params_file, tire_params_file)

    # define acados ODE
    model               = AcadosModel()
    model.f_expl_expr   = pred_model.xdot
    model.x             = pred_model.x
    # model.xdot          = pred_model.xdot
    model.u             = pred_model.u
    # model.z             = pred_model.z
    # model.p             = pred_model.p
    model.name          = pred_model.name
    ocp.model           = model

    ocp.dims.N = N
    unscale =1 # N / Tf

    x = ocp.model.x
    u = ocp.model.u
    # adjust yaw to [0..2*pi]
    yaw = mod(x[2], 2*pi)
    vel_abs = MX.sym("vel_abs")
    yaw = if_else( yaw < 0,yaw + 2*pi , yaw) # adjust for negative angles
    # compute absolute velocity for cost function (as reference velocity, model states give v_lon and v_lat)
    vel_abs = sqrt(x[3]**2 + x[4]**2)

    # cost function formulation
    # ocp.cost.cost_type      = "NONLINEAR_LS" # Cost type at intermediate shooting nodes (1 to N-1)
    # ocp.cost.cost_type_e    = "NONLINEAR_LS" # Cost type at terminal shooting node (N)
    
    # ocp.model.cost_y_expr = vertcat(x[:2], yaw, vel_abs,  u) 
    # ocp.model.cost_y_expr_e = vertcat(x[:2], yaw, vel_abs)

    # # cost function weights
    # ocp.cost.W      = unscale * scipy.linalg.block_diag(Q, R)
    # ocp.cost.W_e    = Qe / unscale

    # # intial references
    # ocp.cost.yref   = np.array([0, 0, 0, 0, 0, 0])
    # ocp.cost.yref_e = np.array([0, 0, 0, 0])
    posx_ref    = MX.sym("posx_ref")
    posy_ref    = MX.sym("posy_ref")
    yaw_ref     = MX.sym("yaw_ref")
    v_ref       = MX.sym("v_ref")
    p = vertcat(posx_ref, posy_ref, yaw_ref, v_ref)    
    ocp.model.p   = p
    ocp.parameter_values = np.array([0, 0, 0, 0]) 

    if xy_deviations_cost:
        cost_x_expr = vertcat(x[0]-posx_ref, x[1]-posy_ref, yaw - yaw_ref, vel_abs - v_ref, u[0], u[1])
        cost_x_expr_e = vertcat(x[0]-posx_ref, x[1]-posy_ref, yaw - yaw_ref, vel_abs - v_ref)
    else:
        dev_long, dev_lat= LonLatDeviations(yaw, x[0], x[1], posx_ref,posy_ref)
        cost_x_expr = vertcat(dev_long, dev_lat, yaw - yaw_ref, vel_abs - v_ref, u[0], u[1])
        cost_x_expr_e = vertcat(dev_long, dev_lat, yaw - yaw_ref, vel_abs - v_ref)

    W       = 0.01 * unscale * scipy.linalg.block_diag(Q, R)
    W_e     = 0.01 * Qe / unscale

    ocp.cost.cost_type      = 'EXTERNAL'    # Cost type at intermediate shooting nodes (1 to N-1)
    ocp.cost.cost_type_e    = 'EXTERNAL'    # Cost type at terminal shooting node (N)
    # cost_x_expr = = vertcat(x[:2], yaw, vel_abs,  u) 
    # ocp.model.cost_expr_ext_cost = (cost_x_expr - ocp.model.p).T @ Q @ (cost_x_expr - ocp.model.p) + u.T @ R @ u
    # ocp.model.cost_expr_ext_cost_e = (cost_x_expr - ocp.model.p).T @ Qe @ (cost_x_expr - ocp.model.p)
    ocp.model.cost_expr_ext_cost = 0.5 * cost_x_expr.T @ W @ cost_x_expr 
    ocp.model.cost_expr_ext_cost_e = 0.5 * cost_x_expr_e.T @ W_e @ cost_x_expr_e

    # --- combined lateral and longitudinal acceleration constraints varying accroding to current velocity
    # Source: Wischnewski, Alexander, et al. "A tube-MPC approach to autonomous multi-vehicle racing on high-speed ovals." IEEE Transactions on Intelligent Vehicles (2022).
    # compute/extract acceleration constraints
    a_lat       = MX.sym("a_lat")
    a_lon       = MX.sym("a_lon")
    acc_limits_ineq1 = MX.sym("acc_limits_ineq1")
    acc_limits_ineq2 = MX.sym("acc_limits_ineq2")
    # extract current acceleration limits based on velocity
    a_lat       = x[3] * x[5] 
    a_lon       = x[7]     
    ay_max = ay_max_interpolant(vel_abs)
    ax_max = ax_max_interpolant(vel_abs)
    ax_max = if_else( a_lon < 0,-pred_model.acc_min , ax_max)  # ax limits are asymetric: >0 -> acceleration, <0: braking --> change the scaling factor for braking
    constraints.a_lat = Function("a_lat", [x], [a_lat]) # define function for evaluation
    # ay_max = constraints.lat_acc_max
    # ax_max = pred_model.acc_max 

    if combined_acc_limits == 0:
        # separated limits
        a_lon_limits_ineq = a_lon/ax_max
        a_lat_limits_ineq = a_lat/ay_max
        # constraints: nonlinear inequalities
        model.con_h_expr = vertcat(a_lon_limits_ineq, a_lat_limits_ineq)
        model.con_h_expr_e = vertcat(a_lon_limits_ineq, a_lat_limits_ineq)
        # nonlinear constraints
        # stage bounds for nonlinear inequalities
        ocp.constraints.lh      = np.array([-1, -1])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
        ocp.constraints.uh      = np.array([1, 1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
        # terminal bounds for nonlinear inequalities
        ocp.constraints.lh_e    = np.array([-1, -1])  # lower bound for nonlinear inequalities at terminal shooting node N
        ocp.constraints.uh_e    = np.array([1, 1])  # upper bound for nonlinear inequalities at terminal shooting node N
    elif combined_acc_limits == 1:
        # Diamond shaped combined lat lon acceleration limits
        acc_limits_ineq1 = a_lon/ax_max + a_lat/ay_max
        acc_limits_ineq2 = a_lon/ax_max - a_lat/ay_max
        # constraints: nonlinear inequalities
        model.con_h_expr = vertcat(acc_limits_ineq1, acc_limits_ineq2)
        model.con_h_expr_e = vertcat(acc_limits_ineq1, acc_limits_ineq2)
        # nonlinear constraints
        # stage bounds for nonlinear inequalities
        ocp.constraints.lh      = np.array([-1, -1])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
        ocp.constraints.uh      = np.array([1, 1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
        # terminal bounds for nonlinear inequalities
        ocp.constraints.lh_e    = np.array([-1, -1])  # lower bound for nonlinear inequalities at terminal shooting node N
        ocp.constraints.uh_e    = np.array([1, 1])  # upper bound for nonlinear inequalities at terminal shooting node N
    else:
        # Circle shaped combined lat lon acceleration limits
        acc_limits_ineq = (a_lon/ax_max)**2 + (a_lat/ay_max)**2
        # constraints: nonlinear inequalities
        model.con_h_expr = vertcat(acc_limits_ineq)
        model.con_h_expr_e = vertcat(acc_limits_ineq)
        # nonlinear constraints
        # stage bounds for nonlinear inequalities
        ocp.constraints.lh      = np.array([0])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
        ocp.constraints.uh      = np.array([1])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
        # terminal bounds for nonlinear inequalities
        ocp.constraints.lh_e    = np.array([0])  # lower bound for nonlinear inequalities at terminal shooting node N
        ocp.constraints.uh_e    = np.array([1])  # upper bound for nonlinear inequalities at terminal shooting node N

    # constraints.a_lon = Function("a_lon", [x], [a_lon])

    # contraints (source reference: https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf)
    # linear constraints
    # stage bounds on x
    ocp.constraints.lbx     = np.array([pred_model.delta_f_min]) # lower bounds on x at intermediate shooting nodes (1 to N-1)
    ocp.constraints.ubx     = np.array([pred_model.delta_f_max]) # upper bounds on x at intermediate shooting nodes (1 to N-1)
    ocp.constraints.idxbx   = np.array([6])   # indices of bounds on x (defines Jbx) at intermediate shooting nodes (1 to N-1)
    # terminal bounds on x
    ocp.constraints.lbx_e   = np.array([pred_model.delta_f_min])   # lower bounds on x at terminal shooting node N
    ocp.constraints.ubx_e   = np.array([pred_model.delta_f_max])   # upper bounds on x at terminal shooting node N
    ocp.constraints.idxbx_e = np.array([6])     # Indices for bounds on x at terminal shooting node N (defines Jebx)
    # stage bounds on u 
    ocp.constraints.lbu     = np.array([pred_model.delta_f_dot_min])
    ocp.constraints.ubu     = np.array([pred_model.delta_f_dot_max])
    ocp.constraints.idxbu   = np.array([1])

    # ocp.constraints.lh      = np.array([constraints.lat_acc_min])  # lower bound for nonlinear inequalities at shooting nodes (0 to N-1)
    # ocp.constraints.uh      = np.array([constraints.lat_acc_max])  # upper bound for nonlinear inequalities at shooting nodes (0 to N-1)
    # # terminal bounds for nonlinear inequalities
    # ocp.constraints.lh_e    = np.array([constraints.lat_acc_min])  # lower bound for nonlinear inequalities at terminal shooting node N
    # ocp.constraints.uh_e    = np.array([constraints.lat_acc_max])  # upper bound for nonlinear inequalities at terminal shooting node N
    
    # soft constraints --> help the solver to find a solution (QP solver sometimes does not find a solution with hard constraints)
    # source: https://discourse.acados.org/t/infeasibility-issues-when-using-hard-nonlinear-constraints/1021 
    # Slack variables are useful to relax the constraints and help the solver to find a solution
    # Define parameters for the slack variables in the cost function
    # L1 is the linear term and L2 is the quadratic term
    # Tuning: if it is too small, the constraint will not be satisfied, if it is too big, it will become a hard constraint
    # and the solver will not find a solution --> tune it to find the right balance.
    
    # L1_pen = 5000     # TODO: add params to MPC config yaml file
    # L2_pen = 70000

    # Slack variables for the stage cost constraints
    # The main idea is to add a cost term in the cost function that penalizes the violation of the constraints
    # The slack variables are added to the cost function and the constraints are relaxed (check eq 1 from link below)
    # https://raw.githubusercontent.com/acados/acados/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Get the number of non linear constraints in the stage cost
    nh = model.con_h_expr.shape[0]
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

    # Slack variables for the terminal cost constraints
    # Get the number of non linear constraints in the terminal cost
    nh_e = model.con_h_expr_e.shape[0]    
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
    # ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 5000
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.levenberg_marquardt = 0.1
    ocp.solver_options.sim_method_num_steps = 15
    ocp.solver_options.qp_solver_iter_max = 50
    # ocp.solver_options.nlp_solver_max_iter = 150
    # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = 'MIRROR'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4        # Runge-Kutta int. stages: (1) RK1, (2) RK2, (4) RK4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.exact_hess_constr = 0
    ocp.solver_options.exact_hess_dyn = 0
    # ocp.solver_options.print_level = 1
    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_NMPC.json", generate = solver_generate_C_code, build=solver_build)
    # ocp.cost.cost_type
    return constraints, pred_model, acados_solver, ocp.cost.cost_type



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
   
