# Created on Tue Jan 10 11:07:38 2023

# Author: Baha Zarrouki (baha.zarrouki@tum.de)
#         Joao Nunes    (joao.nunes@tum.de)
from casadi import *

from Prediction_Models.pred_model_dynamic_stm_pacejka import pred_stm

def pred_stm_disturbed(veh_params_file, tire_params_file = "pacejka_params_file.yaml"):   
    
    model, constraint = pred_stm(veh_params_file, tire_params_file)
   
    # Disturbances
    # w_posx      = MX.sym('w_posx')
    # w_posy      = MX.sym('w_posy')
    w_yaw       = MX.sym('w_yaw')
    w_vlong     = MX.sym('w_vlong')
    w_vlat      = MX.sym('w_vlat')
    w_yawrate   = MX.sym('w_yawrate')
    # w_delta_f   = MX.sym('w_delta_f')
    # w_a         = MX.sym('w_a')

    # W is the disturbance vector, from equation 4 in Zanelli's paper

    w = vertcat(w_yaw, w_vlong, w_vlat, w_yawrate)
    # w = vertcat(w_vlong, w_vlat, w_yawrate)
    # w = vertcat(w_posx, w_posy, w_yaw, w_vlong, w_vlat, w_yawrate, w_delta_f, w_a)
    
    ## CasADi model struct
    model.old_f_expl_expr = model.f_expl_expr[:8]
    model.f_expl_expr   = model.f_expl_expr[:8] + vertcat(0,
                                                      0, 
                                                    #   0,
                                                      w_yaw, 
                                                      w_vlong, 
                                                      w_vlat,
                                                      w_yawrate,
                                                      0,
                                                      0,)
    
    model.f_impl_expr   = model.xdot - model.f_expl_expr
    model.p             = w
    model.name          = 'disturbed_model'
    
    return model, constraint

def export_aug_model(W_dist, veh_params_file, tire_params_file = "pacejka_params_file.yaml"):
    """
    This function creates a casadi model of the vehicle dynamics
    considering the disturbances in the states: ellipsoidal uncertainty set
    augmented model = [nominal model, uncertainty description]
    augmented state = [nominal state, uncertainty description]
    based on: Zanelli, Andrea, et al. 
    "Zero-order robust nonlinear model predictive control with ellipsoidal uncertainty sets." IFAC-PapersOnLine 54.6 (2021): 50-57.
    Code: https://github.com/FreyJo/zoro-NMPC-2021 
    """

    # The disturbance is added to the 4 middle states (yaw, vlong, vlat, yawrate) in the controller prediction model

    # Load the nominal model
    dist_model, constraint = pred_stm_disturbed(veh_params_file, tire_params_file)
    stop_flag = MX.sym('stop_flag')
    p = vertcat(stop_flag)
    aug_model = types.SimpleNamespace()
    
    nx_orig = dist_model.x.size()[0]    

    # This is the uncertainty matrix (defines the ellipsoid where the state is)
    # defined between equations 7 and 8 from Zanelli's paper
    Sigma_vec = MX.sym("Sigma_vec", int((nx_orig+1)*nx_orig/2))         # cross-state uncertainties are taken into account --> reduced ((+1)/2) because symmetric
    Sigma_vec_dot = MX.sym("Sigma_vec_dot", int((nx_orig+1)*nx_orig/2)) # derivative of uncertainty matrix

    x_aug = vertcat(dist_model.x, Sigma_vec)

    # Define the dynamics of the augmented system: compute jacobians w.r.t. nominal state & disturbance to compute the uncertainty dynamics in Eq.(8)
    # B can be analytically derived, as just an identity matrix in the lines where there are disturbances
    # C_fun = Function('C_fun', [dist_model.x, dist_model.u, dist_model.p], [jacobian(dist_model.f_expl_expr, dist_model.x)])
    # B_fun = Function('B_fun', [dist_model.x, dist_model.u, dist_model.p], [jacobian(dist_model.f_expl_expr, dist_model.p)])
    
    # Evaluate in 0: no disturbance
    # C = C_fun(dist_model.x, dist_model.u, [0,0,0,0])
    # B = B_fun(dist_model.x, dist_model.u, [0,0,0,0])

    D = MX.zeros(nx_orig, nx_orig)

    # This C is equivalent to calculating the jacobians. It is hard coded here for performance enhancing
    C = if_else(stop_flag == 1,
                MX.zeros(nx_orig,nx_orig),
                jacobian(dist_model.old_f_expl_expr, dist_model.x))
    # C = jacobian(dist_model.old_f_expl_expr, dist_model.x)
    # This B is equivalent to calculating the jacobians. It is hard coded here for performance enhancing
    B = np.array([[0, 0, 0, 0], 
                  [0, 0, 0, 0], 
                  [1, 0, 0, 0], 
                  [0, 1, 0, 0],        
                  [0, 0, 1, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, 0, 0], 
                  [0, 0, 0, 0]])
    # B = np.array([  [0, 0, 0], 
    #                 [0, 0, 0], 
    #                 [0, 0, 0], 
    #                 [1, 0, 0],        
    #                 [0, 1, 0], 
    #                 [0, 0, 1], 
    #                 [0, 0, 0], 
    #                 [0, 0, 0]])
    # We limit the jacobians according to the max value achieved in the normal simulation

    D = C
    D[4, 4] = 0
    D[5, 3] = 0
    D[5, 4] = 0
    D[5, 5] = 0

    # This code below was used to clip the jacobians according to the maximum absolute value achieved 
    # in the normal simulation. However, it is not used anymore because it is not necessary to clip the jacobians
    # tol_matrix =   np.array([[0, 0, 41,   1,   1,  0,   0,   0],
    #                          [0, 0, 61,   1,   1,  0,   0,   0],
    #                          [0, 0,  0,   0,   0,  0,   0,   0],
    #                          [0, 0,  0, 0.1, 1.2,  6,  12,   0],
    #                          [0, 0,  0,   1,  12, 72, 110, 0.4],
    #                          [0, 0,  0, 0.6,   5, 20, 142, 0.4],
    #                          [0, 0,  0,   0,   0,  0,   0,   0],
    #                          [0, 0,  0,   0,   0,  0,   0,   0]])

    # tol_matrix =   np.array([[0, 0, 41,   1,   1,  0,   0,   0],
    #                          [0, 0, 61,   1,   1,  0,   0,   0],
    #                          [0, 0,  0,   0,   0,  0,   0,   0],
    #                          [0, 0,  0, 0.1, 1.1,  6,  12,   0],
    #                          [0, 0,  0,   1,   0, 72, 110, 0.4],
    #                          [0, 0,  0,   0,   0,  0, 142, 0.4],
    #                          [0, 0,  0,   0,   0,  0,   0,   0],
    #                          [0, 0,  0,   0,   0,  0,   0,   0]])
      
    # for i1 in range(nx_orig):
    #     for i2 in range(nx_orig):
    #         D[i1, i2] = fmax(fmin(C[i1, i2], -tol_matrix[i1, i2]), tol_matrix[i1, i2])

    # In order to propagate the uncertainty matrix dynamics in Casadi, we write the matrix as a vector form (stacking its columns). 
    # However, to write the dynamical equation of the uncertainty matrix propagation, we need it in the matrix form, not vector. 
    # Therefore we use the functions "vec2sym_mat" -> vector to symmetric matrix and "sym_mat2vec" -> symmetric matrix to vector
    # The following is making problems with UPH
    # Sigma_mat = if_else(stop_flag == 1, MX.zeros(nx_orig,nx_orig),vec2sym_mat(Sigma_vec, nx_orig))
    Sigma_mat = vec2sym_mat(Sigma_vec, nx_orig)
    # MX.zeros(nx_orig,nx_orig)

    # Eq.(8): Phi = C * Sigma * C' + B * W * B'
    # We use continuous time Lyapunov dynamics (https://en.wikipedia.org/wiki/Lyapunov_equation)
    # --> Phi = C*Sigma + Sigma*C' + B*W*B'
    Sigmadot_mat = if_else(stop_flag == 1,
                           MX.zeros(nx_orig,nx_orig),
                           D @ Sigma_mat + Sigma_mat @ D.T + B @ W_dist @ B.T)
    # Sigmadot_mat =  D @ Sigma_mat + Sigma_mat @ D.T + B @ W_dist @ B.T
    # Sigmadot_mat = B @ W_dist @ B.T 
    # Sigmadot_mat = C @ Sigma_mat @ C.T + B @ W_dist @ B.T 
    
    Sigmadot_vec = if_else(stop_flag == 1, 
                           MX.zeros(int((nx_orig+1)*nx_orig/2)), 
                           sym_mat2vec(Sigmadot_mat))
    # Sigmadot_vec = sym_mat2vec(Sigmadot_mat)

    # f_expl = Function('f_expl', [dist_model.x, dist_model.u, dist_model.p], [dist_model.f_expl_expr])
    # nominal_f_expl_expr = f_expl(dist_model.x, dist_model.u, [0,0,0,0])
    nominal_f_expl_expr = dist_model.old_f_expl_expr

    # augmented state derivative = [nominal state derivative, uncertainty description derivative]
    xdot = vertcat(dist_model.xdot, Sigma_vec_dot) # ??????????? Sigma_vec_dot or Sigmadot_vec
    # MX.zeros(int((nx_orig+1)*nx_orig/2))
    f_expl_expr = vertcat(nominal_f_expl_expr, Sigmadot_vec)
    

    aug_model.f_expl_expr = f_expl_expr
    aug_model.f_impl_expr = f_expl_expr - xdot

    aug_model.xdot = xdot
    aug_model.u = dist_model.u
    aug_model.x = x_aug
    aug_model.p = p

    aug_model.name = "augmented_model_with_cov_matrix"

    return aug_model, dist_model, constraint, Sigma_mat

def vec2sym_mat(vec, nx):
    # nx = (vec.shape[0])

    # if isinstance(vec, np.ndarray):
    #     mat = np.zeros((nx,nx))
    # else:
    #     mat = MX.zeros(nx,nx)
    mat = MX.zeros(nx,nx)

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        aux = vec[start_mat:end_mat]
        mat[i,i:] = aux.T
        mat[i:,i] = aux
        start_mat += (nx-i)

    return mat

def sym_mat2vec(mat):
    nx = mat.shape[0]

    if isinstance(mat, np.ndarray):
        vec = np.zeros((int((nx+1)*nx/2),))
    else:
        vec = MX.zeros(int((nx+1)*nx/2))

    start_mat = 0
    for i in range(nx):
        end_mat = start_mat + (nx - i)
        vec[start_mat:end_mat] = mat[i:,i]
        start_mat += (nx-i)

    return vec

# Equation 8 from Zanelli's paper
def P_propagation(P, A, B, W):
    #  P_i+1 = A P A^T +  B*W*B^T
    return A @ P @ A.T + B @ W @ B.T

