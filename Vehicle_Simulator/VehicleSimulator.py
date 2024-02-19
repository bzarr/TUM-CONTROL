"""
Created on Wed Jun  7 10:55:24 2023

@author: Baha Zarrouki (baha.zarrouki@tum.de)
"""
from Vehicle_Simulator.sim_model_dynamic_stm_pacejka import sim_stm
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
import casadi as cs

'''
6. Vehicle Simulator
    - initialization: 
        + loading STM simulator params 
        + loading tire model params
        + loading disturbance configuration
        + initializing disturbance simulation
        + loading the simulator / integrator
        + setting the initial state
        + 
    - step: simulation step
        + generate disturbances
        + calculate x_next_sim and x_next_sim_disturbed
    - resetting the simulator/ initial state
'''
class PassengerVehicleSimulatorCodeGen:
    def __init__(self, config_path, sim_main_params, Ts):
        ## --- Model params ---
        veh_params_file_simulator   = sim_main_params['veh_params_file_simulator']
        tire_params_file_simulator  = sim_main_params['tire_params_file_simulator']
        # load STM model
        self.sim_model, self.sim_constraints, self.sim_x_next_integrator, self.sim_x_next_disturbed_integrator = create_simulation_model_acados(config_path+veh_params_file_simulator, config_path + tire_params_file_simulator, Ts)
        self.x = self.sim_model.x
        self.params = self.sim_model.params
        
    def simulator_step(self, x0, u):
        return sim_x_next_mapping(x0, u, self.sim_x_next_integrator)
    
    def simulator_step_disturbed(self, x0, u):
        return sim_x_next_disturbed_mapping(x0, u, self.sim_x_next_disturbed_integrator)
    # def reset_simulator():

class PassengerVehicleSimulator:
    def __init__(self, config_path, sim_main_params, Ts):
        ## --- Model params ---
        veh_params_file_simulator   = sim_main_params['veh_params_file_simulator']
        tire_params_file_simulator  = sim_main_params['tire_params_file_simulator']
        # load STM model
        self.sim_model, self.sim_constraints, self.sim_x_next_mapping, self.sim_x_next_disturbed_mapping = create_simulation_model(config_path+veh_params_file_simulator, config_path + tire_params_file_simulator, Ts)
        self.x = self.sim_model.x
        self.params = self.sim_model.params
        
    def simulator_step(self, x0, u):
        return self.sim_x_next_mapping(x0, u)
    
    def simulator_step_disturbed(self, x0, u):
        return self.sim_x_next_disturbed_mapping(x0, u)
    # def reset_simulator():
    
# create a simulation model
def create_simulation_model(veh_params_file, tire_params_file, Ts):
    sim_model, sim_constraints = sim_stm(veh_params_file, tire_params_file)
        # - states: posx, posy, yaw, vlong, vlat, yawrate, delta_f
        # - inputs: a, steering_rate
        # - state derivatives disturbances: w_posx, w_posy, w_yaw, w_vlong, w_vlat, w_yawrate, w_delta_f
        # - p:  a, steering_rate, w_posx, w_posy, w_yaw, w_vlong, w_vlat, w_yawrate, w_delta_f
    xdot_mapping = cs.Function('xdot_mapping', [sim_model.x, sim_model.u], [sim_model.xdot], ['x','u'], ['ode'])
    xdot_disturbed_mapping = cs.Function('xdot_mapping', [sim_model.x, sim_model.u_disturbed], [sim_model.xdot_disturbed], ['x','u_disturbed'], ['ode_disturbed'])
    # Configure Integrator nominal state
    dae_nominal = {}
    dae_nominal['x']   = sim_model.x    # states
    dae_nominal['p']   = sim_model.u    # fixed during the integration horizon
    dae_nominal['ode'] = xdot_mapping(sim_model.x, sim_model.u) # right-hand side
    intg_options                                = {}
    intg_options['tf']                          = Ts
    intg_options['simplify']                    = True
    intg_options['number_of_finite_elements']   = 4
    intg_nominal = cs.integrator('intg', 'rk', dae_nominal, intg_options) # Runge-Kutta 4 integrator
    # Create a simulation model: (x0,U) -> (x1:xN): create a simulation model that generates next states starting from x0 and applying series of control inputs U
    sol_int_nominal = intg_nominal(x0 = sim_model.x, p = sim_model.u)      # Simplify API to (x,u) -> (x_next) : general integrator solution mapping
    next_x = sol_int_nominal['xf']
    # Simulation Step Function
    sim_x_next_mapping = cs.Function('sim_x_next_mapping', [sim_model.x,sim_model.u], [next_x], ['x','u'], ['x_next'])
    
    # Configure Integrator disturbed state
    dae_disturbed = {}
    dae_disturbed['x']   = sim_model.x    # states
    dae_disturbed['p']   = sim_model.u_disturbed    # fixed during the integration horizon
    dae_disturbed['ode'] = xdot_disturbed_mapping(sim_model.x, sim_model.u_disturbed) # right-hand side
    intg_disturbed = cs.integrator('intg', 'rk', dae_disturbed, intg_options) # Runge-Kutta 4 integrator
    sol_int_disturbed = intg_disturbed(x0 = sim_model.x, p = sim_model.u_disturbed)      # Simplify API to (x,u) -> (x_next) : general integrator solution mapping
    next_x_disturbed = sol_int_disturbed['xf']
    # Simulation Step Function
    sim_x_next_disturbed_mapping = cs.Function('sim_x_next_disturbed_mapping', [sim_model.x,sim_model.u_disturbed], [next_x_disturbed], ['x','u_disturbed'], ['next_x_disturbed'])
    return sim_model, sim_constraints, sim_x_next_mapping, sim_x_next_disturbed_mapping

# create a simulation model
def create_simulation_model_acados(veh_params_file, tire_params_file, Ts):
    sim_model, sim_constraints = sim_stm(veh_params_file, tire_params_file)
        # - states: posx, posy, yaw, vlong, vlat, yawrate, delta_f
        # - inputs: a, steering_rate
        # - state derivatives disturbances: w_posx, w_posy, w_yaw, w_vlong, w_vlat, w_yawrate, w_delta_f
        # - p:  a, steering_rate, w_posx, w_posy, w_yaw, w_vlong, w_vlat, w_yawrate, w_delta_f
    # Configure Integrator nominal state
    model = AcadosModel()
    model.f_expl_expr   = sim_model.xdot
    model.x             = sim_model.x
    model.xdot  = sim_model.xdot
    model.u     = sim_model.u
    # model.p     = sim_model.p
    model.name  = 'VehSim_nominal'
    sim         = AcadosSim()
    sim.model   = model
    sim_time_step = Ts   # [s]
    sim.solver_options.T = sim_time_step
    # set options
    sim.solver_options.integrator_type = 'ERK'
    # sim.solver_options.num_stages = 1 # default:1
    # sim.solver_options.num_steps = 1 # default:1
    # sim.solver_options.newton_iter = 3 # default:3
    # create sim_solver object
    sim_x_next_integrator = AcadosSimSolver(sim,generate=True,build=True)

    # Configure Integrator disturbed state
    dist_model = AcadosModel()
    dist_model.f_expl_expr   = sim_model.xdot_disturbed
    dist_model.x             = sim_model.x
    dist_model.xdot  = sim_model.xdot_disturbed
    dist_model.u     = sim_model.u_disturbed
    # dist_model.p     = sim_model.p
    dist_model.name  = 'VehSim_disturbed'
    sim_dist         = AcadosSim()
    sim_dist.model   = dist_model
    sim_time_step = Ts   # [s]
    sim_dist.solver_options.T = sim_time_step
    # set options
    sim_dist.solver_options.integrator_type = 'ERK'
    # sim_dist.solver_options.num_stages = 1 # default:1
    # sim_dist.solver_options.num_steps = 1 # default:1
    # sim_dist.solver_options.newton_iter = 3 # default:3
    # create sim_solver object
    sim_x_next_disturbed_integrator = AcadosSimSolver(sim_dist,generate=True,build=True)
    return sim_model, sim_constraints, sim_x_next_integrator, sim_x_next_disturbed_integrator

def sim_x_next_mapping(x,u,sim_x_next_integrator):
    sim_x_next_integrator.set("x", x)
    sim_x_next_integrator.set("u", u)
    status = sim_x_next_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))
    return sim_x_next_integrator.get("x")

def sim_x_next_disturbed_mapping(x,u,sim_x_next_disturbed_integrator):
    sim_x_next_disturbed_integrator.set("x", x)
    sim_x_next_disturbed_integrator.set("u", u)
    status = sim_x_next_disturbed_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))
    return sim_x_next_disturbed_integrator.get("x")
