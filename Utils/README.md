## Simulation Components
1. Parameter Handler
    - Simulation params
2. Logging Module
    - initialization: 
        + initialize logging data structs
        + log initial state
        + initialize time
    - step:
        + log simSolverDebug
        + log simREF
        + log CiLX, DisturbedX, MPC_SimX
        + log sim_disturbance_state_estimation, sim_disturbance_derivatives
    - update: save current logs after certain amount of time
    - end_logging:
        + post process logs
3. Trajectory Planner Emulator
    - initialization
4. Trajectory Processor
5. Initialization 
    - Simulator Vehicle initial state 
    - MPC initial state
6. Vehicle Simulator
    - initialization: 
        + loading STM simulator params 
        + loading tire model Params
        + loading disturbance configuration
        + initializing disturbance simulation
        + loading the simulator / integrator
        + setting the initial state
        + 
    - step: simulation step
        + generate disturbances
        + calculate x_next_sim and x_next_sim_disturbed
    - resetting the simulator/ initial state
7. MPC
    - initialisation: 
        + loading MPC params
        + loading Prediciton model params 
        + loading tire model Params
        + loading constraint, model, acados_solver, costfunction_type
        + setting the initial state / constraints
    - step:
        + set current x_next_sim_disturbed (initial state / constraints)
        + set current reference trajectory
        + solve
        + get MPC solution: x0,u0
        + extract current MPC predictions
        + get SolverDebug Stats
8. Visualization Module
    - initialization:
        + loading Track
        + creating figures
        + inializing visualisation setup: initLiveVisuMode1/initLiveVisuMode2
        + initializing GIF Frames
    - step (update)
9. Postprocessing Module
    - performance analysis 
    - generate plots and save them