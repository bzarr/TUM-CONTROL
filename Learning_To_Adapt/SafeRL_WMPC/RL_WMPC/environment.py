import gymnasium as gym
import torch
import numpy as np
import random
import yaml

from gymnasium import spaces
from typing import List
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecMonitor, VecFrameStack, DummyVecEnv
)

from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import (
    Nonlinear_Model_Predictive_Controller
)
from Utils.Logging_Plotting import Logger
from Utils.MPC_sim_utils import PlannerEmulator
from Vehicle_Simulator.VehicleSimulator import PassengerVehicleSimulator
from Utils.SimulationMode_main_class import MPC_Sim
from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.observation import (
    ObservationGenerator
)
from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.reward import RewardGenerator


CONFIG_PATH = 'Config/'
TRAJECTORY_PATH = 'Trajectories/'
LOGS_PATH = 'Logs/'
SIM_MAIN_PARAMS_FILE = "EDGAR/sim_main_params.yaml"
MPC_PARAMS_FILE = "EDGAR/MPC_params.yaml"


class RLEnvironment(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            config: dict,
            trajectory: str,
            random_restarts: bool,
            full_lap: bool,
            evaluation_env: bool
        ) -> None:

        super().__init__()

        # environment settings
        self.n_mpc_steps = config['n_mpc_steps']
        self.max_lat_dev = config['max_lat_dev']
        
        self.episode_steps = 0
        self.crash_counter = 0
        self.start_idx = None
        self.full_lap = full_lap
        self.random_restarts = random_restarts
        self.trajectory = trajectory

        with open(CONFIG_PATH + SIM_MAIN_PARAMS_FILE, 'r') as file:
            sim_main_params = yaml.load(file, Loader=yaml.FullLoader)
        # pre-load simulation module (just to access its attributes...)
        self.simulation = MPC_Sim(
            sim_main_params,
            LOGS_PATH,
        )
        if not evaluation_env: 
            self.episode_length = config['episode_length']
        else:
            self.episode_length = int(self.simulation.Nsim/self.n_mpc_steps)

        # load controller
        self.MPC = Nonlinear_Model_Predictive_Controller(
            CONFIG_PATH, MPC_PARAMS_FILE, self.simulation.sim_main_params,
            self.simulation.X0_MPC
        )

        # define reward settings
        sigmas = config['rew_sigmas']
        normalization_lims = np.concatenate([
            config['rew_lims_lat_dev'],
            config['rew_lims_vel_dev'],
        ]).transpose()
        self.reward_generator = RewardGenerator(sigmas, normalization_lims)

        # load actions and define action space
        actions_file = config['actions_file']
        with open(actions_file, 'r') as file:
            lines = file.readlines()
        parameterizations = torch.empty((len(lines), 7))
        for i, line in enumerate(lines):
            strs = line.strip().split(',')
            params = torch.tensor([float(p) for p in strs])
            parameterizations[i] = params
        self.parameter_sets = parameterizations
        self.n_actions = self.parameter_sets.size()[0]
        self.action_space = spaces.Discrete(self.n_actions)

        # load observation generator and define observation space
        self.observation_generator = ObservationGenerator(
            anticipation_horizon=config['obs_anticipation_horizon'],
            n_anticipation_points=config['obs_n_anticipation_points']
        )
        self.observation_space = spaces.Box(
            low=0., high=1.,
            shape=(self.observation_generator.n_observations,),
            dtype=np.dtype(float)
        )

        # reset environment
        self.reset()

    def step(self, action: int):
        
        # update episode length
        self.episode_steps += 1
        
        # select parameterization according to action
        params = self.parameter_sets[action]
        self.MPC.update_cost_function_weights(params)

        # run an environment step of fixed length
        for i in range(self.n_mpc_steps):
            # plan local trajectory
            current_ref_idx, current_ref_traj = PlannerEmulator(
                self.simulation.ref_traj_set, self.simulation.current_pose,
                self.simulation.N+1, self.simulation.Tp, loop_circuit=True
            )
            
            # solve optimal control problem
            u0, pred_X, MPC_stats = self.MPC.solve(current_ref_traj)
            
            # step vehicle model
            x = self.logger.CiLX[self.logger.current_step, :]
            (
                x_next,
                x_next_MPC,
                x_next_sim,
                x_next_sim_disturbed
            ) = self.simulation.sim_step(self.logger.current_step, x, u0, pred_X)

            # update MPC initial state
            x_next = self.simulation.StateEstimation(x_next)
            self.MPC.set_initial_state(x_next)

            # log system states
            self.logger.logging_step(
                self.logger.current_step, u0, MPC_stats, current_ref_traj,
                x_next_sim, x_next_sim_disturbed, x_next_MPC
            )
            self.logger.current_step += 1

            # truncate if crash condition is fulfilled
            if self._check_crash_condition() == True:
                truncated = True
                self.crash_counter += 1
                print(f"Simulation crashed. Total: {self.crash_counter}")
            else:
                truncated = False
            
            # for full laps, terminate the episode if the endpoint has been reached
            if self.full_lap:
                terminated = current_ref_idx == self.simulation.trajectory_length - 2
            # else terminate the episode after a fixed number of steps
            else:
                terminated = self.episode_steps == self.episode_length

            # break loop if the episode ended
            if terminated or truncated:
                break
        
        # obtain reward
        # if i < 2:
        #     reward = 0
        # else:
        #     reward = self.reward_generator.get_reward(self.logger, step_length=i+1)
        reward = self.reward_generator.get_reward(self.logger, step_length=i+1)
        # obtain normalized observation
        v, lat_dev, vel_dev = self.logger.get_observation_states()
        observation = self.observation_generator.get_observation(
            v, lat_dev, vel_dev, current_ref_traj, self.simulation.Ts
        )
        # NOTE: observations are clipped and cannot exceed the range (0, 1)
        #if np.any([e.item() > 1 or e.item() < 0 for e in observation]):
            #print(f"Ill-defined normalization bounds!")
            #print(', '.join([f'{o:5.3f}' for o in observation]))
        
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # obtain start position
        if self.random_restarts:
            start_idx = random.choice([0, 100, 200, 400, 500, 700, 800])
        else:
            start_idx = 0
        
        #print(f"Reset to {start_idx} on track {self.trajectory}.")
        ## Overwrite standard config file
        with open(CONFIG_PATH + SIM_MAIN_PARAMS_FILE, 'r') as file:
            sim_main_params = yaml.load(file, Loader=yaml.FullLoader)
        sim_main_params['track_file'] = f'track_{self.trajectory}.json'
        sim_main_params['ref_traj_file'] = f'reftraj_{self.trajectory}_edgar.json'
        sim_main_params['idx_ref_start'] = start_idx
        
        # reload simulation module and vehicle
        self.simulation = MPC_Sim(
            sim_main_params,
            LOGS_PATH,
        )
        self.vehicle = PassengerVehicleSimulator(
            CONFIG_PATH, self.simulation.sim_main_params, self.simulation.Ts
        )
        
        # reset MPC
        X0_sim = self.simulation.set_Vehicle(self.vehicle)
        self.MPC.reset(self.simulation.X0_MPC)
        nx, nu = self.MPC.nx, self.MPC.model.u.size()[0]

        # reload logger object
        self.logger = Logger(
            self.simulation, X0_sim, 
            self.vehicle.sim_constraints.alat,
            self.MPC
        )
        self.episode_steps = 0

        # TODO: obtain real initial observation
        #observation = self.observation_generator.get_observation(self.logger)
        # use empty observation for initial step, as no information is available
        observation = np.zeros(self.observation_generator.n_observations)

        info = {}

        return observation, info
    
    def _check_crash_condition(self) -> bool:
        return self.logger.lat_devs[self.logger.current_step-1] > self.max_lat_dev
            

def make_env(
        config: dict,
        trajectory: str,
        random_restarts: bool,
        full_lap: bool = False,
        evaluation_env: bool = False
    ):

    def _init():
        env = RLEnvironment(
            config=config,
            trajectory=trajectory,
            random_restarts=random_restarts,
            full_lap=full_lap,
            evaluation_env=evaluation_env
        )
        return env

    return _init

""" Set up a vectorized RL training environment. """
def setup_environments(
        n_envs: int,
        config: dict,
        trajectories: List[str],
        monitor_path: str,
        random_restarts: bool,
        full_lap: bool = False,
        evaluation_env: bool = False
    ):

    # TODO: externalize solver generation to stabilize parallelization
    #solvers = [setup_solver(sim_config) for _ in range(n_envs)]

    n_trajs = len(trajectories)
    # generate list of environments
    env_list = [
        make_env(
            config,
            #solvers[i],
            trajectories[i % n_trajs],
            random_restarts=random_restarts,
            full_lap=full_lap,
            evaluation_env=evaluation_env
        ) for i in range(n_envs)
    ]
    # assemble to a single vectorized environment
    env = SubprocVecEnv(env_list, 'fork') if n_envs > 1 else DummyVecEnv(env_list)
    # use frame stacking if specified
    if config['n_obs_stack']:
        env = VecFrameStack(env, config['n_obs_stack'], channels_order="first")
    # wrap the vectorized environment in a monitor wrapper
    env = VecMonitor(env, monitor_path)
    
    return env