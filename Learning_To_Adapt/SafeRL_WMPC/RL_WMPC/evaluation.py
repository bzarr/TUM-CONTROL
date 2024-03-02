import numpy as np
import os
from typing import List, Tuple
from collections import namedtuple

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator
)
from stable_baselines3 import PPO

from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.environment import (
    setup_environments
)
from Learning_To_Adapt.SafeRL_WMPC.helpers import get_action_probabilities
from Utils.Logging_Plotting import Logger
from Utils.SimulationMode_main_class import MPC_Sim
from Model_Predictive_Controller.Nominal_NMPC.NMPC_class import (
    Nonlinear_Model_Predictive_Controller
)


class TrainingData:
    """ Extracts and collects training data from an existing Tensorboard di-
    rectory to enable simpler plotting and analysis.
    
    Args:
        run_path (str): absolute or relative path to the run folder
        identifier (str): unique identifer that is assigned to the data object
    """

    def __init__(self, run_path: str, identifier: str):
        # setup path to the tensorboard data
        self.data_path = os.path.join(run_path, 'tensorboard_1')
        # set identifier
        self.identifier = identifier
        # specify the data to store
        keys = [
            'eval/mean_reward',
            'rollout/ep_rew_mean',
            'train/clip_fraction',
            'train/explained_variance',
            'train/entropy_loss',
            'train/loss',
            'train/policy_gradient_loss',
            'train/value_loss'
        ]

        self.data = {}
        for key in keys:
            self.data[key] = self._get_training_data(key)

    def _get_training_data(self, key: str) -> dict:
        # define data structure
        Datarow = namedtuple('Datarow', ['x', 'y'])
        # obtain tensorboard event
        event_accumulator = EventAccumulator(self.data_path)
        event_accumulator.Reload()
        # extract and return data from tensorboard event
        event = event_accumulator.Scalars(key)
        return Datarow(
            x=np.array([x.step for x in event]),
            y=np.array([x.value for x in event]),
        )

def run_policy(
        model: PPO, track: str, config: dict
    ) -> Tuple[
        Logger,
        MPC_Sim,
        Nonlinear_Model_Predictive_Controller,
        List[int],
        np.ndarray
    ]:

    env = setup_environments(
        n_envs=1,
        config=config,
        trajectories=[track],
        monitor_path=None,
        random_restarts=False,
        full_lap=True
    )

    obs = env.reset()
    actions = []
    probs = None
    done = False
    while not done:
        # get action probabilities
        p = get_action_probabilities(model, obs)
        probs = p if probs is None else np.vstack((probs, p))
        # select an action
        action, _ = model.predict(obs, deterministic=True)
        # step the environment
        obs, _, done, _ = env.step(action)
        actions.append(action[0])

        if not done:
            # extract and return environment attributes for plotting purposes
            environment = env.envs[0]
            logger = environment.logger
            simulation = environment.simulation
            mpc = environment.MPC

    return logger, simulation, mpc, actions, probs