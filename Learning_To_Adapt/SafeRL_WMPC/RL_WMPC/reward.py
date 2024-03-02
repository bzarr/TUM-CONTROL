import numpy as np

from Utils.Logging_Plotting import Logger
from Learning_To_Adapt.SafeRL_WMPC.helpers import (
    normalize, get_root_mean_square
)


class RewardGenerator:

    def __init__(self, sigmas: np.ndarray, normalization_lims: np.ndarray):
        self.sigmas = np.array(sigmas)
        self.lims = np.array(normalization_lims)

    def get_reward(self, logger: Logger, step_length: int) -> float:
        
        # obtain the performance metrics
        lat_devs = logger.lat_devs[logger.current_step-step_length:logger.current_step]
        vel_devs = logger.vel_devs[logger.current_step-step_length:logger.current_step]

        metrics = np.array([
            get_root_mean_square(lat_devs), get_root_mean_square(vel_devs)
            # np.max(lat_devs), np.max(vel_devs)
        ])

        # normalize and clip metrics (0 = best performance)
        metrics_nor = np.clip(normalize(metrics, self.lims), 0., 1.)

        # calculate reward
        h = np.power(metrics_nor, 2) / (2 * self.sigmas)
        reward =  1.0 * np.exp(- np.sum(h))

        return reward