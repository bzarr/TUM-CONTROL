import numpy as np

from Learning_To_Adapt.SafeRL_WMPC.helpers import normalize, yaw_from_xy


class ObservationGenerator:

    def __init__(self, anticipation_horizon: int, n_anticipation_points: int):

        # anticipation horizon
        self.horizon = anticipation_horizon
        # number of anticipated points
        self.n_samples = n_anticipation_points
        self.sample_distance = int(self.horizon/self.n_samples)

        self._bounds = np.concatenate((
            np.array([
                # [0., 39.],                              # velocity
                [-3., 3.],                              # lateral deviation
                [-5, 5],                            # velocity deviation
            ]),
            np.tile([0., 39.], (self.n_samples, 1)),    # future velocities
            np.tile([-3.2, 3.2], (self.n_samples, 1)),  # future yaw rates
        )).transpose()

        self.n_observations = self._bounds.shape[1]
    
    def get_observation(
            self,
            v: float,
            lat_dev: float,
            vel_dev: float,
            ref_traj: dict,
            Ts: float
        ) -> np.ndarray:

        # extract momentary states
        current_states = np.array([lat_dev, vel_dev])

        # extract v, yaw and yaw rate from local trajectory
        v_ref = ref_traj['ref_v']
        if not isinstance(v_ref, np.ndarray):
            v_ref = np.array(v_ref)
        # xy_ref = np.column_stack(
        #     (ref_traj['pos_x'], ref_traj['pos_y'])
        # )
        # yaw_ref = yaw_from_xy(xy_ref)
        yaw_ref = ref_traj['ref_yaw']
        yaw_rate_ref = np.diff(np.unwrap(yaw_ref)) / Ts

        # apply moving average to smooth out the noisy yaw rate profile
        N = 10
        yaw_rate_ref = np.convolve(yaw_rate_ref, np.ones(N)/N, mode='valid')
        # define sample indices
        # sample_indices = np.arange(
        #     0, self.horizon, step=self.sample_distance, dtype=int
        # )
        indices_vel = np.linspace(0, len(v_ref) - 1, self.n_samples, dtype=int)
        indices_yawrate = np.linspace(0, len(yaw_rate_ref) - 1, self.n_samples, dtype=int)
        # extract samples from reference trajectory
        v_sampled = v_ref[indices_vel]
        yaw_rate_sampled = yaw_rate_ref[indices_yawrate]

        # assemble and normalize observation
        observation = np.concatenate((
            current_states,
            v_sampled,
            yaw_rate_sampled
        ))
        observation = normalize(observation, self._bounds)

        # clip observation to lie in the range (0, 1)    NOTE: use with caution!
        # observation = np.clip(observation, 0, 1)
        
        return observation
