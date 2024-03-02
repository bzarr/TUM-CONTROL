import numpy as np
import json
from typing import List, Tuple

from Learning_To_Adapt.SafeRL_WMPC.helpers import hysteresis


def curvature_segmentation(
        base_trajectory_name: str,
        base_trajectory_states: dict,
        curv_threshold_lo: float,
        curv_threshold_hi: float,
        overlap: int
    ) -> Tuple[List[dict], List[dict]]:

    # load reference velocity and (unwrapped) yaw profile
    vel = base_trajectory_states['ref_v']
    yaw = np.unwrap(base_trajectory_states['ref_yaw'])
    abs_yaw_rate = np.abs(np.diff(yaw))

    # calculate curvature by dividing yaw rate by velocity
    curvature = abs_yaw_rate/vel[:-1]

    # find indices that mark section changes
    indicator = hysteresis(
        curvature,
        th_lo=curv_threshold_lo,
        th_hi=curv_threshold_hi
    )
    indices = np.where(indicator[:-1] != indicator[1:])[0]
    indices = np.resize(indices, len(indices)+1)

    # generate separate segment objects
    segments = ([], [])
    for i in range(len(indices) - 1):
        start, end = indices[i] - overlap, indices[i+1] + overlap
        # discard very short segments
        if abs(end - start) < 20:
            continue
        type = 0 if curvature[indices[i] + 1] > curv_threshold_lo else 1

        _segment = {
            'start': start,
            'end': end,
            'type': type,
            'trajectory': base_trajectory_name
        }
        segments[type].append(_segment)

    return segments


def get_train_segments(
        curv_threshold_lo=2e-5,
        curv_threshold_hi=1e-3,
        overlap=10,
    ) -> List[List[dict]]:

    trajectory_files = [
        'Trajectories/reftraj_modena_edgar.json',
        'Trajectories/reftraj_monteblanco_edgar.json',
    ]
    trajectory_names = ['modena', 'monteblanco']
    
    segments = [[], []]

    for i in range(len(trajectory_files)):
        # load base trajectory
        with open(trajectory_files[i], 'r') as file:
            base_trajectory = json.load(file)

        _segments = curvature_segmentation(
            base_trajectory_states=base_trajectory,
            base_trajectory_name=trajectory_names[i],
            curv_threshold_lo=curv_threshold_lo,
            curv_threshold_hi=curv_threshold_hi,
            overlap=overlap
        )
        for id, group in enumerate(_segments):
            for seg in group:
                segments[id].append(seg)

    return segments
