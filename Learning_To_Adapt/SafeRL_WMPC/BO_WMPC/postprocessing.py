import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple
from sklearn.cluster import KMeans

from botorch.utils.multi_objective.pareto import is_non_dominated

from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.dataclasses import Trial


def get_pareto_optimal_trials(
        trials: List[Trial], group: int
    ) -> Tuple[List[Trial], List[Trial]]:

    objectives = torch.stack([trial.objectives for trial in trials])

    obj = objectives[:,group,:]

    # obtain the mask that marks the non-dominated objective data
    mask = is_non_dominated(Y=obj)

    nondominated_trials = [i for (i, v) in zip(trials, mask) if v]
    dominated_trials = [i for (i, v) in zip(trials, mask) if not v]

    return nondominated_trials, dominated_trials

def find_closest_point_idx(source_point, points):
    distances = np.sqrt(((points - source_point)**2).sum(axis=1))
    return np.argmin(distances)

def reduce_point_cloud(
        points: Tensor, num_points: int
    ) -> Tuple[np.ndarray, set]:

    n_dimensions = points.shape[1]
    points = np.array(points)
    
    # identify the optimal points along each dimension
    best_point_ids = [np.argmax(points[:, dim]) for dim in range(n_dimensions)]
    best_points = points[best_point_ids]
    
    # adjust num_points if necessary to account for the best_points
    num_points -= len(np.unique(best_points, axis=0))
    
    if num_points <= 0:
        return np.unique(best_points, axis=0)
    
    # remove best points from total points
    remaining_points = np.delete(points, best_point_ids, axis=0)
    
    # perform KMeans clustering to reduce the number of points
    kmeans = KMeans(n_clusters=num_points, n_init=10)
    kmeans.fit(remaining_points)
    centroids = kmeans.cluster_centers_
    # obtain the points that are closest to the KMeans centroids
    closest_ids = [find_closest_point_idx(c, remaining_points) for c in centroids]
    closest_points = remaining_points[closest_ids]
    # obtain the indices of the selected points in the original array 
    indices_in_original_array = [find_in_array(points, p) for p in closest_points]

    # extract all selected points
    selected_indices = set(best_point_ids + indices_in_original_array)
    reduced_points = np.vstack([points[id] for id in selected_indices])

    # return the reduced point cloud
    return reduced_points, selected_indices

def find_in_array(a: np.ndarray, p: np.ndarray) -> int:
    tolerance = 1e-6
    matches = np.all(np.abs(a - p) < tolerance, axis=1)
    index = np.where(matches)[0][0]

    return index
    
def export_params(trials: List[Trial], filepath: str) -> None:

    # extract the parameterization of each trial to obtain the Pareto set
    pareto_set = [trial.params_nat.squeeze().tolist() for trial in trials]
    
    # write the parameters to the specified file
    with open(filepath, 'w+') as file:
        lines = []
        for params in pareto_set:
            l = ','.join([str(round(p, 1)) for p in params]) + '\n'
            lines.append(l)
        file.writelines(lines)
    
    n = len(lines)
    
    print(f"Exported {n} Pareto-optimal parameterizations to {filepath}.")
