import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Union, Callable, List
from matplotlib.widgets import Slider, RadioButtons

from Learning_To_Adapt.SafeRL_WMPC.BO_WMPC.dataclasses import Trial
from Utils.colors import *

from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor

########################### GENERAL HELPER FUNCTIONS ###########################

""" Load a given config file from yaml format and return a dictionary. """
def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    return config

""" Normalize an array/tensor with regard to given bounds. """
def normalize(X: Union[np.ndarray, Tensor], bounds: Union[np.ndarray, Tensor]):
    return (X - bounds[0]) / (bounds[1] - bounds[0])

""" Calculate yaw from a series of xy positions. """
def yaw_from_xy(trajectory):
    dx = np.diff(trajectory[:, 0])
    dy = np.diff(trajectory[:, 1])
    yaws = np.arctan2(dy, dx)    
    return yaws

""" Calculate root-mean square of a given array. """
def get_root_mean_square(x: np.array):
    return np.sqrt(np.mean(x**2))

""" Hysteresis function for a given sequence of values. """
def hysteresis(x, th_lo, th_hi, initial=False):
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    
    if not ind.size:
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi)
    
    return np.where(cnt, hi[ind[cnt-1]], initial)


############################## BO HELPER FUNCTIONS #############################
""" Load a set of BO trials from a given csv file. """
def load_trials_from_csv(filepath: str) -> List[Trial]:

        trials = []

        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split(';')

            params_nat = torch.tensor([float(e) for e in parts[0].split(',')])
            params_nor = torch.tensor([float(e) for e in parts[1].split(',')])
            objectives = torch.tensor([
                [float(e) for e in parts[2].split(',')],
                [float(e) for e in parts[3].split(',')]
            ])
            feasible = bool(int(parts[4]))

            trial = Trial(
                id=0,
                params_nat=params_nat,
                params_nor=params_nor,
                objectives=objectives,
                feasible=feasible
            )
            trials.append(trial)

        return trials


############################# RL HELPER FUNCTIONS ##############################

""" Learning rate scheduler for adaptive learning rate. """
def learning_rate_schedule(
        initial: float, final: float, k: float
    ) -> Callable[[float], float]:

    def func(remaining: float) -> float:
        #return final + (initial - final) * np.exp(-k * (1 - remaining))
        #return initial * pow(k, 1 - remaining)
        return initial * pow((final / initial), (1 - remaining) * k)

    return func

""" Get action probability distribution for a given observation. """
def get_action_probabilities(model: PPO, state) -> np.ndarray:
    obs = obs_as_tensor(state, model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.cpu().detach().numpy()
    return probs_np


############################# RL HELPER FUNCTIONS ##############################

""" Interactive visualization of the BO surrogate models. """
def visualize_surrogate(optimizer, group_id: int) -> None:

    def get_values(params, selected):
        
        # generate a range of test values
        SAMPLES = 100
        X = torch.zeros(SAMPLES, 7)
        X[:,selected] = torch.linspace(0, 1, SAMPLES)
        X[:,0:selected] = torch.ones(SAMPLES, 1) * params[0:selected]
        X[:,selected+1:] = torch.ones(SAMPLES, 1) * params[selected+1:]

        # obtain mean and confidence region for objective function
        mean_objs, std_objs = optimizer.objective_model.evaluate(X)
        mean_objs *= -1
        std_objs *= -1
        lcb_objs = mean_objs - 2 * std_objs
        ucb_objs = mean_objs + 2 * std_objs

        # obtain mean and confidence region for constraint satisfaction
        mean_con, std_con = optimizer.constraint_model.evaluate(X)
        lcb_con = mean_con - 2 * std_con
        ucb_con = mean_con + 2 * std_con

        means = torch.vstack((mean_objs, mean_con))
        lcbs = torch.vstack((lcb_objs, lcb_con))
        ucbs = torch.vstack((ucb_objs, ucb_con))

        return X, means, lcbs, ucbs
    
    def update(val):

        selected = int(radio.value_selected)
        
        parameters = torch.tensor([slider.val for slider in sliders])

        X, means, lcbs, ucbs = get_values(parameters, selected)

        for i, ax in enumerate(axs):
            lines[i].set_ydata(means[i].numpy())
            fills[i].remove()
            ucb = ucbs[i].numpy()
            lcb = lcbs[i].numpy()
            fills[i] = ax.fill_between(X[:,selected], ucb, lcb, color=TUM_BLUE_3)
        fig.canvas.draw_idle()

    def deactivate_slider(val):
        selection = int(val)
        for id, slider in enumerate(sliders):
            slider.set_active(id != selection)

        update(None)

    # generate and fit the surrogate models if they do not yet exist
    optimizer.fit_surrogate_models(segment_group_id=group_id)
    
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(8, 10)
    fig.subplots_adjust(wspace=0, hspace=0.5)
    # define initial parameters
    initial_parameters = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    initial_selection = 0
    # calculate initial solution
    X, means, lcbs, ucbs = get_values(initial_parameters, initial_selection)

    titles = ['Lateral Tracking',
              'Velocity Tracking',
              'Feasibility']
    lines = []
    fills = []
    for i, ax in enumerate(axs):
        mean = means[i].numpy()
        lcb, ucb = lcbs[i].numpy(), ucbs[i].numpy()
        line, = ax.plot(X[:,initial_selection], mean, color=TUM_BLUE)
        lines.append(line)
        fill = ax.fill_between(X[:,initial_selection], ucb, lcb, color=TUM_BLUE_3)
        fills.append(fill)
        ax.set_title(titles[i])

    axs[-1].set_ylim([-0.1, 1.1])

    plt.figure(figsize=(12, 5))
    # radio buttons to select the parameter to show
    rax = plt.axes([0.05, 0.1, 0.3, 0.8], frame_on=False)
    radio = RadioButtons(
        rax,
        [f"{i}" for i in range(7)],
        active=initial_selection,
        activecolor=TUM_BLUE
    )
    radio.on_clicked(deactivate_slider)

    # generate sliders
    sliders = []
    for i in range(7):
        ax = plt.axes([0.2, 0.8-(0.8/7)*i, 0.65, 0.1], facecolor=TUM_BLUE_3)
        slider = Slider(ax, f'', 0.0, 1.0, valinit=0.5)
        slider.on_changed(update)
        sliders.append(slider)

    deactivate_slider(initial_selection)

    plt.tight_layout()
    plt.show()

""" Plot relevant RL training data to assess training stability and success."""
def plot_training_data(
        data: dict,
        show: bool = True, 
        save_to_path: str = None
    ) -> None:
    fig, axs = plt.subplots(2, 4, sharex=True)
    fig.set_size_inches(w=16, h=9)

    for ax, key in zip(axs.ravel(), data.keys()):
        x = data[key].x
        y = data[key].y
        ax.plot(x, y, color=TUM_BLUE)
        ax.set_xlabel('Step')
        ax.set_title(key)

    if save_to_path: fig.savefig(save_to_path)
    if show: plt.show()
