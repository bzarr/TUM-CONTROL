import yaml
import os
import sys
import torch
import argparse
from typing import List

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

# add parent directory to import path
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(current_script_directory))
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from Learning_To_Adapt.SafeRL_WMPC.helpers import (
    load_config, learning_rate_schedule, plot_training_data
)
from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.environment import (
    setup_environments
)
from Learning_To_Adapt.SafeRL_WMPC.RL_WMPC.evaluation import (
    TrainingData, run_policy
)


""" Perform a full training run with specified parameters.

Parameters are taken from the configuration file Config/EDGAR/rl_config.yaml

The training is conducted and the corresponding data, along with the trained
model, is stored in Data/runs/<MODEL_IDENTIFIER> with the identifier specified
in the rl_config.yaml.

Optional:
By running
    python training.py -cont <MODEL_IDENTIFIER>,
an existing model is loaded from Data/runs/ and the training is continued, using
the parameters from rl_config.yaml.

"""


def run(
        config: dict,
        run_path: str,
        continue_run: str = None,
        parent_run_path: str = None
    ):

    if continue_run:
        print(f"Continuing run from {parent_run_path}")
    else:
        print(f"Logging run to {run_path}")
        os.makedirs(run_path, exist_ok=True)

        # store the config file
        config_file = os.path.join(run_path, 'rl_config.yaml')
        with open(config_file, 'x') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    ############################ ENVIRONMENT SETUP #############################

    # training trajectories
    trajectories = ['monteblanco', 'modena']

    # setup training environments
    env = setup_environments(
        n_envs          = config['n_environments'],
        config          = config,
        trajectories    = trajectories,
        monitor_path    = run_path,
        random_restarts = True,
        full_lap        = False,
        evaluation_env  = False
    )

    # setup evaluation environments
    _eval_log_path = os.path.join(run_path, 'eval_results/')
    eval_env = setup_environments(
        n_envs          = config['n_envs_eval'],
        config          = config,
        trajectories    = trajectories,
        monitor_path    = _eval_log_path,
        random_restarts = False,
        full_lap        = False,
        evaluation_env  = True
    )

    # setup evaluation callback
    _best_model_path = os.path.join(run_path, 'best_model')
    eval_callback = EvalCallback(
        eval_env                = eval_env,
        best_model_save_path    = _best_model_path,
        eval_freq               = config['evaluation_frequency'],
        log_path                = _eval_log_path,
        n_eval_episodes         = config['n_eval_episodes'],
        deterministic           = True
    )

    ############################### MODEL SETUP ################################
    # load model from existing run
    if continue_run is not None:
        if parent_run_path is None:
            print(f"Error: You need to specify a parent run path to continue.")
            quit()
        # load the model from the given location
        _model_location = os.path.join(parent_run_path, 'best_model', 'best_model')
        model = PPO.load(_model_location, env=env)
        # re-enable tensorboard logging
        model.tensorboard_log = run_path

    else:
        # setup policy network
        policy_kwargs = {
            'activation_fn': torch.nn.Tanh,
            'net_arch': dict(
                pi = config['net_arch'],
                vf = config['net_arch']
            ),
        }
        # setup learning rate schedule if needed
        if config['use_adaptive_learning_rate']:
            learning_rate   = learning_rate_schedule(
                initial     = config['adaptive_lr_init'],
                final       = config['adaptive_lr_final'],
                k           = config['adaptive_lr_decay']
            )
        else:
            learning_rate = config['learning_rate']
        # setup model
        model = PPO(
            policy              = 'MlpPolicy',
            env                 = env,
            learning_rate       = learning_rate,
            n_steps             = config['n_steps'],
            batch_size          = config['batch_size'],
            n_epochs            = config['n_epochs'],
            gamma               = config['gamma'],
            gae_lambda          = config['gae_lambda'],
            clip_range          = config['clip_range'],
            clip_range_vf       = None,
            normalize_advantage = True,
            ent_coef            = config['ent_coef'],
            vf_coef             = config['vf_coef'],
            max_grad_norm       = config['max_grad_norm'],
            use_sde             = False,
            sde_sample_freq     = -1,
            target_kl           = None,
            tensorboard_log     = run_path,
            policy_kwargs       = policy_kwargs,
            verbose             = 0,
            device              = 'auto'
        )
        
    ############################## MODEL TRAINING ##############################
    # start the learning process
    model.learn(
        total_timesteps = config['n_training_steps'],
        callback = eval_callback,
        log_interval = 1,
        tb_log_name = 'tensorboard',
        progress_bar = True,
        reset_num_timesteps = False
    )

    ################################ EVALUATION ################################
    # load and plot training data
    training_data = TrainingData(run_path, '')
    plot_training_data(
        training_data.data,
        show=False,
        save_to_path=os.path.join(run_path, 'training_data.png')
    )

    # load the best existing model, otherwise use the latest training model
    _model_location = os.path.join(_best_model_path, 'best_model.zip')
    if os.path.exists(_model_location):
        model = PPO.load(_model_location)

    performance_folder = os.path.join(run_path, 'performance')
    os.makedirs(performance_folder, exist_ok=True)

    # test the model on each track and save the results
    # tracks = ['monteblanco', 'modena', 'lvms']
    tracks = ['monteblanco']
    for track in tracks:
        # run the model and record the behavior
        (
            logger,
            sim,
            mpc,
            actions,
            probs
        ) = run_policy(model, track, config)

        # save the simulation logs
        logger.save_logs(
            filepath=os.path.join(performance_folder, f'{track}.npz'),
            sim_main_params=sim.sim_main_params,
            sim_disturbance_derivatives=sim.sim_disturbance_derivatives,
            sim_disturbance_state_estimation=sim.sim_disturbance_state_estimation
        )


if __name__ == '__main__':

    base_path = 'Learning_To_Adapt/SafeRL_WMPC/_models'

    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cont',
        default=0,
        const=0,
        nargs='?'
    )
    args = parser.parse_args()

    continue_run = args.cont if args.cont else None

    # import the model, but nevertheless generate a new folder
    if continue_run:
        parent_run_path = os.path.join(base_path, continue_run)
        run_path = parent_run_path
        config = load_config(parent_run_path + '/rl_config.yaml')
    else:
        parent_run_path = None

        config = load_config(
            'Learning_To_Adapt/SafeRL_WMPC/_config/rl_config.yaml'
        )

        # generate a result directory for the run
        run_path = os.path.join(base_path, config['identifier'])
        os.makedirs(run_path, exist_ok=True)

    run(config, run_path, continue_run, parent_run_path)