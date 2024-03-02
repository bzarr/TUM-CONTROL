# Safe Weights-Varying MPC Using Deep Reinforcement Learning and Pareto Optimal Parameter Sets

Two-stage approach to allow for online weights adaptation while ensuring operational safety. The procedure is based upon two subsequent steps:
- an optimization process based on Bayesian optimization (BO) to obtain Pareto optimal parameter sets
- a learning process based on Reinforcement Learning (RL) to develop an optimal decision policy


## Usage

In order to use a trained model, you must enable WMPC by setting the ```enable_WMPC``` flag in ```MPC_params.yaml```. Furthermore, specify the path to a trained model with ```WMPC_model: Python/TUM_Learning_To_Adapt/Safe_WMPC/_models/<identifier>```. The model needs to be have the standard folder structure as generated during training. The MPC will then automatically update its controller parameters as defined by the trained model.


## Training Procedure

This procedure describes all necessary steps to train a Safe Weights-Varying MPC. The steps must be conducted sequentially, as each step is based upon the results of the previous step. For training, the ```MPC_params.yaml``` must be adapted as follows:
- set ```enable_WMPC``` to false
- disable ```solver_build``` and ```solver_generate_C_code```
This ensures an efficient training without interference and conflicts due to parallel solver code access.

### Bayesian Optimization

In order to obtain a set of Pareto optimal parameterizations, run ```python Python/TUM_Learning_To_Adapt/Safe_WMPC/bo_optimize.py```, after having defined all parameters in ```Python/TUM_Learning_To_Adapt/Safe_WMPC/_config/bo_config.yaml```.

The optimization process performs ```n_initial``` random evaluations, followed by ```n_bayesian_optimization * batch_size``` guided search steps. The search is conducted in a parallel manner, search for parameters that optimize the behavior on straight and curved sections of the training tracks. The result is a set of Pareto optimal parameter sets that is saved in ```.csv``` format to the specified location.

### Parameter Postprocessing

As the obtained Pareto front typically is dense and unevenly distributed, we reduce it to a set of representative points. This is achieved by running ```python bo_postprocess_parameters.py``` with the arguments

- ```-identifier```: the identifier of the BO run to load
- ```-num```: the number of parameter sets to extract from each Pareto front
- ```-save```: whether or not to save the extracted parameter sets

The script generates plots that visualize the two Pareto fronts and the extracted points. If the ```-save``` argument is given, a .csv-formatted parameter set file is generated in ```Python/TUM_Learning_To_Adapt/Safe_WMPC/_parameters```. This file is accessible to the subsequent training step.

### Reinforcement Learning

Finally, the extracted parameter sets are used as the decision space for a RL agent. The learning procedure is started by running ```rl_training.py```, after having configured the process in ```Python/TUM_Learning_To_Adapt/Safe_WMPC/_config/rl_config.yaml```. The process then automatically generates the required folder structure under ```Python/TUM_Learning_To_Adapt/Safe_WMPC/_models/<identifier>```.

It is also possible to continue the training for an exisiting model, using the ```-cont``` argument of the training function, e.g. ```python rl_training.py -cont <identifier>```. In this case, the existing model is loaded and the training is continued, again using the training settings as specified in the global config. The existing model is overwritten during training, which is why it is recommended to save a backup copy.