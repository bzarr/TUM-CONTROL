Simulation parameters are in Config/sim_main_params.yaml
| simulation mode| description|
| ---| --- |
| 0| CiL: simulation model= separate vehicle dynamics model|
| 1| MPC Sim: simulation model= MPC predictions |

##  Time params
| parameter| description| default value|
| ---| --- | --- |
|Tp      |prediction horizon [s] | 3
|Ts_MPC  |MPC prediction discretization step [s] | 0.05
|Ts      |Simulation sampling period [s] | 0.025
|N       |number of discretizaion steps MPC |
|T       |simulation time [s] | 

## Simulation highlevel
- simulation step:
    + simMode = 1 (MPCiL) sim step = next MPC pred
    + simMode = 0 (CiL) 
        - preprocess control input/optimal state
        - call Simulator step method