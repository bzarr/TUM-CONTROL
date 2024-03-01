## Problem Formulation
Problem:

$$
\begin{equation}
\begin{aligned}
& \textbf{Problem 1} && \textbf{Nominal NMPC}\\ 
& \underset{\mathrm{x}(.), \boldsymbol{u}(.)}{\min} &  \int^{T_p}_{\tau=0} & l(\boldsymbol{x}(\tau),\boldsymbol{u}(\tau)) \space  d\tau + m(\boldsymbol{x}(T_p))\\ 
& \text{subject to} & & \text { --- initial state ---} \\
& & & \boldsymbol{x_0} \leq \boldsymbol{x}(0) \leq \boldsymbol{x_0} \text {, }  \\
& & &  \text { --- vehicle dynamics ---} \\
& & & \dot{\boldsymbol{x}}(t) = f(\boldsymbol{x}(t),\boldsymbol{u}(t)) \text {, } & t \in[0, T_p), \\
& & &  \text { --- path constraints ---} \\
& & & \underline{\boldsymbol{h}} \leq h(\boldsymbol{x}(t), \boldsymbol{u}(t)) \leq \bar{\boldsymbol{h}},& t \in[0, T_p), \\
& & & \underline{\boldsymbol{x}} \leq J_x \space \boldsymbol{x}(t) \leq \bar{\boldsymbol{x}},& t \in[0, T_p), \\
& & & \underline{\boldsymbol{u}} \leq J_u \space \boldsymbol{u}(t)\leq \bar{\boldsymbol{u}}, & t \in[0, T_p), \\
& & &  \text { --- terminal constraints ---} \\
& & & \underline{\boldsymbol{h}}^{\mathrm{e}} \leq h^{\mathrm{e}}(\boldsymbol{x}(T_p)) \leq \bar{\boldsymbol{h}}^{\mathrm{e}}, \\
& & & \underline{\boldsymbol{x}}^{\mathrm{e}} \leq J_x^{\mathrm{e}} \space \boldsymbol{x}(T_p) \leq \bar{\boldsymbol{x}}^{\mathrm{e}}, \\
\end{aligned}
\end{equation}
$$


Here, $\boldsymbol{x} \in \mathbb{R}^{n_x}$ denotes the state vector, $\boldsymbol{u} \in \mathbb{R}^{n_u}$ the control vector, $t$ the discrete time, $T_p$ the prediction horizon, $f$ the system dynamics, $h$ and $h^e$ the path and terminal nonlinear inequality constraints, $J_{x}$ and $J_{x}^e$ help express linear path and terminal state constraints, $J_{u}$ helps express control input constraints and $x_0$ the initial state.
Also, $l: \mathbb{R}^{n_{\mathrm{x}}} \times \mathbb{R}^{n_{\mathrm{u}}}  \rightarrow \mathbb{R}$ denotes the stage cost and $m: \mathbb{R}^{n_{\mathrm{x}}}  \rightarrow \mathbb{R}$ the terminal cost.

## Cost function
The stage cost is defined as a nonlinear least square function: 

$$l(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \| \boldsymbol{y}(\boldsymbol{x},\boldsymbol{u}) - \boldsymbol{y_{\mathrm{ref}}}\|_W^2$$  

Similarly, the terminal cost is formulated as $m(\boldsymbol{x})=\frac{1}{2}\|\boldsymbol{y}^e(\boldsymbol{x})-\boldsymbol{y}^e_{\mathrm{ref}}\|_{W^e}^2$. Here, $W$ and $W_e$ represent the weighting matrices for the stage and terminal costs, respectively. $W$ is computed as $W = \text{diag}(Q,R)$, where $Q$ and $R$ are matrices for states and inputs weighting, while $W_e$ is defined as $W_e = Q_e$. The cost terms are defined as follows: 

$$
\begin{aligned}
\boldsymbol{y}(\boldsymbol{x},\boldsymbol{u}) &= [x_{\text{pos}},\space y_{\text{pos}},\space \psi,\space v_{\text{lon}},\space  j, \space \omega_f] \\
\boldsymbol{y_{\mathrm{ref}}} &= [x_{\text{pos,ref}},\space y_{\text{pos,ref}},\space \psi_{\text{ref}},\space v_{\text{ref}},\space  0,\space 0] \\
\boldsymbol{y^e_{\mathrm{ref}}} &= [x_{\text{pos,ref}}^e,\space y_{\text{pos,ref}}^e,\space \psi^e_{\text{ref}},\space v^e_{\text{ref}}]
\end{aligned}  
$$

Moreover, we employ two slack variables, $L1$ for a linear- and $L2$ for a quadratic constraint violation penalization term, relaxing the constraints and helping the solver find a solution.

To determine appropriate matrices $Q$ and $Q_e$ and $R$, we employ Multi-Objective Bayesian Optimization.

## Constraints
We formulate the combined longitudinal and lateral acceleration potential limits for the SNMPC as a nonlinear probabilistic constraint and for the nominal NMPC as a nonlinear hard constraint: 

$$
    h(\boldsymbol{x}, \boldsymbol{u}) = (a_\text{lon}/a_{x_\text{max}})^2 + (a_\text{lat}/a_{y_\text{max}})^2
$$

Here, the longitudinal acceleration is $a_\text{lon} = a$, and the lateral acceleration is $a_\text{lat} = v_\text{lon} \dot{\psi}$. The upper and lower bounds for $h$ are $\bar{h} = 1$ and $\underline{h} = 0$. We adapt the maximum allowed values based on the limits defined by the vehicle's actuator interface software. Specifically, we set $a_{y_\text{max}} = 5.866 m/s²$ , while $a_{x_\text{max}}$ varies based on the current velocity. When decelerating, $a_{x_\text{max}}$ is defined as:

$$
\begin{aligned}
a_{x_\text{max}} = 
\begin{cases}
     |-4.5m/s²|,  &\text{if } 0\leq v_\text{lon} \leq 11 m/s \\
     |-3.5m/s²|,  &\text{if } 11 m/s < v_\text{lon} \leq 37.5 m/s
\end{cases}
\end{aligned}
$$

And during acceleration:

$$
\begin{aligned}
a_{x_\text{max}} = 
\begin{cases}
    3 m/s², & \text{if } 0\leq v_\text{lon} \leq 11 m/s \\
     2.5m/s², & \text{if } 11 m/s < v_\text{lon} \leq 37.5 m/s
\end{cases}
\end{aligned}
$$

Additionally, we impose linear hard constraints on the steering angle and steering rate at the front wheel:

$$
\begin{aligned}
-0.61 rad\leq & \mathbb{E}[\delta_f] \leq 0.61rad \\
-0.322rad/s\leq &\omega_f \leq 0.322rad/s
\end{aligned}
$$

## Bibliography
[1] Zarrouki, B., Wang, C., & Betz, J. (2023). A Stochastic Nonlinear Model Predictive Control with an Uncertainty Propagation Horizon for Autonomous Vehicle Motion Control. arXiv preprint arXiv:2310.18753. http://arxiv.org/abs/2310.18753
