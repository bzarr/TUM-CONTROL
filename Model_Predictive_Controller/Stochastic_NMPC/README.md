## Problem Formulation
Problem 1 transforms a stochastic optimal control problem with probabilistic constraints into a deterministic one.
This framework aims to solve a stochastic OCP subject to chance constraints, offering an efficient method to transform chance constraints into robust deterministic constraints. Furthermore, it utilizes Polynomial Chaos Expansion (PCE) to propagate the uncertainties throughout the prediction horizon and proposes a novel UPH concept to address the infeasibility caused by uncertainty propagation.

In Problem 1, the following symbols are used: $\boldsymbol{x} \in \mathbb{R}^{n_x}$ represents the state vector, $\boldsymbol{u} \in \mathbb{R}^{n_u}$ represents the control vector, $T_p$ stands for the prediction horizon and $f$ denotes the system dynamics. The initial state is denoted as $\boldsymbol{x_0}$. Additionally, $l: \mathbb{R}^{n_{\mathrm{x}}} \times \mathbb{R}^{n_{\mathrm{u}}} \rightarrow \mathbb{R}$ defines the stage cost, while $m: \mathbb{R}^{n_{\mathrm{x}}}\rightarrow \mathbb{R}$ defines the terminal cost. 

$$
\begin{equation}
\begin{aligned}
&\textbf{Problem 1} && \textbf{Deterministic surrogate for}\\ 
&&&\textbf{Stochastic Nonlinear MPC}\\ 
& \underset{\boldsymbol{x}(.), \boldsymbol{u}(.)}{\min} & &  \int^{T_p}_{\tau=0}  l( \mathbb{E}[\boldsymbol{x}(\tau)],\boldsymbol{u}(\tau)) \space  d\tau + m(\mathbb{E}[\boldsymbol{x}(T_p)])\\ 
& \text{subject to} & & \text { --- initial state ---} \\
& & & \boldsymbol{x_0} \leq \boldsymbol{x}(0) \leq \boldsymbol{x_0} \text {, }  \\
& & &  \text { --- vehicle dynamics ---} \\
& & & \dot{\boldsymbol{x}}(t) = f(\mathbb{E}[\boldsymbol{x}(t)],\boldsymbol{u}(t)) \text {, } & t \in[0, T_p), \\
& & &  \text { --- path constraints ---} \\
& & & \underline{\boldsymbol{g}} \leq g(\mathbb{E}[\boldsymbol{x}(t)], \boldsymbol{u}(t)) \leq \bar{\boldsymbol{g}},& t \in[0, T_p), \\
& & & \underline{\boldsymbol{h}} \leq  \mathbb{E}[h(\boldsymbol{x},\boldsymbol{u})]  + \kappa\sqrt{\text{Var}[h(\boldsymbol{x},\boldsymbol{u})]} \leq \bar{\boldsymbol{h}}, & t \in[0, T_p), \\
& & & \underline{\boldsymbol{x}} \leq J\_{\mathrm{bx}} \space \mathbb{E}[\boldsymbol{x}(t)] \leq \bar{\boldsymbol{x}}, & t \in[0, T_p), \\
& & & \underline{\boldsymbol{u}} \leq J\_{\mathrm{bu}} \space \boldsymbol{u}(t)\leq \bar{\boldsymbol{u}}, & t \in[0, T_p], \\
& & & \text { --- terminal constraints ---} \\
& & & \underline{\boldsymbol{g}}^{\mathrm{e}} \leq g^{\mathrm{e}}(\mathbb{E}[\boldsymbol{x}(T_p)]) \leq \bar{\boldsymbol{g}}^{\mathrm{e}} \\
& & & \underline{\boldsymbol{h}}^{\mathrm{e}} \leq \mathbb{E}[h^{\mathrm{e}} (\boldsymbol{x}(T_p))] + \kappa\sqrt{\text{Var}[h^{\mathrm{e}} (\boldsymbol{x}(T_p))]} \leq \bar{\boldsymbol{h}}^{\mathrm{e}}\\
\end{aligned}
\end{equation}
$$

The SNMPC in Problem 1 incorporates hard linear constraints on the states' expectations and on the control inputs formulated with help of $J_{bx}$, $J_{bx}^e$ and $J_{bu}$. Furthermore, it handles hard nonlinear constraints on the states' expectations: $g$ and $g^{\mathrm{e}}$. Problem 1 transforms nonlinear probabilistic inequality constraints into estimated deterministic surrogates in expectation and variance of the nominal path and terminal nonlinear inequality constraints: $h$ and $h^e$. Here, $\kappa = \sqrt{(1- p)/p}$ denotes the nonlinear constraints variance sensitivity factor, i.e. constraints robustification factor, meant to tighten the constraints according to the variance of the nonlinear constraints affected by uncertainties. Here, $p \in (0,1]$ is the desired probability of violating the nonlinear constraints $h$.

The expectation and variance of the states and nonlinear constraints are estimated with Polynomial Chaos Expansion (PCE) method through propagating $n_s$ sampled points around the current measured state to account for uncertainties as in Eq.\ref{eq:propagation of states and constraints}. The propagation of uncertain state samples and constraints through $T_p$ is limited by Uncertainty Propagation Horizon (UPH): $T_u$. After reaching the UPH, the propagation of the samples is stopped and only the last estimated variables at $t = T_u$ are propagated until $T_p$. 

$$
    \begin{cases}
    \begin{aligned}
        &\mathbb{E}[\boldsymbol{x}\_{t}]= \boldsymbol{c}^{(\boldsymbol{x})}\_0\\
        &\mathbb{E}[h(\boldsymbol{x},\boldsymbol{u})] = c^{(h)}\_0\\
        &\text{Var}[h(\boldsymbol{x},\boldsymbol{u})] = \sum\_{k=1}^{L-1}
    (c^{(h)}\_k)^2  
    \end{aligned}
    &\text{, if } t \in \{0,...,N\_{u-1}\}  \\ \\
     \begin{aligned}
        &\mathbb{E}[\boldsymbol{x}\_{t}]= \boldsymbol{x}\_{t} = f(\mathbb{E}[\boldsymbol{x}\_{t-1}],\boldsymbol{u})\\
        &\mathbb{E}[h(\boldsymbol{x},\boldsymbol{u})] = h(\mathbb{E}[\boldsymbol{x}\_t],\boldsymbol{u})\\
        &\text{Var}[h(\boldsymbol{x},\boldsymbol{u})] = 0
    \end{aligned}
    &\text{, if } t \in \{N\_{u},...,N\_{p-1}\}  \\
    \end{cases}
$$

Here, $c^{(h)}\_k$ and $\boldsymbol{c}^{(\boldsymbol{x})}\_0$ represent the PCE coefficients of the nonlinear inequality constraints and the states respectively, $L$ the total number of the PCE terms and $N_{u}$ and $N_{p}$ denote the number of shooting nodes within the UPH and prediction horizon respectively. For further details, we refer to [1].
## Cost function
The stage cost is defined as a nonlinear least square function: 

$$l(\boldsymbol{x}, \boldsymbol{u})=\frac{1}{2}\|\boldsymbol{y}(\boldsymbol{x},\boldsymbol{u})-\boldsymbol{y}_{\mathrm{ref}}\|_W^2$$ 

Similarly, the terminal cost is formulated as $m(\boldsymbol{x})=\frac{1}{2}\|\boldsymbol{y}^e(\boldsymbol{x})-\boldsymbol{y}^e_{\mathrm{ref}}\|_{W^e}^2$. Here, $W$ and $W_e$ represent the weighting matrices for the stage and terminal costs, respectively. $W$ is computed as $W = \text{diag}(Q,R)$, where $Q$ and $R$ are matrices for states and inputs weighting, while $W_e$ is defined as $W_e = Q_e$. The cost terms are defined as follows: 

$$
\begin{aligned}
\boldsymbol{y}(\boldsymbol{x},\boldsymbol{u}) &= [x\_{\text{pos}},\space y_{\text{pos}},\space \psi,\space v_{\text{lon}},\space  j, \space \omega_f] \\
\boldsymbol{y}\_{\text{ref}} &= [x\_{\text{pos,ref}},\space y\_{\text{pos,ref}},\space \psi\_{\text{ref}},\space v\_{\text{ref}},\space  0,\space 0] \\
\boldsymbol{y}^e\_{\text{ref}} &= [x\_{\text{pos,ref}}^e,\space y\_{\text{pos,ref}}^e,\space \psi^e\_{\text{ref}},\space v^e\_{\text{ref}}]\\
\end{aligned}  
$$

Moreover, we employ two slack variables, $L1$ for a linear- and $L2$ for a quadratic constraint violation penalization term, relaxing the constraints and helping the solver find a solution.

To determine appropriate matrices $Q$ and $Q_e$ and $R$, we employ Multi-Objective Bayesian Optimization.
## Constraints
We formulate the combined longitudinal and lateral acceleration potential limits for the SNMPC (Problem 2) as a nonlinear probabilistic constraint using Eq.\ref{eq:deceleration constraints} and for the nominal NMPC (Problem 1) as a nonlinear hard constraint: 

$$
    h(\boldsymbol{x}, \boldsymbol{u}) = (a_\text{lon}/a_{x_\text{max}})^2 + (a_\text{lat}/a_{y_\text{max}})^2
$$

Here, the longitudinal acceleration is $a\_\text{lon} = a$, and the lateral acceleration is $a_\text{lat} = v_\text{lon} \dot{\psi}$. The upper and lower bounds for $h$ are $\bar{h} = 1$ and $\underline{h} = 0$. We adapt the maximum allowed values based on the limits defined by the vehicle's actuator interface software. Specifically, we set $a_{y_\text{max}} = 5.866 m/s²$ , while $a\_{x_\text{max}}$ varies based on the current velocity. When decelerating, $a_{x_\text{max}}$ is defined as:

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
