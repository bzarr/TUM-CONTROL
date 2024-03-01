# Vehicle Dynamics Models
Vehicle Dynamics models for the MPC formulated using CasADi. Current available  models:
- Nonlinear dynamic single track model: The default tire model is Pacejka 'magic formula'. 

# Dynamic single track model
## Inputs  
$[a,  \omega_f ]$, where 
- $a$ is the longitudinal acceleration 
- $\omega_f$ is the steering rate at the front wheel

## States
We define the vehicle state vector as follows:

$$
\begin{equation}
\begin{aligned}
\boldsymbol{x} &= [x_{\text{pos}},\space y_{\text{pos}},\space \psi,\space v_{\text{lon}},\space v_{\text{lat}},\space \dot{\psi},\space \delta_f]^T
\end{aligned}
\end{equation}
$$

Here, $x_{\text{pos}}$ and $y_{\text{pos}}$ represent the x- and y-coordinates of the ego vehicle, $\psi$ is the yaw angle, $v_{\text{lon}}$ and $v_{\text{lat}}$ indicate velocities in the longitudinal and lateral directions, $\dot{\psi}$ represents the yaw rate, $\delta_f$ corresponds to the steering angle at the front wheel, and $a$ signifies acceleration. 
The control vector is defined as $u = [a, \space \omega_f ]^T$, where $a$ represents the longitudinal acceleration, and $\omega_f$ represents the steering rate at the front wheel.

## Differential equations:
We adopt a dynamic nonlinear single-track model combined with Pacejka Magic Formula to account for essential dynamic effects. The system dynamics are defined as: 

$$
f(\boldsymbol{x},\boldsymbol{u}) =
\begin{aligned}
\begin{bmatrix}
v_\text{lon} \cos(\psi) - v_\text{lat} \sin(\psi) \space \\
v_\text{lon} \sin(\psi) + v_\text{lat} \cos(\psi) \space \\
\dot{\psi} \\
\frac{1}{m}\left(F_{x_r} - F_{y_f} \sin(\delta_f) + F_{x_f}  \cos(\delta_f) + m  v_\text{lat} \dot{\psi}\right) \\
\frac{1}{m}\left(F_{y_r} + F_{y_f} \cos(\delta_f) + F_{x_f} \sin(\delta_f) - m v_\text{lon} \dot{\psi}\right) \\
\frac{1}{I_z}\left(l_f \left(F_{y_f} \cos(\delta_f) + F_{x_f} \sin(\delta_f)\right) - l_r  F_{y_r}\right) \\
\omega_f 
\end{bmatrix} 
\end{aligned}
$$

For the lateral forces $Fy_{\{f,r\}}$, we account for the combined slip of lateral and longitudinal dynamics as in [5], where ${{f,r}}$ refers to the front or rear tires. 

$$
\begin{aligned}
Fy_{\{f,r\}} &= F_{{\{f,r\},\text{tire}}} \cos\left(\arcsin\left(F_{x_{\{f,r\}}}/F_{\text{max}_{\{f,r\}}}\right)\right)
\end{aligned}
$$

To avoid singularity problems, we clip $F_{x_{\{f,r\}}}/F_{\text{max}_{\{f,r\}}}$ at 0.98 as in [5]. 
The lateral front and rear tire forces are defined using the reduced Pacejka magic formula: 

$$
\begin{aligned}
F_{{\{f,r\},\text{tire}}} &= D_{\{f,r\}} \sin(C_{\{f,r\}} \arctan(B_{\{f,r\}} \alpha_{\{f,r\}} \\
&- E_{\{f,r\}} (B_{\{f,r\}} \alpha_{\{f,r\}}- \arctan(B_{\{f,r\}} \alpha_{\{f,r\}}))))
\end{aligned}
$$

The side slip angles are defined as follows: 

$$
\begin{aligned}
\alpha_f &= 
\delta_f - \arctan\left((v_\text{lat}+l_f \cdot \dot{\psi})/v_\text{lon}\right)  \\
\alpha_r &= 
\arctan\left((l_r \cdot \dot{\psi} - v_\text{lat}) /v_\text{lon}\right)
\end{aligned}
$$

The tire sideslip angle formula has a singularity issue with longitudinal velocity. To address this, we assume that the tire sideslip angle is negligible at low velocities, avoiding the need for two different models.

The longitudinal forces are defined as following:

$$
\begin{aligned} 
Fx_f &= - Fr_f \\
Fx_r &= F_d  - Fr_r  - F_\text{aero}\\
\end{aligned}
$$

Here, the driving force at the wheel is defined as $ F_d = m \cdot a$ and the rolling resistance forces [2] are defined as
$Fr_{\{f,r\}} = fr \cdot F_{z,{\{f,r\}}}$. The rolling constant $f_r$ is defined as $fr = fr_0 + fr_1 \cdot \frac{v}{100} + fr_4 \cdot \left(\frac{v}{100}\right)^4$, where $v$ represents the absolute velocity in km/h [2].
The aerodynamic force is calculated as $F_\text{aero} = 0.5\cdot \rho \cdot S \cdot Cd \cdot v_\text{lon}^2 $ [2] and $F_{z,\{f/r\}}$ represents the vertical static tire load at the front and rear axles $F_{z,{\{f,r\}}} = \frac{m \cdot g \cdot l_{\{r,f\}}}{l_f + l_r}$. 

## Parameter Identification for EDGAR vehicle (VW T7 Multivan)
To identify the model parameters, we perform ISO 4138-compliant steady-state circular driving tests. 

Validating the chosen model with real-world data and identifying parameter values are crucial for ensuring accuracy and reliability. We measure some parameters, including the position of the center of gravity and vehicle mass. The vehicle configuration employed in this study comprises five mounted seats and accommodates a driver and no passengers.

To identify further parameters, we conduct steady-state circular driving behavior tests compliant with ISO 4138. Our focus is on the constant steering-wheel angle approach, and we employ two variations: discrete- and continuous speed increase tests. We collect motion and steering data from the GPS-IMU and VW Series motion sensors.
To cover both low- and high-velocity ranges, we conduct tests at velocities from 5 km/h up to 13km/h, with steering wheel angles ranging from 45° to 540° in both turning directions. Under normal circumstances, i.\,e., clear weather, negligible wind speed, and an outside temperature of 23°C, we use Bridgestone 235/50R18 101H summer tires.

|                | Value | Unit                | Description                           |
|----------------|-------|---------------------|---------------------------------------|
| $l$            | 3.128 | m               | Wheelbase                             |
| $l_{\mathrm{f}}$ | 1.484 | m               | Front axle to center of gravity       |
| $l_{\mathrm{r}}$ | 1.644 | m               | Rear axle to center of gravity        |
| $m$            | 2520  | kg            | Vehicle mass                          |
| $I_{\mathrm{z}}$ | 13600 | kg.m² | Moment of inertia in yaw          |
| $\rho$         | 1.225 | $kg.m^{-3}$ | Air density                      |
| $A$            | 2.9   | m²       | Cross-sectional frontal area          |
| $c_{\mathrm{d}}$ | 0.35  |                     | Drag coefficient                      |


Our goal is to estimate the single track model parameters that minimize the deviation between a series of true measured states $\boldsymbol{X_\text{true}} \in  \mathbb{R}^{n_s} \times \mathbb{R} ^{n_x}$ and the open-loop predicted 
states $\boldsymbol{X_\text{pred}} \in  \mathbb{R}^{n_s} \times \mathbb{R} ^{n_x}$, starting from the same initial state $\boldsymbol{x_\text{true}}^{(0)} = \boldsymbol{x_\text{pred}}^{(0)}$ and applying the same series of control vector inputs $\boldsymbol{U} \in  \mathbb{R}^{n_s} \times \mathbb{R} ^{n_u}$ for $n_s$ prediction steps. Here, $\boldsymbol{X_\text{true}} = [\boldsymbol{x_\text{true}}^{(1)}, \ldots, \boldsymbol{x_\text{true}}^{(n_s)}]$, $\boldsymbol{X_\text{pred}} = [\boldsymbol{x_\text{pred}}^{(1)}, \ldots, \boldsymbol{x_\text{pred}}^{(n_s)}]$, and $\boldsymbol{U} = [\boldsymbol{u}^{(0)}, \ldots, \boldsymbol{u}^{(n_s-1)}]$, where $\boldsymbol{x_\text{true}}^{(i)}$,  $\boldsymbol{x_\text{pred}}^{(i)}$, and $\boldsymbol{u}^{(0)}, \forall i \in \{0,\dots,n_s\}$ are defined as in Equation (1). Our model operates on a discretization time of $T_s = 0.02s$, and it is essential to highlight that the prediction model remains uninformed about the real states, i.\,e. the model does not receive updates during the simulation.

We refine our parameter estimates through an iterative process of manual tuning. The resulting single-track and tire model parameters are listed in Table \ref{tab:stm_parameters} and \ref{tab:tire_model}, respectively, where $F_{z,\{f/r\}}$ represents the vertical static tire load at the front and rear axles.

| Parameter | Front | Rear | Description        |
|-----------|-------|------|--------------------|
| $B$       | 10    | 10   | Stiffness factor  |
| $C$       | 1.3   | 1.6  | Shape factor       |
| $D$       | $1.2⋅F_{z,f}$ | $2.1⋅F_{z,r}$ | Peak value    |
| $E$       | 0.97  | 0.97 | Curvature factor   |

<p align="center">
<img
  src="../Utils/Veh_dyn_Param_Ident_EDGAR.png"
  alt=""
  title=""
  style="margin: 0 auto; max-width: 100px">
</p>

In this figure, we present the open-loop prediction assessment spanning $n_s = 1000$ prediction steps, corresponding to a 20s ride. During this evaluation, the steering-wheel angle remains fixed at 90°, while the vehicle speed undergoes a steady increase. 

It is important to emphasize that the accuracy of parameter estimation relies heavily on both the selected model and the quality of the measured data. Additionally, our approach did not employ any state estimation and sensor fusion techniques to enhance state estimations.

## Symbols
|variable | Definition|
| --------| ----------|
|$j$                    |longitudinal jerk|
|$\omega_f$	            |Steering rate at the front wheel|
|$x$	                |Position in x-axis|
|$y$	                |Position in y-axis|
|$\psi$                 |the orientation (\psi)|
|$v_{long}$, $v_{lat}$  |the velocities in longitudinal and lateral directions| 
|$\dot{\psi}$           |the \psi rate |
|$\delta_f$             |the steering angle|
|$a$                    |the acceleration at the front wheel|
|$F_d$                  |the driving force at the wheel|
|$F_{y_r}$	            |Lateral force at the rear wheel
|$F_{y_f}$	            |Lateral force at the front wheel
|$F_{x_r}$	            |Longitudinal force at the rear wheel|
|$F_{x_f}$	            |Longitudinal force at the front wheel|
|$F_{max_f}$	        |Maximum force that can be applied to the front tires|
|$F_{max_r}$	        |Maximum force that can be applied to the rear tires|
|$F_{aero}$             |the aerodynamic force|
|$Fr_f$                 |the rolling resistance at the front wheel|
|$Fr_r$                 |the rolling resistance at the rear wheel|
|$F_{banking_x}$        |the banking force on x-ax due to the road banking|
|$F_{banking_y}$        |the force on y-ax due to the road banking|
|$Fz_f$                 |the static tyre load at the front axle|
|$Fz_r$                 |the static tyre load at the rear axle|
|lr                   |the distance between the center of gravity and the rear axle|
|lf                   |the distance between the center of gravity and the front axle|
|m                    |the mass of the vehicle|
|$I_z$                  |the moment of inertia about the vertical axis passing through the center of mass of the vehicle|
|g                    |the acceleration due to gravity|
|banking              |the angle of the road banking|
|$\mu$                  |the coefficient of friction between the tyre and the road|
|$\rho$                 |the density of air|
|S                    |the cross-sectional area of the vehicle|
|Cd                   |the drag coefficient|
|fr                   |the friction coefficient|
|fr0, fr1, fr4        |rolling resistance constants|


# Bibliography
[1] Ge, Qiang, et al. "Numerically stable dynamic bicycle model for discrete-time control." 2021 IEEE Intelligent Vehicles Symposium Workshops (IV Workshops). IEEE, 2021.

[2] Gerdts, M. "The single track model." (2003).

[3] Effects of Model Complexity on the Performance of Automated Vehicle Steering Controllers: Model Development, Validation and Comparison.

[4] Kong, Jason, et al. "Kinematic and dynamic vehicle models for autonomous driving control design." 2015 IEEE intelligent vehicles symposium (IV). IEEE, 2015.

[5] Raji, Ayoub, et al. "Motion planning and control for multi vehicle autonomous racing at high speeds." 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2022.
