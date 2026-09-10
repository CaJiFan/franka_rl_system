Ran command: `find /home/userlab/cjimenez/riemannian_rl -name "*train*.py" -o -name "*config*.yaml" -o -name "*config*.py"`
Ran command: `ls -la /home/userlab/cjimenez`
Searched for "min_kp"
Viewed rl_env_node.py:395-415

If you have the opportunity to retrain, tailoring the simulation environment to match physical reality will produce a **dramatically better Sim2Real policy**. 

Here is the exact blueprint of parameters, limits, domain randomization (DR), and reward adjustments to make:

---

### 1. Decouple Translational & Rotational Stiffness ($K_p$)

In Robosuite, using a symmetric `[1, 300]` range for both translation and orientation is one of the biggest causes of Sim2Real failure. They represent completely different physical quantities:

* **Translational Stiffness ($K_{p,\text{trans}}$ in $\text{N/m}$)**:
  * A displacement error of $1\,\text{cm}$ ($0.01\,\text{m}$) at $K_p = 200\,\text{N/m}$ generates **$2.0\,\text{N}$** of restoring force. This is gentle, predictable, and compliant.
* **Rotational Stiffness ($K_{p,\text{ori}}$ in $\text{Nm/rad}$)**:
  * An angle error of just **$5.7^\circ$ ($0.1\,\text{rad}$)** at $K_p = 150\,\text{Nm/rad}$ demands **$15.0\,\text{Nm}$** of torque!
  * On a real Franka, the continuous torque limit of Joints 5, 6, and 7 is only **$12\,\text{Nm}$**.
  * In simulation (rigid physics, zero latency, no sensor noise), the policy learns to crank $K_{p,\text{ori}} \to 150 - 300\,\text{Nm/rad}$ to lock the wrist. On real hardware, that exact stiffness creates intense high-frequency wrist shaking, actuator saturation, and collision stops.

#### Recommended Decoupled Ranges for Retraining:
| Stiffness Parameter | Default Robosuite | **Recommended for Retraining** | Rationale |
| :--- | :--- | :--- | :--- |
| **$K_{p,\text{trans}}$** | `[1.0, 300.0]` | **`[20.0, 200.0]` $\text{N/m}$** | Floor at $20\,\text{N/m}$ prevents lateral friction stalls; ceiling at $200\,\text{N/m}$ ensures contact force stays safely below $10 - 12\,\text{N}$ even with a $2\,\text{cm}$ position error. |
| **$K_{p,\text{ori}}$** | `[1.0, 300.0]` | **`[1.0, 12.0]` $\text{Nm/rad}$** | Soft rotational compliance allows the felt pad to naturally seat flat against the board without fighting reaction moments. |

---

### 2. Domain Randomization (DR) Upgrades

To make the policy robust to physical whiteboard variations without needing manual crutches on hardware:

1. **Surface Incline & Orientation Perturbation**:
   - Randomize the whiteboard angle: nominal $36^\circ \pm 4^\circ$ tilt.
   - Randomize slight roll/yaw of the board ($\pm 2^\circ$). This forces the policy to learn active compliant orientation tracking rather than memorizing a fixed geometric plane.
2. **Surface Friction Randomization ($\mu$)**:
   - Felt-on-whiteboard dry-erase friction varies widely as ink accumulates: set $\mu \in [0.2, 0.7]$.
   - This prevents the policy from assuming zero lateral drag.
3. **Observation Noise**:
   - Add Gaussian noise to:
     - End-effector position: $\sigma = 2\,\text{mm}$
     - Tool orientation (quaternion): $\sigma = 0.02$ ($\sim 2^\circ$)
     - Force reading: $\sigma = 0.5\,\text{N}$
     - Marker positions from vision: $\sigma = 3\,\text{mm}$
4. **Action Delay / Latency (Critical for Stability)**:
   - In simulation, commands are applied instantly (zero latency).
   - In reality, camera capture + neural net inference + ROS 2 network + Franka 1 kHz filter introduces **$50 - 80\,\text{ms}$ ($1 - 2$ control steps)** of latency.
   - Randomize a 1-step action delay buffer in simulation during training (`action_t` executed at `t+1`). This single addition eliminates almost all Sim2Real trembling and overshoot.

---

### 3. Reward Function: Eliminating "Hopping" Between Markers

Why did the policy lift off between markers?
In standard wiping tasks, if the reward heavily penalizes friction/effort or does not reward continuous contact, the policy discovers an optimal shortcut:
$$\text{Wipe M0} \longrightarrow \text{Lift off board to escape friction} \longrightarrow \text{Fly through air} \longrightarrow \text{Pound down on M1}$$

To train the policy to **glide smoothly over the board**:

1. **Continuous Contact Maintenance Reward**:
   Add a reward bonus for maintaining contact within the target force window while translating:
   $$R_{\text{contact}} = \begin{cases} +c_1, & \text{if } 3\,\text{N} \le F_n \le 8\,\text{N} \\ -c_2, & \text{if } F_n < 1\,\text{N} \text{ (lifted during wiping episode)} \end{cases}$$
2. **Penalize Surface Separation**:
   Penalize distance $d_{\perp} = \max(0, z_{\text{tool}} - z_{\text{surface}})$ when moving between uncompleted markers.
3. **Tangential Velocity Reward**:
   Reward velocity projected *along the plane of the whiteboard* towards the active marker ($\vec{v} \cdot \hat{u}_{\text{tangent}}$).

---

### Summary Checklist

1. **Stiffness**: Set `kp_trans_range = [20.0, 200.0]` and `kp_ori_range = [1.0, 12.0]` in the environment action space.
2. **DR**: Add board slope variation ($\pm 4^\circ$), friction $[0.2, 0.7]$, and a 1-step action delay.
3. **Reward**: Penalize lifting off the surface ($F_n < 1\,\text{N}$) while markers remain to be cleared.

With these parameters, the policy will naturally learn to maintain a continuous, compliant, glide along the board on real hardware without any artificial waypoint guidance or safety tethers!