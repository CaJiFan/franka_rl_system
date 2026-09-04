Viewed rl_env_node.py:406-454
Viewed rl_env_node.py:340-480
Viewed rl_env_node.py:480-520

Here is a complete, transparent breakdown of **all control bounds, safety guards, and post-processing filters** currently active between the policy's raw neural network output and the Franka hardware controller.

---

### 1. Stiffness & Damping Mapping ($K_p, K_d$)

| Control Variable | Parameter / Equation | Description |
|---|---|---|
| **Stiffness Bounds** | $K_{p,\text{min}} = 1.0\text{ N/m}$, $K_{p,\text{max}} = 160.0\text{ N/m}$ | Physical stiffness range mapped from raw policy output $[-1, 1] \to [1.0, 160.0]\text{ N/m}$. |
| **Damping Matrix** | $K_d = 2 \sqrt{K_p}$ | Enforces **critical damping** to eliminate oscillations when impedance matrices change. |
| **SPD Manifold Log-Map** | $K_p = \exp(S)$ | Ensures the $3 \times 3$ translational stiffness matrix $K_p$ remains **Symmetric Positive Definite (SPD)**. |
| **Fixed Baseline Override** | `FIXED_KP = 150.0 N/m` (when `POSITION_ONLY_MODE = True`) | Replaces policy stiffness outputs with fixed isotropic stiffness for baseline comparison. |

---

### 2. Action Delta Scaling & Max Deltas

| Control Variable | Value | Description |
|---|---|---|
| **Step Scale ($XY$ Wiping Plane)** | $0.25\text{ cm/step}$ ($2.5\text{ mm}$) $\to 5.0\text{ cm/s}$ at 20 Hz | Scales horizontal target progression toward active marker for smooth, continuous wiping. |
| **Step Scale ($Z$ Pressing Axis)** | $0.20\text{ cm/step}$ ($2.0\text{ mm}$) downward | Controls downward target integration rate until board contact plane is reached. |
| **Orientation Delta Cap** | $|\Delta \theta| \le 0.1\text{ rad/step}$ ($\approx 5.7^\circ$) | Clamps maximum rotational step to prevent rapid wrist spinning. |
| **Orientation Fixed Override** | `POSITION_ONLY_ORI = True` | Fixes wrist orientation to initial pose, routing all control capability through position deltas. |

---

### 3. Decoupled Safety Tethers (Lead-Distance Limits)

To prevent integrated target pose $x_{\text{target}}$ from pulling too far ahead of real physical robot pose $x_{\text{real}}$ when encountering surface friction:

| Axis | Max Tether Limit | Resulting Max Force Cap ($K_p = 150\text{ N/m}$) | Purpose |
|---|---|---|---|
| **$XY$ Wiping Plane** | $\Delta xy_{\text{max}} = 10.0\text{ cm}$ | $F_{xy,\text{max}} = 15.0\text{ N}$ | Ensures enough lateral pushing force to overcome felt pad friction without runaway target accumulation. |
| **$Z$ Pressing Axis** | $\Delta z_{\text{max}} = 2.5\text{ cm}$ | $F_{z,\text{max}} = 3.75\text{ N}$ | Limits downward target lead distance so contact force remains near $\sim 2.5 - 3.5\text{ N}$ (ideal ICRA evaluation range). |

---

### 4. Workspace Boundaries & Whiteboard Surface Limits

| Boundary | Value / Equation | Purpose |
|---|---|---|
| **$X$ Bounds** | $[0.35\text{ m}, 0.75\text{ m}]$ | Restricts reach within robot front workspace. |
| **$Y$ Bounds** | $[-0.30\text{ m}, +0.30\text{ m}]$ | Limits lateral motion across whiteboard. |
| **Dynamic $Z_{\text{MIN}}$ (Tilted Board Plane)** | $z_{\text{surface}}(x) = 0.1546 - 0.727 (x - 0.4137)$<br>$z_{\text{min\_dynamic}} = z_{\text{surface}}(x) - 0.020\text{ m}$ | Adapts minimum allowable $Z$ target as a function of $X$ to match the $37^\circ$ board incline, allowing target pose to press at most $2.0\text{ cm}$ below local board plane. |
| **$Z_{\text{MAX}}$ Bounds** | $0.40\text{ m}$ | Prevents target from lifting above the workspace. |

---

### 5. Exponential Moving Average (EMA) Smoothing

To eliminate high-frequency policy jitter before sending commands to C++ controller:

$$\text{Command}_t = \alpha \cdot \text{NewCommand}_t + (1 - \alpha) \cdot \text{Command}_{t-1}$$

- **Position Smoothing**: $\alpha_{\text{pos}} = 0.4$
- **Orientation Smoothing**: $\alpha_{\text{ori}} = 0.3$ (with quaternion sign alignment to avoid slerp wraparound)
- **Stiffness Smoothing**: $\alpha_{\text{stif}} = 0.2$