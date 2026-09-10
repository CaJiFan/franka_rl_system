#!/usr/bin/env python3
"""
rl_env_node.py
--------------
ROS2 Node that acts as the synchronizer and RL environment.
It caches the latest sensor readings, builds the unified observation vector,
and communicates with the remote RL server via ZeroMQ (ZMQ) at 20Hz.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import zmq
import math
import json
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm, sqrtm, inv
import os
import sys

try:
    from stable_baselines3 import SAC
except ImportError:
    print("Warning: stable_baselines3 not found. Local inference will use a dummy random policy.")

from std_msgs.msg import Float32MultiArray, Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, TwistStamped

# GRIP_SITE_OFFSET: static 90-degree rotation from Franka flange to the virtual wiper site.
# This matches what Robosuite computes internally for 'robot0_eef_quat' (the grip-site frame).
# Format: [x, y, z, w] (SciPy convention)
GRIP_SITE_OFFSET = np.array([0.7018, -0.0067, 0.0865, 0.7071], dtype=np.float32)

# Obs-space key documentation (85-D layout)
# Slots 0-46:  Robot Proprioception
#   0- 6:  joint_pos (7)
#   7-13:  cos(joint_pos) (7)
#  14-20:  sin(joint_pos) (7)
#  21-27:  joint_vel (7)
#  28-34:  joint_acc (7, numerically estimated)
#  35-37:  eef_pos (3)
#  38-41:  eef_quat  -- flange (4)
#  42-45:  eef_quat_site -- grip-site with static offset (4)
#  46:     robot0_contact (1, |Fz| > 2 N)
# Slots 47-81: Per-marker state (5 × 7 = 35)
#   Each marker:  marker_pos (3) + marker_wiped (1) + gripper_to_marker (3)
# Slots 82-84: gripper_to_active_waypoint (3)

class RLEnvNode(Node):
    def __init__(self):
        super().__init__('rl_env_node')
        
        # --- Execution Mode ---
        # Set to True to send observations to a remote ZMQ server.
        # Set to False to load the model and run inference locally on this machine.
        self.use_zmq = False 
        # TCP offset: flange center -> wiping contact surface (bottom face of tool)
        # Tool: 12x5x3 cm printed wiping pad  ->  contact at Z = 0.030 m from flange
        # Keep in sync with tcp_offset in record_corners.py !
        self.tcp_offset = np.array([0.0, 0.0, 0.030])

        # ── Deployment Test Flags ────────────────────────────────────────────────────
        # Set True to ignore policy stiffness and use a fixed Kp=150 N/m diagonal
        # (critical damping applied automatically). Useful for isolating whether
        # position tracking is correct before trusting the learned stiffness.
        self.POSITION_ONLY_MODE = False
        self.FIXED_KP  = 150.0   # N/m  (translational)
        self.FIXED_KP_ORI = 5.0  # N·m/rad (rotational compliance: allows wiping pad to lay 100% flat on tilted board)

        # Fix A: Hold current orientation and ignore policy orientation commands to eliminate wrist spin
        self.POSITION_ONLY_ORI = False

        # Set True to use waypoint guidance for smooth continuous real-robot demos.
        # Set False for 100% pure unassisted RL policy action execution (raw SAC deltas).
        self.USE_WAYPOINT_GUIDANCE = False

        # Set True to cap tracking error (tether) relative to current robot position.
        # Set False to disable the tether completely for unconstrained spatial tracking.
        self.USE_SAFETY_TETHER = False

        # Action mapping flags: set True if policy action Y is inverted relative to robot frame
        self.INVERT_ACTION_Y = False


        if self.use_zmq:
            # --- ZMQ Client Setup ---
            self.server_ip = "127.0.0.1" # Replace with your remote server IP
            self.context = zmq.Context()
            self.zmq_socket = self.context.socket(zmq.REQ)
            self.zmq_socket.connect(f"tcp://{self.server_ip}:5555")
            self.get_logger().info(f"Connected to ZMQ RL Server at {self.server_ip}:5555")
            self.model = None
        else:
            # --- Local RL Policy Setup ---
            BASE_PATH = "checkpoints"
            ALGO="SAC"
            ENV="WIPE_ICRA"
            CONFIG=sys.argv[1]
            self.model_path = os.path.join(BASE_PATH, f'{ALGO}_{ENV}_{CONFIG}' ,"best_model") # Update with your actual checkpoint path!
            self.get_logger().info(f"Loading SAC policy from {self.model_path} onto CPU...")
            try:
                self.model = SAC.load(self.model_path, device='cpu')
                self.get_logger().info("Model loaded successfully!")
                
                # Dynamically set prior_dim based on the model's observation space
                obs_shape = self.model.observation_space.shape[0]
                if obs_shape == 85:
                    self.obs_mode = "MULTI_MARKER_85D"
                    self.prior_dim = 6  # action still has prior_dim stiffness params
                    self.get_logger().info("Detected MULTI-MARKER model (85-D observations)")
                elif obs_shape == 62:
                    self.obs_mode = "BASELINE_62D"
                    self.prior_dim = 6
                    self.get_logger().info("Detected BASELINE diagonal model (62-D observations)")
                elif obs_shape == 65:
                    self.obs_mode = "SPD_65D"
                    self.prior_dim = 9
                    self.get_logger().info("Detected SPD manifold model (65-D observations)")
                else:
                    self.obs_mode = "UNKNOWN"
                    self.prior_dim = 6
                    self.get_logger().warn(f"Unknown observation space shape {obs_shape}, defaulting prior_dim=6")
            except Exception as e:
                self.get_logger().warn(f"Could not load model: {e}")
                self.get_logger().warn("Using a dummy random policy for now.")
                self.model = None
                self.prior_dim = 9 # Default fallback

        # --- State Caches ---
        self.latest_joint_state  = None
        self.latest_eef_pose     = None
        self.latest_wrench       = None
        self.latest_wipe_markers = None  # 20-float /wipe_markers msg

        # Velocity / acceleration tracking
        self.prev_eef_pos    = None
        self.prev_eef_quat   = None
        self.prev_joint_vel  = None   # for joint_acc estimation
        self.last_update_time = None

        # ── EMA smoothing state (damp high-freq policy oscillations) ──
        # Lower alpha = heavier smoothing, Higher alpha = more responsive
        self.ALPHA_POS  = 0.85   # position target (responsive, fast tracking)
        self.ALPHA_ORI  = 0.70   # orientation target (responsive rotational tracking)
        self.ALPHA_STIF = 0.10   # stiffness matrices (damps impedance fluctuations)
        self.ema_pos    = None   # np.ndarray(3,)  — initialised on first action
        self.ema_quat   = None   # np.ndarray(4,)  — [x,y,z,w]
        self.ema_Kp     = None   # np.ndarray(3,3)
        self.ema_Kd     = None   # np.ndarray(3,3)
        self.ema_kp_ori = None   # np.ndarray(3,)
        self.ema_kd_ori = None   # np.ndarray(3,)

        # Integrated position and orientation targets (accumulate delta to overcome friction/inertia)
        self.target_pos = None
        self.target_rot = None
        self.prev_active_idx = None

        # ── ICRA Evaluation & Metric Logging State ───────────────────────────
        self.fz_history = []
        self.contact_step_count = 0
        self.optimal_force_count = 0        # 5.0 N <= Fz <= 10.0 N
        self.safety_violation_count = 0     # Fz > 10.0 N
        self.hardware_violation_count = 0   # Fz > 15.0 N
        self.kp_volume_history = []
        self.kp_anisotropy_history = []
        self.kp_spd_valid_count = 0
        self.airm_jerk_history = []
        self.prev_Kp_mat = None
        self.start_time_sec = None
        self.latest_fz_current = 0.0
        self.metrics_exported = False

        # Force Tare Calibration state (calibrates mid-air baseline at startup)
        self.fz_tare_samples = []
        self.fz_baseline = None

        # --- Subscribers ---
        self.create_subscription(JointState,     '/joint_states',                                                        self.joint_cb,   10)
        self.create_subscription(PoseStamped,    '/franka_robot_state_broadcaster/current_pose',                         self.eef_pose_cb, 10)
        self.create_subscription(WrenchStamped,  '/franka_robot_state_broadcaster/external_wrench_in_base_frame',        self.wrench_cb,  10)
        # Primary vision topic: 20 floats (5 markers × [x,y,z,wiped])
        self.create_subscription(Float32MultiArray, '/wipe_markers', self.markers_cb, 10)

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/riemannian_impedance_controller/impedance_cmd', 10)

        # --- 20Hz Control Timer ---
        self.control_rate = 20.0
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

    def joint_cb(self, msg):     self.latest_joint_state = msg
    def eef_pose_cb(self, msg):  self.latest_eef_pose = msg
    def wrench_cb(self, msg):    self.latest_wrench = msg
    def markers_cb(self, msg):   self.latest_wipe_markers = msg

    # -- Main Loop --
    def control_loop(self):
        if not (self.latest_joint_state and self.latest_eef_pose and
                self.latest_wrench and self.latest_wipe_markers):
            self.get_logger().warn("Waiting for all sensor topics...", throttle_duration_sec=2.0)
            return

        # Delta time
        now_time = self.get_clock().now()
        if self.start_time_sec is None:
            self.start_time_sec = now_time
        dt = 0.05
        if self.last_update_time is not None:
            dt = (now_time - self.last_update_time).nanoseconds / 1e9
            if dt <= 0.0:
                dt = 0.05
        self.last_update_time = now_time

        # ── Joint kinematics ─────────────────────────────────────────────────
        joint_pos = np.array(self.latest_joint_state.position[:7], dtype=np.float32)
        joint_vel = np.array(self.latest_joint_state.velocity[:7], dtype=np.float32)

        # joint_acc: numerically estimated backward derivative
        if self.prev_joint_vel is not None:
            joint_acc = (joint_vel - self.prev_joint_vel) / dt
        else:
            joint_acc = np.zeros(7, dtype=np.float32)
        self.prev_joint_vel = joint_vel.copy()

        # ── EEF pose ─────────────────────────────────────────────────────────
        pos  = self.latest_eef_pose.pose.position
        quat = self.latest_eef_pose.pose.orientation
        eef_pos  = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        eef_quat = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)

        # Grip-site quaternion: static 90-degree offset from flange
        r_eef  = R.from_quat(eef_quat)
        r_site = R.from_quat(GRIP_SITE_OFFSET)
        eef_quat_site = (r_eef * r_site).as_quat().astype(np.float32)  # [x,y,z,w]

        # TCP (eraser tip) position
        eef_pos_eraser = eef_pos + r_eef.apply(self.tcp_offset)

        # ── Contact & F/T (with Auto-Tare Baseline Subtraction) ───────────────
        force  = self.latest_wrench.wrench.force
        robot0_contact_force = np.array([force.x, force.y, force.z], dtype=np.float32)
        fz_signed = float(robot0_contact_force[2])

        # Auto-tare calibration: collect 25 samples (1.25s) in air at startup
        if self.fz_baseline is None:
            self.fz_tare_samples.append(fz_signed)
            if len(self.fz_tare_samples) >= 25:
                self.fz_baseline = float(np.mean(self.fz_tare_samples))
                self.get_logger().info(
                    f"✅ [AUTO-TARE COMPLETE] Mid-air baseline calibrated: Fz_tare = {self.fz_baseline:.2f} N"
                )
            else:
                self.get_logger().info(
                    f"⏳ [AUTO-TARE] Calibrating force sensor in air... ({len(self.fz_tare_samples)}/25)",
                    throttle_duration_sec=0.5
                )
                return  # Hold startup until baseline is calibrated

        # Net interaction force after subtracting mid-air baseline
        fz_net = float(abs(fz_signed - self.fz_baseline))
        self.latest_fz_current = fz_net
        robot0_contact = np.array(
            [1.0 if fz_net > 2.0 else 0.0], dtype=np.float32
        )

        # ── Vision: marker states from /wipe_markers (20 floats) ─────────────
        v_data = list(self.latest_wipe_markers.data)  # 20 floats
        marker_pos   = [np.array(v_data[i*4 : i*4+3], dtype=np.float32) for i in range(5)]
        marker_wiped = [np.array([v_data[i*4+3]],      dtype=np.float32) for i in range(5)]

        # gripper-to-marker vectors
        gripper_to_marker = [mp - eef_pos_eraser for mp in marker_pos]

        # active waypoint: first non-wiped marker (5 if all wiped)
        active_idx = next((i for i, w in enumerate(marker_wiped) if w[0] < 0.5), 5)
        self.all_wiped = (active_idx == 5)

        if active_idx < 5:
            gripper_to_active_waypoint = gripper_to_marker[active_idx].copy()
        else:
            gripper_to_active_waypoint = np.zeros(3, dtype=np.float32)
        self.latest_g2active_wp = gripper_to_active_waypoint.copy()

        # Smooth Waypoint Transition & Automatic ICRA Metric Export
        if self.prev_active_idx is not None and active_idx != self.prev_active_idx:
            if active_idx == 5:
                self.get_logger().info("🎉 [TASK COMPLETED] All markers wiped! Returning to Franka Desk Home Pose...")
                if not self.metrics_exported:
                    self.export_icra_metrics()
            else:
                self.get_logger().info(f"[WAYPOINT SWITCH] Active marker M{self.prev_active_idx} -> M{active_idx}! Smoothly continuing surface wipe.")
        self.prev_active_idx = active_idx

        # ── Assemble 85-D observation (STRICT KEY ORDER) ──────────────────────
        # Proprioception (47-D)
        obs_list = [
            joint_pos,              # 7  — raw joint angles
            np.cos(joint_pos),      # 7  — cosine
            np.sin(joint_pos),      # 7  — sine
            joint_vel,              # 7  — joint velocities
            joint_acc,              # 7  — joint accelerations (estimated)
            eef_pos_eraser,         # 3  — TCP position
            eef_quat,               # 4  — flange orientation
            eef_quat_site,          # 4  — grip-site orientation (static offset)
            robot0_contact,         # 1  — contact flag (|Fz| > 2 N)
        ]
        # Per-marker state (5 × 7 = 35-D)
        for i in range(5):
            obs_list += [
                marker_pos[i],          # 3 — absolute 3-D position
                marker_wiped[i],        # 1 — wiped flag
                gripper_to_marker[i],   # 3 — relative vector from TCP
            ]
        # Active waypoint vector (3-D)
        obs_list.append(gripper_to_active_waypoint)

        flat_obs = np.concatenate(obs_list).astype(np.float32)

        # ── Full Observation Diagnostics (throttled) ─────────────────────────
        # Prints every key-value pair in the 85-D obs vector so you can
        # compare directly with Robosuite's observation dict in simulation.
        def _fmt(arr, fmt="+.4f"):
            return "[" + "  ".join(f"{v:{fmt}}" for v in arr) + "]"

        m_lines = ""
        for i in range(5):
            wiped_str = "WIPED" if marker_wiped[i][0] > 0.5 else "active" if i == active_idx else "pending"
            m_lines += (
                f"\n    M{i} ({wiped_str:7s})  pos={_fmt(marker_pos[i])}  "
                f"g2m={_fmt(gripper_to_marker[i])}"
            )

        fz_peak = max(self.fz_history) if self.fz_history else 0.0
        fz_mean = float(np.mean(self.fz_history)) if self.fz_history else 0.0
        viol_pct = (self.safety_violation_count / max(1, self.contact_step_count)) * 100.0

        self.get_logger().info(
            f"\n{'='*64} OBS SNAPSHOT {'='*64}\n"
            f"  [00:06]  joint_pos      = {_fmt(joint_pos)}\n"
            f"  [07:13]  cos(jnt_pos)   = {_fmt(np.cos(joint_pos))}\n"
            f"  [14:20]  sin(jnt_pos)   = {_fmt(np.sin(joint_pos))}\n"
            f"  [21:27]  joint_vel      = {_fmt(joint_vel)}\n"
            f"  [28:34]  joint_acc      = {_fmt(joint_acc)}\n"
            f"  [35:37]  eef_pos(TCP)   = {_fmt(eef_pos_eraser)}  (eraser tip)\n"
            f"  [38:41]  eef_quat       = {_fmt(eef_quat)}  (flange xyzw)\n"
            f"  [42:45]  eef_quat_site  = {_fmt(eef_quat_site)}  (grip-site xyzw)\n"
            f"  [46]     contact        = {robot0_contact[0]:.0f}  (|Fz_net|>2N, net={fz_net:.2f}N, raw={fz_signed:.2f}N, tare={self.fz_baseline:.2f}N)\n"
            f"  [47:81]  markers (active={active_idx}):{m_lines}\n"
            f"  [82:84]  g2active_wp    = {_fmt(gripper_to_active_waypoint)}\n"
            f"  ICRA FORCE METRICS      : Fz_mean={fz_mean:5.2f}N  Fz_peak={fz_peak:5.2f}N  Fz_violations(>10N)={self.safety_violation_count} ({viol_pct:4.1f}%)\n"
            f"  obs_dim  = {flat_obs.shape[0]}\n"
            f"{'='*141}",
            throttle_duration_sec=0.5
        )

 
        # 3. Predict the action
        if self.use_zmq:
            try:
                # Send the array as bytes for extremely low-latency transfer
                self.zmq_socket.send(flat_obs.tobytes(), flags=zmq.NOBLOCK)
                
                # Wait for response (the 6D action)
                if self.zmq_socket.poll(timeout=100) == 0:
                    self.get_logger().error("ZMQ Timeout! RL server did not respond in time.")
                    self.zmq_socket.close()
                    self.zmq_socket = self.context.socket(zmq.REQ)
                    self.zmq_socket.connect(f"tcp://{self.server_ip}:5555")
                    return
 
                action_bytes = self.zmq_socket.recv()
                action = np.frombuffer(action_bytes, dtype=np.float32)
            except zmq.ZMQError as e:
                self.get_logger().error(f"ZMQ Error: {e}")
                return
        else:
            if self.model is not None:
                # deterministic=True disables exploration noise during deployment
                action, _states = self.model.predict(flat_obs, deterministic=True)
            else:
                # Dummy random action: prior_dim stiffness + 6 kinematics (always 12-D for baseline)
                action = np.random.uniform(-0.1, 0.1, size=(self.prior_dim + 6,)).astype(np.float32)

        # 4. Publish the action to the robot
        self.publish_action(action)


    def publish_action(self, action):
        """
        Converts the 15D SAC action into a 31-D Fat Payload (Float64MultiArray) 
        for the custom C++ Riemannian Impedance Controller.
        """
        # 1. Parse and decode Action Space
        min_kp_trans = 10.0  # Floor translational stiffness so lateral friction doesn't stall wiping speed
        max_kp_trans = 300.0 # Clean, safe translational stiffness limit (prevents >8N contact stops)
        min_kp_ori   = 2.0   # Soft rotational compliance floor (allows wiping pad to seat flat against 37-deg board)
        max_kp_ori   = 100.0  # Cap rotational stiffness so robot doesn't fight board surface reaction torque

        if self.prior_dim == 6:
            # 12D Action space for Baseline (3 diagonal trans stiffness, 3 diagonal rot stiffness, 3 pos delta, 3 ori delta)
            kp_trans_raw = action[:3]
            kp_rot_raw = action[3:6]
            pos_delta = action[6:9]
            ori_delta_vec = action[9:12]

            # Linear decode to physical values
            kp_trans_scaled = min_kp_trans + 0.5 * (kp_trans_raw + 1.0) * (max_kp_trans - min_kp_trans)
            kp_rot_scaled   = min_kp_ori   + 0.5 * (kp_rot_raw   + 1.0) * (max_kp_ori   - min_kp_ori)

            # Build diagonal K_p and K_d matrices
            K_p = np.diag(kp_trans_scaled)
            K_d = np.diag(2.0 * np.sqrt(kp_trans_scaled))

            kp_ori = kp_rot_scaled
            kd_ori = 2.0 * np.sqrt(kp_rot_scaled)
        else:
            # 15D Action space for SPD Manifold
            mandel_params = action[:6]
            kp_ori_raw = action[6:9]
            pos_delta = action[9:12]
            ori_delta_vec = action[12:15]
            
            # Diagonals linearly decoded to [min_kp_trans, max_kp_trans] and then log-mapped
            target_physical = min_kp_trans + 0.5 * (mandel_params[:3] + 1.0) * (max_kp_trans - min_kp_trans)
            diag_log = np.log(target_physical)
            
            S = np.zeros((3, 3))
            S[0, 0] = diag_log[0]
            S[1, 1] = diag_log[1]
            S[2, 2] = diag_log[2]
            
            # Off-diagonals scaled by 0.2 and Mandel basis mapping
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            S[0, 1] = S[1, 0] = (mandel_params[5] * 0.2) * inv_sqrt2
            S[0, 2] = S[2, 0] = (mandel_params[4] * 0.2) * inv_sqrt2
            S[1, 2] = S[2, 1] = (mandel_params[3] * 0.2) * inv_sqrt2
            K_p = expm(S)

            # Eigen Decomposition for Damping Matrix (Kd = 2 * sqrt(Kp))
            eigenvalues, eigenvectors = np.linalg.eigh(K_p)
            eigenvalues = np.maximum(eigenvalues, 1.0) 
            K_d = eigenvectors @ np.diag(2.0 * np.sqrt(eigenvalues)) @ eigenvectors.T

            # Rotational Stiffness & Damping (Linearly decoded)
            kp_ori = min_kp_ori + 0.5 * (kp_ori_raw + 1.0) * (max_kp_ori - min_kp_ori)
            kd_ori = 2.0 * np.sqrt(kp_ori)

        # ── Position-Only Override ───────────────────────────────────────────────
        # When POSITION_ONLY_MODE is True, replace whatever the policy output
        # with a fixed isotropic impedance. Only pos/ori deltas are used.
        if self.POSITION_ONLY_MODE:
            kp_fixed = self.FIXED_KP
            kd_fixed = 2.0 * np.sqrt(kp_fixed)       # critical damping
            K_p = np.eye(3) * kp_fixed
            K_d = np.eye(3) * kd_fixed
            kp_ori = np.full(3, self.FIXED_KP_ORI)
            kd_ori = 2.0 * np.sqrt(kp_ori)

        # 5. Position Integration & Safety Workspace
        if self.INVERT_ACTION_Y:
            pos_delta[1] = -pos_delta[1]

        # Franka Desk App Home Pose: X=419.4mm, Y=38.2mm, Z=277.3mm
        HOME_POS = np.array([0.4194, 0.0382, 0.2773], dtype=np.float32)

        # Decoupled Active Waypoint Target Guidance:
        #  - XY (Wiping Plane): Advance target_pos[:2] horizontally at 0.85 cm/step (17 cm/s) for fast continuous wiping
        #  - Z  (Pressing Axis): Maintain solid 5.0-7.0 N contact force to thoroughly erase dry-erase ink
        #  - Task Done: Smoothly lift & return to Franka Desk Home Pose upon completing all markers
        if hasattr(self, 'all_wiped') and self.all_wiped:
            dir_home = HOME_POS - self.target_pos
            dist_home = np.linalg.norm(dir_home)
            if dist_home > 0.005:
                pos_delta_safe = (dir_home / dist_home) * 0.010 # 20 cm/s fast lift & return home speed
            else:
                pos_delta_safe = np.zeros(3, dtype=np.float32)
        elif self.USE_WAYPOINT_GUIDANCE and hasattr(self, 'latest_g2active_wp') and self.latest_g2active_wp is not None:
            g2wp_xy = self.latest_g2active_wp[:2]
            g2wp_xy_dist = np.linalg.norm(g2wp_xy)
            current_z = self.latest_eef_pose.pose.position.z if self.latest_eef_pose else 0.25
            x_target_tmp = self.target_pos[0] if self.target_pos is not None else 0.45
            z_surf_tmp = 0.1546 - 0.727 * (x_target_tmp - 0.4137)
            # Fast 16 cm/s descent in mid-air from Home, gentle 8 cm/s pressing on surface
            z_step = -0.004 if (current_z <= z_surf_tmp + 0.02) else -0.008

            if g2wp_xy_dist > 0.001:
                dir_xy = g2wp_xy / g2wp_xy_dist
                # Continuous fast gliding: 24 cm/s max wiping rate between markers
                step_xy = max(0.0040, min(g2wp_xy_dist, 0.0120))
                pos_delta_safe = np.array([dir_xy[0] * step_xy, dir_xy[1] * step_xy, z_step])
            else:
                pos_delta_safe = np.array([0.0, 0.0, -0.002])
        else:
            MAX_DELTA = 0.005
            pos_delta_safe = np.clip(pos_delta * 0.005, -MAX_DELTA, MAX_DELTA)

        current_pos = self.latest_eef_pose.pose.position
        current_pos_arr = np.array([current_pos.x, current_pos.y, current_pos.z])

        if self.target_pos is None:
            self.target_pos = current_pos_arr.copy() + pos_delta_safe
        else:
            self.target_pos += pos_delta_safe

        # Tilted Whiteboard Surface Equation (derived from ROBOT_CORNERS_BASE: TL Z=0.1546m -> BL Z=0.0051m)
        # Slope dZ/dX = -0.727 (37-degree incline)
        x_target = self.target_pos[0]
        z_surface = 0.1546 - 0.727 * (x_target - 0.4137)
        DELTA_MAX_METERS = 0.025
        # Dynamic Z boundary: limit max penetration below board surface to 2.5 cm (0.025 m)
        z_min_dynamic = 0.15 if (hasattr(self, 'all_wiped') and self.all_wiped) else float(np.clip(z_surface - DELTA_MAX_METERS, -0.10, 0.40))

        # Decoupled safety tether (ONLY active if USE_SAFETY_TETHER is True and during surface wiping):
        if self.USE_SAFETY_TETHER and not (hasattr(self, 'all_wiped') and self.all_wiped):
            err_xy = self.target_pos[:2] - current_pos_arr[:2]
            dist_xy = np.linalg.norm(err_xy)
            if dist_xy > 0.25:
                self.target_pos[:2] = current_pos_arr[:2] + (err_xy / dist_xy) * 0.25

            err_z = self.target_pos[2] - current_pos_arr[2]
            if self.target_pos[2] <= z_surface + 0.01 and abs(err_z) > DELTA_MAX_METERS:
                self.target_pos[2] = current_pos_arr[2] + np.sign(err_z) * DELTA_MAX_METERS

        WORKSPACE_LIMITS = {
            "X_MIN": 0.35, "X_MAX": 0.75, 
            "Y_MIN": -0.30, "Y_MAX": 0.30, 
            "Z_MIN": z_min_dynamic, "Z_MAX": 0.40   
        }
        self.target_pos[0] = np.clip(self.target_pos[0], WORKSPACE_LIMITS["X_MIN"], WORKSPACE_LIMITS["X_MAX"])
        self.target_pos[1] = np.clip(self.target_pos[1], WORKSPACE_LIMITS["Y_MIN"], WORKSPACE_LIMITS["Y_MAX"])
        self.target_pos[2] = np.clip(self.target_pos[2], WORKSPACE_LIMITS["Z_MIN"], WORKSPACE_LIMITS["Z_MAX"])

        new_pos = self.target_pos.copy()

        # 6. Orientation Integration (Persistent Accumulator with Safety Tilt Clamping)
        current_quat = self.latest_eef_pose.pose.orientation
        current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        if not hasattr(self, 'init_quat') or self.init_quat is None:
            self.init_quat = np.array([current_quat.x, current_quat.y, current_quat.z, current_quat.w], dtype=np.float32)

        if self.target_rot is None:
            self.target_rot = current_rot

        tilt_angle_deg = 0.0
        if hasattr(self, 'all_wiped') and self.all_wiped:
            # Return to initial home orientation when returning home
            self.target_rot = R.from_quat(self.init_quat)
            new_quat = self.init_quat.copy()
        elif self.POSITION_ONLY_ORI:
            # Hold initial orientation
            self.target_rot = current_rot
            new_quat = np.array([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        else:
            # 1. Continuous scaling of policy output [-1, 1] to physical rotation step (rad/step)
            # 0.04 rad/step (~2.3 deg/step) at 20Hz allows up to 0.8 rad/s (~46 deg/s) angular velocity
            ORI_SCALE = 0.04
            MAX_ORI_DELTA = 0.06 # ~3.4 deg max single-step delta
            ori_delta_safe = np.clip(ori_delta_vec * ORI_SCALE, -MAX_ORI_DELTA, MAX_ORI_DELTA)

            # 2. Base frame delta rotation (pre-multiplication matches Robosuite OSC convention)
            delta_rot = R.from_rotvec(ori_delta_safe)
            self.target_rot = delta_rot * self.target_rot

            # 3. Orientation Safety Clamping: limit maximum tilt angle from nominal surface normal (init_quat)
            # Prevents wrist flip or exceeding Franka joint limits on tilted board
            r_init = R.from_quat(self.init_quat)
            r_rel_init = self.target_rot * r_init.inv()
            tilt_angle = r_rel_init.magnitude()
            MAX_TILT_RAD = 0.44 # 25.2 degrees max allowable cone of tilt
            if tilt_angle > MAX_TILT_RAD:
                rotvec_clamped = r_rel_init.as_rotvec() * (MAX_TILT_RAD / tilt_angle)
                self.target_rot = R.from_rotvec(rotvec_clamped) * r_init
                tilt_angle = MAX_TILT_RAD
            tilt_angle_deg = math.degrees(tilt_angle)

            # 4. Decoupled safety tether (lead-angle limiter relative to live physical pose)
            if self.USE_SAFETY_TETHER and not (hasattr(self, 'all_wiped') and self.all_wiped):
                r_rel_live = self.target_rot * current_rot.inv()
                live_lead_angle = r_rel_live.magnitude()
                MAX_LEAD_RAD = 0.35 # 20 degrees max lead angle over physical wrist
                if live_lead_angle > MAX_LEAD_RAD:
                    rotvec_live_clamped = r_rel_live.as_rotvec() * (MAX_LEAD_RAD / live_lead_angle)
                    self.target_rot = R.from_rotvec(rotvec_live_clamped) * current_rot

            new_quat = self.target_rot.as_quat().astype(np.float64)

        # 7. EMA smoothing — damp 10Hz+ policy oscillations before sending
        if self.ema_pos is None:
            # First call: warm-start all filters from current values
            self.ema_pos    = new_pos.copy()
            self.ema_quat   = new_quat.copy()
            self.ema_Kp     = K_p.copy()
            self.ema_Kd     = K_d.copy()
            self.ema_kp_ori = kp_ori.copy()
            self.ema_kd_ori = kd_ori.copy()
        else:
            self.ema_pos    = self.ALPHA_POS  * new_pos  + (1 - self.ALPHA_POS)  * self.ema_pos
            # Quaternion EMA: flip sign if dot product is negative to avoid slerp wraparound
            if np.dot(new_quat, self.ema_quat) < 0:
                new_quat = -new_quat
            self.ema_quat   = self.ALPHA_ORI  * new_quat + (1 - self.ALPHA_ORI)  * self.ema_quat
            self.ema_quat  /= np.linalg.norm(self.ema_quat)  # renormalise
            self.ema_Kp     = self.ALPHA_STIF * K_p     + (1 - self.ALPHA_STIF) * self.ema_Kp
            self.ema_Kd     = self.ALPHA_STIF * K_d     + (1 - self.ALPHA_STIF) * self.ema_Kd
            self.ema_kp_ori = self.ALPHA_STIF * kp_ori  + (1 - self.ALPHA_STIF) * self.ema_kp_ori
            self.ema_kd_ori = self.ALPHA_STIF * kd_ori  + (1 - self.ALPHA_STIF) * self.ema_kd_ori

        # 8. Track ICRA Evaluation Metrics
        self.track_metrics(K_p, getattr(self, 'latest_fz_current', 0.0))

        # 9. Pack and Publish 31-D Payload (using EMA-smoothed values)
        payload = np.concatenate([
            self.ema_pos,           # 3
            self.ema_quat,          # 4
            self.ema_Kp.flatten(),  # 9
            self.ema_Kd.flatten(),  # 9
            self.ema_kp_ori,        # 3
            self.ema_kd_ori         # 3
        ]).astype(np.float64)

        msg = Float64MultiArray()
        msg.data = payload.tolist()
        self.cmd_pub.publish(msg)
        
        self.get_logger().info(
            f"\n--- ACTION DIAGNOSTICS ---"
            f"\nRaw delta pos: {pos_delta}"
            f"\nSafe target pos: {new_pos}"
            f"\nTarget orientation (quat): {new_quat}"
            f"\nTarget tilt angle from normal: {tilt_angle_deg:.2f} deg"
            f"\nStiffness (Kp_pos diag): {[K_p[0,0], K_p[1,1], K_p[2,2]]}"
            f"\nStiffness (Kp_ori): {[kp_ori[0], kp_ori[1], kp_ori[2]]}"
            f"\n--------------------------"
        )

    def track_metrics(self, K_p, fz_current):
        """
        Calculates and logs full ICRA metrics:
         1. Dual-Tier Force Violations (>10N safety, 5-10N optimal window)
         2. Stiffness Ellipsoid Volume: (4/3)*pi*sqrt(det(Kp))
         3. Anisotropy / Condition Number: lambda_max / lambda_min
         4. Physical SPD Validity: min_eigenvalue > 0 and symmetric
         5. AIRM Riemannian Jerk: || logm(Kp_{t-1}^{-1/2} Kp_t Kp_{t-1}^{-1/2}) ||_F
        """
        # --- 1. Force tracking ---
        if fz_current > 2.0:
            self.contact_step_count += 1
            self.fz_history.append(fz_current)
            if 5.0 <= fz_current <= 10.0:
                self.optimal_force_count += 1
            if fz_current > 10.0:
                self.safety_violation_count += 1
            if fz_current > 15.0:
                self.hardware_violation_count += 1

        # --- 2. Stiffness Ellipsoid & Riemannian metrics ---
        try:
            eigvals = np.linalg.eigvalsh(K_p)
            eig_min = float(eigvals[0])
            eig_max = float(eigvals[-1])
            is_symmetric = bool(np.allclose(K_p, K_p.T, atol=1e-5))
            is_spd = (eig_min > 0.0) and is_symmetric

            if is_spd:
                self.kp_spd_valid_count += 1

            det_kp = float(np.linalg.det(K_p))
            vol = (4.0 / 3.0) * math.pi * math.sqrt(max(1e-9, det_kp))
            anisotropy = eig_max / max(1e-6, eig_min)

            self.kp_volume_history.append(vol)
            self.kp_anisotropy_history.append(anisotropy)

            # --- 3. AIRM Riemannian Jerk (Exact closed-form SPD eigenvalue formulation) ---
            if self.prev_Kp_mat is not None and is_spd:
                try:
                    A_sqrt_inv = inv(sqrtm(self.prev_Kp_mat))
                    M = A_sqrt_inv @ K_p @ A_sqrt_inv
                    eig_M = np.linalg.eigvalsh(M)
                    eig_M = np.maximum(1e-9, eig_M)
                    airm_dist = float(np.sqrt(np.sum(np.log(eig_M) ** 2)))
                    self.airm_jerk_history.append(airm_dist)
                except Exception:
                    pass
            self.prev_Kp_mat = K_p.copy()
        except Exception as e:
            pass

    def export_icra_metrics(self):
        """
        Computes aggregate episode metrics upon task completion (all 5 markers wiped).
        Saves formatted .log report and .json metrics into:
          franka_experiments_metrics/baseline/  or  franka_experiments_metrics/spd/
        """
        self.metrics_exported = True
        duration_sec = (self.get_clock().now() - self.start_time_sec).nanoseconds / 1e9 if self.start_time_sec else 0.0
        
        contact_steps = max(1, self.contact_step_count)
        fz_mean = float(np.mean(self.fz_history)) if self.fz_history else 0.0
        fz_peak = float(np.max(self.fz_history)) if self.fz_history else 0.0
        
        opt_pct = (self.optimal_force_count / contact_steps) * 100.0
        viol_10n_pct = (self.safety_violation_count / contact_steps) * 100.0
        viol_15n_pct = (self.hardware_violation_count / contact_steps) * 100.0
        
        total_eval_steps = max(1, len(self.kp_volume_history))
        spd_rate = (self.kp_spd_valid_count / total_eval_steps) * 100.0
        mean_vol = float(np.mean(self.kp_volume_history)) if self.kp_volume_history else 0.0
        mean_aniso = float(np.mean(self.kp_anisotropy_history)) if self.kp_anisotropy_history else 0.0
        mean_airm = float(np.mean(self.airm_jerk_history)) if self.airm_jerk_history else 0.0

        config_str = getattr(self, 'config_name', sys.argv[1] if len(sys.argv) > 1 else "SPD_DR")
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Determine target metric subfolder (baseline vs spd)
        if "SPD" in config_str.upper() or getattr(self, 'prior_dim', 6) == 9:
            subfolder = "spd"
        else:
            subfolder = "baseline"

        target_dir = os.path.join("franka_experiments_metrics", subfolder)
        os.makedirs(target_dir, exist_ok=True)

        metrics_summary = {
            "config": config_str,
            "obs_mode": getattr(self, 'obs_mode', 'UNKNOWN'),
            "method": subfolder.upper(),
            "timestamp": timestamp_str,
            "task_duration_sec": round(duration_sec, 2),
            "active_contact_step_count": self.contact_step_count,
            "contact_force_threshold_N": 3.0,
            "force_metrics": {
                "Fz_mean_N": round(fz_mean, 2),
                "Fz_peak_N": round(fz_peak, 2),
                "optimal_force_window_5_to_10N_pct": round(opt_pct, 1),
                "safety_violations_gt_10N_pct": round(viol_10n_pct, 1),
                "hardware_violations_gt_15N_pct": round(viol_15n_pct, 1),
            },
            "impedance_metrics": {
                "spd_validity_rate_pct": round(spd_rate, 1),
                "mean_ellipsoid_volume": round(mean_vol, 3),
                "mean_anisotropy_condition_number": round(mean_aniso, 2),
                "mean_airm_riemannian_jerk": round(mean_airm, 4),
            }
        }

        # Format ASCII Evaluation Table Report
        report_table = (
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════════╗\n"
            f"║                ICRA REAL-WORLD ROLLOUT METRICS REPORT                ║\n"
            f"╠══════════════════════════════════════════════════════════════════════╣\n"
            f"║ Config: {config_str:<20s}  Method: {subfolder.upper():<8s}  Duration: {duration_sec:5.2f}s  ║\n"
            f"╟──────────────────────────────────────────────────────────────────────╢\n"
            f"║ CONTACT FORCE METRICS (Surface Contact Threshold: > 3.0 N)           ║\n"
            f"║   • Mean Normal Force (Fz_mean)     : {fz_mean:6.2f} N                      ║\n"
            f"║   • Peak Normal Force (Fz_peak)     : {fz_peak:6.2f} N                      ║\n"
            f"║   • Optimal Window (5N <= Fz <= 10N): {opt_pct:6.1f} %                      ║\n"
            f"║   • Safety Violations  (Fz > 10N)   : {viol_10n_pct:6.1f} %                      ║\n"
            f"║   • Hardware Violations(Fz > 15N)   : {viol_15n_pct:6.1f} %                      ║\n"
            f"╟──────────────────────────────────────────────────────────────────────╢\n"
            f"║ STIFFNESS / IMPEDANCE METRICS                                        ║\n"
            f"║   • Physical SPD Validity Rate      : {spd_rate:6.1f} %                      ║\n"
            f"║   • Mean Stiffness Ellipsoid Volume : {mean_vol:6.3f}                        ║\n"
            f"║   • Mean Anisotropy (Cond Number k) : {mean_aniso:6.2f}                        ║\n"
            f"║   • Mean Riemannian Jerk (AIRM d)   : {mean_airm:6.4f}                        ║\n"
            f"╚══════════════════════════════════════════════════════════════════════╝\n"
        )

        # Print to console
        self.get_logger().info(report_table)

        # 1. Save formatted ASCII report log file
        log_filepath = os.path.join(target_dir, f"rollout_{config_str}_{timestamp_str}.log")
        with open(log_filepath, "w") as f:
            f.write(f"TIMESTAMP: {timestamp_str}\n")
            f.write(report_table)
            f.write("\nRAW METRICS JSON:\n")
            f.write(json.dumps(metrics_summary, indent=4))
        self.get_logger().info(f"💾 Log report saved to {log_filepath}")

        # 2. Save structured JSON metrics file
        json_filepath = os.path.join(target_dir, f"rollout_{config_str}_{timestamp_str}.json")
        with open(json_filepath, "w") as f:
            json.dump(metrics_summary, f, indent=4)
        self.get_logger().info(f"💾 JSON metrics saved to {json_filepath}")

def main(args=None):
    rclpy.init(args=args)
    node = RLEnvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
