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
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
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
        self.POSITION_ONLY_MODE = True
        self.FIXED_KP  = 150.0   # N/m  (translational)
        self.FIXED_KP_ORI = 50.0 # N·m/rad (rotational, ~10% of translational)

        # Fix A: Hold current orientation and ignore policy orientation commands to eliminate wrist spin
        self.POSITION_ONLY_ORI = True


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

        # ── EMA smoothing state (Fix 1 & Fix B: damp high-freq policy oscillations) ──
        # Lower alpha = heavier smoothing, Higher alpha = more responsive
        self.ALPHA_POS  = 0.40   # position target (Fix B: increased from 0.15 to allow faster position tracking)
        self.ALPHA_ORI  = 0.20   # orientation target (SLERP-like approximation)
        self.ALPHA_STIF = 0.10   # stiffness matrices (slowest — avoids torque spikes)
        self.ema_pos    = None   # np.ndarray(3,)  — initialised on first action
        self.ema_quat   = None   # np.ndarray(4,)  — [x,y,z,w]
        self.ema_Kp     = None   # np.ndarray(3,3)
        self.ema_Kd     = None   # np.ndarray(3,3)
        self.ema_kp_ori = None   # np.ndarray(3,)
        self.ema_kd_ori = None   # np.ndarray(3,)

        # Integrated position target (accumulates delta to overcome static joint friction)
        self.target_pos = None

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

        # ── Contact & F/T ────────────────────────────────────────────────────
        force  = self.latest_wrench.wrench.force
        torque = self.latest_wrench.wrench.torque
        robot0_contact_force  = np.array([force.x,  force.y,  force.z],  dtype=np.float32)

        # Contact: threshold on Z-axis normal force (|Fz| > 2 N), matching sim
        robot0_contact = np.array(
            [1.0 if abs(robot0_contact_force[2]) > 2.0 else 0.0], dtype=np.float32
        )

        # ── Vision: marker states from /wipe_markers (20 floats) ─────────────
        v_data = list(self.latest_wipe_markers.data)  # 20 floats
        marker_pos   = [np.array(v_data[i*4 : i*4+3], dtype=np.float32) for i in range(5)]
        marker_wiped = [np.array([v_data[i*4+3]],      dtype=np.float32) for i in range(5)]

        # gripper-to-marker vectors
        gripper_to_marker = [mp - eef_pos_eraser for mp in marker_pos]

        # active waypoint: first non-wiped marker
        active_idx = next((i for i, w in enumerate(marker_wiped) if w[0] < 0.5), 4)
        gripper_to_active_waypoint = gripper_to_marker[active_idx].copy()

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
            f"  [46]     contact        = {robot0_contact[0]:.0f}  (|Fz|>2N)\n"
            f"  [47:81]  markers (active={active_idx}):{m_lines}\n"
            f"  [82:84]  g2active_wp    = {_fmt(gripper_to_active_waypoint)}\n"
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
        min_kp = 1.0
        max_kp = 160.0 # Clean, safe stiffness limit

        if self.prior_dim == 6:
            # 12D Action space for Baseline (3 diagonal trans stiffness, 3 diagonal rot stiffness, 3 pos delta, 3 ori delta)
            kp_trans_raw = action[:3]
            kp_rot_raw = action[3:6]
            pos_delta = action[6:9]
            ori_delta_vec = action[9:12]

            # Linear decode to physical values
            kp_trans_scaled = min_kp + 0.5 * (kp_trans_raw + 1.0) * (max_kp - min_kp)
            kp_rot_scaled = min_kp + 0.5 * (kp_rot_raw + 1.0) * (max_kp - min_kp)

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
            
            # Diagonals linearly decoded to [min_kp, max_kp] and then log-mapped
            target_physical = min_kp + 0.5 * (mandel_params[:3] + 1.0) * (max_kp - min_kp)
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
            kp_ori = min_kp + 0.5 * (kp_ori_raw + 1.0) * (max_kp - min_kp)
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

        # 5. Position Integration & Safety Workspace (Integrated target with 4cm safety tether)
        # Cap max delta to 1cm per step (0.2 m/s at 20Hz)
        pos_delta_safe = np.clip(pos_delta, -0.01, 0.01) 
        current_pos = self.latest_eef_pose.pose.position
        current_pos_arr = np.array([current_pos.x, current_pos.y, current_pos.z])

        if self.target_pos is None:
            self.target_pos = current_pos_arr.copy() + pos_delta_safe
        else:
            self.target_pos += pos_delta_safe

        # Safety tether: clamp target error to max 4 cm from live robot position
        # (Allows force to scale up to 6 N at Kp=150 N/m to break joint stiction, but prevents runaway)
        err = self.target_pos - current_pos_arr
        dist = np.linalg.norm(err)
        MAX_TETHER = 0.04  # 4 cm
        if dist > MAX_TETHER:
            self.target_pos = current_pos_arr + (err / dist) * MAX_TETHER

        WORKSPACE_LIMITS = {
            "X_MIN": 0.35, "X_MAX": 0.85, 
            "Y_MIN": -0.30, "Y_MAX": 0.30, 
            "Z_MIN": 0.02, "Z_MAX": 0.40   
        }
        self.target_pos[0] = np.clip(self.target_pos[0], WORKSPACE_LIMITS["X_MIN"], WORKSPACE_LIMITS["X_MAX"])
        self.target_pos[1] = np.clip(self.target_pos[1], WORKSPACE_LIMITS["Y_MIN"], WORKSPACE_LIMITS["Y_MAX"])
        self.target_pos[2] = np.clip(self.target_pos[2], WORKSPACE_LIMITS["Z_MIN"], WORKSPACE_LIMITS["Z_MAX"])

        new_pos = self.target_pos.copy()

        # 6. Orientation Integration (Axis-Angle to Quaternion)
        current_quat = self.latest_eef_pose.pose.orientation
        if self.POSITION_ONLY_ORI:
            # Fix A: Hold current orientation to prevent policy-induced wrist spinning
            new_quat = np.array([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        else:
            ori_delta_safe = np.clip(ori_delta_vec, -0.1, 0.1) # Max rotation per step
            current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
            delta_rot = R.from_rotvec(ori_delta_safe)
            new_rot = delta_rot * current_rot # Local frame rotation
            new_quat = new_rot.as_quat() # [x, y, z, w]

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

        # 8. Pack and Publish 31-D Payload (using EMA-smoothed values)
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
            f"\nStiffness (Kp_pos diag): {[K_p[0,0], K_p[1,1], K_p[2,2]]}"
            f"\nStiffness (Kp_ori): {[kp_ori[0], kp_ori[1], kp_ori[2]]}"
            f"\n--------------------------"
        )

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
