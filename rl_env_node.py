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

# Define the exact order of keys your RL agent expects.
# You can easily re-order this array to match Robosuite!
OBS_KEYS = [
    "joint_pos",                # 7D
    "joint_vel",                # 7D
    "eef_pos",                  # 3D
    "eef_quat",                 # 4D
    "proportion_wiped",         # 1D
    "gripper_to_wipe_centroid", # 3D
    "robot0_contact",           # 1D
]

class RLEnvNode(Node):
    def __init__(self):
        super().__init__('rl_env_node')
        
        # --- Execution Mode ---
        # Set to True to send observations to a remote ZMQ server.
        # Set to False to load the model and run inference locally on this machine.
        self.use_zmq = False 
        self.tcp_offset = np.array([0.0, 0.0, 0.185])

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
                if obs_shape == 62:
                    self.prior_dim = 6
                    self.get_logger().info("Detected BASELINE diagonal model (62-D observations)")
                elif obs_shape == 65:
                    self.prior_dim = 9
                    self.get_logger().info("Detected SPD manifold model (65-D observations)")
                else:
                    self.prior_dim = 9 # Fallback
                    self.get_logger().warn(f"Unknown observation space shape {obs_shape}, defaulting prior_dim=9")
            except Exception as e:
                self.get_logger().warn(f"Could not load model: {e}")
                self.get_logger().warn("Using a dummy random policy for now.")
                self.model = None
                self.prior_dim = 9 # Default fallback

        # --- State Caches ---
        self.latest_joint_state = None
        self.latest_eef_pose = None
        self.latest_wrench = None
        self.latest_wipe_state = None

        # Velocity tracking variables
        self.prev_eef_pos = None
        self.prev_eef_quat = None
        self.last_update_time = None

        # --- Subscribers ---
        # Subscribe to standard ROS2 topics (adjust topic names based on your Franka setup)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.eef_pose_cb, 10)
        self.create_subscription(WrenchStamped, '/franka_robot_state_broadcaster/external_wrench_in_base_frame', self.wrench_cb, 10)
        
        # Subscribe to our custom Vision Node
        self.create_subscription(Float32MultiArray, '/wipe_state', self.vision_cb, 10)

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/riemannian_impedance_controller/impedance_cmd', 10)

        # --- 20Hz Control Timer ---
        self.control_rate = 20.0
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

    # -- Callbacks to update caches --
    def joint_cb(self, msg):
        self.latest_joint_state = msg

    def eef_pose_cb(self, msg):
        self.latest_eef_pose = msg

    def wrench_cb(self, msg):
        self.latest_wrench = msg

    def vision_cb(self, msg):
        self.latest_wipe_state = msg

    # -- Main Loop --
    def control_loop(self):
        # Ensure we have received at least one message from all sensors
        if not (self.latest_joint_state and self.latest_eef_pose and 
                self.latest_wrench and self.latest_wipe_state):
            self.get_logger().warn("Waiting for all sensor topics...", throttle_duration_sec=2.0)
            return

        # Compute delta time for velocity estimation
        now_time = self.get_clock().now()
        dt = 0.05
        if self.last_update_time is not None:
            dt = (now_time - self.last_update_time).nanoseconds / 1e9
            if dt <= 0.0:
                dt = 0.05
        self.last_update_time = now_time

        # Extract values
        joint_pos = np.array(self.latest_joint_state.position[:7], dtype=np.float32)
        joint_vel = np.array(self.latest_joint_state.velocity[:7], dtype=np.float32)

        pos = self.latest_eef_pose.pose.position
        quat = self.latest_eef_pose.pose.orientation
        eef_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        eef_quat = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)

        # Compute position of the eraser tip
        r_curr = R.from_quat(eef_quat)
        eef_pos_eraser = eef_pos + r_curr.apply(self.tcp_offset)

        # Estimate EEF velocities of the eraser tip
        if self.prev_eef_pos is not None:
            eef_vel_lin = (eef_pos_eraser - self.prev_eef_pos) / dt
            r_prev = R.from_quat(self.prev_eef_quat)
            r_diff = r_prev.inv() * r_curr
            eef_vel_ang = r_diff.as_rotvec() / dt
        else:
            eef_vel_lin = np.zeros(3, dtype=np.float32)
            eef_vel_ang = np.zeros(3, dtype=np.float32)
        
        self.prev_eef_pos = eef_pos_eraser.copy()
        self.prev_eef_quat = eef_quat.copy()

        # Contact and force torque
        force = self.latest_wrench.wrench.force
        torque = self.latest_wrench.wrench.torque
        robot0_contact_force = np.array([force.x, force.y, force.z], dtype=np.float32)
        robot0_contact_torque = np.array([torque.x, torque.y, torque.z], dtype=np.float32)
        
        # 5 Newtons contact threshold
        robot0_contact = np.array([1.0 if np.linalg.norm(robot0_contact_force) > 5.0 else 0.0], dtype=np.float32)

        # Vision State
        v_data = self.latest_wipe_state.data
        wipe_centroid = np.array(v_data[0:3], dtype=np.float32)
        proportion_wiped = np.array([v_data[4]], dtype=np.float32)
        
        # Safety check: If no ink is detected, the vision node publishes [0,0,0].
        # We override this to the current eraser tip position to prevent the robot from diving to the base origin.
        if np.all(wipe_centroid == 0.0):
            wipe_centroid = eef_pos_eraser.copy()
            
        gripper_to_wipe_centroid = wipe_centroid - eef_pos_eraser

        # Construct 55-D base observation list in exact insertion order matching Robosuite Wipe env
        obs_list = [
            np.cos(joint_pos),                  # 7
            np.sin(joint_pos),                  # 7
            joint_pos,                          # 7
            joint_vel,                          # 7
            eef_pos_eraser,                     # 3 (TCP position, not flange!)
            eef_quat,                           # 4
            eef_vel_lin,                        # 3
            eef_vel_ang,                        # 3
            robot0_contact,                     # 1
            robot0_contact_force,               # 3
            robot0_contact_torque,              # 3
            wipe_centroid,                      # 3
            proportion_wiped,                   # 1
            gripper_to_wipe_centroid            # 3
        ]

        obs_list_names = [
            "cos(joint_pos)",                     # 7
            "sin(joint_pos)",                     # 7
            "joint_pos",                          # 7
            "joint_vel",                          # 7
            "eef_pos",                            # 3
            "eef_quat",                           # 4
            "eef_vel_lin",                        # 3
            "eef_vel_ang",                        # 3
            "robot0_contact",                     # 1
            "robot0_contact_force",               # 3
            "robot0_contact_torque",              # 3
            "wipe_centroid",                      # 3
            "proportion_wiped",                   # 1
            "gripper_to_wipe_centroid"            # 3
        ]

        for key, value in zip(obs_list_names, obs_list):
            self.get_logger().info(f"{key}: {value}")

        # Add LLM residual prior states (prior_dim for prior, 1 for weight) -> 7-D or 10-D (Total 62-D or 65-D)
        obs_list.append(np.zeros(self.prior_dim, dtype=np.float32))     # current_prior
        obs_list.append(np.array([0.0], dtype=np.float32))              # current_w
 
        # Flatten into the 62D or 65D vector
        flat_obs = np.concatenate(obs_list).astype(np.float32)
 
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
                # Dummy random action if no model is loaded (prior_dim + 6 size)
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

        # 5. Position Integration & Safety Workspace
        # Cap max delta to 1cm per step (0.2 m/s at 20Hz)
        pos_delta_safe = np.clip(pos_delta, -0.01, 0.01) 
        current_pos = self.latest_eef_pose.pose.position
        new_pos = np.array([current_pos.x, current_pos.y, current_pos.z]) + pos_delta_safe

        WORKSPACE_LIMITS = {
            "X_MIN": 0.35, "X_MAX": 0.85, 
            "Y_MIN": -0.30, "Y_MAX": 0.30, 
            "Z_MIN": 0.02, "Z_MAX": 0.40   
        }
        new_pos[0] = np.clip(new_pos[0], WORKSPACE_LIMITS["X_MIN"], WORKSPACE_LIMITS["X_MAX"])
        new_pos[1] = np.clip(new_pos[1], WORKSPACE_LIMITS["Y_MIN"], WORKSPACE_LIMITS["Y_MAX"])
        new_pos[2] = np.clip(new_pos[2], WORKSPACE_LIMITS["Z_MIN"], WORKSPACE_LIMITS["Z_MAX"])

        # 6. Orientation Integration (Axis-Angle to Quaternion)
        ori_delta_safe = np.clip(ori_delta_vec, -0.1, 0.1) # Max rotation per step
        current_quat = self.latest_eef_pose.pose.orientation
        current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        delta_rot = R.from_rotvec(ori_delta_safe)
        new_rot = delta_rot * current_rot # Local frame rotation
        new_quat = new_rot.as_quat() # [x, y, z, w]

        # 7. Pack and Publish 31-D Payload
        payload = np.concatenate([
            new_pos,                # 3
            new_quat,               # 4
            K_p.flatten(),          # 9
            K_d.flatten(),          # 9
            kp_ori,                 # 3
            kd_ori                  # 3
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
