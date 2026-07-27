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

try:
    from stable_baselines3 import SAC
except ImportError:
    print("Warning: stable_baselines3 not found. Local inference will use a dummy random policy.")

from std_msgs.msg import Float32MultiArray
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
            self.model_path = "sac_wipe_policy.zip" # Update with your actual checkpoint path!
            self.get_logger().info(f"Loading SAC policy from {self.model_path} onto CPU...")
            try:
                self.model = SAC.load(self.model_path, device='cpu')
                self.get_logger().info("Model loaded successfully!")
            except Exception as e:
                self.get_logger().warn(f"Could not load model: {e}")
                self.get_logger().warn("Using a dummy random policy for now.")
                self.model = None

        # --- State Caches ---
        self.latest_joint_state = None
        self.latest_eef_pose = None
        self.latest_wrench = None
        self.latest_wipe_state = None

        # --- Subscribers ---
        # Subscribe to standard ROS2 topics (adjust topic names based on your Franka setup)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(PoseStamped, '/franka/eef_pose', self.eef_pose_cb, 10)
        self.create_subscription(WrenchStamped, '/franka/force_torque_ext', self.wrench_cb, 10)
        
        # Subscribe to our custom Vision Node
        self.create_subscription(Float32MultiArray, '/wipe_state', self.vision_cb, 10)

        # --- Publishers ---
        # TODO: Adjust this to match your actual robot controller (e.g., cartesian velocity)
        self.cmd_pub = self.create_publisher(TwistStamped, '/franka/cmd_vel', 10)

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

        # 1. Construct the observation dictionary
        obs_dict = self.build_obs_dict()

        # 2. Flatten into the exact 1D array order the agent expects
        flat_obs = np.concatenate([obs_dict[k] for k in OBS_KEYS]).astype(np.float32)

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
                # Dummy random action if no model is loaded (6D cartesian velocity)
                action = np.random.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)

        # 4. Publish the action to the robot
        self.publish_action(action)

    def build_obs_dict(self):
        """
        Extracts raw ROS2 messages and formats them into a dictionary matching Robosuite.
        """
        # Robot Kinematics
        # Assume first 7 are arm joints
        joint_pos = np.array(self.latest_joint_state.position[:7])
        joint_vel = np.array(self.latest_joint_state.velocity[:7])

        # End Effector Pose
        pos = self.latest_eef_pose.pose.position
        quat = self.latest_eef_pose.pose.orientation
        eef_pos = np.array([pos.x, pos.y, pos.z])
        eef_quat = np.array([quat.x, quat.y, quat.z, quat.w])

        # Contact Flag (Thresholding Z-axis force)
        fz = self.latest_wrench.wrench.force.z
        robot0_contact = np.array([1.0 if abs(fz) > 5.0 else 0.0]) # 5 Newtons threshold

        # Vision State
        # vision_node publishes: [cent_x, cent_y, cent_z, radius, proportion_wiped, eef2c_x, eef2c_y, eef2c_z]
        v_data = self.latest_wipe_state.data
        wipe_centroid = np.array(v_data[0:3])
        proportion_wiped = np.array([v_data[4]])
        
        # Calculate dynamic EEF to Centroid vector!
        gripper_to_wipe_centroid = wipe_centroid - eef_pos

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "proportion_wiped": proportion_wiped,
            "gripper_to_wipe_centroid": gripper_to_wipe_centroid,
            "robot0_contact": robot0_contact
        }

    def publish_action(self, action):
        """
        Converts the 6D SAC action (e.g. [vx, vy, vz, wx, wy, wz]) into a ROS2 command.
        """
        # Example for Cartesian Velocity control using a Twist message:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(action[0])
        msg.twist.linear.y = float(action[1])
        msg.twist.linear.z = float(action[2])
        msg.twist.angular.x = float(action[3])
        msg.twist.angular.y = float(action[4])
        msg.twist.angular.z = float(action[5])
        
        self.cmd_pub.publish(msg)

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
