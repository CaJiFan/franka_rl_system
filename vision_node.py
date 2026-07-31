#!/usr/bin/env python3
"""
vision_node.py
--------------
ROS2 Node that wraps the OpenCV vision module.
Runs the UI on a background thread and publishes the detected wipe state.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import threading
import sys

# Import your existing generic CV2 module
import vision_module_cv2

class WipeVisionNode(Node):
    def __init__(self):
        super().__init__('wipe_vision_node')
        
        # Publish the vision state
        # We use a standard Float32MultiArray to avoid compiling custom .msg files.
        self.pub_vision = self.create_publisher(Float32MultiArray, '/wipe_state', 10)
        
        # Subscribe to EEF pose to mask out the robot arm/tool
        self.sub_eef = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.eef_callback,
            10
        )
        
        # Timer to read the shared vision state and publish it at the control rate
        self.timer = self.create_timer(1.0 / vision_module_cv2.CONTROL_HZ, self.timer_callback)
        
        # Launch the OpenCV window loop in a background thread
        # This allows cv2.imshow() to run without blocking the ROS2 spin loop.
        # self.vision_thread = threading.Thread(target=vision_module_cv2.run_vision_loop, daemon=True)
        cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        self.vision_thread = threading.Thread(target=vision_module_cv2.run_vision_loop, kwargs={'cam_index': cam_index}, daemon=True)
        self.vision_thread.start()
        
        self.get_logger().info(f"Wipe Vision Node started on camera {cam_index}.")

    def eef_callback(self, msg):
        pos = msg.pose.position
        quat = msg.pose.orientation
        vision_module_cv2.vision_state.set_eef_pose(
            [pos.x, pos.y, pos.z],
            [quat.x, quat.y, quat.z, quat.w]
        )

    def timer_callback(self):
        # Retrieve the latest observation from the vision module's thread-safe state
        obs_vec = vision_module_cv2.vision_state.get_obs()
        
        # obs_vec is: [cent_x, cent_y, cent_z, radius, proportion_wiped, eef2c_x, eef2c_y, eef2c_z]
        # Note: The eef_to_centroid vector will be 0 here since this node doesn't track EEF.
        # The RL Env node will compute the real EEF-to-centroid vector later.
        
        msg = Float32MultiArray()
        msg.data = obs_vec.tolist()
        self.pub_vision.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WipeVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
