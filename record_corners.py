import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class CornerRecorder(Node):
    def __init__(self):
        super().__init__('corner_recorder')
        self.sub = self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.pose_cb, 10)
        self.latest_pose = None
        self.tcp_offset = np.array([0.0, 0.0, 0.185]) # 18.5cm eraser offset along wrist Z-axis

    def pose_cb(self, msg):
        self.latest_pose = msg

    def record(self):
        corners = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
        recorded_corners = {}
        
        print("\n==================================================")
        print("Whiteboard 3D Corner Recorder")
        print("==================================================")
        print("Instructions:")
        print("1. Set the robot to gravity compensation / manual guidance mode.")
        print("2. Move the eraser tip to touch the target corner.")
        print("3. Press Enter to record the coordinate.\n")
        
        for corner in corners:
            input(f"--> Move eraser to {corner} and press Enter to record...")
            
            # Spin once to get the latest pose
            rclpy.spin_once(self, timeout_sec=1.0)
            
            if self.latest_pose is None:
                print("Error: Did not receive pose from /franka_robot_state_broadcaster/current_pose!")
                print("Make sure the robot controller is running.")
                return
                
            pos = self.latest_pose.pose.position
            quat = self.latest_pose.pose.orientation
            
            eef_pos = np.array([pos.x, pos.y, pos.z])
            eef_quat = np.array([quat.x, quat.y, quat.z, quat.w])
            
            # Compute eraser tip position using the TCP offset
            r_curr = R.from_quat(eef_quat)
            eef_pos_eraser = eef_pos + r_curr.apply(self.tcp_offset)
            
            print(f"Recorded {corner}:")
            print(f"  Raw Flange: {eef_pos}")
            print(f"  Eraser Tip: {eef_pos_eraser}\n")
            recorded_corners[corner.split(" ")[0]] = eef_pos_eraser
            
        print("==================================================")
        print("Copy and paste this dictionary into your scripts:")
        print("==================================================")
        print("ROBOT_CORNERS_BASE = {")
        for name, coords in recorded_corners.items():
            print(f"    \"{name}\": np.array([{coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}]),")
        print("}")
        print("==================================================")

def main():
    rclpy.init()
    recorder = CornerRecorder()
    try:
        recorder.record()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
