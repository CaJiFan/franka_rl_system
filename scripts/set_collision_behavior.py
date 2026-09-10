#!/usr/bin/env python3
"""
Utility script to configure Franka collision thresholds via ROS 2 service.
Matches the wiping/contact-task configuration used in the lab (allowing up to 85N contact force
while protecting the wrist joints 5-7 at 11 Nm).
"""

import sys
import argparse
import rclpy
from rclpy.node import Node
from franka_msgs.srv import SetFullCollisionBehavior

# Factory / ROS 2 Default Thresholds (from default_robot_behavior_utils.hpp)
DEFAULT_CONFIG = {
    "lower_torque_nominal": [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0],
    "upper_torque_nominal": [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0],
    "lower_torque_accel":   [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0],
    "upper_torque_accel":   [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0],
    "lower_force_nominal":  [30.0, 30.0, 30.0, 25.0, 25.0, 25.0],
    "upper_force_nominal":  [40.0, 40.0, 40.0, 35.0, 35.0, 35.0],
    "lower_force_accel":    [30.0, 30.0, 30.0, 25.0, 25.0, 25.0],
    "upper_force_accel":    [40.0, 40.0, 40.0, 35.0, 35.0, 35.0],
}

# Recommended RL Wiping Configuration:
# - Raised Cartesian pushing forces (85N) and base torques (85Nm) to allow firm wiping contact
# - Preserves Franka's factory wrist torque headroom (J5-J7 at 29, 27, 24 Nm; Moments at 35 Nm)
#   so the policy's orientation/tilting freedom is not restricted by false collision stops.
RL_WIPING_CONFIG = {
    "lower_torque_nominal": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "upper_torque_nominal": [85.0, 85.0, 85.0, 85.0, 29.0, 27.0, 24.0],
    "lower_torque_accel":   [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    "upper_torque_accel":   [85.0, 85.0, 85.0, 85.0, 29.0, 27.0, 24.0],
    "lower_force_nominal":  [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "upper_force_nominal":  [85.0, 85.0, 85.0, 35.0, 35.0, 35.0],
    "lower_force_accel":    [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    "upper_force_accel":    [85.0, 85.0, 85.0, 35.0, 35.0, 35.0],
}

# Lab Mate Teleoperation Configuration (strict 11Nm wrist protection for teleop)
LAB_MATE_CONFIG = {
    "lower_torque_nominal": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "upper_torque_nominal": [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0],
    "lower_torque_accel":   [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    "upper_torque_accel":   [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0],
    "lower_force_nominal":  [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "upper_force_nominal":  [85.0, 85.0, 85.0, 11.0, 11.0, 11.0],
    "lower_force_accel":    [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    "upper_force_accel":    [85.0, 85.0, 85.0, 11.0, 11.0, 11.0],
}


def set_collision_behavior(node, config, service_name="/service_server/set_full_collision_behavior"):
    client = node.create_client(SetFullCollisionBehavior, service_name)
    
    node.get_logger().info(f"Waiting for '{service_name}'...")
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().error(f"Service '{service_name}' not found! Is the Franka hardware bringup running?")
        return False

    req = SetFullCollisionBehavior.Request()
    req.lower_torque_thresholds_nominal = [float(v) for v in config["lower_torque_nominal"]]
    req.upper_torque_thresholds_nominal = [float(v) for v in config["upper_torque_nominal"]]
    req.lower_torque_thresholds_acceleration = [float(v) for v in config["lower_torque_accel"]]
    req.upper_torque_thresholds_acceleration = [float(v) for v in config["upper_torque_accel"]]
    req.lower_force_thresholds_nominal = [float(v) for v in config["lower_force_nominal"]]
    req.upper_force_thresholds_nominal = [float(v) for v in config["upper_force_nominal"]]
    req.lower_force_thresholds_acceleration = [float(v) for v in config["lower_force_accel"]]
    req.upper_force_thresholds_acceleration = [float(v) for v in config["upper_force_accel"]]

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    res = future.result()
    if res is not None and res.success:
        node.get_logger().info(" Successfully updated Franka collision thresholds!")
        return True
    else:
        err_msg = res.error if res is not None else "Service call timed out or failed"
        node.get_logger().error(f"❌ Failed to set collision behavior: {err_msg}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Set Franka collision thresholds")
    parser.add_argument("--default", action="store_true", help="Restore factory/ROS 2 default thresholds")
    parser.add_argument("--lab-mate", action="store_true", help="Use lab mate's teleop config (11Nm wrist clamping)")
    parser.add_argument("--service", type=str, default="/service_server/set_full_collision_behavior", help="Service name")
    args = parser.parse_args()

    if args.default:
        cfg = DEFAULT_CONFIG
        mode_name = "DEFAULT (Franka factory/ROS 2 defaults: 40N force, 35Nm moments)"
    elif args.lab_mate:
        cfg = LAB_MATE_CONFIG
        mode_name = "LAB MATE TELEOP (85N base, 11Nm wrist clamp)"
    else:
        cfg = RL_WIPING_CONFIG
        mode_name = "RECOMMENDED RL WIPING (85N contact force, full factory wrist freedom: 29/27/24 Nm)"

    rclpy.init()
    node = rclpy.create_node("franka_threshold_setter")
    node.get_logger().info(f"Applying mode: {mode_name}")
    
    success = set_collision_behavior(node, cfg, service_name=args.service)
    
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
