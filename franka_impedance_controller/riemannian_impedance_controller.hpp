#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <franka_semantic_components/franka_robot_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <realtime_tools/realtime_buffer.hpp>

namespace franka_example_controllers {

struct ImpedanceCommand {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  Eigen::Matrix3d K_p_pos{Eigen::Matrix3d::Identity() * 100.0};
  Eigen::Matrix3d K_d_pos{Eigen::Matrix3d::Identity() * 20.0};
  Eigen::Vector3d K_p_ori{Eigen::Vector3d::Constant(10.0)};
  Eigen::Vector3d K_d_ori{Eigen::Vector3d::Constant(6.32)};
};

class RiemannianImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string arm_id_;
  const int k_num_joints = 7;
  
  // Fat Payload Subscription
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_impedance_cmd_;
  
  // Real-Time Safe Lock-Free Command Buffer
  realtime_tools::RealtimeBuffer<ImpedanceCommand> cmd_buffer_;

  // State
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Matrix3d K_p_pos_;
  Eigen::Matrix3d K_d_pos_;
  Eigen::Vector3d K_p_ori_;
  Eigen::Vector3d K_d_ori_;

  // Previous torque for slew-rate limiting (max 1.0 Nm / ms)
  Vector7d tau_prev_{Vector7d::Zero()};

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  const std::string state_interface_name_{"robot_state"};
  const std::string robot_model_interface_name_{"robot_model"};

  // Dead-man's switch: freeze target if publisher stops sending
  rclcpp::Time last_cmd_time_;
  bool cmd_received_{false};          // true after first message
  const double CMD_TIMEOUT_SEC{0.5};  // freeze after 500 ms of silence

  void payloadCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
};

}  // namespace franka_example_controllers
