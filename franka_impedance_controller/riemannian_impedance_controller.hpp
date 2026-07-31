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

namespace franka_example_controllers {

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
  
  // State
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Matrix3d K_p_pos_;
  Eigen::Matrix3d K_d_pos_;
  Eigen::Vector3d K_p_ori_;
  Eigen::Vector3d K_d_ori_;

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  const std::string state_interface_name_{"robot_state"};
  const std::string robot_model_interface_name_{"robot_model"};

  void payloadCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
};

}  // namespace franka_example_controllers
