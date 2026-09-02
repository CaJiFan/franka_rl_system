#include <franka_example_controllers/riemannian_impedance_controller.hpp>
#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka/robot_state.h>

namespace franka_example_controllers {

controller_interface::CallbackReturn RiemannianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration RiemannianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= k_num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration RiemannianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= k_num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  config.names.push_back(arm_id_ + "/" + state_interface_name_);
  config.names.push_back(arm_id_ + "/" + robot_model_interface_name_);
  return config;
}

controller_interface::CallbackReturn RiemannianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  
  auto qos = rclcpp::SystemDefaultsQoS();
  qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  
  sub_impedance_cmd_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/impedance_cmd", qos, 
      std::bind(&RiemannianImpedanceController::payloadCallback, this, std::placeholders::_1));

  // Initialize safe defaults
  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  K_p_pos_.setIdentity(); K_p_pos_ *= 100.0;
  K_d_pos_.setIdentity(); K_d_pos_ *= 2.0 * std::sqrt(100.0);
  K_p_ori_.setOnes(); K_p_ori_ *= 10.0;
  K_d_ori_.setOnes(); K_d_ori_ *= 2.0 * std::sqrt(10.0);

  return controller_interface::CallbackReturn::SUCCESS;
}

void RiemannianImpedanceController::payloadCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  if (msg->data.size() != 31) {
    RCLCPP_ERROR(get_node()->get_logger(), "Expected 31 floats in payload, got %zu", msg->data.size());
    return;
  }
  // 3 pos
  position_d_ << msg->data[0], msg->data[1], msg->data[2];
  // 4 quat (x, y, z, w)
  orientation_d_.coeffs() << msg->data[3], msg->data[4], msg->data[5], msg->data[6];
  
  // 9 Kp_pos (Flattened row-major from Python)
  K_p_pos_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&msg->data[7]);
  // 9 Kd_pos
  K_d_pos_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&msg->data[16]);
  
  // 3 Kp_ori
  K_p_ori_ << msg->data[25], msg->data[26], msg->data[27];
  // 3 Kd_ori
  K_d_ori_ << msg->data[28], msg->data[29], msg->data[30];

  // Dead-man's switch: record time of last valid command
  last_cmd_time_ = get_node()->get_clock()->now();
  cmd_received_  = true;
}

controller_interface::CallbackReturn RiemannianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + robot_model_interface_name_,
                                                   arm_id_ + "/" + state_interface_name_));
                                                   
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  
  // Read current pose to prevent jumps on activation
  std::array<double, 16> initial_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose.data()));
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn RiemannianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->release_interfaces();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type RiemannianImpedanceController::update(
    const rclcpp::Time& time, const rclcpp::Duration& /*period*/) {

  // ── Dead-man's switch ────────────────────────────────────────────────────
  // If the RL publisher dies (Ctrl+C), freeze target at current robot pose
  // to prevent continued tracking of a stale oscillating target.
  if (cmd_received_) {
    double elapsed = (time - last_cmd_time_).seconds();
    if (elapsed > CMD_TIMEOUT_SEC) {
      // Snap target to live robot pose so error_pos / error_ori -> 0
      std::array<double, 16> live_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
      Eigen::Affine3d live_tf(Eigen::Matrix4d::Map(live_pose.data()));
      position_d_    = live_tf.translation();
      orientation_d_ = Eigen::Quaterniond(live_tf.rotation());
      // Log once per second at most
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "[RiemannianImpedanceController] No cmd for %.2f s — holding current pose.", elapsed);
    }
  }

  std::array<double, 42> jacobian_array = franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  
  Eigen::Matrix<double, 7, 1> dq;
  for (int i = 0; i < 7; ++i) {
    dq(i) = state_interfaces_.at(2 * i + 1).get_value();
  }
  
  std::array<double, 16> pose_array = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose_array.data()));
  Eigen::Vector3d position = transform.translation();
  Eigen::Quaterniond orientation(transform.rotation());

  // Error math (match Robosuite / Franka OSC logic)
  Eigen::Vector3d error_pos = position_d_ - position;
  
  Eigen::Vector3d error_ori;
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error_ori = error_quaternion.vec(); 
  error_ori = transform.rotation() * error_ori; // Transform error to base frame

  // Velocity error
  Eigen::Matrix<double, 6, 1> velocity = jacobian * dq;
  Eigen::Vector3d error_vel_pos = -velocity.head(3); // Desired velocity is 0
  Eigen::Vector3d error_vel_ori = -velocity.tail(3);

  // CRITICAL: DENSE MATRIX MULTIPLICATION FOR POSITIONAL STIFFNESS!
  Eigen::Vector3d force_cmd = K_p_pos_ * error_pos + K_d_pos_ * error_vel_pos;
  
  // ORIENTATION STIFFNESS (Element-wise multiplication using arrays)
  Eigen::Vector3d torque_cmd = (error_ori.array() * K_p_ori_.array()) + (error_vel_ori.array() * K_d_ori_.array());

  Eigen::Matrix<double, 6, 1> wrench_cmd;
  wrench_cmd.head(3) << force_cmd;
  wrench_cmd.tail(3) << torque_cmd;

  // Compute joint torques
  Vector7d tau_cmd = jacobian.transpose() * wrench_cmd;

  for (size_t i = 0; i < 7; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }
  
  return controller_interface::return_type::OK;
}

} // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::RiemannianImpedanceController,
                       controller_interface::ControllerInterface)
