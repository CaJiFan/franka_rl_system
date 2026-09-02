# franka_rl_system


# Terminal 4 (or any terminal with the workspace sourced)
source /opt/ros/humble/setup.bash
source /home/userlab/cjimenez/ros2_ws/install/setup.bash   # loadcj does this

# Load the controller (makes it known to the controller manager)
ros2 control load_controller riemannian_impedance_controller

# Configure and activate it
ros2 control set_controller_state riemannian_impedance_controller configured
ros2 control set_controller_state riemannian_impedance_controller active
