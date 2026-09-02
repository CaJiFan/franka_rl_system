#!/bin/bash

source /home/userlab/cjimenez/ros2_ws/install/setup.bash

ros2 control load_controller riemannian_impedance_controller
ros2 control set_controller_state riemannian_impedance_controller inactive
ros2 control set_controller_state riemannian_impedance_controller active
