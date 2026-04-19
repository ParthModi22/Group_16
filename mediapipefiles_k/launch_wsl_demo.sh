#!/bin/bash
# A quick launch script to run RViz2 and the Robot State Publisher

echo "Starting WSL Hackathon Demo Visualization..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. Start the Robot State Publisher in the background
# This converts /joint_states coming from wsl_bridge.py into 3D TF frames for RViz
echo "Publishing URDF state from: $(pwd)/softbot.urdf"
URDF_CONTENT=$(cat softbot.urdf)
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$URDF_CONTENT" \
  -p use_sim_time:=false &

# 2. Wait a bit so the publisher starts
sleep 3

# 3. Start RViz2 so we can see the humanoid moving live
echo "Launching RViz2..."
rviz2 &

# Keep the script running
wait
