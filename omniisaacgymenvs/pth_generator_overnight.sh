#!/bin/bash

# Set the base path for the script location
PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh
BASE_PATH="$HOME/RANS-ASV-IROS2024/omniisaacgymenvs"

# Function to run training
run_training() {
    echo "Starting training with config: $1 and experiment name: $2"
    $PYTHON_PATH scripts/rlgames_train.py task=USV/batch/$1 train=USV/USV_PPOcontinuous_MLP headless=True enable_livestream=False wandb_activate=True experiment=$2 seed=$3 max_iterations=1000 num_envs=1024
    echo "Training for $2 completed."
}

# Ensure PYTHON_PATH is set
if [[ -z "$PYTHON_PATH" ]]; then
    echo "PYTHON_PATH is not set. Please set PYTHON_PATH to your Python executable."
    exit 1
fi

# Navigate to the base path
cd $BASE_PATH

# Run the training sessions
run_training "USV_Virtual_GoToXY_SysID-DR0" "Capture_SysID-DR0-42" "42"
run_training "USV_Virtual_GoToXY_SysID-DR10" "Capture_SysID-DR10-42" "42"
run_training "USV_Virtual_GoToXY_SysID-DR50" "Capture_SysID-DR50-42" "42"
run_training "USV_Virtual_GoToXY_Nominal-DR0" "Capture_Nominal-DR0-42" "42"
run_training "USV_Virtual_GoToXY_Nominal-DR10" "Capture_Nominal-DR10-42" "42"
run_training "USV_Virtual_GoToXY_Nominal-DR50" "Capture_Nominal-DR50-42" "42"

run_training "USV_Virtual_GoToXY_SysID-DR0" "Capture_SysID-DR0-43" "43"
run_training "USV_Virtual_GoToXY_SysID-DR10" "Capture_SysID-DR10-43" "43"
run_training "USV_Virtual_GoToXY_SysID-DR50" "Capture_SysID-DR50-43" "43"
run_training "USV_Virtual_GoToXY_Nominal-DR0" "Capture_Nominal-DR0-43" "43"
run_training "USV_Virtual_GoToXY_Nominal-DR10" "Capture_Nominal-DR10-43" "43"
run_training "USV_Virtual_GoToXY_Nominal-DR50" "Capture_Nominal-DR50-43" "43"

run_training "USV_Virtual_GoToXY_SysID-DR0" "Capture_SysID-DR0-44" "44"
run_training "USV_Virtual_GoToXY_SysID-DR10" "Capture_SysID-DR10-44" "44"
run_training "USV_Virtual_GoToXY_SysID-DR50" "Capture_SysID-DR50-44" "44"
run_training "USV_Virtual_GoToXY_Nominal-DR0" "Capture_Nominal-DR0-44" "44"
run_training "USV_Virtual_GoToXY_Nominal-DR10" "Capture_Nominal-DR10-44" "44"
run_training "USV_Virtual_GoToXY_Nominal-DR50" "Capture_Nominal-DR50-44" "44"

run_training "USV_Virtual_GoToXY_SysID-DR0" "Capture_SysID-DR0-45" "45"
run_training "USV_Virtual_GoToXY_SysID-DR10" "Capture_SysID-DR10-45" "45"
run_training "USV_Virtual_GoToXY_SysID-DR50" "Capture_SysID-DR50-45" "45"
run_training "USV_Virtual_GoToXY_Nominal-DR0" "Capture_Nominal-DR0-45" "45"
run_training "USV_Virtual_GoToXY_Nominal-DR10" "Capture_Nominal-DR10-45" "45"
run_training "USV_Virtual_GoToXY_Nominal-DR50" "Capture_Nominal-DR50-45" "45"

run_training "USV_Virtual_GoToXY_SysID-DR0" "Capture_SysID-DR0-46" "46"
run_training "USV_Virtual_GoToXY_SysID-DR10" "Capture_SysID-DR10-46" "46"
run_training "USV_Virtual_GoToXY_SysID-DR50" "Capture_SysID-DR50-46" "46"
run_training "USV_Virtual_GoToXY_Nominal-DR0" "Capture_Nominal-DR0-46" "46"
run_training "USV_Virtual_GoToXY_Nominal-DR10" "Capture_Nominal-DR10-46" "46"
run_training "USV_Virtual_GoToXY_Nominal-DR50" "Capture_Nominal-DR50-46" "46"
echo "All training sessions completed."