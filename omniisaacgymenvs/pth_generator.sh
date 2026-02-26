#!/bin/bash

# Set the base path for the script location
PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh
BASE_PATH="$HOME/RANS-ASV-IROS2024/omniisaacgymenvs"

# Function to run training
run_training() {
    echo "Starting training with config: $1 and experiment name: $2"
    $PYTHON_PATH scripts/rlgames_train.py task=USV/batch/$1 train=USV/USV_PPOcontinuous_MLP headless=True enable_livestream=True wandb_activate=True experiment=$2 seed=$3
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
run_training "USV_Virtual_GoToXY_SysID-DR10" "Capture_SysID-DR10-TEST44" "44"

echo "All training sessions completed."