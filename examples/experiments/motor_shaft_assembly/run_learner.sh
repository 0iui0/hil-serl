#!/bin/bash
# Peg-in-Hole Learner (runs on training machine with 2x RTX 5090)
# Usage: bash run_learner.sh

cd "$(dirname "$0")/../../.."
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python examples/train_rlpd.py \
    --learner \
    --exp_name=motor_shaft_assembly \
    --ip=0.0.0.0 \
    --checkpoint_path=experiments/motor_shaft_assembly/checkpoints \
    --demo_path=experiments/motor_shaft_assembly/demos.pkl
