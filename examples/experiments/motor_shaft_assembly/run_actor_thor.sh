#!/bin/bash
# Peg-in-Hole Actor (runs on Jetson Thor)
# Usage: bash run_actor_thor.sh <LEARNER_IP>

LEARNER_IP="${1:-192.168.1.100}"

cd /home/tophand/workspaces/hil-serl
source .venv/bin/activate

# Start Marvin robot server in background
python serl_robot_infra/robot_servers/marvin_server.py \
    --robot_ip=192.168.1.190 \
    --arm=A \
    --flask_url=127.0.0.1 \
    --flask_port=5000 &
MARVIN_PID=$!
sleep 3

echo "Marvin server started (PID: $MARVIN_PID)"
echo "Connecting to Learner at $LEARNER_IP"

python examples/train_rlpd.py \
    --actor \
    --exp_name=motor_shaft_assembly \
    --ip=$LEARNER_IP \
    --checkpoint_path=examples/experiments/motor_shaft_assembly/checkpoints

# Cleanup
kill $MARVIN_PID 2>/dev/null
