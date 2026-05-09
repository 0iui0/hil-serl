#!/bin/bash
# Deploy minimal code to Thor for peg-in-hole assembly actor
# Usage: bash thor_deploy.sh

set -e

THOR_USER="tophand"
THOR_HOST="thor"
THOR_DIR="/home/${THOR_USER}/workspaces/hil-serl"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Syncing core code to ${THOR_HOST}:${THOR_DIR} ==="

rsync -avz --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.egg-info/' \
    --exclude='.claude/' \
    --exclude='.vscode/' \
    --exclude='.gitignore' \
    --exclude='checkpoints/' \
    --exclude='videos/' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/kinematicsSDK/' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/contrlSDK/' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/DEMO_C++/' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/DEMO_PYTHON/' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/*.pdf' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/*.md' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/*.sh' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/*.bat' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/.venv/' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/.gitignore' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/README.md' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/LICENSE' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/ccs_m3.MvKDCfg' \
    --exclude='marvin_sdk/TJ_FX_ROBOT_CONTRL_SDK/robot_M3_CCS.ini' \
    "${REPO_DIR}/" "${THOR_USER}@${THOR_HOST}:${THOR_DIR}/"

echo ""
echo "=== Cleaning __pycache__ on Thor ==="
ssh "${THOR_USER}@${THOR_HOST}" "find ${THOR_DIR} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; echo 'done'"

echo ""
echo "=== Verifying Thor disk usage ==="
ssh "${THOR_USER}@${THOR_HOST}" "du -sh ${THOR_DIR} --exclude=.venv && du -sh ${THOR_DIR}/.venv 2>/dev/null"

echo ""
echo "=== Deploy complete ==="
