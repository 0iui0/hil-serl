"""Peg-in-Hole task wrapper for Marvin M6.

Supports two modes via USE_GRIPPER config:
- fixed-flange: shaft pre-attached, 6D action (GripperCloseEnv wrapper)
- learned-gripper: pick shaft from holder, insert, release

Workflow (learned-gripper):
1. Reset: open gripper → move to GRASP_POSE → close → lift → move to RESET_POSE
2. Episode: DQN controls 6D pose + gripper
3. Success: pose + force threshold
"""
import copy
import time

import gymnasium as gym
import numpy as np
import requests

from franka_env.utils.rotations import euler_2_quat
from marvin_env.envs.marvin_env import MarvinEnv, DefaultMarvinEnvConfig
from scipy.spatial.transform import Rotation as R


class PegInHoleEnv(MarvinEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_gripper = getattr(self.config, "USE_GRIPPER", False)
        self.grasp_pose = getattr(self.config, "GRASP_POSE", None)
        self.grasp_force_threshold = getattr(self.config, "GRASP_FORCE_THRESHOLD", 2.0)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self._save_video_recording()

        self._recover()

        if self.use_gripper and self.grasp_pose is not None:
            self._reset_with_grasp()
        else:
            self.go_to_reset(joint_reset=False)

        self._recover()
        self.curr_path_length = 0
        self.success = False

        self._update_currpos()
        obs = self._get_obs()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {"succeed": False}

    def _reset_with_grasp(self):
        """Reset for learned-gripper mode: pick up shaft from holder."""
        # Open gripper
        self._send_gripper_command(1.0)
        time.sleep(0.5)

        # Precision mode for grasp
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.3)

        # Move above grasp pose
        above_grasp = copy.deepcopy(self.grasp_pose)
        above_grasp[2] += 0.03
        self.interpolate_move(above_grasp, timeout=1)
        time.sleep(0.3)

        # Descend to grasp pose
        self.interpolate_move(self.grasp_pose, timeout=0.5)
        time.sleep(0.3)

        # Close gripper (hard-coded during reset)
        self._send_gripper_command(-1.0)
        time.sleep(1.0)

        # Verify grasp
        self._update_currpos()
        grasp_force = np.linalg.norm(self.currforce)
        if grasp_force < self.grasp_force_threshold:
            print(f"WARNING: Grasp force {grasp_force:.2f}N < threshold {self.grasp_force_threshold}N")

        # Lift up
        lift_pose = copy.deepcopy(self.grasp_pose)
        lift_pose[2] += 0.04
        self.interpolate_move(lift_pose, timeout=0.5)
        time.sleep(0.3)

        # Move to reset pose above hole
        self.go_to_reset(joint_reset=False)

    def go_to_reset(self, joint_reset=False):
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Pull up to avoid collision with hole
        self._update_currpos()
        pull_up_pose = copy.deepcopy(self.currpos)
        pull_up_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(pull_up_pose, timeout=1)

        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Cartesian reset with optional randomization
        if self.randomreset:
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self._send_pos_command(reset_pose)
        else:
            self._send_pos_command(self.resetpos.copy())
        time.sleep(0.5)

        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        current_rot = R.from_quat(current_pose[3:]).as_matrix()
        target_rot = R.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = R.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        pose_ok = np.all(delta < self._REWARD_THRESHOLD)

        if self.use_gripper:
            force = np.linalg.norm(obs["state"]["tcp_force"])
            force_ok = force > 3.0
            return pose_ok and force_ok
        return pose_ok


class GripperPenaltyWrapper(gym.Wrapper):
    """Penalty for unnecessary gripper toggling."""

    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,), "GripperPenaltyWrapper requires 7D action"
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
