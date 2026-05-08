"""Peg-in-Hole task wrapper for Marvin M6.

Custom reset: pull up before returning to avoid collision with the hole.
"""
import copy
import time
import numpy as np
import requests

from franka_env.utils.rotations import euler_2_quat
from marvin_env.envs.marvin_env import MarvinEnv


class PegInHoleEnv(MarvinEnv):
    def go_to_reset(self, joint_reset=False):
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

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

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self._save_video_recording()

        self._recover()
        self.go_to_reset(joint_reset=False)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}
