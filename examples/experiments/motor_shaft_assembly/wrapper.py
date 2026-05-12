"""Motor shaft peg-in-hole environment wrapper for CR5AF.

Custom reset: pull up → joint/cartesian reset → compliance mode.
Task: insert 7mm motor shaft into hole (top-down, interference fit).
"""
import copy
import time

import numpy as np
import requests

from cr5af_env.envs.cr5af_env import CR5AFEnv, DefaultCR5AFEnvConfig
from franka_env.utils.rotations import euler_2_quat


class MotorShaftEnv(CR5AFEnv):
    """Peg-in-hole task for motor shaft assembly on CR5AF."""

    def go_to_reset(self, joint_reset=False):
        """Pull up to clear shaft, then reset to start pose."""
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Pull up above workpiece
        self._update_currpos()
        pull_up = self.currpos.copy()
        pull_up[2] = self.resetpos[2] + 0.04
        self.interpolate_move(pull_up, timeout=1)

        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Cartesian reset
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
