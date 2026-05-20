"""Motor shaft pick-and-insert environment for CR5AF.

Supports two modes:
- fixed-flange (USE_GRIPPER=False): shaft pre-attached, 6D action
- learned-gripper (USE_GRIPPER=True): pick shaft from holder, insert, release

Workflow (learned-gripper mode):
1. Reset: open gripper → move to grasp pose → close gripper → lift → move to reset pose
2. Episode: DQN controls 6D pose + gripper (insert then release)
3. Success: classifier detects inserted shaft + force threshold
"""
import copy
import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
import requests

from cr5af_env.envs.cr5af_env import CR5AFEnv, DefaultCR5AFEnvConfig


class MotorShaftEnv(CR5AFEnv):
    """Motor shaft pick-and-insert on CR5AF."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_gripper = getattr(self.config, "USE_GRIPPER", False)
        self.grasp_pose = getattr(self.config, "GRASP_POSE", None)
        self.grasp_force_threshold = getattr(self.config, "GRASP_FORCE_THRESHOLD", 2.0)

    def reset(self, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        self._recover()

        if self.use_gripper and self.grasp_pose is not None:
            # Learned-gripper mode: pick up shaft from holder
            self._reset_with_grasp()
        else:
            # Fixed-flange mode: shaft already attached
            self.go_to_reset(joint_reset=False)

        self._recover()
        self.curr_path_length = 0
        self.success = False

        self._update_currpos()
        # Override pose: RT cache may lag after MovL; we know where the robot is
        self.currpos = self.resetpos.copy()
        self._target_pos = None  # force re-init on first step of new episode
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def _reset_with_grasp(self):
        """Reset procedure for learned-gripper mode."""
        # Open gripper
        self._send_gripper_command(1.0)
        time.sleep(0.5)

        # Move to precision mode for grasp
        self._post("update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.3)

        # Move above grasp pose
        above_grasp = copy.deepcopy(self.grasp_pose)
        above_grasp[2] += 0.03
        self.interpolate_move(above_grasp, timeout=1)
        time.sleep(0.3)

        # Descend to grasp pose
        self.interpolate_move(self.grasp_pose, timeout=0.5)
        time.sleep(0.3)

        # Close gripper (hard-coded grasp during reset)
        self._send_gripper_command(-1.0)
        time.sleep(1.0)

        # Verify grasp (force feedback)
        self._update_currpos()
        grasp_force = np.linalg.norm(self.currforce)
        if grasp_force < self.grasp_force_threshold:
            print(f"WARNING: Grasp force {grasp_force:.2f}N < threshold {self.grasp_force_threshold}N. Shaft may not be securely held.")

        # Lift up
        lift_pose = copy.deepcopy(self.grasp_pose)
        lift_pose[2] += 0.04
        self.interpolate_move(lift_pose, timeout=0.5)
        time.sleep(0.3)

        # Move to reset pose (above hole)
        self.go_to_reset(joint_reset=False)

    def go_to_reset(self, joint_reset=False):
        """Move to reset pose above hole via MovL (smooth, robot-planned)."""
        self._post("update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.3)

        # Pull up to clear workpiece via MovL (blocking)
        self._update_currpos()
        pull_up = self.currpos.copy()
        pull_up[2] = self.resetpos[2] + 0.04
        self._post("movl_wait", json={"arr": pull_up.tolist(), "v": 3}, timeout=30)

        if joint_reset:
            print("JOINT RESET")
            self._post("jointreset")
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
            reset_pose[3:] = R.from_euler("XYZ", euler_random, degrees=True).as_quat()
        else:
            reset_pose = self.resetpos.copy()

        # Move to reset pose via MovL (blocking, smooth)
        self._post("movl_wait", json={"arr": reset_pose.tolist(), "v": 3}, timeout=30)

        self._post("update_param", json=self.config.COMPLIANCE_PARAM)

        # Trust MovL result rather than RT cache (which may lag)
        self.currpos = reset_pose.copy()

    def compute_reward(self, obs) -> bool:
        """Reward: insertion success (pose + force threshold)."""
        current_pose = obs["state"]["tcp_pose"]
        current_rot = current_pose[3:]
        from scipy.spatial.transform import Rotation as R
        current_rot = R.from_quat(current_rot).as_matrix()
        target_rot = R.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = R.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))

        pose_ok = np.all(delta < self._REWARD_THRESHOLD)

        # For learned-gripper: also check insertion force (interference fit)
        if self.use_gripper:
            force = np.linalg.norm(obs["state"]["tcp_force"])
            # During insertion, force should be elevated (interference)
            force_ok = force > 3.0  # N, tunable
            return pose_ok and force_ok

        return pose_ok


class GripperPenaltyWrapper(gym.Wrapper):
    """Penalty for unnecessary gripper toggling (from USB task)."""

    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,), "GripperPenaltyWrapper requires 7D action"
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]  # gripper_pose in SERLObs
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        # Penalize gripper toggle when already in desired state
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info


class ServerSpacemouseIntervention(gym.ActionWrapper):
    """Reads SpaceMouse from cr5af_server via HTTP (/get_spacemouse).

    Drop-in replacement for SpacemouseIntervention that avoids pyspacemouse/HID
    issues on Jetson/ARM. The SpaceMouse is read by cr5af_server via evdev and
    exposed as an HTTP endpoint.
    """

    def __init__(self, env, server_url="http://127.0.0.1:5000/"):
        super().__init__(env)
        self.server_url = server_url
        self.gripper_enabled = self.action_space.shape == (7,)
        self.left = False
        self.right = False
        self._req_session = requests.Session()
        self._req_session.trust_env = False
        self._zero_offset: np.ndarray | None = None
        self._calibrate_zero()

    def _calibrate_zero(self, samples: int = 30):
        """Sample SpaceMouse at rest to get zero offset."""
        print("Calibrating SpaceMouse zero offset via /get_spacemouse...")
        vals = []
        for _ in range(samples):
            try:
                resp = self._req_session.post(
                    self.server_url + "get_spacemouse", timeout=2.0,
                ).json()
                vals.append(np.array(resp["action"], dtype=np.float32))
            except Exception:
                pass
            time.sleep(0.02)
        if vals:
            self._zero_offset = np.median(vals, axis=0)
            print(f"SpaceMouse zero offset: {self._zero_offset}")
        else:
            self._zero_offset = np.zeros(6, dtype=np.float32)
            print("WARNING: SpaceMouse calibration failed, using zeros")

    _step_count: int = 0
    _ema_alpha: float = 0.5  # EMA smoothing factor (0.5 = ~5 sample settling time)

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        try:
            resp = self._req_session.post(
                self.server_url + "get_spacemouse", timeout=1.0,
            ).json()
            expert_a = np.array(resp["action"], dtype=np.float32)
            buttons = resp["buttons"]
        except Exception as e:
            if np.random.random() < 0.05:
                print(f"[SpaceMouse] /get_spacemouse failed: {e}")
            return action, False

        if self._zero_offset is not None:
            expert_a[:6] -= self._zero_offset

        # EMA smoothing to suppress sensor jitter (standalone teleop doesn't need
        # this at 33Hz, but at 25Hz the per-step delta is large enough that raw
        # sensor noise causes visible vibration)
        if not hasattr(self, '_smoothed_a'):
            self._smoothed_a = expert_a[:6].copy()
        else:
            self._smoothed_a = (self._ema_alpha * expert_a[:6]
                                + (1 - self._ema_alpha) * self._smoothed_a)
        expert_a[:6] = self._smoothed_a

        self.left, self.right = buttons[0], buttons[1]
        intervened = False

        # Deadman switch: hold left button to enable motion (same as teleop)
        if not self.left:
            return action, False

        # Per-axis dead zone (match teleop_cr5af.py threshold of 0.15)
        if np.max(np.abs(expert_a[:6])) < 0.15:
            return action, False

        intervened = True

        if self.gripper_enabled:
            if self.right:
                # Right button: open gripper
                gripper_action = np.array([1.0], dtype=np.float32)
                intervened = True
            else:
                gripper_action = np.zeros((1,), dtype=np.float32)
            expert_a = np.concatenate((expert_a, gripper_action))

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        ServerSpacemouseIntervention._step_count += 1
        if replaced:
            print(f"[SM #{ServerSpacemouseIntervention._step_count}] "
                  f"INTERVENE action={np.round(new_action[:6], 3).tolist()}")
        elif ServerSpacemouseIntervention._step_count % 30 == 0:
            print(f"[SM #{ServerSpacemouseIntervention._step_count}] no intervention")
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info
