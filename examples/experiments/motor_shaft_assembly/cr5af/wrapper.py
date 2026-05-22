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
import glob
import threading
import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

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

        self._zero_force_sensor()
        self._update_currpos()
        self._target_pos = None  # force re-init on first step of new episode
        self._servop_active = False
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


class HidrawSpaceMouse:
    """Reads SpaceMouse via hidraw (easyhid) — works on both USB and Bluetooth.

    Background thread polls hidraw for HID reports, caches latest state.
    Same implementation as cr5af_server.HidrawSpaceMouse, duplicated here so
    the env can read SpaceMouse directly without HTTP round trips.
    """

    _SUPPORTED_IDS = [(0x256F, 0xC63A), (0x256F, 0xC62E)]

    def __init__(self, device_path: str = ""):
        self._device = None
        self._axes = [0.0] * 6
        self._buttons = [0, 0]
        self._running = True

        try:
            from easyhid import Enumeration
        except ImportError:
            print("WARNING: easyhid not installed. SpaceMouse will return zeros.")
            return

        hid = Enumeration()
        all_hids = hid.find()

        found_dev = None
        if device_path:
            for d in all_hids:
                if d.path == device_path:
                    for vid, pid in self._SUPPORTED_IDS:
                        if d.vendor_id == vid and d.product_id == pid:
                            found_dev = d
                            break
                    if found_dev:
                        break
        else:
            for d in all_hids:
                for vid, pid in self._SUPPORTED_IDS:
                    if d.vendor_id == vid and d.product_id == pid:
                        found_dev = d
                        break
                if found_dev:
                    break

        if found_dev is None:
            print("WARNING: No SpaceMouse found via hidraw. Action returns zeros.")
            return

        try:
            found_dev.open()
            found_dev.set_nonblocking(True)
        except Exception as e:
            print(f"WARNING: Failed to open SpaceMouse hidraw: {e}")
            print("  Try: sudo chmod 666 /dev/hidraw*")
            return

        self._device = found_dev
        self._bytes_to_read = 13

        self._thread = threading.Thread(target=self._hidraw_loop, daemon=True)
        self._thread.start()
        print(f"SpaceMouse connected: {found_dev.path} (vid=0x{found_dev.vendor_id:04X} pid=0x{found_dev.product_id:04X})")

    def _to_int16(self, lo, hi):
        val = lo | (hi << 8)
        if val >= 32768:
            val = -(65536 - val)
        return val

    def _hidraw_loop(self):
        while self._running:
            try:
                data = self._device.read(self._bytes_to_read)
                if not data:
                    data = self._device.read(self._bytes_to_read, timeout_ms=50)
                if data and len(data) >= 3:
                    channel = data[0]
                    if channel == 1 and len(data) >= 13:
                        self._axes[0] = self._to_int16(data[1], data[2]) / 350.0
                        self._axes[1] = self._to_int16(data[3], data[4]) / -350.0
                        self._axes[2] = self._to_int16(data[5], data[6]) / -350.0
                        self._axes[3] = self._to_int16(data[7], data[8]) / -350.0
                        self._axes[4] = self._to_int16(data[9], data[10]) / -350.0
                        self._axes[5] = self._to_int16(data[11], data[12]) / 350.0
                    elif channel == 3 and len(data) >= 2:
                        btn_byte = data[1]
                        self._buttons[0] = 1 if (btn_byte & 0x01) else 0
                        self._buttons[1] = 1 if (btn_byte & 0x02) else 0
            except Exception:
                pass
            time.sleep(0.001)

    def get_state(self) -> tuple:
        return self._axes[:], self._buttons[:]

    def close(self):
        self._running = False
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass


class ServerSpacemouseIntervention(gym.ActionWrapper):
    """Reads SpaceMouse directly via hidraw (no HTTP).

    Drop-in replacement that eliminates the /get_spacemouse HTTP round trip.
    The SpaceMouse device must not be opened by cr5af_server — start server
    with --no_spacemouse when using this wrapper.
    """

    def __init__(self, env, server_url="http://127.0.0.1:5000/"):
        super().__init__(env)
        self.gripper_enabled = self.action_space.shape == (7,)
        self.left = False
        self.right = False
        self._sm = HidrawSpaceMouse()
        self._zero_offset: np.ndarray | None = None
        self._calibrate_zero()

    def _calibrate_zero(self, samples: int = 30):
        print("Calibrating SpaceMouse zero offset...")
        vals = []
        for _ in range(samples):
            action, _ = self._sm.get_state()
            vals.append(np.array(action, dtype=np.float32))
            time.sleep(0.02)
        if vals:
            self._zero_offset = np.median(vals, axis=0)
            print(f"SpaceMouse zero offset: {self._zero_offset}")
        else:
            self._zero_offset = np.zeros(6, dtype=np.float32)
            print("WARNING: SpaceMouse calibration failed, using zeros")

    _step_count: int = 0

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        raw_action, buttons = self._sm.get_state()
        expert_a = np.array(raw_action, dtype=np.float32)

        if self._zero_offset is not None:
            expert_a[:6] -= self._zero_offset

        self.left, self.right = buttons[0], buttons[1]

        # Deadman switch: hold left button to enable motion
        if not self.left:
            return action, False

        # Per-axis dead zone
        if np.max(np.abs(expert_a[:6])) < 0.15:
            return action, False

        if self.gripper_enabled:
            if self.right:
                gripper_action = np.array([1.0], dtype=np.float32)
            else:
                gripper_action = np.zeros((1,), dtype=np.float32)
            expert_a = np.concatenate((expert_a, gripper_action))

        return expert_a, True

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
