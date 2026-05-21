"""Gym Interface for CR5AF robot (6-DOF).

Standalone CR5AF environment following HIL-SERL architecture:
- 6-axis force sensor (SixForceValue) via RT port 30004
- FC mode for impedance-like control (stiffness/damping params)
- MovL for Cartesian pose commands (equilibrium point in FC mode)
- Same observation/action space as FrankaEnv for wrapper compatibility

Imports camera, spacemouse, and utils from franka_env (shared infrastructure).
"""
import os
import copy
import time
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

import cv2
import numpy as np
import gymnasium as gym
import requests
from scipy.spatial.transform import Rotation as R

from franka_env.camera.rs_capture import RSCapture


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        first = True
        while True:
            try:
                img_array = self.queue.get(timeout=0.05)
            except queue.Empty:
                cv2.waitKey(10)
                continue
            if img_array is None:
                break
            panels = []
            for k, v in img_array.items():
                if "full" in k:
                    continue
                full = img_array.get(k + "_full", v)
                panels.append(full)
            frame = np.concatenate(panels, axis=1)
            cv2.imshow(self.name, frame)
            if first:
                cv2.resizeWindow(self.name, frame.shape[1], frame.shape[0])
                first = False
            cv2.waitKey(10)


class DefaultCR5AFEnvConfig:
    """CR5AF environment configuration. Override in experiment config."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "TBD",
        "wrist_2": "TBD",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0
    # Per-step delta limiting (mimics Franka's per-cycle impedance clips)
    MAX_TRANSLATION_DELTA_MM: float = 3.0    # cap single-step translation per axis
    MAX_ROTATION_DELTA_DEG: float = 3.0      # cap single-step rotation
    MIN_DELTA_MM: float = 0.1                # skip ServoP below this (lower = less filtering)


class CR5AFEnv(gym.Env):
    """CR5AF robot environment compatible with HIL-SERL wrappers."""

    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultCR5AFEnvConfig = None,
    ):
        self.fake_env = fake_env
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.max_translation_delta = config.MAX_TRANSLATION_DELTA_MM / 1000.0  # per-axis
        self.max_rotation_delta = np.deg2rad(config.MAX_ROTATION_DELTA_DEG)
        self.min_delta = config.MIN_DELTA_MM / 1000.0
        self._servop_active = False
        self._target_pos: np.ndarray | None = None  # tracked ServoP target, not RT cache

        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], R.from_euler("XYZ", config.RESET_POSE[3:], degrees=True).as_quat()]
        )
        if not fake_env:
            self._update_currpos()
        else:
            self.currpos = self.resetpos.copy()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        self.save_video = save_video
        if self.save_video:
            self.recording_frames = []

        # Safety bounding box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        # Action/Observation space (same as FrankaEnv for wrapper compatibility)
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                     for key in config.REALSENSE_CAMERAS}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if not fake_env:
            from pynput import keyboard
            self.terminate = False

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized CR5AF Env")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        sign = np.sign(euler[0])
        euler[0] = sign * np.clip(
            np.abs(euler[0]), self.rpy_bounding_box.low[0], self.rpy_bounding_box.high[0]
        )
        euler[1:] = np.clip(euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:])
        pose[3:] = R.from_euler("xyz", euler).as_quat()
        return pose

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        gripper_action = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_action)

        # Initialize tracked target on first step after reset.
        # Trust self.currpos (set by reset()/go_to_reset() after MovL) —
        # _update_currpos() would read stale RT cache that lags behind MovL.
        if self._target_pos is None:
            self._target_pos = self.currpos.copy()

        state_updated = False

        if np.max(np.abs(action[:6])) > 1e-6:
            dx, dy, dz, droll, dpitch, dyaw = action[:6]
            xyz_delta_m = np.array([
                -dx * self.action_scale[0],
                 dy * self.action_scale[0],
                -dz * self.action_scale[0],
            ])
            rot_delta = np.array([dpitch, droll, dyaw]) * self.action_scale[1]

            xyz_delta_m = np.clip(xyz_delta_m, -self.max_translation_delta, self.max_translation_delta)
            rot_delta = np.clip(rot_delta, -self.max_rotation_delta, self.max_rotation_delta)

            if np.max(np.abs(xyz_delta_m)) >= self.min_delta or np.max(np.abs(rot_delta)) >= self.min_delta:
                # Apply delta to TRACKED target (not RT cache), eliminating feedback oscillation
                self.nextpos = self._target_pos.copy()
                self.nextpos[:3] = self.nextpos[:3] + xyz_delta_m
                self.nextpos[3:] = (
                    R.from_rotvec(rot_delta)
                    * R.from_quat(self._target_pos[3:])
                ).as_quat()

                self._send_pos_command(self.clip_safety_box(self.nextpos))
                self._target_pos = self.nextpos.copy()
                self._servop_active = True
                state_updated = True
            elif self._servop_active:
                self._send_pos_command(self.clip_safety_box(self._target_pos))
                self._servop_active = False
                state_updated = True
        elif self._servop_active:
            self._send_pos_command(self.clip_safety_box(self._target_pos))
            self._servop_active = False
            state_updated = True

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        if not state_updated:
            self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        current_rot = R.from_quat(current_pose[3:]).as_matrix()
        target_rot = R.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = R.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        return bool(np.all(delta < self._REWARD_THRESHOLD))

    def get_im(self) -> Dict[str, np.ndarray]:
        images = {}
        display_images = {}
        full_res_images = {}
        for key, cap in self.cap.items():
            ret, rgb = cap.read()
            if not ret:
                input(f"{key} camera read failed. Check connect, then press enter to relaunch...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()
            cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
            resized = cv2.resize(
                cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = resized[..., ::-1]
            display_images[key] = resized
            display_images[key + "_full"] = cropped_rgb
            full_res_images[key] = copy.deepcopy(cropped_rgb)

        if self.save_video:
            self.recording_frames.append(full_res_images)
        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], R.from_euler("XYZ", goal[3:], degrees=True).as_quat()])
        steps = int(timeout * self.hz)
        self._update_currpos()

        start_pos = self.currpos[:3].copy()
        start_quat = self.currpos[3:].copy()
        goal_pos = goal[:3].copy()
        goal_quat = goal[3:].copy()

        # Ensure shortest quaternion path (q and -q are the same rotation)
        if np.dot(start_quat, goal_quat) < 0:
            goal_quat = -goal_quat

        for i in range(steps):
            t = (i + 1) / steps
            pos = (1 - t) * start_pos + t * goal_pos
            quat = self._slerp(start_quat, goal_quat, t)
            p = np.concatenate([pos, quat])
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = goal
        self._update_currpos()

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for unit quaternions."""
        dot = np.dot(q0, q1)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot) * t
        q_perp = q1 - q0 * dot
        norm = np.linalg.norm(q_perp)
        if norm < 1e-12:
            return q0.copy()
        q_perp = q_perp / norm
        return q0 * np.cos(theta) + q_perp * np.sin(theta)

    def go_to_reset(self, joint_reset=False):
        """Move to reset pose via ServoP interpolate_move (no mode switch).

        Uses ServoP throughout — no stoprobot/MovL — so _target_pos stays
        continuous and RT cache doesn't drift behind the actual pose.
        """
        self._post("update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.3)

        # Pull up to clear workpiece via ServoP.
        # Use current XYZ but reset orientation — orientation may have drifted
        # under FC compliance (e.g. shaft stuck in hole).
        self._update_currpos()
        pull_up = self.currpos.copy()
        pull_up[3:] = self.resetpos[3:]
        pull_up[2] = self.resetpos[2] + 0.04
        self.interpolate_move(pull_up, timeout=2.0)

        if joint_reset:
            print("JOINT RESET")
            # CR5AF MovJ cannot run while ServoP/FC is active; briefly exit
            self._post("stoprobot")
            time.sleep(0.1)
            self._post("jointreset")
            time.sleep(0.5)
            self._post("update_param", json=self.config.PRECISION_PARAM)
            time.sleep(0.3)

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
            reset_pose[:2] += np.random.uniform(-0.05, 0.05, (2,))
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(-2.0, 2.0)
            reset_pose[3:] = R.from_euler("XYZ", euler_random, degrees=True).as_quat()

        # Move to reset pose via ServoP interpolate_move
        self.interpolate_move(reset_pose, timeout=4.0)

        self._post("update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        if self.fake_env:
            self.curr_path_length = 0
            self.terminate = False
            return self._get_obs(), {"succeed": False}

        self.last_gripper_act = time.time()
        self._post("update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        self._zero_force_sensor()
        self._target_pos = None  # force re-init on first step of new episode
        self._servop_active = False
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                for camera_key in self.recording_frames[0].keys():
                    video_path = f'./videos/{camera_key}_{timestamp}.mp4'
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height),
                    )
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            self.cap[cam_name] = RSCapture(name=cam_name, **kwargs)

    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        self._post("full_recovery")

    def _send_pos_command(self, pos: np.ndarray) -> dict | None:
        """Send ServoP and return parsed state from response (combines /pose + /getstate)."""
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        r = self._post("pose", json=data)
        if r.status_code != 200:
            return None
        ps = r.json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])
        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        self.curr_gripper_pos = np.array(ps["gripper_pos"])
        sf = ps.get("six_force")
        if sf is not None:
            sf = np.array(sf)
            if sf.shape == (6,) and np.any(sf != 0):
                self.currforce = sf[:3]
                self.currtorque = sf[3:6]

        return ps

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Gripper commands are no-ops until gripper hardware is connected."""
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):
                self._post("close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):
                self._post("open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else:
                return

    def _post(self, endpoint: str, **kwargs):
        """POST to local server, bypassing any system proxy for localhost."""
        kwargs.setdefault("timeout", 5)
        if not hasattr(self, "_req_session"):
            self._req_session = requests.Session()
            self._req_session.trust_env = False
        return self._req_session.post(self.url + endpoint, **kwargs)

    def _zero_force_sensor(self):
        """Zero the six-axis force sensor via Dobot's SixForceHome()."""
        if self.fake_env:
            return
        try:
            self._post("force_home", timeout=5)
        except Exception as e:
            print(f"WARNING: force_home failed: {e}")

    def _update_currpos(self):
        if self.fake_env:
            return
        for attempt in range(5):
            try:
                r = self._post("getstate", timeout=5)
                if r.status_code == 200 and r.text:
                    ps = r.json()
                    break
                print(f"_update_currpos attempt {attempt}: status={r.status_code}, text_len={len(r.text)}, text={r.text[:200]}")
            except Exception as e:
                print(f"_update_currpos attempt {attempt}: {type(e).__name__}: {e}")
                if attempt < 4:
                    time.sleep(1.0)
        else:
            raise RuntimeError(f"Failed to get state from server after 5 attempts")
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])
        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])
        self.curr_gripper_pos = np.array(ps["gripper_pos"])

        # Prefer 6-axis force sensor data when available
        six_force = ps.get("six_force")
        if six_force is not None:
            sf = np.array(six_force)
            if sf.shape == (6,) and np.any(sf != 0):
                self.currforce = sf[:3]
                self.currtorque = sf[3:6]

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
