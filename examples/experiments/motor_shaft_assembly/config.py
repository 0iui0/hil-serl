import os

import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from marvin_env.envs.marvin_env import DefaultMarvinEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.motor_shaft_assembly.wrapper import PegInHoleEnv, GripperPenaltyWrapper


class EnvConfig(DefaultMarvinEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "352122272331",
            "dim": (640, 480),
            "exposure": 40000,
        },
        "side": {
            "serial_number": "352122271675",
            "dim": (640, 480),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[100:400, 200:500],
        "side": lambda img: img[100:500, 150:1100],
    }
    # TODO: measure actual target and reset poses from robot
    TARGET_POSE = np.array([0.4, 0.0, 0.2, np.pi, 0, 0])
    GRASP_POSE = np.array([0.4, 0.1, 0.2, np.pi, 0, 0])  # Shaft holder position
    RESET_POSE = np.array([0.4, 0.0, 0.25, np.pi, 0, 0])
    ABS_POSE_LIMIT_LOW = np.array([0.35, -0.05, 0.15, np.pi - 0.05, -0.1, -0.4])
    ABS_POSE_LIMIT_HIGH = np.array([0.45, 0.05, 0.30, np.pi + 0.05, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.005
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.001, 0.04, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 120  # Longer for grasp + insert
    REWARD_THRESHOLD = np.array([0.002, 0.002, 0.003, 0.05, 0.05, 0.05])

    # Gripper mode switch
    USE_GRIPPER = False  # False = fixed-flange, True = learned-gripper
    GRASP_FORCE_THRESHOLD = 2.0  # N, minimum force to confirm grasp

    # Marvin Cartesian impedance parameters (K: N/m, Nm/rad; D: ratio 0-1)
    COMPLIANCE_PARAM = {
        "cart_stiffness": [500, 500, 500, 30, 30, 30, 10],
        "cart_damping": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5],
    }
    PRECISION_PARAM = {
        "cart_stiffness": [2000, 2000, 2000, 100, 100, 100, 20],
        "cart_damping": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5],
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "side"]
    classifier_keys = ["wrist", "side"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper" if EnvConfig.USE_GRIPPER else "single-arm-fixed-gripper"
    gripper_penalty = -0.05

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = PegInHoleEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )

        # Fixed-flange: mask out gripper action
        if not EnvConfig.USE_GRIPPER:
            env = GripperCloseEnv(env)

        if not fake_env:
            env = SpacemouseIntervention(env)

        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier_fn(obs)) > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        # Learned-gripper: penalty for unnecessary gripper toggling
        if EnvConfig.USE_GRIPPER:
            env = GripperPenaltyWrapper(env, penalty=self.gripper_penalty)

        return env
