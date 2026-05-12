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
from cr5af_env.envs.cr5af_env import DefaultCR5AFEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.motor_shaft_assembly.wrapper import MotorShaftEnv


# TODO: fill in real values when CR5AF is connected and workspace is calibrated
class EnvConfig(DefaultCR5AFEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "TBD",  # TODO: D405 serial on Jetson Thor
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "TBD",  # TODO: D405 serial on Jetson Thor
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }

    # TODO: calibrate with real robot
    TARGET_POSE = np.zeros((6,))
    RESET_POSE = np.zeros((6,))
    REWARD_THRESHOLD = np.zeros((6,))
    ACTION_SCALE = (0.01, 0.06, 1)
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100

    # FC impedance params (CR5AF FC mode stiffness/damping)
    # Mapped from Franka impedance params — tune on real robot
    COMPLIANCE_PARAM = {
        "stiffness": [500, 500, 500, 30, 30, 30],
        "damping": [10, 10, 10, 1, 1, 1],
    }
    PRECISION_PARAM = {
        "stiffness": [2000, 2000, 2000, 100, 100, 100],
        "damping": [30, 30, 30, 5, 5, 5],
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = MotorShaftEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
