import os

import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from cr5af_env.envs.cr5af_env import DefaultCR5AFEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.motor_shaft_assembly.cr5af.wrapper import (
    MotorShaftEnv, GripperPenaltyWrapper, ServerSpacemouseIntervention,
)



# TODO: fill in real values when CR5AF is connected and workspace is calibrated
class EnvConfig(DefaultCR5AFEnvConfig):
    SERVER_URL = "http://192.168.16.158:5000/"
    # Single external camera (hand-eye). Add more entries for additional cameras.
    REALSENSE_CAMERAS = {
        "external": {
            "serial_number": "333422302713",  # D455 #1 on Jetson Thor
            "dim": (640, 480),
            "exposure": 40000,
            "fps": 30,
        },
        "wrist": {
            "serial_number": "352122272331",  # D405 wrist
            "dim": (640, 480),
            "exposure": 40000,
            "fps": 30,
        },
    }
    IMAGE_CROP = {
        "external": lambda img: img[46:411, 197:493],
        "wrist": lambda img: img[132:479, 229:640],
    }

    # Calibrated 2025-05-18 with CR5AF
    # Units: XYZ in meters, rotation in degrees
    RESET_POSE = np.array([0.7300, -0.1370, 0.1400, 178.28, -3.44, -90.00])
    GRASP_POSE = np.array([0.7000, -0.1750, 0.2000, -180.00, -0.00, 0.00])
    TARGET_POSE = np.array([0.7300, -0.1370, 0.0895, 178.28, -3.44, -90.0])
    REWARD_THRESHOLD = np.array([0.005, 0.005, 0.005, 2.0, 2.0, 2.0])
    ACTION_SCALE = (0.003, 0.05, 1)
    ABS_POSE_LIMIT_LOW = np.array([0.600, -0.250, 0.086, -180, -90, -180])
    ABS_POSE_LIMIT_HIGH = np.array([0.850, -0.050, 0.160, 180, 90, 180])

    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 200            # 8s at 25Hz (aligned with Franka ~6-10s)
    # Constrain TCP orientation to RESET_POSE rotation ± this many degrees
    MAX_ORIENTATION_DEVIATION_DEG = 1.0
    # Per-step delta caps for RL safety (25Hz × 3mm = 75mm/s)
    MAX_TRANSLATION_DELTA_MM = 3.0
    MAX_ROTATION_DELTA_DEG = 3.0
    MIN_DELTA_MM = 0.05                 # lower threshold for more responsive fine control

    # Gripper mode switch
    USE_GRIPPER = False                 # False = fixed-flange (current), True = learned-gripper (future)
    GRASP_FORCE_THRESHOLD = 2.0         # N, minimum force to confirm grasp
    FORCE_THRESHOLD = 1.0              # N, |fz| threshold for insertion success
    CLASSIFIER_THRESHOLD = 0.9         # sigmoid score threshold for insertion detection

    # FC impedance params (CR5AF FC mode stiffness/damping)
    # Higher damping prevents oscillation; moderate stiffness tracks SpaceMouse crisply
    COMPLIANCE_PARAM = {
        "stiffness": [400, 400, 400, 60, 60, 60],
        "damping": [120, 120, 120, 25, 25, 25],
    }
    PRECISION_PARAM = {
        "stiffness": [2000, 2000, 2000, 200, 200, 200],
        "damping": [30, 30, 30, 10, 10, 10],
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["external", "wrist"]
    classifier_keys = ["external", "wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    # Auto-switch training mode based on gripper availability
    setup_mode = "single-arm-learned-gripper" if EnvConfig.USE_GRIPPER else "single-arm-fixed-gripper"
    # Match BC checkpoint architecture (BCAgent uses [512, 512, 512])
    policy_network_kwargs = {"hidden_dims": [512, 512, 512]}
    gripper_penalty = -0.05

    def get_environment(self, fake_env=False, save_video=False, classifier=False, server_url=None, server_teleop=False):
        env_config = EnvConfig()
        if server_url is not None:
            env_config.SERVER_URL = server_url
        env = MotorShaftEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
            hz=25,
        )

        # Fixed-flange: mask out gripper action
        if not EnvConfig.USE_GRIPPER:
            env = GripperCloseEnv(env)

        if not fake_env and not server_teleop:
            env = ServerSpacemouseIntervention(env, server_url=env_config.SERVER_URL)

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
                return int(sigmoid(classifier(obs)) > env_config.CLASSIFIER_THRESHOLD)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        # Learned-gripper: add penalty for unnecessary gripper toggling
        if EnvConfig.USE_GRIPPER:
            env = GripperPenaltyWrapper(env, penalty=self.gripper_penalty)

        return env
