import datetime
import os
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import numpy as np
import wandb

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def _to_scalar(v):
    """Convert JAX/numpy arrays to Python scalars for wandb logging."""
    try:
        import jax.numpy as jnp
        if isinstance(v, jnp.ndarray):
            return float(v.item()) if v.ndim == 0 else float(v.mean())
    except ImportError:
        pass
    if isinstance(v, np.ndarray):
        return float(v.item()) if v.ndim == 0 else float(v.mean())
    if isinstance(v, (int, float)):
        return v
    return v


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        debug=False,
        sync_tensorboard=False,
        tensorboard_log_dir=None,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if debug:
            mode = "disabled"
        else:
            mode = "online"

        self._tb_writer = None
        self._sync_tensorboard = sync_tensorboard
        if sync_tensorboard and _HAS_TF:
            if tensorboard_log_dir is None:
                tensorboard_log_dir = os.path.join(wandb_output_dir, "tensorboard")
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            self._tb_writer = tf.summary.create_file_writer(tensorboard_log_dir)

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=self.config.tag,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
            sync_tensorboard=sync_tensorboard,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: _to_scalar(v) for k, v in zip(*data_flat)}
        wandb.log(data, step=step)

        if self._tb_writer is not None and step is not None:
            with self._tb_writer.as_default(step=step):
                for k, v in data.items():
                    try:
                        scalar_val = float(v)
                        tf.summary.scalar(k, scalar_val)
                    except (TypeError, ValueError):
                        pass
