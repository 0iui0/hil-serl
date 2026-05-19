"""Verify reward classifier on live camera feed (run on Thor).

Displays real-time camera feed with classifier confidence overlay.
Press 'q' to quit.

Usage:
    # Copy classifier_ckpt/ to Thor first:
    # scp -r classifier_ckpt thor:~/workspaces/hil-serl/

    # On Thor:
    .venv/bin/python serl_robot_infra/robot_servers/verify_reward_classifier.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "serl_robot_infra")
sys.path.insert(0, "serl_launcher")
sys.path.insert(0, "examples")

import jax
import jax.numpy as jnp

# JAX 0.6+ compatibility: tree functions removed from top-level jax
for _name in ('tree_leaves', 'tree_map', 'tree_flatten', 'tree_unflatten',
              'tree_structure', 'tree_transpose'):
    if not hasattr(jax, _name):
        setattr(jax, _name, getattr(jax.tree_util, _name))

# Compatibility: older pickled JAX arrays include 'named_shape' which
# was removed from ShapedArray.update() in newer JAX versions.
try:
    from jax._src.core import ShapedArray as _SA
    _orig_update = _SA.update
    def _patched_update(self, **kwargs):
        kwargs.pop("named_shape", None)
        return _orig_update(self, **kwargs)
    _SA.update = _patched_update
except Exception:
    pass

from franka_env.camera.rs_capture import RSCapture
from serl_launcher.networks.reward_classifier import create_classifier
from flax.training import checkpoints


CAMERA_CONFIG = {
    "serial_number": "333422302713",
    "dim": (640, 480),
    "exposure": 40000,
}

IMAGE_CROP = lambda img: img[100:400, 150:500]

CHECKPOINT_PATH = os.path.join(os.getcwd(), "classifier_ckpt")
IMAGE_KEY = "external"


def build_classifier():
    """Build classifier and load checkpoint weights."""
    dummy_image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    dummy_state = np.zeros((1, 19), dtype=np.float32)
    sample = {IMAGE_KEY: dummy_image, "state": dummy_state}

    key = jax.random.PRNGKey(42)
    classifier = create_classifier(key, sample, [IMAGE_KEY])
    classifier = checkpoints.restore_checkpoint(CHECKPOINT_PATH, target=classifier)
    print(f"Loaded classifier from {CHECKPOINT_PATH}")

    predict_fn = jax.jit(
        lambda obs: classifier.apply_fn({"params": classifier.params}, obs, train=False)
    )
    return predict_fn


def preprocess(rgb: np.ndarray) -> dict:
    """Crop + resize to match training pipeline, return observation dict."""
    cropped = IMAGE_CROP(rgb)
    resized = cv2.resize(cropped, (128, 128))
    return {
        IMAGE_KEY: resized[np.newaxis, ...].astype(np.uint8),
        "state": np.zeros((1, 19), dtype=np.float32),
    }


def main():
    print("Initializing classifier...")
    predict_fn = build_classifier()

    print(f"Connecting to camera (serial: {CAMERA_CONFIG['serial_number']})...")
    camera = RSCapture(
        name="external",
        serial_number=CAMERA_CONFIG["serial_number"],
        dim=CAMERA_CONFIG["dim"],
        fps=15,
        depth=False,
        exposure=CAMERA_CONFIG["exposure"],
    )
    print("Camera connected. Press 'q' to quit.\n")

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue

            obs = preprocess(frame)
            logit_arr = predict_fn(obs)
            logit = float(logit_arr.ravel()[0])
            prob = 1.0 / (1.0 + float(jnp.exp(-logit)))
            label = "SUCCESS" if prob > 0.5 else "FAILURE"

            color = (0, 255, 0) if label == "SUCCESS" else (0, 0, 255)
            display = frame.copy()
            cv2.putText(display, f"{label} {prob:.3f} (logit={logit:.2f})",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, "Press 'q' to quit",
                        (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("CR5AF Reward Classifier Verification", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
