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


CAMERAS = {
    "external": {
        "serial_number": "333422302713",
        "dim": (640, 480),
        "exposure": 40000,
        "fps": 30,
        "crop": lambda img: img[158:418, 234:500],
    },
    "wrist": {
        "serial_number": "352122272331",
        "dim": (640, 480),
        "exposure": 40000,
        "fps": 30,
        "crop": lambda img: img[111:364, 101:387],
    },
}

CHECKPOINT_PATH = os.path.join(os.getcwd(), "classifier_ckpt")
IMAGE_KEYS = ["external", "wrist"]


def build_classifier():
    """Build classifier and load checkpoint weights."""
    dummy_image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    dummy_state = np.zeros((1, 19), dtype=np.float32)
    sample = {**{k: dummy_image for k in IMAGE_KEYS}, "state": dummy_state}

    key = jax.random.PRNGKey(42)
    classifier = create_classifier(key, sample, IMAGE_KEYS)
    classifier = checkpoints.restore_checkpoint(CHECKPOINT_PATH, target=classifier)
    print(f"Loaded classifier from {CHECKPOINT_PATH}")

    predict_fn = jax.jit(
        lambda obs: classifier.apply_fn({"params": classifier.params}, obs, train=False)
    )
    return predict_fn


def preprocess(frames: dict[str, np.ndarray]) -> dict:
    """Crop + resize all camera frames, return observation dict."""
    obs = {"state": np.zeros((1, 19), dtype=np.float32)}
    for name, rgb in frames.items():
        cfg = CAMERAS[name]
        cropped = cfg["crop"](rgb)
        resized = cv2.resize(cropped, (128, 128))
        resized = resized[..., ::-1]  # BGR → RGB
        obs[name] = resized[np.newaxis, ...].astype(np.uint8)
    return obs


def main():
    print("Initializing classifier...")
    predict_fn = build_classifier()

    # Open all cameras
    caps = {}
    for name, cfg in CAMERAS.items():
        print(f"Connecting to camera {name} (serial: {cfg['serial_number']})...")
        caps[name] = RSCapture(
            name=name,
            serial_number=cfg["serial_number"],
            dim=cfg["dim"],
            fps=cfg["fps"],
            depth=False,
            exposure=cfg["exposure"],
        )
    print("All cameras connected. Press 'q' to quit.\n")

    try:
        while True:
            frames = {}
            for name, cap in caps.items():
                success, frame = cap.read()
                if not success or frame is None:
                    continue
                frames[name] = frame

            if not frames:
                time.sleep(0.05)
                continue

            obs = preprocess(frames)
            logit_arr = predict_fn(obs)
            logit = float(logit_arr.ravel()[0])
            prob = 1.0 / (1.0 + float(jnp.exp(-logit)))
            label = "SUCCESS" if prob > 0.5 else "FAILURE"

            # Build side-by-side display
            panels = []
            for name, frame in frames.items():
                display = frame.copy()
                color = (0, 255, 0) if label == "SUCCESS" else (0, 0, 255)
                cv2.putText(display, f"{name}: {label} {prob:.3f}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                panels.append(display)

            if panels:
                max_h = max(p.shape[0] for p in panels)
                padded = []
                for p in panels:
                    if p.shape[0] < max_h:
                        pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=p.dtype)
                        p = np.concatenate([p, pad], axis=0)
                    padded.append(p)
                combined = np.concatenate(padded, axis=1)
                cv2.putText(combined, "Press 'q' to quit",
                            (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow("CR5AF Reward Classifier Verification", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.close()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    main()
