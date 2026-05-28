"""Preview camera feeds with crop overlay in real-time."""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from franka_env.camera.rs_capture import RSCapture

CAMERAS = {
    "external": {
        "sn": "333422302713",
        "crop": (46, 411, 197, 493),  # y1, y2, x1, x2
    },
    "wrist": {
        "sn": "352122272331",
        "crop": (132, 479, 229, 640),
    },
}


def main():
    caps = {}
    for name, cfg in CAMERAS.items():
        cap = RSCapture(
            name=name,
            serial_number=cfg["sn"],
            dim=(640, 480),
            exposure=40000,
            fps=30,
        )
        # warm up
        for _ in range(5):
            cap.read()
        caps[name] = (cap, cfg["crop"])

    print("Press Q to quit")

    try:
        while True:
            for name, (cap, (y1, y2, x1, x2)) in caps.items():
                ok, img = cap.read()
                if not ok:
                    continue

                display = img.copy()
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{name} [{y1}:{y2}, {x1}:{x2}]",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                cropped = img[y1:y2, x1:x2]
                ch, cw = cropped.shape[:2]
                # stack crop below full frame
                label = np.zeros((25, display.shape[1], 3), dtype=np.uint8)
                cv2.putText(label, f"cropped ({cw}x{ch})",
                            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # pad crop to full frame width
                pad_right = display.shape[1] - cw
                cropped_padded = cv2.copyMakeBorder(cropped, 0, 0, 0, pad_right,
                                                     cv2.BORDER_CONSTANT, value=0)
                canvas = np.vstack([display, label, cropped_padded])

                cv2.imshow(f"{name}", canvas)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for cap, _ in caps.values():
            cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
