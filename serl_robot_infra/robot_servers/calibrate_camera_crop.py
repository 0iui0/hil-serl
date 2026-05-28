"""
Interactive Camera Crop Calibration Tool for CR5AF.

Captures full-frame images from each camera, draws grid overlay,
then opens an interactive window where you can drag to select
the crop region. Selected coordinates are printed as the
IMAGE_CROP lambda you can paste into config.py.

Usage:
    python calibrate_camera_crop.py

Steps:
1. Run this script
2. For each camera, a window opens with the grid overlay
3. Drag a rectangle to select the workspace region
4. Press SPACE/ENTER to confirm, or C to re-select
5. Copy the printed lambda into config.py IMAGE_CROP
"""

import sys
import os
import cv2

# Add camera module path
CAM_DIR = os.path.join(os.path.dirname(__file__), "..", "franka_env")
if CAM_DIR not in sys.path:
    sys.path.insert(0, CAM_DIR)

from franka_env.camera.rs_capture import RSCapture

# Camera configs (from cr5af/config.py)
CAMERAS = {
    "external": {
        "serial_number": "333422302713",
        "dim": (640, 480),
        "exposure": 40000,
    },
    "wrist": {
        "serial_number": "352122272331",
        "dim": (640, 480),
        "exposure": 40000,
    },
}


def draw_grid_overlay(img, grid_step=40):
    """Draw grid lines with coordinate labels on image."""
    overlay = img.copy()
    h, w = overlay.shape[:2]

    # Minor grid (every 40px) — green
    for y in range(0, h, grid_step):
        cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(overlay, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                     0.4, (0, 255, 0), 1)
    for x in range(0, w, grid_step):
        cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(overlay, str(x), (x + 5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                     0.4, (0, 255, 0), 1)

    # Major grid (every 80px) — yellow, thicker
    for y in range(0, h, 80):
        cv2.line(overlay, (0, y), (w, y), (255, 255, 0), 2)
    for x in range(0, w, 80):
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 0), 2)

    # Dimensions label
    cv2.putText(overlay, f"{w}x{h}", (w - 80, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay


def interactive_select_crop(name, frame, grid_img):
    """Open interactive window, let user drag to select crop region."""
    print(f"\n  >>> Select crop region for '{name}' <<<")
    print(f"  Drag rectangle in the window, then press:")
    print(f"    SPACE / ENTER  — confirm selection")
    print(f"    C              — re-select (cancel current)")
    print(f"    ESC            — skip this camera (no crop)")

    window_name = f"Select crop: {name} (SPACE=confirm, C=re-select, ESC=skip)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # selectROI returns (x, y, w, h) or empty list if cancelled
    roi = cv2.selectROI(window_name, grid_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print(f"  Skipped — no crop selected for '{name}'")
        return None

    x1, y1, w_sel, h_sel = roi
    x2 = x1 + w_sel
    y2 = y1 + h_sel

    # Clamp to image bounds
    h_img, w_img = frame.shape[:2]
    y1 = max(0, min(y1, h_img))
    y2 = max(0, min(y2, h_img))
    x1 = max(0, min(x1, w_img))
    x2 = max(0, min(x2, w_img))

    crop_h = y2 - y1
    crop_w = x2 - x1

    print(f"  Selected crop region: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
    print(f"  Crop size: {crop_w} x {crop_h}")
    print(f"  Lambda: lambda img: img[{y1}:{y2}, {x1}:{x2}]")

    # Save cropped image for verification
    cropped = frame[y1:y2, x1:x2]
    crop_path = f"/tmp/{name}_selected_crop.jpg"
    cv2.imwrite(crop_path, cropped)
    print(f"  Cropped image saved: {crop_path}  (shape: {cropped.shape})")

    # Save full frame with grid + red crop box
    calib_img = grid_img.copy()
    cv2.rectangle(calib_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(calib_img, f"({x1},{y1})", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
    cv2.putText(calib_img, f"({x2},{y2})", (x2 - 60, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
    calib_path = f"/tmp/{name}_calib.jpg"
    cv2.imwrite(calib_path, calib_img)
    print(f"  Full frame + crop box saved: {calib_path}")

    return (y1, y2, x1, x2)


def capture_and_calibrate(name, config):
    """Capture frame, draw grid, let user interactively select crop."""
    print(f"\n=== {name} (S/N: {config['serial_number']}) ===")
    try:
        cap = RSCapture(
            name=name,
            serial_number=config["serial_number"],
            dim=config["dim"],
            fps=15,
            depth=False,
            exposure=config["exposure"],
        )
        print(f"  Camera initialized OK")

        # Warm up — discard first few frames
        for _ in range(5):
            cap.read()

        success, frame = cap.read()
        if not success or frame is None:
            print(f"  FAILED to capture frame")
            cap.close()
            return None

        print(f"  Frame shape: {frame.shape}")

        # Save raw full frame
        cv2.imwrite(f"/tmp/{name}_full.jpg", frame)

        # Draw grid overlay
        grid_img = draw_grid_overlay(frame, grid_step=40)

        # Interactive crop selection
        crop_coords = interactive_select_crop(name, frame, grid_img)

        cap.close()
        print(f"  Camera closed OK")
        return crop_coords

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("=" * 60)
    print("Interactive Camera Crop Calibration Tool")
    print("=" * 60)
    print("\nFor each camera:")
    print("  1. A window opens showing the frame with grid overlay")
    print("  2. Drag a rectangle around the workspace region")
    print("  3. Press SPACE to confirm, C to re-select, ESC to skip")
    print("=" * 60)

    results = {}
    for name, config in CAMERAS.items():
        coords = capture_and_calibrate(name, config)
        if coords is not None:
            results[name] = coords

    # Summary — ready to paste into config.py
    print("\n" + "=" * 60)
    print("COPY THIS INTO config.py IMAGE_CROP:")
    print("=" * 60)

    if results:
        print("IMAGE_CROP = {")
        for name, (y1, y2, x1, x2) in results.items():
            print(f'    "{name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
        print("}")
    else:
        print("  No crops selected.")

    print("\nVerification files in /tmp/:")
    for name in results:
        print(f"  {name}_selected_crop.jpg  — cropped region")
        print(f"  {name}_calib.jpg          — full frame + red box")

    print("=" * 60)


if __name__ == "__main__":
    main()