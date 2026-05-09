"""
Test RealSense cameras on Thor: D405 and L515.
Captures frames, saves images for visual inspection.
"""
import sys
import os
import cv2
import numpy as np

SDK_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "marvin_sdk", "TJ_FX_ROBOT_CONTRL_SDK", "SDK_PYTHON"
)
SDK_DIR = os.path.abspath(SDK_DIR)
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

# Add camera module path
CAM_DIR = os.path.join(os.path.dirname(__file__), "..", "franka_env")
if CAM_DIR not in sys.path:
    sys.path.insert(0, CAM_DIR)

from franka_env.camera.rs_capture import RSCapture


def test_camera(name: str, serial: str, dim=(640, 480), exposure=40000):
    print(f"\n=== Testing {name} (S/N: {serial}) ===")
    try:
        cap = RSCapture(
            name=name,
            serial_number=serial,
            dim=dim,
            fps=15,
            depth=False,
            exposure=exposure,
        )
        print(f"  Camera initialized OK")

        success, frame = cap.read()
        if success and frame is not None:
            print(f"  Frame captured: shape={frame.shape}, dtype={frame.dtype}")
            print(f"  Value range: [{frame.min():.1f}, {frame.max():.1f}]")

            out_path = f"/tmp/{name}_test.jpg"
            cv2.imwrite(out_path, frame)
            print(f"  Saved to {out_path}")

            # Basic checks
            assert frame.shape == (dim[1], dim[0], 3), f"Shape mismatch: {frame.shape}"
            assert frame.mean() > 1, "Image seems empty (mean < 1)"
            print(f"  Image mean pixel value: {frame.mean():.1f}")
        else:
            print(f"  FAILED: read() returned success={success}, frame={None if frame is None else 'not None'}")

        cap.close()
        print(f"  Camera closed OK")
        return success and frame is not None

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def find_all_cameras():
    """Query all RealSense devices via pyrealsense2 directly."""
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    results = []
    for d in devices:
        results.append({
            "name": d.get_info(rs.camera_info.name),
            "serial": d.get_info(rs.camera_info.serial_number),
            "product": d.get_info(rs.camera_info.product_id),
            "fw": d.get_info(rs.camera_info.firmware_version),
            "usb": d.get_info(rs.camera_info.usb_type_descriptor),
        })
    return results


def main():
    print("=" * 60)
    print("RealSense Camera Test on Thor")
    print("=" * 60)

    cameras = find_all_cameras()
    print(f"\npyrealsense2 detected {len(cameras)} camera(s):")
    for c in cameras:
        print(f"  {c['name']}  S/N: {c['serial']}  FW: {c['fw']}  USB: {c['usb']}")

    if not cameras:
        print("\nNo cameras detected. Check USB connections and udev rules.")
        sys.exit(1)

    # Test each detected camera
    results = []
    for c in cameras:
        ok = test_camera(c["name"], c["serial"])
        results.append((c["name"], ok))

    print("\n" + "=" * 60)
    print("Results Summary:")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
