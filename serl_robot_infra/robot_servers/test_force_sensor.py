"""
Marvin M6 force sensor test.
Read-only mode. Matches the C++ wrapper's approach:
  force[i] = fb_joint_them[i] / 10000   (i=0..5)

Also reads all other force-related fields for comparison.
"""
import sys
import os
import time
import argparse

SDK_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "marvin_sdk", "TJ_FX_ROBOT_CONTRL_SDK", "SDK_PYTHON"
)
SDK_DIR = os.path.abspath(SDK_DIR)
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from fx_robot import Marvin_Robot, DCSS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="192.168.1.190")
    parser.add_argument("--arm", default="A", choices=["A", "B"])
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--set_tool", action="store_true",
                        help="Call set_tool (non-safe, but needed for force est)")
    args = parser.parse_args()

    arm_idx = {"A": 0, "B": 1}[args.arm]
    arm = args.arm

    dcss = DCSS()
    robot = Marvin_Robot()

    print(f"[Connecting to {args.robot_ip} ...]")
    ret = robot.connect(args.robot_ip)
    if ret == 0:
        print("ERROR: connect failed")
        return
    print("Connected.")

    robot.check_error_and_clear(dcss)

    # Verify UDP
    frame_update = None
    motion_tag = 0
    for i in range(5):
        sub_data = robot.subscribe(dcss)
        if sub_data is None:
            continue
        fs = sub_data["outputs"][0]["frame_serial"]
        if fs != 0 and fs != frame_update:
            motion_tag += 1
            frame_update = fs
        time.sleep(0.01)
    if motion_tag > 0:
        print("UDP data channel: OK")
    else:
        print("WARNING: no frame update (robot may be off)")

    # Optionally inject tool params (matches C++ OnSetTool_A)
    if args.set_tool:
        # From C++ code: TCP z-offset = 78mm, mass = 0.232kg
        kine_params = [0, 0, 78, 0.0, 0.0, 0.0]
        dyn_params = [
            0.231932,           # mass (kg)
            93.926344,    # COM x (mm)
            95.696747,   # COM y (mm)
            -46.129158,      # COM z (mm)
            0.030662,   # Ixx
            0.0,          # Ixy
            0.0,          # Ixz
            0.014517,    # Iyy
            0.0,          # Iyz
            0.016582     # Izz
        ]
        robot.clear_set()
        robot.set_tool(arm=arm, kineParams=kine_params, dynamicParams=dyn_params)
        robot.send_cmd()
        print(f"set_tool injected: kine={kine_params}, dyn={dyn_params}")
    else:
        print("NOTE: --set_tool not used; force data may be invalid without tool params.")
        print("      The C++ code calls OnSetTool_A(kineParams, dynParams) during connect.")

    print("\n=== Force Sensor Test ===")
    print("Format:  Fx, Fy, Fz (N)  |  Tx, Ty, Tz (Nm)")
    print("Press Ctrl+C to stop.\n")

    header = (
        f"{'C++ method [them/10000]':<48} "
        f"{'est_cart_fn (SDK)':<48} "
        f"{'est_joint_force':<42} "
        f"{'joint_sToq':<35}"
    )
    print(header)
    print("-" * 175)

    try:
        while True:
            sub_data = robot.subscribe(dcss)
            if sub_data is None:
                time.sleep(args.interval)
                continue

            out = sub_data["outputs"][arm_idx]

            fb_joint_them = out.get("fb_joint_them", [0.0] * 7)
            est_cart_fn = out.get("est_cart_fn", [0.0] * 6)
            est_joint_force = out.get("est_joint_force", [0.0] * 7)
            fb_joint_sToq = out.get("fb_joint_sToq", [0.0] * 7)

            # C++ method: them[i] / 10000 for i = 0..5
            force_cpp = [fb_joint_them[i] / 10000.0 for i in range(6)]

            cpp_str = f"[F: {force_cpp[0]:+8.4f} {force_cpp[1]:+8.4f} {force_cpp[2]:+8.4f}  |  T: {force_cpp[3]:+8.4f} {force_cpp[4]:+8.4f} {force_cpp[5]:+8.4f}]"
            est_str = f"[{', '.join(f'{v:+8.4f}' for v in est_cart_fn[:3])}  |  {', '.join(f'{v:+8.4f}' for v in est_cart_fn[3:6])}]"
            jnt_str = f"[{', '.join(f'{v:+7.3f}' for v in est_joint_force)}]"
            stoq_str = f"[{', '.join(f'{v:+7.3f}' for v in fb_joint_sToq)}]"

            print(f"{cpp_str:<48} {est_str:<48} {jnt_str:<42} {stoq_str:<35}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        robot.release_robot()
        print("Released.")


if __name__ == "__main__":
    main()
