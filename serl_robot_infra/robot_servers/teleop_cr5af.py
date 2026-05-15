"""SpaceMouse → CR5AF teleoperation with persistent connections.

Usage (on thor):
    cd ~/workspaces/hil-serl
    .venv/bin/python serl_robot_infra/robot_servers/teleop_cr5af.py --robot_ip 192.168.5.1

Controls:
    SpaceMouse push/twist -> delta pose -> MovL command
    Left button  -> close gripper (no-op until gripper arrives)
    Right button -> open gripper
    Ctrl+C -> exit
"""
import sys
import time
import socket
import struct
import select
import argparse
import threading
import numpy as np

sys.path.insert(0, "serl_robot_infra")

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert

RT_TOOL_VECTOR = 624
RT_FRAME_MAGIC = 0x123456789abcdef
RT_TEST_VALUE = 48


class CR5AFConnection:
    """Persistent TCP connections to CR5AF for low-latency teleop."""

    def __init__(self, robot_ip, dashboard_port=29999, rt_port=30004):
        self.robot_ip = robot_ip
        self.dashboard_port = dashboard_port
        self.rt_port = rt_port

        # Persistent RT connection + background reader
        self._rt_sock = None
        self._pose = None  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        self._lock = threading.Lock()
        self._running = True

        # Persistent Dashboard connection
        self._ds_sock = None

    def connect(self):
        self._connect_rt()
        self._connect_dashboard()

    def _connect_rt(self):
        self._rt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._rt_sock.settimeout(5)
        self._rt_sock.connect((self.robot_ip, self.rt_port))
        t = threading.Thread(target=self._rt_loop, daemon=True)
        t.start()

    def _rt_loop(self):
        while self._running:
            try:
                data = self._rt_sock.recv(1440)
                if len(data) >= 648:
                    magic = struct.unpack_from("<Q", data, RT_TEST_VALUE)[0]
                    if magic == RT_FRAME_MAGIC:
                        pose = list(struct.unpack_from("<6d", data, RT_TOOL_VECTOR))
                        with self._lock:
                            self._pose = pose
            except Exception:
                if self._running:
                    try:
                        self._rt_sock.close()
                    except Exception:
                        pass
                    self._rt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._rt_sock.settimeout(5)
                    try:
                        self._rt_sock.connect((self.robot_ip, self.rt_port))
                    except Exception:
                        time.sleep(0.5)

    def get_pose(self):
        with self._lock:
            return list(self._pose) if self._pose else None

    def _connect_dashboard(self):
        self._ds_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ds_sock.settimeout(3)
        self._ds_sock.connect((self.robot_ip, self.dashboard_port))
        # drain welcome
        time.sleep(0.3)
        while select.select([self._ds_sock], [], [], 0.3)[0]:
            self._ds_sock.recv(4096)

    def dashboard_cmd(self, cmd, wait=0.5):
        self._ds_sock.sendall((cmd + "\r\n").encode())
        time.sleep(wait)
        data = b""
        while select.select([self._ds_sock], [], [], 0.5)[0]:
            chunk = self._ds_sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return data.decode().strip()

    def movl(self, x, y, z, rx, ry, rz):
        cmd = f"MovL(pose={{{x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f}}})\r\n"
        try:
            self._ds_sock.sendall(cmd.encode())
        except Exception:
            pass

    def close(self):
        self._running = False
        for s in (self._rt_sock, self._ds_sock):
            if s:
                try:
                    s.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="192.168.5.1")
    parser.add_argument("--dashboard_port", type=int, default=29999)
    parser.add_argument("--rt_port", type=int, default=30004)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--action_scale", type=float, default=3.0,
                        help="mm per unit of SpaceMouse input per step")
    parser.add_argument("--rot_scale", type=float, default=1.5,
                        help="deg per unit of SpaceMouse rotation per step")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    dt = 1.0 / args.hz

    print("=" * 60)
    print("SpaceMouse -> CR5AF Teleoperation (persistent connections)")
    print("=" * 60)
    print(f"Robot: {args.robot_ip}  Speed: {args.speed}%")
    print(f"Scale: {args.action_scale} mm/unit  Rate: {args.hz} Hz")
    print("Press Ctrl+C to exit.")
    print("=" * 60)

    # Connect
    conn = CR5AFConnection(args.robot_ip, args.dashboard_port, args.rt_port)
    conn.connect()
    print("Connected.")

    # Setup robot
    print("  EnableRobot:", conn.dashboard_cmd("EnableRobot()"))
    print("  SpeedFactor:", conn.dashboard_cmd(f"SpeedFactor({int(args.speed)})"))

    # Wait for first RT data
    for _ in range(20):
        pose = conn.get_pose()
        if pose:
            break
        time.sleep(0.1)
    print(f"  Pose: x={pose[0]:.2f} y={pose[1]:.2f} z={pose[2]:.2f} "
          f"rx={pose[3]:.2f} ry={pose[4]:.2f} rz={pose[5]:.2f}")
    print()

    # Open SpaceMouse
    print("Opening SpaceMouse...")
    expert = SpaceMouseExpert()
    time.sleep(0.5)
    print("Ready! Move SpaceMouse to control robot.\n")

    try:
        while True:
            action, buttons = expert.get_action()

            # Check raw SpaceMouse input (same as Franka SpacemouseIntervention)
            if np.linalg.norm(action) < 0.001:
                time.sleep(dt)
                continue

            dx, dy, dz, droll, dpitch, dyaw = action
            # Scale to mm and degrees
            delta_x = -dx * args.action_scale
            delta_y = dy * args.action_scale
            delta_z = dz * args.action_scale
            delta_rx = droll * args.rot_scale
            delta_ry = dpitch * args.rot_scale
            delta_rz = dyaw * args.rot_scale

            pose = conn.get_pose()
            if pose:
                target = [
                    pose[0] + delta_x,
                    pose[1] + delta_y,
                    pose[2] + delta_z,
                    pose[3] + delta_rx,
                    pose[4] + delta_ry,
                    pose[5] + delta_rz,
                ]
                conn.movl(*target)
                print(f"\r  x={target[0]:.1f} y={target[1]:.1f} "
                      f"z={target[2]:.1f} rz={target[5]:.1f}  "
                      f"btn={buttons}    ", end="", flush=True)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\nExiting...")

    expert.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
