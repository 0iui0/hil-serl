"""SpaceMouse -> CR5AF teleoperation using ServoP for real-time control.

ServoP is an interruptible Cartesian servo command.
Each new ServoP immediately interrupts the previous one, so stopping
the SpaceMouse stops the robot instantly — no queued-command lag.

CR5AF only allows ONE TCP connection to port 29999, so all commands
(Dashboard + ServoP) share a single persistent socket.

Usage (on thor):
    cd ~/workspaces/hil-serl
    .venv/bin/python serl_robot_infra/robot_servers/teleop_cr5af.py --robot_ip 192.168.5.1

Controls:
    SpaceMouse push/twist -> delta pose -> ServoP command
    Left button  -> close gripper (no-op until gripper arrives)
    Right button -> open gripper
    Ctrl+C -> exit
"""
import sys
import time
import socket
import struct
import argparse
import threading
import glob
import numpy as np

sys.path.insert(0, "serl_robot_infra")

import evdev
import struct as _struct
import socket as _socket


class EvdevSpaceMouse:
    """Reads SpaceMouse via evdev (reliable on Jetson/ARM where hidraw is blocked).

    Returns the same action/button format as SpaceMouseExpert:
      action = [-ev_Y, ev_X, ev_Z, -ev_RX, -ev_RY, -ev_RZ]
      buttons = [BTN_0, BTN_1]
    """

    def __init__(self, device_path="/dev/input/event6"):
        self.device = evdev.InputDevice(device_path)
        self._axes = {i: 0.0 for i in range(6)}
        self._buttons = [0, 0]
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._ev_loop, daemon=True)
        self._thread.start()

    def _ev_loop(self):
        try:
            for event in self.device.read_loop():
                if not self._running:
                    break
                if event.type == evdev.ecodes.EV_ABS and event.code <= 5:
                    self._axes[event.code] = event.value / 350.0
                elif event.type == evdev.ecodes.EV_KEY:
                    if event.code == 256:
                        self._buttons[0] = event.value
                    elif event.code == 257:
                        self._buttons[1] = event.value
        except Exception:
            pass

    def get_action(self):
        a = self._axes
        action = [-a[1], a[0], a[2], -a[3], -a[4], -a[5]]
        return np.array(action, dtype=np.float32), self._buttons.copy()

    def close(self):
        self._running = False
        try:
            self.device.close()
        except Exception:
            pass


class UDPSpaceMouseReceiver:
    """Receives SpaceMouse data from a remote x86_64 machine via UDP.

    Protocol (from sm_udp_bridge.py): seq(4) + 6x float32(24) + 2x int32(8) = 36 bytes
    """

    def __init__(self, port=9999):
        self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        self._sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.setblocking(False)
        self._action = np.zeros(6, dtype=np.float32)
        self._buttons = [0, 0]
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._thread.start()

    def _udp_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(64)
                if len(data) >= 36:
                    seq, *action, b0, b1 = _struct.unpack("<I6f2i", data[:36])
                    self._action = np.array(action, dtype=np.float32)
                    self._buttons = [b0, b1]
            except BlockingIOError:
                import time
                time.sleep(0.001)
            except Exception:
                pass

    def get_action(self):
        return self._action.copy(), self._buttons.copy()

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass


RT_TOOL_VECTOR = 624
RT_FRAME_MAGIC = 0x123456789abcdef


class CR5AFConnection:
    """Persistent TCP connections to CR5AF for low-latency teleop.

    CR5AF only allows ONE connection to port 29999, so all text commands
    (Dashboard + ServoP) share a single persistent socket.

    Ports:
      29999 — Command socket (EnableRobot, SpeedFactor, ServoP, etc.)
      30004 — RT feedback (1440-byte binary every 8ms)
    """

    def __init__(self, robot_ip, cmd_port=29999, rt_port=30004):
        self.robot_ip = robot_ip
        self.cmd_port = cmd_port
        self.rt_port = rt_port

        # RT feedback
        self._rt_sock = None
        self._pose = None  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        self._lock = threading.Lock()
        self._running = True

        # Single persistent command socket for ALL commands
        self._cmd_sock = None
        self._cmd_lock = threading.Lock()

    def connect(self):
        self._connect_rt()
        self._connect_cmd()

    # --- RT feedback (port 30004) ---

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
                    magic = struct.unpack_from("<Q", data, 48)[0]
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

    # --- Command socket (port 29999) ---

    def _connect_cmd(self):
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_sock.settimeout(5)
        self._cmd_sock.connect((self.robot_ip, self.cmd_port))
        self._cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Drain any stale data
        self._cmd_sock.setblocking(False)
        try:
            self._cmd_sock.recv(4096)
        except BlockingIOError:
            pass
        self._cmd_sock.setblocking(True)

    def _send_cmd(self, cmd, read_response=True, timeout=3.0):
        """Send a command and optionally read the response on the persistent socket."""
        with self._cmd_lock:
            try:
                self._cmd_sock.sendall(cmd.encode())
                if read_response:
                    self._cmd_sock.settimeout(timeout)
                    resp = bytearray()
                    while True:
                        c = self._cmd_sock.recv(1)
                        if not c or c == b";":
                            break
                        resp.extend(c)
                    return resp.decode().strip()
                return None
            except Exception as e:
                return f"ERROR: {e}"

    def dashboard_cmd(self, cmd, timeout=3.0):
        """Send a Dashboard-style command and read response."""
        return self._send_cmd(cmd, read_response=True, timeout=timeout)

    def servop(self, x, y, z, rx, ry, rz):
        """Send ServoP command.

        ServoP is an interruptible Cartesian servo: each new command
        immediately replaces the previous one.  No queuing, no lag.
        """
        cmd = f"ServoP({x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f})"
        with self._cmd_lock:
            try:
                # Drain previous response to keep buffer clear
                self._cmd_sock.setblocking(False)
                try:
                    while True:
                        self._cmd_sock.recv(4096)
                except BlockingIOError:
                    pass
                # Send ServoP (fire-and-forget for low latency)
                self._cmd_sock.setblocking(True)
                self._cmd_sock.sendall(cmd.encode())
            except Exception as e:
                print(f"\n[ServoP ERROR] {e}", flush=True)
                self._reconnect_cmd()

    def _reconnect_cmd(self):
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        try:
            self._connect_cmd()
        except Exception as e:
            print(f"\n[CMD RECONNECT FAILED] {e}", flush=True)

    def close(self):
        self._running = False
        for sock in (self._rt_sock, self._cmd_sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", default="192.168.5.1")
    parser.add_argument("--cmd_port", type=int, default=29999)
    parser.add_argument("--rt_port", type=int, default=30004)
    parser.add_argument("--speed", type=float, default=50.0,
                        help="Robot speed percentage (0-100)")
    parser.add_argument("--action_scale", type=float, default=8.0,
                        help="mm per unit of SpaceMouse input per step")
    parser.add_argument("--rot_scale", type=float, default=3.0,
                        help="deg per unit of SpaceMouse rotation per step")
    parser.add_argument("--hz", type=float, default=33.0,
                        help="ServoP send rate (Hz). Recommended: 33")
    parser.add_argument("--dead_zone", type=float, default=0.15,
                        help="Per-axis raw abs value below this is treated as zero")
    parser.add_argument("--delta_threshold", type=float, default=0.3,
                        help="Min delta (mm or deg) to send a ServoP command")
    parser.add_argument("--no-deadman", action="store_true",
                        help="Disable deadman switch (motion always enabled)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug info every frame")
    parser.add_argument("--udp", type=int, default=0, metavar="PORT",
                        help="Receive SpaceMouse data via UDP on PORT (from sm_udp_bridge.py)")
    parser.add_argument("--device", type=str, default="",
                        help="Path to SpaceMouse evdev device (auto-detect if empty)")
    args = parser.parse_args()

    dt = 1.0 / args.hz

    print("=" * 60)
    print("SpaceMouse -> CR5AF Teleoperation (ServoP mode)")
    print("=" * 60)
    print(f"Robot: {args.robot_ip}  Speed: {args.speed}%")
    print(f"Scale: {args.action_scale} mm/unit  Rot: {args.rot_scale} deg/unit  Rate: {args.hz} Hz")
    print(f"Dead zone: {args.dead_zone}  Delta threshold: {args.delta_threshold} mm/deg")
    print(f"Cmd port: {args.cmd_port}  RT port: {args.rt_port}")
    print("Press Ctrl+C to exit.")
    print("=" * 60)

    # Connect
    conn = CR5AFConnection(args.robot_ip, args.cmd_port, args.rt_port)
    conn.connect()
    print("Connected (RT + Command).")

    # Setup robot
    print("  EnableRobot:", conn.dashboard_cmd("EnableRobot()"))
    print("  SpeedFactor:", conn.dashboard_cmd(f"SpeedFactor({int(args.speed)})"))
    # Clear any pending motion commands
    print("  ResetRobot:", conn.dashboard_cmd("ResetRobot()"))
    time.sleep(0.2)

    # Wait for first RT data
    pose = None
    for _ in range(20):
        pose = conn.get_pose()
        if pose:
            break
        time.sleep(0.1)
    if not pose:
        print("ERROR: No RT pose data received. Check port 30004.")
        sys.exit(1)
    print(f"  Pose: x={pose[0]:.2f} y={pose[1]:.2f} z={pose[2]:.2f} "
          f"rx={pose[3]:.2f} ry={pose[4]:.2f} rz={pose[5]:.2f}")
    print()

    # Open SpaceMouse (evdev local, or UDP from remote x86_64)
    if args.udp:
        print(f"Receiving SpaceMouse via UDP on port {args.udp}...")
        expert = UDPSpaceMouseReceiver(port=args.udp)
        time.sleep(0.3)
        zero_offset = np.zeros(6, dtype=np.float32)
        print("Ready! (zero offset from sender side)\n")
    else:
        device_path = args.device
        if not device_path:
            for ev in sorted(glob.glob("/dev/input/event*")):
                try:
                    name = open(f"/sys/class/input/{ev.split('/')[-1]}/device/name").read().strip()
                    if "SpaceMouse" in name:
                        device_path = ev
                        break
                except Exception:
                    pass
            if not device_path:
                print("ERROR: SpaceMouse not found! Specify --device /dev/input/eventX")
                sys.exit(1)
        print(f"Opening SpaceMouse via evdev ({device_path})...")
        expert = EvdevSpaceMouse(device_path)
        time.sleep(0.3)
        print("Calibrating zero offset (keep SpaceMouse untouched)...")
        zero_samples = []
        for _ in range(int(1.0 / dt)):
            a, _ = expert.get_action()
            zero_samples.append(np.array(a[:6], dtype=np.float32))
            time.sleep(dt)
        zero_offset = np.median(zero_samples, axis=0)
        print(f"Zero offset: {zero_offset}\n")
        print("Ready! Move SpaceMouse to control robot.\n")

    frame = 0
    last_sent = time.monotonic()

    try:
        while True:
            frame += 1
            loop_start = time.monotonic()
            action, buttons = expert.get_action()

            # Subtract zero offset
            action = np.array(action, dtype=np.float32)
            action[:6] -= zero_offset

            # Deadman switch: hold left button to enable motion
            if not args.no_deadman and not buttons[0]:
                if args.verbose and frame % 10 == 0:
                    print(f"\r[deadman] btn={buttons}    ", end="", flush=True)
                time.sleep(dt)
                continue

            # Per-axis dead zone
            max_abs = np.max(np.abs(action[:6]))
            if max_abs < args.dead_zone:
                if args.verbose and frame % 10 == 0:
                    print(f"\r[deadzone] max_abs={max_abs:.3f}    ", end="", flush=True)
                time.sleep(dt)
                continue

            dx, dy, dz, droll, dpitch, dyaw = action[:6]
            delta_x = -dx * args.action_scale
            delta_y = dy * args.action_scale
            delta_z = -dz * args.action_scale
            delta_rx = droll * args.rot_scale
            delta_ry = dpitch * args.rot_scale
            delta_rz = dyaw * args.rot_scale

            delta = np.array([delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz])

            # Ignore tiny deltas
            if np.max(np.abs(delta)) < args.delta_threshold:
                if args.verbose and frame % 10 == 0:
                    print(f"\r[threshold] |delta|={np.max(np.abs(delta)):.2f}    ", end="", flush=True)
                time.sleep(dt)
                continue

            pose = conn.get_pose()
            if not pose:
                if args.verbose and frame % 10 == 0:
                    print(f"\r[no pose]    ", end="", flush=True)
                time.sleep(dt)
                continue

            target = [
                pose[0] + delta[0],
                pose[1] + delta[1],
                pose[2] + delta[2],
                pose[3] + delta[3],
                pose[4] + delta[4],
                pose[5] + delta[5],
            ]

            conn.servop(*target)

            elapsed_ms = (time.monotonic() - last_sent) * 1000
            last_sent = time.monotonic()
            print(f"\r  x={target[0]:.1f} y={target[1]:.1f} "
                  f"z={target[2]:.1f} rz={target[5]:.1f}  "
                  f"dt={elapsed_ms:.0f}ms max_abs={max_abs:.3f} btn={buttons}    ",
                  end="", flush=True)

            # Sleep to maintain target Hz
            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nExiting...")

    expert.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
