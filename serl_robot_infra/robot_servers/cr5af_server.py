"""
CR5AF Robot Control Server (Flask)

Wraps the CR5AF TCP protocol into a Flask HTTP API
compatible with the HIL-SERL franka_env gym interface.

Uses persistent command socket + ServoP for responsive teleop.
CR5AF allows only ONE TCP connection to port 29999 at a time, so all
commands (Dashboard + ServoP) share a single persistent socket.

The CR5AF communicates via TCP:
  - Dashboard port 29999: text commands (MovL, ServoP, EnableRobot, etc.)
  - Real-time port 30004: binary state feedback (1440 bytes)

Routes match franka_server.py so the gym env can connect without modification.
"""
import glob
import os
import sys
import time
import struct
import socket
import threading
import logging
import numpy as np
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "192.168.5.1", "Robot controller IP")
flags.DEFINE_integer("dashboard_port", 29999, "Dashboard TCP port")
flags.DEFINE_integer("rt_port", 30004, "Real-time data TCP port")
flags.DEFINE_string("flask_url", "0.0.0.0", "Flask bind address")
flags.DEFINE_integer("flask_port", 5000, "Flask port")
flags.DEFINE_boolean("safe", False, "Safe mode: read-only, no motion commands")
flags.DEFINE_float("speed", 50.0, "Motion speed percentage (0-100)")
flags.DEFINE_float("acceleration", 50.0, "Motion acceleration percentage (0-100)")
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -90, 0, 45],
    "Home joint angles in degrees for jointreset",
)
flags.DEFINE_string("spacemouse_device", "", "SpaceMouse evdev device path (auto-detect if empty)")
flags.DEFINE_boolean("no_spacemouse", False, "Disable SpaceMouse teleop server")
flags.DEFINE_boolean("teleop", False, "Enable background SpaceMouse teleop (for standalone testing)")
flags.DEFINE_float("teleop_action_scale", 8.0, "Teleop: mm per unit of SpaceMouse input")
flags.DEFINE_float("teleop_rot_scale", 3.0, "Teleop: deg per unit of SpaceMouse rotation")
flags.DEFINE_float("teleop_hz", 33.0, "Teleop: ServoP send rate (Hz)")
flags.DEFINE_float("teleop_dead_zone", 0.15, "Teleop: per-axis dead zone")
flags.DEFINE_float("teleop_delta_threshold", 0.3, "Teleop: min delta (mm or deg) to send ServoP")

MM_TO_M = 0.001
M_TO_MM = 1000.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
NUM_JOINTS = 6


class EvdevSpaceMouse:
    """Reads SpaceMouse via evdev for Jetson/ARM where hidraw is blocked.

    Background thread caches latest axes/buttons state.
    Returns the same action format as SpaceMouseExpert:
      action = [-ev_Y, ev_X, ev_Z, -ev_RX, -ev_RY, -ev_RZ]
      buttons = [BTN_0, BTN_1]
    """

    def __init__(self, device_path: str = ""):
        import evdev

        if not device_path:
            for ev in sorted(glob.glob("/dev/input/event*")):
                try:
                    name = open(
                        f"/sys/class/input/{ev.split('/')[-1]}/device/name"
                    ).read().strip()
                    if "SpaceMouse" in name or "Space Navigator" in name:
                        device_path = ev
                        break
                except Exception:
                    pass

        self._device = None
        self._axes = [0.0] * 6
        self._buttons = [0, 0]
        self._running = True

        if not device_path:
            print("WARNING: No SpaceMouse found. /get_spacemouse returns zeros.")
            return

        self._device = evdev.InputDevice(device_path)
        self._thread = threading.Thread(target=self._ev_loop, daemon=True)
        self._thread.start()
        print(f"SpaceMouse connected: {device_path}")

    def _ev_loop(self):
        import evdev
        try:
            for event in self._device.read_loop():
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

    def get_state(self) -> tuple[list[float], list[int]]:
        a = self._axes
        action = [-a[1], a[0], a[2], -a[3], -a[4], -a[5]]
        return action, self._buttons.copy()

    def close(self):
        self._running = False
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass

# Real-time data struct offsets (packed, no padding)
RT_HEADER = 0
RT_ROBOT_MODE = 24
RT_TEST_VALUE = 48
RT_Q_ACTUAL = 432
RT_QD_ACTUAL = 480
RT_TOOL_VECTOR = 624
RT_TCP_SPEED = 672
RT_TCP_FORCE = 720
RT_SIX_FORCE = 1304
RT_ACTUAL_QUAT = 1384
RT_CURRENT_COMMAND_ID = 1112


def _recv_exact(sock: socket.socket, n: int, timeout: float = 3.0) -> bytes:
    """Receive exactly n bytes from socket with timeout."""
    sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


RT_FRAME_MAGIC = 0x123456789abcdef


def _parse_rt_data(data: bytes) -> dict | None:
    """Parse the 1440-byte real-time data struct into a dict."""
    def _doubles(offset: int, count: int = 6) -> list:
        return list(struct.unpack_from("<" + "d" * count, data, offset))

    fields = struct.unpack_from("<H", data, RT_HEADER)
    rt_len = fields[0]
    if rt_len != 1440:
        raise ValueError(f"Unexpected RT data length: {rt_len}")

    test_val = struct.unpack_from("<Q", data, RT_TEST_VALUE)[0]
    if test_val != RT_FRAME_MAGIC:
        return None

    return {
        "robot_mode": struct.unpack_from("<Q", data, RT_ROBOT_MODE)[0],
        "q_actual": _doubles(RT_Q_ACTUAL),
        "qd_actual": _doubles(RT_QD_ACTUAL),
        "tool_vector": _doubles(RT_TOOL_VECTOR),
        "tcp_speed": _doubles(RT_TCP_SPEED),
        "tcp_force": _doubles(RT_TCP_FORCE),
        "six_force": _doubles(RT_SIX_FORCE),
        "actual_quat": _doubles(RT_ACTUAL_QUAT, 4),
        "current_command_id": struct.unpack_from("<Q", data, RT_CURRENT_COMMAND_ID)[0],
    }


class CR5AFServer:
    """CR5AF robot server with persistent command socket + ServoP for teleop.

    Uses a single persistent TCP connection to port 29999 for all commands.
    ServoP provides interruptible, low-latency Cartesian servo for real-time
    control (teleop, RL step). MovL/MovJ used for point-to-point moves (reset).
    """

    def __init__(self, robot_ip: str, dashboard_port: int, rt_port: int,
                 speed: float, accel: float, safe: bool = False):
        self.robot_ip = robot_ip
        self.dashboard_port = dashboard_port
        self.rt_port = rt_port
        self.safe = safe
        self.speed_pct = speed
        self.accel_pct = accel

        # State cache (SI units: position in meters, rotation in radians)
        self.lock = threading.Lock()
        self.pos = np.zeros(7)
        self.vel = np.zeros(6)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.q = np.zeros(NUM_JOINTS)
        self.dq = np.zeros(NUM_JOINTS)
        self.gripper_pos = 1.0
        self.robot_mode = 0
        self.current_command_id = 0
        self.six_force = np.zeros(6)
        self._connected = False

        # Connect to real-time data feed
        self._rt_sock = None
        self._running = True
        self._connect_rt()

        # Persistent command socket (CR5AF allows only ONE connection to port 29999)
        self._cmd_sock = None
        self._cmd_lock = threading.Lock()

        if not safe:
            self._connect_cmd()
            self._enable_robot()
        else:
            print("SAFE MODE: reading state only, no motion commands")

    # ── RT feedback (port 30004) ────────────────────────────────────────────

    def _connect_rt(self):
        """Connect to the real-time data feed in a background thread."""
        self._rt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._rt_sock.settimeout(5.0)
        try:
            self._rt_sock.connect((self.robot_ip, self.rt_port))
            self._connected = True
            print(f"Connected to RT feed at {self.robot_ip}:{self.rt_port}")
        except Exception as e:
            print(f"Warning: RT feed connection failed: {e}")
            self._connected = False
            return

        self._rt_thread = threading.Thread(target=self._rt_loop, daemon=True)
        self._rt_thread.start()

    def _rt_loop(self):
        """Background loop reading real-time data."""
        while self._running:
            if self._rt_sock is None:
                time.sleep(0.5)
                continue
            try:
                data = _recv_exact(self._rt_sock, 1440, timeout=3.0)
                rt = _parse_rt_data(data)
            except Exception as e:
                print(f"RT read error: {e}, reconnecting...")
                try:
                    self._rt_sock.close()
                except Exception:
                    pass
                self._rt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._rt_sock.settimeout(5.0)
                try:
                    self._rt_sock.connect((self.robot_ip, self.rt_port))
                    continue
                except Exception:
                    time.sleep(1.0)
                    continue

            if rt is None:
                continue

            with self.lock:
                self.robot_mode = rt["robot_mode"]

                tv = rt["tool_vector"]
                xyz = np.array(tv[:3]) * MM_TO_M
                rxyz = np.array(tv[3:6]) * DEG2RAD
                rot = R.from_euler("XYZ", rxyz)
                quat = rot.as_quat()

                self.pos = np.concatenate([xyz, quat])

                spd = rt["tcp_speed"]
                linear_vel = np.array(spd[:3]) * MM_TO_M        # mm/s → m/s
                angular_vel = np.array(spd[3:6]) * DEG2RAD      # deg/s → rad/s
                self.vel = np.concatenate([linear_vel, angular_vel])

                ft = rt["tcp_force"]
                self.force = np.array(ft[:3])
                self.torque = np.array(ft[3:6])

                self.q = np.array(rt["q_actual"]) * DEG2RAD
                self.dq = np.array(rt["qd_actual"]) * DEG2RAD

                self.current_command_id = rt["current_command_id"]

                self.six_force = np.array(rt["six_force"])

    # ── Persistent command socket (port 29999) ──────────────────────────────

    def _connect_cmd(self):
        """Connect persistent command socket for all dashboard + ServoP commands."""
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_sock.settimeout(5.0)
        self._cmd_sock.connect((self.robot_ip, self.dashboard_port))
        self._cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Drain stale data from welcome message
        self._cmd_sock.setblocking(False)
        try:
            self._cmd_sock.recv(4096)
        except BlockingIOError:
            pass
        self._cmd_sock.setblocking(True)
        print(f"Connected command socket to {self.robot_ip}:{self.dashboard_port}")

    def _send_cmd(self, cmd: str, read_response: bool = True, timeout: float = 5.0) -> str:
        """Send command via persistent socket. Thread-safe via _cmd_lock."""
        with self._cmd_lock:
            try:
                # Drain stale responses before sending (like servop does)
                self._cmd_sock.setblocking(False)
                try:
                    while True:
                        self._cmd_sock.recv(4096)
                except BlockingIOError:
                    pass
                self._cmd_sock.setblocking(True)
                self._cmd_sock.sendall(cmd.encode("utf-8"))
                if read_response:
                    self._cmd_sock.settimeout(timeout)
                    resp = bytearray()
                    while True:
                        c = self._cmd_sock.recv(1)
                        if not c or c == b";":
                            break
                        resp.extend(c)
                    return resp.decode("utf-8").strip()
                return ""
            except Exception as e:
                print(f"[CMD ERROR] {e}")
                self._reconnect_cmd()
                return f"ERROR: {e}"

    def servop(self, x_mm: float, y_mm: float, z_mm: float,
               rx_deg: float, ry_deg: float, rz_deg: float):
        """Send ServoP (interruptible Cartesian servo, fire-and-forget)."""
        cmd = f"ServoP({x_mm:.3f},{y_mm:.3f},{z_mm:.3f},{rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f})"
        print(f"[ServoP] x={x_mm:.1f} y={y_mm:.1f} z={z_mm:.1f} "
              f"rx={rx_deg:.2f} ry={ry_deg:.2f} rz={rz_deg:.2f}")
        with self._cmd_lock:
            try:
                # Drain stale response to keep buffer clear
                self._cmd_sock.setblocking(False)
                try:
                    while True:
                        self._cmd_sock.recv(4096)
                except BlockingIOError:
                    pass
                # Fire-and-forget for low latency
                self._cmd_sock.setblocking(True)
                self._cmd_sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                print(f"[ServoP ERROR] {e}")
                self._reconnect_cmd()

    def servop_pose(self, pose: np.ndarray):
        """Send ServoP for Cartesian pose [x,y,z, qx,qy,qz,qw]."""
        xyz_m = pose[:3]
        quat = pose[3:7]
        rot = R.from_quat(quat)
        rxyz_deg = rot.as_euler("XYZ", degrees=True)
        self.servop(
            xyz_m[0] * M_TO_MM, xyz_m[1] * M_TO_MM, xyz_m[2] * M_TO_MM,
            rxyz_deg[0], rxyz_deg[1], rxyz_deg[2],
        )

    def _reconnect_cmd(self):
        """Reconnect the command socket."""
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        try:
            self._connect_cmd()
        except Exception as e:
            print(f"[CMD RECONNECT FAILED] {e}")

    # ── Robot control ───────────────────────────────────────────────────────

    def _enable_robot(self):
        """Enable robot and set speed/acceleration via persistent socket."""
        try:
            resp = self._send_cmd("EnableRobot()")
            print(f"EnableRobot: {resp}")
            time.sleep(0.5)
            resp = self._send_cmd(f"SpeedFactor({int(self.speed_pct)})")
            print(f"SpeedFactor: {resp}")
            resp = self._send_cmd(f"AccL({int(self.accel_pct)})")
            print(f"AccL: {resp}")
            resp = self._send_cmd(f"VelL({int(self.speed_pct)})")
            print(f"VelL: {resp}")
        except Exception as e:
            print(f"Warning: EnableRobot failed: {e}")

    def disable_robot(self):
        self._send_cmd("DisableRobot()")

    def clear_error(self):
        try:
            self._send_cmd("ClearError()")
            time.sleep(1.0)
            self._send_cmd("EnableRobot()")
            time.sleep(0.5)
            self._send_cmd(f"SpeedFactor({int(self.speed_pct)})")
            self._send_cmd(f"AccL({int(self.accel_pct)})")
        except Exception as e:
            print(f"ClearError failed: {e}")

    def get_error(self, language: str = "en") -> str:
        return self._send_cmd(f"GetError({language})")

    def move_to_joint(self, joint_deg: np.ndarray):
        """Move to joint position [j1..j6] in degrees via MovJ."""
        cmd = (
            "MovJ(joint={"
            f"{joint_deg[0]:.3f},{joint_deg[1]:.3f},{joint_deg[2]:.3f},"
            f"{joint_deg[3]:.3f},{joint_deg[4]:.3f},{joint_deg[5]:.3f}"
            "})"
        )
        resp = self._send_cmd(cmd)
        if resp and resp[0] != '0':
            print(f"MovJ error: {resp}")

    def move_to_pose(self, pose: np.ndarray, v: int = -1):
        """Move to Cartesian pose [x,y,z, qx,qy,qz,qw] via MovL (blocking)."""
        xyz_m = pose[:3]
        quat = pose[3:7]
        rot = R.from_quat(quat)
        rxyz_deg = rot.as_euler("XYZ", degrees=True)
        print(f"[MovL] x={xyz_m[0]*M_TO_MM:.1f} y={xyz_m[1]*M_TO_MM:.1f} z={xyz_m[2]*M_TO_MM:.1f} "
              f"rx={rxyz_deg[0]:.2f} ry={rxyz_deg[1]:.2f} rz={rxyz_deg[2]:.2f} v={v}")
        cmd = (
            "MovL(pose={"
            f"{xyz_m[0]*M_TO_MM:.3f},{xyz_m[1]*M_TO_MM:.3f},"
            f"{xyz_m[2]*M_TO_MM:.3f},{rxyz_deg[0]:.3f},"
            f"{rxyz_deg[1]:.3f},{rxyz_deg[2]:.3f}"
            "}"
        )
        if v > 0:
            cmd += f",v={v}"
        cmd += ")"
        for attempt in range(5):
            resp = self._send_cmd(cmd, timeout=30.0)
            if resp and resp[0] != '0':
                print(f"MovL error: {resp} (attempt {attempt+1}/5)")
                time.sleep(0.5)
            else:
                return
        print(f"MovL failed after all attempts")

    def move_to_pose_no_wait(self, pose: np.ndarray):
        """Send MovL fire-and-forget (for non-critical queued moves)."""
        xyz_m = pose[:3]
        quat = pose[3:7]
        rot = R.from_quat(quat)
        rxyz_deg = rot.as_euler("XYZ", degrees=True)
        print(f"[MovL_nowait] x={xyz_m[0]*M_TO_MM:.1f} y={xyz_m[1]*M_TO_MM:.1f} z={xyz_m[2]*M_TO_MM:.1f} "
              f"rx={rxyz_deg[0]:.2f} ry={rxyz_deg[1]:.2f} rz={rxyz_deg[2]:.2f}")
        cmd = (
            "MovL(pose={"
            f"{xyz_m[0]*M_TO_MM:.3f},{xyz_m[1]*M_TO_MM:.3f},"
            f"{xyz_m[2]*M_TO_MM:.3f},{rxyz_deg[0]:.3f},"
            f"{rxyz_deg[1]:.3f},{rxyz_deg[2]:.3f}"
            "})"
        )
        self._send_cmd(cmd, read_response=False)

    def update_state(self):
        with self.lock:
            pass

    def reset_joint(self):
        joints = FLAGS.reset_joint_target
        cmd = (
            "MovJ(joint={"
            + ",".join(str(float(j)) for j in joints)
            + "})"
        )
        self._send_cmd(cmd)

    def close(self):
        self._running = False
        for sock in (self._rt_sock, self._cmd_sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    # ── Teleop (SpaceMouse → ServoP background thread) ──────────────────

    def start_teleop(self, spacemouse: "EvdevSpaceMouse",
                     action_scale: float = 8.0, rot_scale: float = 3.0,
                     hz: float = 33.0, dead_zone: float = 0.15,
                     delta_threshold: float = 0.3):
        """Start background SpaceMouse → ServoP teleop thread."""
        self._teleop_sm = spacemouse
        self._teleop_hz = hz
        self._teleop_action_scale = action_scale
        self._teleop_rot_scale = rot_scale
        self._teleop_dead_zone = dead_zone
        self._teleop_delta_threshold = delta_threshold
        self._teleop_running = True

        # Calibrate zero offset
        dt = 1.0 / hz
        print("Calibrating SpaceMouse zero offset (keep untouched)...")
        samples = []
        for _ in range(int(1.0 / dt)):
            a, _ = spacemouse.get_state()
            samples.append(np.array(a[:6], dtype=np.float32))
            time.sleep(dt)
        self._teleop_zero = np.median(samples, axis=0)
        print(f"Zero offset: {self._teleop_zero}")

        t = threading.Thread(target=self._teleop_loop, daemon=True)
        t.start()
        print(f"Teleop thread started ({hz}Hz)")

    def _teleop_loop(self):
        """Background loop: SpaceMouse → delta → ServoP."""
        dt = 1.0 / self._teleop_hz
        while self._teleop_running and self._running:
            loop_start = time.monotonic()
            try:
                action, buttons = self._teleop_sm.get_state()
                action = np.array(action, dtype=np.float32)
                action[:6] -= self._teleop_zero

                # Deadman: hold left button to enable motion
                if not buttons[0]:
                    time.sleep(dt)
                    continue

                # Dead zone
                if np.max(np.abs(action[:6])) < self._teleop_dead_zone:
                    time.sleep(dt)
                    continue

                dx, dy, dz, droll, dpitch, dyaw = action[:6]
                delta = np.array([
                    -dx * self._teleop_action_scale,
                    dy * self._teleop_action_scale,
                    -dz * self._teleop_action_scale,
                    dpitch * self._teleop_rot_scale,
                    droll * self._teleop_rot_scale,
                    dyaw * self._teleop_rot_scale,
                ])

                if np.max(np.abs(delta)) < self._teleop_delta_threshold:
                    time.sleep(dt)
                    continue

                # Current pose from RT cache (mm, deg)
                with self.lock:
                    if self.pos is None:
                        time.sleep(dt)
                        continue
                    cur_xyz_mm = self.pos[:3] * M_TO_MM
                    cur_euler_deg = R.from_quat(self.pos[3:]).as_euler("XYZ", degrees=True)

                target = [
                    cur_xyz_mm[0] + delta[0],
                    cur_xyz_mm[1] + delta[1],
                    cur_xyz_mm[2] + delta[2],
                    cur_euler_deg[0] + delta[3],
                    cur_euler_deg[1] + delta[4],
                    cur_euler_deg[2] + delta[5],
                ]

                self.servop(*target)
            except Exception as e:
                print(f"[TELEOP ERROR] {e}")

            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ── Force Control (FC) mode ──────────────────────────────────────────

    def fc_force_mode(self, directions: list[int], target_forces: list[int],
                      reference: int = -1, user: int = -1, tool: int = -1) -> str:
        dir_str = "{" + ",".join(str(d) for d in directions) + "}"
        f_str = "{" + ",".join(str(f) for f in target_forces) + "}"
        cmd = f"FCForceMode({dir_str},{f_str}"
        if reference != -1:
            cmd += f",reference={reference}"
        if user != -1:
            cmd += f",user={user}"
        if tool != -1:
            cmd += f",tool={tool}"
        cmd += ")"
        resp = self._send_cmd(cmd)
        print(f"FCForceMode: {resp}")
        return resp

    def fc_set_stiffness(self, vals: list[float]) -> str:
        cmd = "FCSetStiffness(" + ",".join(f"{v}" for v in vals) + ")"
        return self._send_cmd(cmd, read_response=False)

    def fc_set_damping(self, vals: list[float]) -> str:
        cmd = "FCSetDamping(" + ",".join(f"{v}" for v in vals) + ")"
        return self._send_cmd(cmd, read_response=False)

    def fc_set_mass(self, vals: list[float]) -> str:
        cmd = "FCSetMass(" + ",".join(f"{v}" for v in vals) + ")"
        return self._send_cmd(cmd, read_response=False)

    def fc_set_force_limit(self, vals: list[float]) -> str:
        cmd = "FCSetForceLimit(" + ",".join(f"{v}" for v in vals) + ")"
        return self._send_cmd(cmd)

    def fc_set_force_speed_limit(self, vals: list[int]) -> str:
        cmd = "FCSetForceSpeedLimit(" + ",".join(str(v) for v in vals) + ")"
        return self._send_cmd(cmd)

    def fc_set_force(self, vals: list[float]) -> str:
        cmd = "FCSetForce(" + ",".join(f"{v}" for v in vals) + ")"
        return self._send_cmd(cmd)

    def fc_off(self) -> str:
        resp = self._send_cmd("FCOff()")
        print(f"FCOff: {resp}")
        return resp

    def six_force_home(self) -> str:
        return self._send_cmd("SixForceHome()")

    def get_force_sensor(self, tool: int = -1) -> str:
        cmd = "GetForce()" if tool == -1 else f"GetForce({tool})"
        return self._send_cmd(cmd)


def main(argv):
    webapp = Flask(__name__)

    # Suppress Flask HTTP request logs (noise that drowns out ServoP commands)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    server = CR5AFServer(
        robot_ip=FLAGS.robot_ip,
        dashboard_port=FLAGS.dashboard_port,
        rt_port=FLAGS.rt_port,
        speed=FLAGS.speed,
        accel=FLAGS.acceleration,
        safe=FLAGS.safe,
    )

    spacemouse: EvdevSpaceMouse | None = None
    if not FLAGS.no_spacemouse:
        spacemouse = EvdevSpaceMouse(device_path=FLAGS.spacemouse_device)
        if FLAGS.teleop and spacemouse._device is not None and not server.safe:
            server.start_teleop(
                spacemouse,
                action_scale=FLAGS.teleop_action_scale,
                rot_scale=FLAGS.teleop_rot_scale,
                hz=FLAGS.teleop_hz,
                dead_zone=FLAGS.teleop_dead_zone,
                delta_threshold=FLAGS.teleop_delta_threshold,
            )

    def _require_not_safe():
        if server.safe:
            return jsonify(
                {"error": "Server is in safe mode. Restart without --safe to enable control."}
            ), 403
        return None

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        server.update_state()
        return jsonify({
            "pose": server.pos.tolist(),
            "vel": server.vel.tolist(),
            "force": server.force.tolist(),
            "torque": server.torque.tolist(),
            "q": server.q.tolist(),
            "dq": server.dq.tolist(),
            "jacobian": np.zeros((6, 6)).tolist(),
            "gripper_pos": server.gripper_pos,
            "robot_mode": server.robot_mode,
            "current_command_id": server.current_command_id,
            "six_force": server.six_force.tolist(),
        })

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        server.update_state()
        return jsonify({"pose": server.pos.tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": server.vel.tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": server.force.tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": server.torque.tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": server.q.tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": server.dq.tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.zeros((6, 6)).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": server.gripper_pos})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        err = _require_not_safe()
        if err:
            return err
        server.reset_joint()
        return "Reset Joint"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        """Lightweight error clear (like Franka). Just send ClearError(), return immediately."""
        if not server.safe:
            server._send_cmd("ClearError()")
        return "Clear"

    @webapp.route("/full_recovery", methods=["POST"])
    def full_recovery():
        """Full error recovery: ClearError + re-enable robot + set speed/accel."""
        err = _require_not_safe()
        if err:
            return err
        server.clear_error()
        return "Full Recovery"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        """Send ServoP and return current state (combines /pose + /getstate)."""
        err = _require_not_safe()
        if err:
            return err
        pos = np.array(request.json["arr"])
        server.servop_pose(pos)
        with server.lock:
            return jsonify({
                "pose": server.pos.tolist(),
                "vel": server.vel.tolist(),
                "force": server.force.tolist(),
                "torque": server.torque.tolist(),
                "q": server.q.tolist(),
                "dq": server.dq.tolist(),
                "gripper_pos": server.gripper_pos,
                "six_force": server.six_force.tolist(),
            })

    @webapp.route("/stoprobot", methods=["POST"])
    def stop_robot():
        """Stop robot and exit ServoP mode before MovL."""
        if not server.safe:
            server._send_cmd("StopRobot()")
        return "Stopped"

    @webapp.route("/movl", methods=["POST"])
    def movl():
        """Move to Cartesian pose via MovL (queued, for point-to-point moves)."""
        err = _require_not_safe()
        if err:
            return err
        pos = np.array(request.json["arr"])
        server.move_to_pose_no_wait(pos)
        return "Moved"

    @webapp.route("/movl_wait", methods=["POST"])
    def movl_wait():
        """Blocking MovL — waits for robot to finish motion before responding."""
        err = _require_not_safe()
        if err:
            return err
        pos = np.array(request.json["arr"])
        speed = request.json.get("v", -1)
        server.move_to_pose(pos, v=speed)
        return "Moved"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        err = _require_not_safe()
        if err:
            return err
        body = request.json or {}

        stiffness = body.get("stiffness", [500, 500, 500, 30, 30, 30])
        damping = body.get("damping", [10, 10, 10, 1, 1, 1])
        server.fc_set_stiffness(stiffness)
        server.fc_set_damping(damping)

        directions = [1, 1, 1, 1, 1, 1]
        target_forces = [0, 0, 0, 0, 0, 0]
        resp = server.fc_force_mode(directions, target_forces)
        return jsonify({"fc_mode": resp})

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        err = _require_not_safe()
        if err:
            return err
        resp = server.fc_off()
        return jsonify({"fc_off": resp})

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        err = _require_not_safe()
        if err:
            return err
        body = request.json or {}
        results = {}
        if "stiffness" in body:
            results["stiffness"] = server.fc_set_stiffness(body["stiffness"])
        if "damping" in body:
            results["damping"] = server.fc_set_damping(body["damping"])
        if "mass" in body:
            results["mass"] = server.fc_set_mass(body["mass"])
        if "force_limit" in body:
            results["force_limit"] = server.fc_set_force_limit(body["force_limit"])
        if "target_forces" in body:
            results["target_force"] = server.fc_set_force(body["target_forces"])
        return jsonify(results) if results else "No params updated"

    @webapp.route("/geterror", methods=["POST"])
    def get_error():
        lang = request.json.get("language", "en") if request.json else "en"
        resp = server.get_error(language=lang)
        return jsonify({"error_info": resp})

    @webapp.route("/movej", methods=["POST"])
    def move_joint():
        err = _require_not_safe()
        if err:
            return err
        joint_deg = np.array(request.json["arr"])
        server.move_to_joint(joint_deg)
        return "Moved Joint"

    # ── Force Control (FC) optional routes ───────────────────────────────

    @webapp.route("/fc_force_mode", methods=["POST"])
    def fc_force_mode():
        err = _require_not_safe()
        if err:
            return err
        body = request.json
        resp = server.fc_force_mode(
            body.get("directions", [1, 1, 1, 1, 1, 1]),
            body.get("target_forces", [0, 0, 0, 0, 0, 0]),
            reference=body.get("reference", -1),
            user=body.get("user", -1),
            tool=body.get("tool", -1),
        )
        return jsonify({"fc_force_mode": resp})

    @webapp.route("/set_stiffness", methods=["POST"])
    def set_stiffness():
        err = _require_not_safe()
        if err:
            return err
        resp = server.fc_set_stiffness(request.json["values"])
        return jsonify({"stiffness": resp})

    @webapp.route("/set_damping", methods=["POST"])
    def set_damping():
        err = _require_not_safe()
        if err:
            return err
        resp = server.fc_set_damping(request.json["values"])
        return jsonify({"damping": resp})

    @webapp.route("/fc_set_force", methods=["POST"])
    def fc_set_force():
        err = _require_not_safe()
        if err:
            return err
        resp = server.fc_set_force(request.json["values"])
        return jsonify({"target_force": resp})

    @webapp.route("/fc_off", methods=["POST"])
    def fc_off():
        err = _require_not_safe()
        if err:
            return err
        resp = server.fc_off()
        return jsonify({"fc_off": resp})

    @webapp.route("/force_home", methods=["POST"])
    def force_home():
        err = _require_not_safe()
        if err:
            return err
        resp = server.six_force_home()
        return jsonify({"six_force_home": resp})

    @webapp.route("/get_force_sensor", methods=["POST"])
    def get_force_sensor():
        tool = request.json.get("tool", -1) if request.json else -1
        resp = server.get_force_sensor(tool=tool)
        return jsonify({"force_sensor": resp})

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        return "Opened (no-op)"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        return "Closed (no-op)"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        return "No gripper (no-op)"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        return "No gripper (no-op)"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        return "No gripper (no-op)"

    @webapp.route("/get_spacemouse", methods=["POST"])
    def get_spacemouse():
        if spacemouse is None:
            return jsonify({"action": [0.0] * 6, "buttons": [0, 0]})
        action, buttons = spacemouse.get_state()
        return jsonify({"action": action, "buttons": buttons})

    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port,
               threaded=True)


if __name__ == "__main__":
    app.run(main)
