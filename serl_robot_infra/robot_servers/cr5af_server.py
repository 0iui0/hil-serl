"""
CR5AF Robot Control Server (Flask)

Wraps the CR5AF TCP protocol into a Flask HTTP API
compatible with the HIL-SERL franka_env gym interface.

The CR5AF communicates via TCP:
  - Dashboard port 29999: text commands (MovL, EnableRobot, etc.)
  - Real-time port 30004: binary state feedback (1440 bytes)

Routes match franka_server.py so the gym env can connect without modification.
"""
import os
import sys
import time
import struct
import socket
import threading
import numpy as np
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "192.168.1.6", "Robot controller IP")
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

MM_TO_M = 0.001
M_TO_MM = 1000.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
NUM_JOINTS = 6

# Real-time data struct offsets (packed, no padding)
# Refer to dobot_api.py MyType for full struct layout.
# uint16_t len at 0
# uint64_t robot_mode at 24
# uint64_t test_value at 48 — magic 0x123456789abcdef for frame validation
# double q_actual[6] at 432
# double qd_actual[6] at 480
# double tool_vector_actual[6] at 624 — [x,y,z,rx,ry,rz] (mm, deg)
# double TCP_speed_actual[6] at 672
# double TCP_force[6] at 720 — current-based estimation
# double SixForceValue[6] at 1304 — 6-axis force sensor
# double ActualQuaternion[4] at 1384 — [w,x,y,z]
# uint64_t current_command_id at 1112
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


def _dashboard_cmd(ip: str, port: int, cmd: str, timeout: float = 5.0) -> str:
    """Send a text command to the dashboard port and return the response."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        sock.sendall(cmd.encode("utf-8"))
        resp = bytearray()
        while True:
            c = sock.recv(1)
            if not c or c == b";":
                break
            resp.extend(c)
        return resp.decode("utf-8").strip()
    finally:
        sock.close()


def _dashboard_cmd_no_wait(ip: str, port: int, cmd: str):
    """Send a text command without waiting for response (fire-and-forget)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((ip, port))
        sock.sendall(cmd.encode("utf-8"))
    except Exception:
        pass
    finally:
        sock.close()


RT_FRAME_MAGIC = 0x123456789abcdef


def _parse_rt_data(data: bytes) -> dict | None:
    """Parse the 1440-byte real-time data struct into a dict.

    Returns None if the frame magic (TestValue) is invalid.
    """
    def _doubles(offset: int, count: int = 6) -> list:
        return list(struct.unpack_from("<" + "d" * count, data, offset))

    fields = struct.unpack_from("<H", data, RT_HEADER)
    rt_len = fields[0]
    if rt_len != 1440:
        raise ValueError(f"Unexpected RT data length: {rt_len}")

    # Validate frame magic
    test_val = struct.unpack_from("<Q", data, RT_TEST_VALUE)[0]
    if test_val != RT_FRAME_MAGIC:
        return None

    return {
        "robot_mode": struct.unpack_from("<Q", data, RT_ROBOT_MODE)[0],
        "q_actual": _doubles(RT_Q_ACTUAL),
        "qd_actual": _doubles(RT_QD_ACTUAL),
        "tool_vector": _doubles(RT_TOOL_VECTOR),   # [x,y,z,rx,ry,rz] mm,deg
        "tcp_speed": _doubles(RT_TCP_SPEED),
        "tcp_force": _doubles(RT_TCP_FORCE),
        "six_force": _doubles(RT_SIX_FORCE),
        "actual_quat": _doubles(RT_ACTUAL_QUAT, 4),  # [w,x,y,z]
        "current_command_id": struct.unpack_from("<Q", data, RT_CURRENT_COMMAND_ID)[0],
    }


class CR5AFServer:
    """CR5AF robot server wrapping the TCP protocol for Flask API."""

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
        self.pos = np.zeros(7)        # [x,y,z, qx,qy,qz,qw]
        self.vel = np.zeros(6)        # tcp velocity (m/s, rad/s)
        self.force = np.zeros(3)      # tcp force (N)
        self.torque = np.zeros(3)     # tcp torque (Nm)
        self.q = np.zeros(NUM_JOINTS)     # joint positions (rad)
        self.dq = np.zeros(NUM_JOINTS)    # joint velocities (rad/s)
        self.gripper_pos = 1.0
        self.robot_mode = 0
        self.current_command_id = 0
        self.six_force = np.zeros(6)       # six-axis force sensor (N, Nm)
        self._connected = False

        # Connect to real-time data feed
        self._rt_sock = None
        self._running = True
        self._connect_rt()

        if safe:
            print("SAFE MODE: reading state only, no motion commands")
        else:
            self._enable_robot()

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
                # Invalid frame magic, skip
                continue

            with self.lock:
                self.robot_mode = rt["robot_mode"]

                # tool_vector: [x,y,z,rx,ry,rz] in mm and degrees → m and radians
                tv = rt["tool_vector"]
                xyz = np.array(tv[:3]) * MM_TO_M
                rxyz = np.array(tv[3:6]) * DEG2RAD
                rot = R.from_euler("XYZ", rxyz)
                quat = rot.as_quat()  # [qx, qy, qz, qw]

                self.pos = np.concatenate([xyz, quat])

                # TCP speed
                spd = rt["tcp_speed"]
                self.vel = np.array(spd[:3]) * MM_TO_M  # convert mm/s to m/s
                # angular speed is already rad/s (tcp_speed[3:6])

                # Force/torque from current-based estimation
                ft = rt["tcp_force"]
                self.force = np.array(ft[:3])
                self.torque = np.array(ft[3:6])

                # Joint positions and velocities
                self.q = np.array(rt["q_actual"]) * DEG2RAD
                self.dq = np.array(rt["qd_actual"]) * DEG2RAD

                # Current command ID for movement tracking
                self.current_command_id = rt["current_command_id"]

                # Six-axis force sensor data (from SixForceValue)
                self.six_force = np.array(rt["six_force"])

    def _enable_robot(self):
        """Enable robot and set speed/acceleration."""
        try:
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port,
                                  "EnableRobot()")
            print(f"EnableRobot: {resp}")
            time.sleep(0.5)
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port,
                                  f"SpeedFactor({int(self.speed_pct)})")
            print(f"SpeedFactor: {resp}")
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port,
                                  f"AccL({int(self.accel_pct)})")
            print(f"AccL: {resp}")
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port,
                                  f"VelL({int(self.speed_pct)})")
            print(f"VelL: {resp}")
        except Exception as e:
            print(f"Warning: EnableRobot failed: {e}")

    def disable_robot(self):
        """Disable robot."""
        try:
            _dashboard_cmd(self.robot_ip, self.dashboard_port,
                           "DisableRobot()")
        except Exception:
            pass

    def clear_error(self):
        """Clear robot errors and re-enable."""
        try:
            _dashboard_cmd(self.robot_ip, self.dashboard_port,
                           "ClearError()")
            time.sleep(1.0)
            _dashboard_cmd(self.robot_ip, self.dashboard_port,
                           "EnableRobot()")
            time.sleep(0.5)
            _dashboard_cmd(self.robot_ip, self.dashboard_port,
                           f"SpeedFactor({int(self.speed_pct)})")
            _dashboard_cmd(self.robot_ip, self.dashboard_port,
                           f"AccL({int(self.accel_pct)})")
        except Exception as e:
            print(f"ClearError failed: {e}")

    def get_error(self, language: str = "en") -> str:
        """Get robot error/alarm information."""
        try:
            return _dashboard_cmd(self.robot_ip, self.dashboard_port,
                                  f"GetError({language})")
        except Exception as e:
            return f"GetError failed: {e}"

    def move_to_joint(self, joint_deg: np.ndarray):
        """Move to joint position [j1..j6] in degrees via MovJ."""
        cmd = (
            "MovJ(joint={"
            f"{joint_deg[0]:.3f},{joint_deg[1]:.3f},{joint_deg[2]:.3f},"
            f"{joint_deg[3]:.3f},{joint_deg[4]:.3f},{joint_deg[5]:.3f}"
            "})"
        )
        try:
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port, cmd)
            if resp and resp[0] != '0':
                print(f"MovJ error: {resp}")
        except Exception as e:
            print(f"MovJ failed: {e}")

    def move_to_pose(self, pose: np.ndarray):
        """Move to Cartesian pose [x,y,z, qx,qy,qz,qw] via MovL."""
        xyz_m = pose[:3]
        quat = pose[3:7]
        rot = R.from_quat(quat)
        rxyz_deg = rot.as_euler("XYZ", degrees=True)

        # Convert m to mm for robot
        x_mm = xyz_m[0] * M_TO_MM
        y_mm = xyz_m[1] * M_TO_MM
        z_mm = xyz_m[2] * M_TO_MM

        cmd = (
            "MovL(pose={"
            f"{x_mm:.3f},{y_mm:.3f},{z_mm:.3f},"
            f"{rxyz_deg[0]:.3f},{rxyz_deg[1]:.3f},{rxyz_deg[2]:.3f}"
            "})"
        )

        try:
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port, cmd)
            if resp and resp[0] != '0':
                print(f"MovL error: {resp}")
        except Exception as e:
            print(f"MovL failed: {e}")

    def move_to_pose_no_wait(self, pose: np.ndarray):
        """Send MovL but don't wait for completion."""
        xyz_m = pose[:3]
        quat = pose[3:7]
        rot = R.from_quat(quat)
        rxyz_deg = rot.as_euler("XYZ", degrees=True)
        cmd = (
            "MovL(pose={"
            f"{xyz_m[0]*M_TO_MM:.3f},{xyz_m[1]*M_TO_MM:.3f},"
            f"{xyz_m[2]*M_TO_MM:.3f},{rxyz_deg[0]:.3f},"
            f"{rxyz_deg[1]:.3f},{rxyz_deg[2]:.3f}"
            "})"
        )
        _dashboard_cmd_no_wait(self.robot_ip, self.dashboard_port, cmd)

    def update_state(self):
        """Read current robot state from the cache (updated by RT thread)."""
        with self.lock:
            pass  # state is already up-to-date from RT loop

    def reset_joint(self):
        """Reset to home joint position via MovJ."""
        joints = FLAGS.reset_joint_target
        cmd = (
            "MovJ(joint={"
            + ",".join(str(float(j)) for j in joints)
            + "})"
        )
        try:
            _dashboard_cmd(self.robot_ip, self.dashboard_port, cmd)
        except Exception as e:
            print(f"Joint reset failed: {e}")

    def close(self):
        """Clean up connections."""
        self._running = False
        if self._rt_sock:
            try:
                self._rt_sock.close()
            except Exception:
                pass

    # ── Force Control (FC) mode ──────────────────────────────────────────

    def fc_force_mode(self, directions: list[int], target_forces: list[int],
                      reference: int = -1, user: int = -1, tool: int = -1) -> str:
        """Start force control with per-direction enable flags and target forces.

        directions:   6 ints [0|1] per axis (x,y,z,rx,ry,rz) — 1=enabled
        target_forces: 6 ints — target force per axis (N for xyz, N/m for rz)
        """
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
        try:
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port, cmd)
            print(f"FCForceMode: {resp}")
            return resp
        except Exception as e:
            msg = f"FCForceMode failed: {e}"
            print(msg)
            return msg

    def fc_set_stiffness(self, vals: list[float]) -> str:
        """Set stiffness (elastic coefficient) per axis [kx,ky,kz,krx,kry,krz]."""
        cmd = "FCSetStiffness(" + ",".join(f"{v}" for v in vals) + ")"
        return self._fc_cmd("FCSetStiffness", cmd)

    def fc_set_damping(self, vals: list[float]) -> str:
        """Set damping coefficient per axis [dx,dy,dz,drx,dry,drz]."""
        cmd = "FCSetDamping(" + ",".join(f"{v}" for v in vals) + ")"
        return self._fc_cmd("FCSetDamping", cmd)

    def fc_set_mass(self, vals: list[float]) -> str:
        """Set mass/inertia coefficient per axis [mx,my,mz,mrx,mry,mrz]."""
        cmd = "FCSetMass(" + ",".join(f"{v}" for v in vals) + ")"
        return self._fc_cmd("FCSetMass", cmd)

    def fc_set_force_limit(self, vals: list[float]) -> str:
        """Set max force limit per axis [x,y,z,rx,ry,rz]."""
        cmd = "FCSetForceLimit(" + ",".join(f"{v}" for v in vals) + ")"
        return self._fc_cmd("FCSetForceLimit", cmd)

    def fc_set_force_speed_limit(self, vals: list[int]) -> str:
        """Set force control adjustment speed per axis [x,y,z,rx,ry,rz]."""
        cmd = "FCSetForceSpeedLimit(" + ",".join(str(v) for v in vals) + ")"
        return self._fc_cmd("FCSetForceSpeedLimit", cmd)

    def fc_set_force(self, vals: list[float]) -> str:
        """Adjust target force in real-time [fx,fy,fz,frx,fry,frz]."""
        cmd = "FCSetForce(" + ",".join(f"{v}" for v in vals) + ")"
        return self._fc_cmd("FCSetForce", cmd)

    def fc_off(self) -> str:
        """Exit force control mode."""
        try:
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port, "FCOff()")
            print(f"FCOff: {resp}")
            return resp
        except Exception as e:
            msg = f"FCOff failed: {e}"
            print(msg)
            return msg

    def six_force_home(self) -> str:
        """Tare the six-axis force sensor."""
        try:
            return _dashboard_cmd(self.robot_ip, self.dashboard_port, "SixForceHome()")
        except Exception as e:
            return f"SixForceHome failed: {e}"

    def get_force_sensor(self, tool: int = -1) -> str:
        """Read six-axis force sensor values."""
        cmd = "GetForce()" if tool == -1 else f"GetForce({tool})"
        try:
            return _dashboard_cmd(self.robot_ip, self.dashboard_port, cmd)
        except Exception as e:
            return f"GetForce failed: {e}"

    def _fc_cmd(self, name: str, cmd: str) -> str:
        """Send an FC command and return the response."""
        try:
            resp = _dashboard_cmd(self.robot_ip, self.dashboard_port, cmd)
            return resp
        except Exception as e:
            msg = f"{name} failed: {e}"
            print(msg)
            return msg


def main(argv):
    webapp = Flask(__name__)

    server = CR5AFServer(
        robot_ip=FLAGS.robot_ip,
        dashboard_port=FLAGS.dashboard_port,
        rt_port=FLAGS.rt_port,
        speed=FLAGS.speed,
        accel=FLAGS.acceleration,
        safe=FLAGS.safe,
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
        err = _require_not_safe()
        if err:
            return err
        server.clear_error()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        """Move to Cartesian pose. Uses no-wait MovL for continuous control."""
        err = _require_not_safe()
        if err:
            return err
        pos = np.array(request.json["arr"])
        server.move_to_pose_no_wait(pos)
        return "Moved"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        """Start FC-based impedance-like mode.

        Configures FC with zero target forces (pure compliance).
        Stiffness/damping provide spring-damper behavior similar to
        Franka's cartesian_impedance_controller.
        MovL commands sent during FC mode act as equilibrium pose changes.
        """
        err = _require_not_safe()
        if err:
            return err
        body = request.json or {}

        # Set compliance parameters before entering FC mode
        stiffness = body.get("stiffness", [500, 500, 500, 30, 30, 30])
        damping = body.get("damping", [10, 10, 10, 1, 1, 1])
        server.fc_set_stiffness(stiffness)
        server.fc_set_damping(damping)

        # FC mode: all directions compliant, zero target force
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

    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port,
               threaded=True)


if __name__ == "__main__":
    app.run(main)
