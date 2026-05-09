"""
Marvin M6 Robot Control Server (Flask)

Wraps the Marvin SDK (Marvin_Robot + Marvin_Kine) into a Flask HTTP API
compatible with the HIL-SERL franka_env gym interface.

Usage:
    python marvin_server.py --robot_ip 192.168.1.190 --arm A

Routes match franka_server.py so the gym env can connect without modification.
"""
import os
import sys
import time
import ctypes
import numpy as np
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags

# Add SDK path
SDK_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "marvin_sdk", "TJ_FX_ROBOT_CONTRL_SDK", "SDK_PYTHON"
)
SDK_DIR = os.path.abspath(SDK_DIR)
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from fx_robot import Marvin_Robot, DCSS

KINE_DIR = SDK_DIR  # fx_kine.py is in same dir as fx_robot.py
try:
    from fx_kine import Marvin_Kine
except ImportError:
    Marvin_Kine = None
    print("Warning: Marvin_Kine not available, FK/IK will be disabled")

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "192.168.1.190", "Robot controller IP")
flags.DEFINE_string("arm", "A", "Arm to control: A or B")
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -90, 0, 0, 0],
    "Home joint angles in degrees",
)
flags.DEFINE_string("flask_url", "127.0.0.1", "Flask bind address")
flags.DEFINE_string("kine_config", "ccs_m6_40.MvKDCfg", "Kinematics config file")
flags.DEFINE_integer("flask_port", 5000, "Flask port")
flags.DEFINE_boolean("safe", False, "Safe mode: read-only, no motion commands")

ARM_IDX = {"A": 0, "B": 1}
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


class MarvinServer:
    """Marvin M6 robot server wrapping the SDK for Flask API."""

    def __init__(self, robot_ip: str, arm: str, reset_joint_target: list,
                 kine_config: str, safe: bool = False):
        self.arm = arm
        self.arm_idx = ARM_IDX[arm]
        self.reset_joint_target = reset_joint_target
        self.safe = safe

        # Initialize robot
        self.robot = Marvin_Robot()
        connected = self.robot.connect(robot_ip)
        if not connected:
            raise ConnectionError(f"Failed to connect to robot at {robot_ip}")
        print(f"Robot connected to {robot_ip}")

        # Initialize data subscription structure
        self.dcss = DCSS()

        # Check and clear errors
        self.robot.check_error_and_clear(self.dcss)

        # Verify UDP data channel
        self._verify_connection()

        # Initialize kinematics
        self.kine = None
        if Marvin_Kine is not None:
            self._init_kinematics(kine_config)

        # Cached state
        self.pos = np.zeros(7)      # xyz + quat (x,y,z,qx,qy,qz,qw)
        self.vel = np.zeros(6)      # tcp velocity
        self.force = np.zeros(3)    # tcp force
        self.torque = np.zeros(3)   # tcp torque
        self.q = np.zeros(7)        # joint positions (radians)
        self.dq = np.zeros(7)       # joint velocities
        self.gripper_pos = 1.0      # no gripper = always open

        if safe:
            print("SAFE MODE: reading state only, no motion commands will be sent")
        else:
            # Start in impedance mode
            self._start_impedance()

    def _verify_connection(self):
        """Verify UDP data channel is working."""
        motion_tag = 0
        frame_update = None
        for _ in range(5):
            sub_data = self.robot.subscribe(self.dcss)
            fs = sub_data["outputs"][self.arm_idx]["frame_serial"]
            if fs != 0 and frame_update != fs:
                motion_tag += 1
                frame_update = fs
            time.sleep(0.01)
        if motion_tag == 0:
            raise ConnectionError("UDP data channel not receiving frames")
        print(f"Robot connected at {FLAGS.robot_ip}, arm={self.arm}")

    def _init_kinematics(self, config_name: str):
        """Initialize kinematics for FK/IK."""
        try:
            config_path = os.path.join(KINE_DIR, config_name)
            if not os.path.exists(config_path):
                # Try demo path
                config_path = os.path.join(
                    os.path.dirname(SDK_DIR), "DEMO_PYTHON", config_name
                )
            self.kine = Marvin_Kine()
            self.kine.log_switch(0)
            ini = self.kine.load_config(
                arm_type=self.arm_idx, config_path=config_path
            )
            result = self.kine.initial_kine(
                robot_type=ini["TYPE"][self.arm_idx],
                dh=ini["DH"][self.arm_idx],
                pnva=ini["PNVA"][self.arm_idx],
                j67=ini["BD"][self.arm_idx],
            )
            if not result:
                print("Warning: Kinematics initialization failed")
                self.kine = None
            else:
                print("Kinematics initialized successfully")
        except Exception as e:
            print(f"Warning: Kinematics init failed: {e}")
            self.kine = None

    def _start_impedance(self):
        """Start Cartesian impedance mode for compliant control."""
        # Default compliance params for peg-in-hole (low stiffness)
        default_K = [500, 500, 500, 30, 30, 30, 10]
        default_D = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5]

        self.robot.clear_set()
        self.robot.set_cart_kd_params(
            arm=self.arm, K=default_K, D=default_D, type=2
        )
        self.robot.send_cmd()
        time.sleep(0.3)

        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=3)  # torque mode
        self.robot.set_impedance_type(arm=self.arm, type=2)  # cartesian impedance
        self.robot.set_vel_acc(arm=self.arm, velRatio=50, AccRatio=50)
        self.robot.send_cmd()
        time.sleep(0.3)

    def get_cartesian_pose(self) -> np.ndarray:
        """Get current EE pose as [x,y,z, qx,qy,qz,qw] via FK."""
        sub_data = self.robot.subscribe(self.dcss)
        joint_deg = sub_data["outputs"][self.arm_idx]["fb_joint_pos"]

        if self.kine is not None:
            fk_mat = self.kine.fk(joints=joint_deg)
            xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=fk_mat)
            xyz = xyzabc[:3]
            # Convert ABC (degrees, xyz extrinsic) to quaternion
            rot = R.from_euler("XYZ", [xyzabc[3], xyzabc[4], xyzabc[5]], degrees=True)
            quat = rot.as_quat()  # [qx, qy, qz, qw]
            return np.concatenate([xyz, quat])
        else:
            # Without kinematics, return zeros (FK unavailable)
            return np.zeros(7)

    def move_to_pose(self, pose: np.ndarray):
        """Move to Cartesian pose [x,y,z, qx,qy,qz,qw] via IK + joint cmd."""
        if self.kine is None:
            print("Warning: Kinematics not available, cannot move to Cartesian pose")
            return

        xyz = pose[:3]
        quat = pose[3:7]
        rot = R.from_quat(quat)
        abc_deg = rot.as_euler("XYZ", degrees=True)
        target_xyzabc = np.concatenate([xyz, abc_deg])

        # Get current joints for seed
        sub_data = self.robot.subscribe(self.dcss)
        cur_joints = sub_data["outputs"][self.arm_idx]["fb_joint_pos"]

        # IK
        ik_result = self.kine.ik(
            xyzabc=target_xyzabc.tolist(),
            ref_joints=cur_joints,
        )
        if ik_result is not None:
            self.robot.clear_set()
            self.robot.set_joint_cmd_pose(arm=self.arm, joints=ik_result)
            self.robot.send_cmd()

    def update_state(self):
        """Read current robot state from subscription."""
        sub_data = self.robot.subscribe(self.dcss)
        out = sub_data["outputs"][self.arm_idx]

        # Joint positions (degrees -> radians)
        self.q = np.array(out["fb_joint_pos"]) * DEG2RAD
        self.dq = np.array(out["fb_joint_vel"]) * DEG2RAD

        # Force/torque: C++ method — fb_joint_them[i] / 10000
        # (m_FB_Joint_Them is repurposed to carry 6D force data)
        fb_joint_them = out.get("fb_joint_them", [0.0] * 7)
        cart_fn = [fb_joint_them[i] / 10000.0 for i in range(6)]
        self.force = np.array(cart_fn[:3])
        self.torque = np.array(cart_fn[3:6])

        # Cartesian pose via FK
        self.pos = self.get_cartesian_pose()

        # TCP velocity (approximate from Jacobian if available)
        # For now, use finite difference or zeros
        self.vel = np.zeros(6)

    def clear_error(self):
        """Clear robot errors."""
        self.robot.clear_error(self.arm)
        time.sleep(0.1)

    def reset_joint(self):
        """Reset to home joint position via position mode, then back to impedance."""
        # Switch to position mode
        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=1)
        self.robot.send_cmd()
        time.sleep(0.5)

        # Move to reset position
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(
            arm=self.arm, joints=self.reset_joint_target
        )
        self.robot.send_cmd()

        # Wait for convergence
        for _ in range(30):
            time.sleep(1)
            sub_data = self.robot.subscribe(self.dcss)
            cur = np.array(sub_data["outputs"][self.arm_idx]["fb_joint_pos"])
            target = np.array(self.reset_joint_target)
            if np.allclose(cur, target, atol=0.5):
                break

        # Switch back to impedance mode
        self._start_impedance()

    def update_compliance(self, params: dict):
        """Update Cartesian impedance parameters."""
        K = params.get("cart_stiffness", [500, 500, 500, 30, 30, 30, 10])
        D = params.get("cart_damping", [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5])
        self.robot.clear_set()
        self.robot.set_cart_kd_params(arm=self.arm, K=K, D=D, type=2)
        self.robot.send_cmd()
        time.sleep(0.1)


def main(argv):
    webapp = Flask(__name__)

    robot_ip = FLAGS.robot_ip
    arm = FLAGS.arm
    reset_joints = [float(x) for x in FLAGS.reset_joint_target]

    server = MarvinServer(
        robot_ip=robot_ip,
        arm=arm,
        reset_joint_target=reset_joints,
        kine_config=FLAGS.kine_config,
        safe=FLAGS.safe,
    )

    def _require_not_safe():
        if server.safe:
            return jsonify({"error": "Server is in safe mode. Restart without --safe to enable control."}), 403
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
            "jacobian": np.zeros((6, 7)).tolist(),  # placeholder
            "gripper_pos": server.gripper_pos,
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
        return jsonify({"jacobian": np.zeros((6, 7)).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": server.gripper_pos})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        err = _require_not_safe()
        if err: return err
        server.reset_joint()
        return "Reset Joint"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        err = _require_not_safe()
        if err: return err
        server.clear_error()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        err = _require_not_safe()
        if err: return err
        pos = np.array(request.json["arr"])
        server.move_to_pose(pos)
        return "Moved"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        err = _require_not_safe()
        if err: return err
        server._start_impedance()
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        err = _require_not_safe()
        if err: return err
        server.robot.clear_set()
        server.robot.set_state(arm=server.arm, state=0)
        server.robot.send_cmd()
        return "Stopped impedance"

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        err = _require_not_safe()
        if err: return err
        server.update_compliance(request.json)
        return "Updated compliance parameters"

    # Gripper stubs (no gripper for peg-in-hole)
    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        server.gripper_pos = 1.0
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        server.gripper_pos = 0.0
        return "Closed"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        return "No gripper"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        return "No gripper"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        return "No gripper"

    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)


if __name__ == "__main__":
    app.run(main)
