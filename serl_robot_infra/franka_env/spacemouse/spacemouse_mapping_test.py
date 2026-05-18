"""SpaceMouse → Robot End-Effector Mapping Test.

Run this script WITHOUT the robot connected to familiarize yourself with
the SpaceMouse control mapping before operating the real arm.

The SpaceMouse outputs a 6-DOF action [x, y, z, roll, pitch, yaw].
In HIL-SERL, this action is interpreted in the END-EFFECTOR FRAME
(not the world/base frame). This means:

  - Pushing the SpaceMouse FORWARD  -> moves the end-effector along its own Y axis
  - Pushing the SpaceMouse RIGHT    -> moves the end-effector along its own X axis
  - Pulling the SpaceMouse UP       -> moves the end-effector along its own Z axis
  - Twisting/tilting the SpaceMouse -> rotates the end-effector around its own axes

IMPORTANT SAFETY NOTE:
If the robot end-effector is pointing DOWN (e.g., for top-down insertion),
the end-effector Z axis points DOWNWARD. Therefore:
  - SpaceMouse "UP" (z+)  -> robot moves DOWN (into the hole)
  - SpaceMouse "DOWN" (z-)-> robot moves UP (away from the hole)

Press Ctrl+C to exit.
"""
import time
import numpy as np
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


def describe_action(action: np.ndarray) -> str:
    """Return a human-readable description of the active axes."""
    parts = []
    x, y, z, roll, pitch, yaw = action
    thresh = 0.05

    if abs(x) > thresh:
        parts.append(f"X={'RIGHT' if x > 0 else 'LEFT':<5} ({x:+.3f})")
    if abs(y) > thresh:
        parts.append(f"Y={'BACK ' if y > 0 else 'FWD  ':<5} ({y:+.3f})")
    if abs(z) > thresh:
        parts.append(f"Z={'DOWN ' if z > 0 else 'UP   ':<5} ({z:+.3f})")
    if abs(roll) > thresh:
        parts.append(f"Roll={'+' if roll > 0 else '-':<5} ({roll:+.3f})")
    if abs(pitch) > thresh:
        parts.append(f"Pitch={'+' if pitch > 0 else '-':<5} ({pitch:+.3f})")
    if abs(yaw) > thresh:
        parts.append(f"Yaw={'+' if yaw > 0 else '-':<5} ({yaw:+.3f})")

    return " | ".join(parts) if parts else "(idle)"


def main():
    print("=" * 70)
    print("SpaceMouse → Robot Mapping Test")
    print("=" * 70)
    print("""
Action frame: END-EFFECTOR FRAME (not world frame!)

SpaceMouse physical motion   ->   Robot end-effector motion
------------------------------------------------------------
Push forward (away from you)  ->  Y+  (along end-effector Y axis)
Pull back   (toward you)      ->  Y-
Push right                    ->  X+  (along end-effector X axis)
Push left                     ->  X-
Pull up                       ->  Z+  (along end-effector Z axis)
Push down                     ->  Z-
Twist clockwise (top view)    ->  Yaw+
Tilt right                    ->  Roll+
Tilt forward (nose down)      ->  Pitch+

SAFETY: If the end-effector points DOWN, Z+ means DOWNWARD (into the hole).
        Z- means UPWARD (retract).

Left button  = CLOSE gripper (if gripper enabled)
Right button = OPEN gripper  (if gripper enabled)
""")
    print("=" * 70)
    print("Operate the SpaceMouse now. Press Ctrl+C to exit.\n")

    expert = SpaceMouseExpert()

    with np.printoptions(precision=3, suppress=True):
        try:
            while True:
                action, buttons = expert.get_action()
                desc = describe_action(action)
                btn_str = f"L={buttons[0]} R={buttons[1]}"
                print(f"action={action}  [{btn_str}]  ->  {desc}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExited.")
            expert.close()


if __name__ == "__main__":
    main()
