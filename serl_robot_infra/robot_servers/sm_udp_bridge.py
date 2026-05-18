"""SpaceMouse UDP bridge: sender (run on x86_64 machine where SpaceMouse works)."""
import socket
import struct
import time
import sys
import numpy as np

sys.path.insert(0, "serl_robot_infra")
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert

THOR_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.5.2"
UDP_PORT = 9999


def main():
    print("Opening SpaceMouse...")
    expert = SpaceMouseExpert()
    time.sleep(0.5)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)

    # Calibration
    dt = 0.02
    print("Calibrating zero offset...")
    zero_samples = []
    for _ in range(50):
        a, _ = expert.get_action()
        zero_samples.append(np.array(a[:6], dtype=np.float32))
        time.sleep(dt)
    zero_offset = np.median(zero_samples, axis=0)
    print("Zero offset:", zero_offset)
    print("Sending to %s:%d..." % (THOR_IP, UDP_PORT))

    seq = 0
    try:
        while True:
            action, buttons = expert.get_action()
            action = np.array(action, dtype=np.float32)
            action[:6] -= zero_offset

            # Pack: seq(4) + 6x float32(24) + 2x int32 buttons(8) = 36 bytes
            data = struct.pack("<I6f2i", seq, *action[:6], int(buttons[0]), int(buttons[1]))
            sock.sendto(data, (THOR_IP, UDP_PORT))
            seq += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        expert.close()


if __name__ == "__main__":
    main()
