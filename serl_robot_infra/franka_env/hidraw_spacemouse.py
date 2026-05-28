"""Shared HidrawSpaceMouse — reads 3Dconnexion SpaceMouse via hidraw.

Both cr5af_server teleop and ServerSpacemouseIntervention wrapper import
from here so only one copy of the HID parsing logic exists.
"""

import time
import threading


class HidrawSpaceMouse:
    """Reads SpaceMouse via hidraw (easyhid) — works on both USB and Bluetooth.

    Background thread polls hidraw for HID reports, caches latest state.
    Returns the same action format as SpaceMouseExpert:
      action = [x, y, z, roll, pitch, yaw]  (normalized to ~[-1, 1])
      buttons = [BTN_0, BTN_1]
    """

    # SpaceMouse Wireless BT: vid=0x256F, pid=0xC63A
    # SpaceMouse Wireless (USB dongle): vid=0x256F, pid=0xC62E
    _SUPPORTED_IDS = [(0x256F, 0xC63A), (0x256F, 0xC62E)]

    def __init__(self, device_path: str = ""):
        self._device = None
        self._axes = [0.0] * 6  # [x, y, z, roll, pitch, yaw] normalized
        self._buttons = [0, 0]
        self._running = True

        try:
            from easyhid import Enumeration
        except ImportError:
            print("WARNING: easyhid not installed. SpaceMouse will return zeros.")
            print("  Install: pip install easyhid")
            return

        hid = Enumeration()
        all_hids = hid.find()

        found_dev = None
        if device_path:
            for d in all_hids:
                if d.path == device_path:
                    for vid, pid in self._SUPPORTED_IDS:
                        if d.vendor_id == vid and d.product_id == pid:
                            found_dev = d
                            break
                    if found_dev:
                        break
        else:
            for d in all_hids:
                for vid, pid in self._SUPPORTED_IDS:
                    if d.vendor_id == vid and d.product_id == pid:
                        found_dev = d
                        break
                if found_dev:
                    break

        if found_dev is None:
            print("WARNING: No SpaceMouse found via hidraw. Teleop will return zeros.")
            return

        try:
            found_dev.open()
            found_dev.set_nonblocking(True)
        except Exception as e:
            print(f"WARNING: Failed to open SpaceMouse hidraw: {e}")
            print("  Try: sudo chmod 666 /dev/hidraw*")
            return

        self._device = found_dev
        self._product_id = found_dev.product_id
        # pid C63A (BT) and C62E (Wireless dongle) use 13-byte reports:
        # channel 1: [1, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, rx_lo, rx_hi, ry_lo, ry_hi, rz_lo, rz_hi]
        # channel 3: [3, ..., btn_byte, ...]
        self._bytes_to_read = 13

        self._thread = threading.Thread(target=self._hidraw_loop, daemon=True)
        self._thread.start()
        print(f"SpaceMouse connected: {found_dev.path} "
              f"(vid=0x{found_dev.vendor_id:04X} pid=0x{found_dev.product_id:04X})")

    def _hidraw_loop(self):
        """Background loop: poll hidraw device for HID reports."""

        def _to_int16(lo, hi):
            val = lo | (hi << 8)
            if val >= 32768:
                val = -(65536 - val)
            return val

        while self._running:
            try:
                data = self._device.read(self._bytes_to_read)
                # Fallback: try with short timeout if nonblocking returns empty
                if not data:
                    data = self._device.read(self._bytes_to_read, timeout_ms=50)
                if data and len(data) >= 3:
                    channel = data[0]
                    if channel == 1 and len(data) >= 13:
                        self._axes[0] = _to_int16(data[1], data[2]) / 350.0   # x
                        self._axes[1] = _to_int16(data[3], data[4]) / -350.0  # y (inverted)
                        self._axes[2] = _to_int16(data[5], data[6]) / 350.0   # z
                        self._axes[3] = _to_int16(data[7], data[8]) / -350.0  # pitch (inverted)
                        self._axes[4] = _to_int16(data[9], data[10]) / -350.0 # roll (inverted)
                        self._axes[5] = _to_int16(data[11], data[12]) / 350.0 # yaw
                    elif channel == 3 and len(data) >= 2:
                        btn_byte = data[1]
                        self._buttons[0] = 1 if (btn_byte & 0x01) else 0
                        self._buttons[1] = 1 if (btn_byte & 0x02) else 0
            except Exception:
                pass
            time.sleep(0.001)  # ~1ms poll interval, ~1000Hz max

    def get_state(self) -> tuple[list[float], list[int]]:
        """Return (action, buttons) matching SpaceMouseExpert format.

        action = [tx, ty, tz, roll, pitch, yaw] (normalized ~[-1, 1])
        """
        a = self._axes  # [x, y, z, pitch, roll, yaw]
        action = [a[1], -a[0], a[2], a[4], a[3], a[5]]  # swap x<->y, pitch<->roll
        return action, self._buttons[:]

    def close(self):
        self._running = False
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
