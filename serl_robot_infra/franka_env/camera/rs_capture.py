import threading
import time

import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.s = self.profile.get_device().query_sensors()[0]
        self.s.set_option(rs.option.exposure, exposure)

        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Background thread for non-blocking reads
        self._latest_frame = None
        self._latest_depth = None
        self._frame_ready = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait for first frame
        if not self._frame_ready.wait(timeout=10.0):
            print(f"[RSCapture {self.name}] WARNING: no frame received within 10s")

    def _capture_loop(self):
        """Background thread: continuously read frames and cache the latest."""
        while self._running:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=5000)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()

                if color_frame and color_frame.is_video_frame():
                    image = np.asarray(color_frame.get_data())
                    self._latest_frame = image
                    if self.depth:
                        depth_frame = aligned_frames.get_depth_frame()
                        if depth_frame and depth_frame.is_depth_frame():
                            self._latest_depth = np.expand_dims(
                                np.asarray(depth_frame.get_data()), axis=2
                            )
                    self._frame_ready.set()
            except RuntimeError:
                pass
            except Exception:
                time.sleep(0.001)

    def read(self):
        """Return latest cached frame (non-blocking)."""
        if self._latest_frame is None:
            return False, None
        if self.depth and self._latest_depth is not None:
            return True, np.concatenate((self._latest_frame.copy(), self._latest_depth), axis=-1)
        return True, self._latest_frame.copy()

    def close(self):
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.pipe.stop()
        self.cfg.disable_all_streams()
