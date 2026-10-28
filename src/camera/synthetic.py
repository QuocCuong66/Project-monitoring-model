"""Synthetic fallback camera stream generating animated test room frames."""

import time
from typing import Optional, Tuple
import cv2
import numpy as np
from src.camera.stream import CameraStream


class SyntheticCameraStream(CameraStream):
    """Fallback camera stream that generates synthetic test frames with animated human stick figure."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        max_queue_size: int = 2,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self._source_name = "synthetic_camera"
        self._last_read_time = 0.0
        self._step = 0

    def _open_capture(self) -> bool:
        return True

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._last_read_time > 0:
            target_dt = 1.0 / max(1.0, self.target_fps)
            elapsed = time.time() - self._last_read_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
        self._last_read_time = time.time()
        self._step += 1

        # Create room background
        frame = np.full((self.target_height, self.target_width, 3), (35, 30, 30), dtype=np.uint8)

        # Floor line
        floor_y = int(self.target_height * 0.8)
        cv2.line(frame, (0, floor_y), (self.target_width, floor_y), (60, 60, 60), 2)

        # Simulated walking/standing figure
        cycle = (self._step % 300) / 300.0  # 10 second cycle
        cx = int(self.target_width * (0.3 + 0.4 * np.sin(cycle * 2 * np.pi)))
        cy = int(floor_y - 180)

        # Draw a human-like silhouette
        # Head
        cv2.circle(frame, (cx, cy - 40), 25, (200, 200, 200), -1)
        # Torso
        cv2.rectangle(frame, (cx - 20, cy - 15), (cx + 20, cy + 90), (180, 160, 140), -1)
        # Legs
        cv2.line(frame, (cx - 10, cy + 90), (cx - 15, floor_y - 5), (120, 120, 160), 10)
        cv2.line(frame, (cx + 10, cy + 90), (cx + 15, floor_y - 5), (120, 120, 160), 10)
        # Arms
        cv2.line(frame, (cx - 20, cy + 5), (cx - 35, cy + 60), (180, 160, 140), 8)
        cv2.line(frame, (cx + 20, cy + 5), (cx + 35, cy + 60), (180, 160, 140), 8)

        # Watermark notice
        text = f"SYNTHETIC CAMERA FEED (Simulation Mode) | FPS: {self.target_fps}"
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)

        return True, frame

    def _close_capture(self) -> None:
        pass

    def get_fps(self) -> float:
        return float(self.target_fps)

    def get_resolution(self) -> Tuple[int, int]:
        return (self.target_width, self.target_height)
