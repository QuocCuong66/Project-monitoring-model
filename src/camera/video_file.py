"""Local video file stream implementation for evaluation and testing."""

import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from src.camera.stream import CameraStream


class VideoFileStream(CameraStream):
    """Streams frames from a recorded video file (e.g. MP4, AVI)."""

    def __init__(
        self,
        video_path: str | Path,
        loop: bool = False,
        simulate_realtime: bool = True,
        max_queue_size: int = 5,
        fps: Optional[float] = 15.0,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size, target_fps=float(fps or 15.0))
        self.video_path = str(video_path)
        self.loop = loop
        self.simulate_realtime = simulate_realtime
        self._source_name = Path(self.video_path).name
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = float(fps or 15.0)
        self._last_read_time = 0.0

    def _open_capture(self) -> bool:
        if not Path(self.video_path).exists():
            return False
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            return False
        native_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if native_fps > 0 and self.target_fps <= 0:
            self._fps = native_fps
        return True

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        effective_fps = self.target_fps if self.target_fps > 0 else self._fps
        if self.simulate_realtime and self._last_read_time > 0:
            target_dt = 1.0 / max(1.0, effective_fps)
            elapsed = time.time() - self._last_read_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
        self._last_read_time = time.time()

        success, frame = self._cap.read()
        if not success:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self._cap.read()
            return False, None

        return True, frame

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_fps(self, fps: float) -> None:
        super().set_fps(fps)
        self._fps = float(fps)

    def get_fps(self) -> float:
        return float(self.target_fps) if self.target_fps > 0 else self._fps

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (1280, 720)
