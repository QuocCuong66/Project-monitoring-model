"""Webcam and USB camera capture stream implementation."""

import sys
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from src.camera.stream import CameraStream
from src.utils.config import load_config

BUSY_MESSAGE = (
    "Webcam ngừng gửi hình (có thể đang bị ứng dụng khác chiếm: Zalo / Teams / Camera / "
    "điện thoại liên kết Phone Link...). Đang tự động kết nối lại..."
)


class WebcamStream(CameraStream):
    """Captures video frames from physical webcam or USB camera.

    Another application can take the camera over while we stream (Windows reports
    MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED / "can't grab frame"). Instead of reading a dead handle
    forever, the stream closes it once no frame has arrived for `reopen_after_seconds` and reopens it
    with an increasing delay, reporting the state like network cameras do.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        api_preference: str = "default",
        max_queue_size: int = 2,
        warmup_seconds: Optional[float] = None,
        reopen_after_seconds: Optional[float] = None,
        max_reconnect_delay_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.api_preference = api_preference
        self._source_name = f"webcam_{device_index}"
        self._cap: Optional[cv2.VideoCapture] = None

        try:
            cfg: dict[str, Any] = load_config("camera.yaml").get("webcam", {}) or {}
        except Exception:
            cfg = {}
        # An opened handle only counts once it delivers a frame within this time
        self.warmup_seconds = float(warmup_seconds if warmup_seconds is not None else cfg.get("warmup_seconds", 2.0))
        # No frame for this long = camera lost / taken over: close and reopen
        self.reopen_after_seconds = float(
            reopen_after_seconds if reopen_after_seconds is not None else cfg.get("reopen_after_seconds", 2.0)
        )
        self.max_reconnect_delay_seconds = float(
            max_reconnect_delay_seconds if max_reconnect_delay_seconds is not None
            else cfg.get("max_reconnect_delay_seconds", 10.0)
        )

        self.last_error: Optional[str] = None
        self.last_frame_time = 0.0
        self._failed_attempts = 0
        self._next_retry_at = 0.0
        self._first_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ opening
    def _backends(self) -> list[int]:
        """Backends to try, in order.

        On Windows Media Foundation numbers cameras differently from DirectShow (and also lists a
        phone linked through Phone Link), so "webcam N" under MSMF can be another device. MSMF is only
        tried when DirectShow does not list this index at all.
        """
        pref = self.api_preference.lower()
        if pref == "v4l2":
            return [cv2.CAP_V4L2]
        if pref == "msmf":
            return [cv2.CAP_MSMF]
        if sys.platform == "win32" and pref in ("dshow", "default"):
            try:
                from src.camera.devices import list_webcam_devices
                listed = list_webcam_devices()
            except Exception:
                listed = None
            if listed is not None and any(d.get("device_index") == self.device_index for d in listed):
                return [cv2.CAP_DSHOW]
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        return [cv2.CAP_ANY]

    def _try_backend(self, backend: int) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.device_index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        deadline = time.time() + self.warmup_seconds
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self._first_frame = frame
                return cap
            time.sleep(0.05)
        cap.release()                       # opens but never delivers: busy / taken over
        return None

    def _open_capture(self) -> bool:
        for backend in self._backends():
            cap = self._try_backend(backend)
            if cap is not None:
                self._cap = cap
                self._failed_attempts = 0
                self.last_error = None
                self.last_frame_time = time.time()
                return True
        self._cap = None
        self.last_error = BUSY_MESSAGE
        return False

    def _schedule_retry(self) -> None:
        self._failed_attempts += 1
        delay = min(self.max_reconnect_delay_seconds, 0.5 * (2 ** (self._failed_attempts - 1)))
        self._next_retry_at = time.time() + delay

    # ------------------------------------------------------------------ reading
    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._first_frame is not None:  # frame grabbed while checking the camera works
            frame, self._first_frame = self._first_frame, None
            return True, frame

        if self._cap is None or not self._cap.isOpened():
            wait = self._next_retry_at - time.time()
            if wait > 0:
                time.sleep(min(wait, 0.5))  # short sleeps so release() is not blocked by a long backoff
                return False, None
            if not self._open_capture():
                self._schedule_retry()
                return False, None

        ok, frame = self._cap.read()
        if ok and frame is not None and frame.size > 0:
            self._failed_attempts = 0
            self.last_error = None
            self.last_frame_time = time.time()
            return True, frame

        if time.time() - self.last_frame_time >= self.reopen_after_seconds:
            print(f"[Camera] Webcam {self.device_index}: không nhận được hình {self.reopen_after_seconds:.0f}s, "
                  f"đóng và kết nối lại (lần {self._failed_attempts + 1}).")
            self.last_error = BUSY_MESSAGE
            self._close_capture()
            self._schedule_retry()
        else:
            time.sleep(0.03)                # brief hiccup: retry soon, without spinning
        return False, None

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def connection_status(self) -> dict[str, Any]:
        """Connection health for the dashboard (same shape as network cameras)."""
        connected = self._cap is not None and time.time() - self.last_frame_time < max(5.0, self.reopen_after_seconds)
        return {
            "state": "connected" if connected else ("reconnecting" if self._running else "stopped"),
            "reconnect_attempts": self._failed_attempts,
            "seconds_since_last_frame": round(time.time() - self.last_frame_time, 1) if self.last_frame_time else None,
            "last_error": self.last_error,
        }

    def get_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return float(self._cap.get(cv2.CAP_PROP_FPS)) or float(self.target_fps)
        return float(self.target_fps)

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (self.target_width, self.target_height)
