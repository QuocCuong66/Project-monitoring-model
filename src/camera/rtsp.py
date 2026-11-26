"""Network / remote IP camera stream (RTSP, RTSPS, RTMP, HTTP MJPEG, HLS) with auto-reconnection.

Remote cameras (over the internet, a VPN or a 4G router) differ from LAN ones: UDP packets
get dropped or blocked by NAT, connections hang instead of failing, and links drop for
minutes. So RTSP is forced over TCP, opens and reads have timeouts, and reconnection backs
off exponentially and never gives up (the monitor must recover on its own, 24/7).
"""

import os
import re
import threading
import time
from typing import Any, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
from src.camera.stream import CameraStream

SUPPORTED_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https")

# OPENCV_FFMPEG_CAPTURE_OPTIONS is process-wide and read when a capture opens
_ffmpeg_env_lock = threading.Lock()


def mask_url(url: str) -> str:
    """Hide the password in a camera URL for display and logs."""
    return re.sub(r"://([^:/@]+):([^@]*)@", r"://\1:****@", url or "")


def build_stream_url(url: str, username: Optional[str] = None, password: Optional[str] = None) -> str:
    """Insert credentials into a stream URL, percent-encoding them.

    Passwords with '@', ':', '#' or '/' break a hand-typed rtsp://user:pass@host URL; entering
    them separately and encoding here avoids that. Credentials already in the URL are replaced
    only when a username is given.
    """
    url = (url or "").strip()
    if not username:
        return url
    parts = urlsplit(url)
    host = parts.netloc.rsplit("@", 1)[-1]
    auth = quote(username, safe="")
    if password:
        auth += ":" + quote(password, safe="")
    return urlunsplit((parts.scheme, f"{auth}@{host}", parts.path, parts.query, parts.fragment))


def validate_stream_url(url: str) -> Optional[str]:
    """Return an error message (Vietnamese, shown in the UI) or None when the URL is usable."""
    parts = urlsplit((url or "").strip())
    if parts.scheme.lower() not in SUPPORTED_SCHEMES:
        return (
            "Đường dẫn camera phải bắt đầu bằng rtsp://, rtsps://, rtmp://, http:// hoặc https:// "
            "(ví dụ: rtsp://admin:pass@tenmien.ddns.net:554/stream1)."
        )
    if not parts.hostname:
        return "Đường dẫn camera thiếu địa chỉ IP hoặc tên miền."
    return None


class RTSPStream(CameraStream):
    """Captures video frames from a network camera with resilience to network drops."""

    def __init__(
        self,
        url: str,
        reconnect_delay_seconds: float = 2.0,
        max_reconnect_delay_seconds: float = 30.0,
        max_reconnect_attempts: int = 0,
        max_queue_size: int = 2,
        transport: str = "tcp",
        open_timeout_ms: int = 10000,
        read_timeout_ms: int = 8000,
        buffer_size: int = 1,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.url = url
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_delay_seconds = max(reconnect_delay_seconds, max_reconnect_delay_seconds)
        self.max_reconnect_attempts = max_reconnect_attempts  # 0 = retry forever
        self.transport = transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.buffer_size = buffer_size
        self._source_name = "rtsp_stream"
        self._cap: Optional[cv2.VideoCapture] = None
        self._failed_attempts = 0
        self._next_retry_at = 0.0
        self.last_error: Optional[str] = None
        self.last_frame_time = 0.0
        # When True, start() succeeds even if the camera is unreachable right now and keeps
        # retrying in the background (app startup with a remote camera that is still offline).
        # Interactive switching leaves it False so the user gets immediate feedback.
        self.allow_offline_start = False

    @classmethod
    def from_config(cls, url: str, cfg: Optional[dict[str, Any]] = None, max_queue_size: int = 2) -> "RTSPStream":
        """Build from the `rtsp:` section of camera.yaml (shipped defaults + user overrides)."""
        if cfg is None:
            try:
                from src.camera.camera_settings import load_camera_config
                cfg = load_camera_config().get("rtsp", {}) or {}
            except Exception:
                cfg = {}
        return cls(
            url=url,
            reconnect_delay_seconds=float(cfg.get("reconnect_delay_seconds", 2.0)),
            max_reconnect_delay_seconds=float(cfg.get("max_reconnect_delay_seconds", 30.0)),
            max_reconnect_attempts=int(cfg.get("max_reconnect_attempts", 0)),
            max_queue_size=max_queue_size,
            transport=str(cfg.get("transport", "tcp")),
            open_timeout_ms=int(cfg.get("open_timeout_ms", 10000)),
            read_timeout_ms=int(cfg.get("read_timeout_ms", 8000)),
            buffer_size=int(cfg.get("buffer_size", 1)),
        )

    def start(self) -> bool:
        if super().start():
            return True
        if not self.allow_offline_start:
            return False
        self._schedule_retry()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraWorker")
        self._thread.start()
        return True

    def _ffmpeg_options(self) -> str:
        opts = []
        scheme = urlsplit(self.url).scheme.lower()
        if scheme in ("rtsp", "rtsps") and self.transport in ("tcp", "udp", "http"):
            opts.append(f"rtsp_transport;{self.transport}")
        return "|".join(opts)

    def _open_capture(self) -> bool:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms,
        ]
        with _ffmpeg_env_lock:
            previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._ffmpeg_options()
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            finally:
                if previous is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous
        if not cap.isOpened():
            cap.release()
            self._cap = None
            self.last_error = "Không mở được luồng camera (sai địa chỉ / tài khoản, hoặc không truy cập được mạng)."
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)  # Avoid buffering stale frames in the backend
        self._cap = cap
        return True

    def _schedule_retry(self) -> None:
        self._failed_attempts += 1
        delay = min(self.max_reconnect_delay_seconds, self.reconnect_delay_seconds * (2 ** (self._failed_attempts - 1)))
        self._next_retry_at = time.time() + delay

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            if self.max_reconnect_attempts > 0 and self._failed_attempts >= self.max_reconnect_attempts:
                time.sleep(0.5)
                return False, None
            wait = self._next_retry_at - time.time()
            if wait > 0:
                time.sleep(min(wait, 0.5))  # short sleeps so release() is not blocked by a long backoff
                return False, None
            if not self._open_capture():
                self._schedule_retry()
                return False, None

        success, frame = self._cap.read()
        if not success or frame is None:
            # Stream drop (or read timeout): release and let the next loop reconnect with backoff
            self.last_error = "Mất kết nối với camera, đang tự động kết nối lại..."
            self._close_capture()
            self._schedule_retry()
            return False, None
        self._failed_attempts = 0
        self.last_error = None
        self.last_frame_time = time.time()
        return True, frame

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def connection_status(self) -> dict[str, Any]:
        """Connection health for the dashboard."""
        connected = self._cap is not None and self._cap.isOpened() and time.time() - self.last_frame_time < 5.0
        return {
            "state": "connected" if connected else ("reconnecting" if self._running else "stopped"),
            "reconnect_attempts": self._failed_attempts,
            "seconds_since_last_frame": round(time.time() - self.last_frame_time, 1) if self.last_frame_time else None,
            "last_error": self.last_error,
        }

    def get_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            if 0 < fps <= 120:
                return fps
        return 25.0

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (1920, 1080)

    def get_width(self) -> int:
        return self.get_resolution()[0]

    def get_height(self) -> int:
        return self.get_resolution()[1]


def probe_stream(url: str, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Open a stream once, read one frame and report what came back, without keeping it."""
    stream = RTSPStream.from_config(url, cfg)
    t0 = time.perf_counter()
    if not stream._open_capture():
        return {"ok": False, "message": stream.last_error}
    try:
        ok, frame = stream._cap.read()
        if not ok or frame is None:
            return {"ok": False, "message": "Kết nối được nhưng không nhận được hình ảnh (kiểm tra kênh/luồng phụ của camera)."}
        h, w = frame.shape[:2]
        return {
            "ok": True,
            "message": f"Kết nối thành công: {w}x{h}, {stream.get_fps():.0f} FPS.",
            "resolution": f"{w}x{h}",
            "fps": round(stream.get_fps(), 1),
            "connect_ms": round((time.perf_counter() - t0) * 1000.0),
        }
    finally:
        stream._close_capture()
