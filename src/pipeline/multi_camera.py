"""Multi-camera monitoring: extra cameras next to the primary (UI-switchable) camera.

Every extra camera runs as a CameraSession: its own stream, ingestion thread and its own
DecoupledPipeline session (tracker, temporal buffers, risk state machine, alert cooldowns and
fall-clip pre-roll), so people tracked in one room never leak into another room's temporal
reasoning. The heavy pose / temporal / anomaly models are loaded once and shared through
DecoupledPipeline.create_camera_session(), which keeps VRAM flat as cameras are added.
"""

import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from src.camera.camera_settings import save_extra_cameras
from src.camera.rtsp import RTSPStream, mask_url, validate_stream_url
from src.camera.stream import CameraStream, FramePacket
from src.camera.video_file import VideoFileStream
from src.camera.webcam import WebcamStream

logger = logging.getLogger(__name__)

SOURCE_TYPES = ("webcam", "rtsp", "video")
EMERGENCY_STATES = ("HIGH_RISK", "ALERT_SENT")

StreamFactory = Callable[[dict[str, Any]], CameraStream]
SessionFactory = Callable[[str, str], Any]


class CameraConfigError(ValueError):
    """Invalid extra camera definition (message is shown to the user)."""


def normalize_camera_entry(entry: dict[str, Any], existing_ids: set[str]) -> dict[str, Any]:
    """Validate a camera definition from the UI / config and fill in an id and a name."""
    source_type = str(entry.get("source_type") or "rtsp").strip().lower()
    if source_type not in SOURCE_TYPES:
        raise CameraConfigError(f"Loại nguồn không hỗ trợ: {source_type}")

    cam: dict[str, Any] = {"source_type": source_type, "enabled": bool(entry.get("enabled", True))}
    if source_type == "webcam":
        try:
            cam["device_index"] = int(entry.get("device_index", 0))
        except (TypeError, ValueError):
            raise CameraConfigError("Số thứ tự webcam không hợp lệ.")
    elif source_type == "rtsp":
        url = str(entry.get("rtsp_url") or "").strip()
        error = validate_stream_url(url)
        if error:
            raise CameraConfigError(error)
        cam["rtsp_url"] = url
    else:
        path = str(entry.get("video_path") or "").strip()
        if not path:
            raise CameraConfigError("Vui lòng nhập đường dẫn tệp video.")
        cam["video_path"] = path

    cam_id = str(entry.get("id") or "").strip()
    if not cam_id:
        cam_id = f"cam_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", cam_id):
        raise CameraConfigError("Mã camera chỉ được chứa chữ, số, '_' hoặc '-'.")
    if cam_id in existing_ids or cam_id == "main":
        raise CameraConfigError(f"Mã camera '{cam_id}' đã tồn tại.")
    cam["id"] = cam_id

    name = str(entry.get("name") or "").strip()[:60]
    cam["name"] = name or _default_name(cam)
    return cam


def _default_name(cam: dict[str, Any]) -> str:
    if cam["source_type"] == "webcam":
        return f"Webcam {cam['device_index']}"
    if cam["source_type"] == "video":
        return f"Video {Path(cam['video_path']).name}"
    return f"Camera IP ({mask_url(cam['rtsp_url'])})"


def public_camera_entry(cam: dict[str, Any]) -> dict[str, Any]:
    """Camera definition safe to send to the browser (RTSP password masked)."""
    out = dict(cam)
    if out.get("rtsp_url"):
        out["rtsp_url"] = mask_url(out["rtsp_url"])
    return out


def default_stream_factory(camera_cfg: dict[str, Any]) -> StreamFactory:
    """Build camera streams with the shipped / user camera.yaml settings."""
    webcam_cfg = camera_cfg.get("webcam", {}) or {}
    rtsp_cfg = camera_cfg.get("rtsp", {}) or {}

    def create(cam: dict[str, Any]) -> CameraStream:
        if cam["source_type"] == "webcam":
            return WebcamStream(
                device_index=int(cam["device_index"]),
                width=webcam_cfg.get("width", 1280),
                height=webcam_cfg.get("height", 720),
                fps=webcam_cfg.get("fps", 30),
                api_preference=webcam_cfg.get("api_preference", "dshow"),
            )
        if cam["source_type"] == "rtsp":
            stream = RTSPStream.from_config(cam["rtsp_url"], rtsp_cfg)
            # Remote cameras may be offline right now (router / 4G restarting): keep retrying
            stream.allow_offline_start = True
            return stream
        return VideoFileStream(video_path=cam["video_path"], loop=True)

    return create


class CameraSession:
    """One extra camera: stream -> its own AI pipeline session -> latest JPEG for the web tile."""

    def __init__(
        self,
        cam: dict[str, Any],
        stream: CameraStream,
        pipeline: Any,
        tile_fps: float = 10.0,
        tile_max_width: int = 640,
        jpeg_quality: int = 80,
        stale_after_seconds: float = 5.0,
    ) -> None:
        self.cam = cam
        self.stream = stream
        self.pipeline = pipeline
        self.tile_period = 1.0 / max(0.5, float(tile_fps))
        self.tile_max_width = int(tile_max_width)
        self.jpeg_quality = int(jpeg_quality)
        self.stale_after_seconds = float(stale_after_seconds)

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._last_frame_time = 0.0
        self._last_tile_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.stream_started = False

    @property
    def camera_id(self) -> str:
        return self.cam["id"]

    def start(self) -> None:
        self.stream_started = bool(self.stream.start())
        if hasattr(self.pipeline, "start"):
            self.pipeline.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"Camera-{self.camera_id}")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if hasattr(self.pipeline, "stop"):
            try:
                self.pipeline.stop()
            except Exception:
                pass
        try:
            self.stream.release()
        except Exception:
            pass

    def _loop(self) -> None:
        while self._running:
            try:
                if not self.stream.is_opened():
                    time.sleep(0.1)
                    continue
                packet = self.stream.read(timeout=0.5)
                if packet is None or packet.frame is None:
                    continue
                self.process_packet(packet)
            except Exception:
                logger.exception("[CameraSession %s] frame processing failed", self.camera_id)
                time.sleep(0.1)

    def process_packet(self, packet: FramePacket) -> None:
        """AI gets every fresh frame; the web tile is rendered / encoded at tile_fps only."""
        self._last_frame_time = time.time()
        self.pipeline.submit_frame(packet)
        if self._last_frame_time - self._last_tile_time < self.tile_period:
            return
        self._last_tile_time = self._last_frame_time
        annotated, _alerts = self.pipeline.render_frame(packet.frame, packet.timestamp)
        self._publish(annotated)

    def _publish(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.tile_max_width > 0 and w > self.tile_max_width:
            scale = self.tile_max_width / float(w)
            frame = cv2.resize(frame, (self.tile_max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self._lock:
                self._latest_jpeg = jpeg.tobytes()
                self._jpeg_seq += 1

    def latest_jpeg(self) -> tuple[Optional[bytes], int]:
        with self._lock:
            return self._latest_jpeg, self._jpeg_seq

    def set_blur(self, enabled: bool) -> None:
        if hasattr(self.pipeline, "blur_faces"):
            self.pipeline.blur_faces = bool(enabled)

    def status(self) -> dict[str, Any]:
        age = time.time() - self._last_frame_time if self._last_frame_time else None
        online = age is not None and age < self.stale_after_seconds
        shared = getattr(self.pipeline, "shared_state", None)
        tracks = 0
        emergency = False
        ai_fps = 0.0
        if shared is not None:
            with shared.lock:
                tracks = len(shared.tracks)
                emergency = any(
                    a.state in EMERGENCY_STATES for tid, a in shared.assessments.items() if tid in shared.tracks
                )
                ai_fps = float(shared.ai_fps)
        connection = self.stream.connection_status() if hasattr(self.stream, "connection_status") else None
        return {
            **public_camera_entry(self.cam),
            "online": online,
            "last_frame_age_s": round(age, 1) if age is not None else None,
            "tracked_persons": tracks,
            "emergency": emergency,
            "ai_fps": round(ai_fps, 1),
            "connection": connection,
        }


class MultiCameraMonitor:
    """Starts, stops and persists the extra cameras; the API layer only talks to this class."""

    def __init__(
        self,
        settings: Optional[dict[str, Any]] = None,
        stream_factory: Optional[StreamFactory] = None,
        session_factory: Optional[SessionFactory] = None,
        persist: Optional[Callable[[list[dict[str, Any]]], Any]] = save_extra_cameras,
        jpeg_quality: int = 80,
    ) -> None:
        settings = settings or {}
        self.max_cameras = int(settings.get("max_cameras", 3))
        self.tile_fps = float(settings.get("tile_fps", 10.0))
        self.tile_max_width = int(settings.get("tile_max_width", 640))
        self.stale_after_seconds = float(settings.get("stale_after_seconds", 5.0))
        self.jpeg_quality = int(jpeg_quality)
        self.stream_factory = stream_factory
        self.session_factory = session_factory
        self.persist = persist
        self.blur_faces = False

        self._lock = threading.RLock()
        self._sessions: dict[str, CameraSession] = {}
        self._cameras: list[dict[str, Any]] = []  # every saved camera, enabled or not, in order

    def attach_base_pipeline(self, pipeline: Any) -> None:
        """Share the primary camera's loaded models (DecoupledPipeline) with the extra cameras."""
        if hasattr(pipeline, "create_camera_session"):
            self.session_factory = pipeline.create_camera_session
            self.blur_faces = bool(getattr(pipeline, "blur_faces", False))

    def _require_factories(self) -> None:
        if self.stream_factory is None:
            self.stream_factory = default_stream_factory({})
        if self.session_factory is None:
            # Primary is not a DecoupledPipeline (sequential fallback): load the models once here
            from src.pipeline.decoupled_pipeline import DecoupledPipeline
            base = DecoupledPipeline()
            self.session_factory = base.create_camera_session

    def load_saved(self, cameras: list[dict[str, Any]]) -> list[str]:
        """Restore cameras saved earlier; start the enabled ones. Returns error messages."""
        errors: list[str] = []
        for entry in cameras or []:
            try:
                with self._lock:
                    cam = normalize_camera_entry(entry, {c["id"] for c in self._cameras})
                    if len(self._cameras) >= self.max_cameras:
                        raise CameraConfigError(f"Đã đạt tối đa {self.max_cameras} camera thêm.")
                    self._cameras.append(cam)
                if cam["enabled"]:
                    self._start_session(cam)
            except Exception as e:
                errors.append(f"{entry.get('name') or entry.get('id') or 'camera'}: {e}")
                logger.warning("[MultiCamera] skip saved camera: %s", e)
        return errors

    def add_camera(self, entry: dict[str, Any], primary_webcam_index: Optional[int] = None) -> dict[str, Any]:
        with self._lock:
            if len(self._cameras) >= self.max_cameras:
                raise CameraConfigError(
                    f"Đã đạt tối đa {self.max_cameras} camera thêm (mọi camera dùng chung một GPU). "
                    "Có thể tăng extra_cameras.max_cameras trong configs/camera.yaml nếu máy đủ mạnh."
                )
            cam = normalize_camera_entry(entry, {c["id"] for c in self._cameras})
            if cam["source_type"] == "webcam":
                in_use = {c.get("device_index") for c in self._cameras if c["source_type"] == "webcam"}
                if primary_webcam_index is not None:
                    in_use.add(primary_webcam_index)
                if cam["device_index"] in in_use:
                    raise CameraConfigError(f"Webcam {cam['device_index']} đang được dùng bởi camera khác.")
            self._cameras.append(cam)
        try:
            session = self._start_session(cam)
        except Exception:
            with self._lock:
                self._cameras = [c for c in self._cameras if c["id"] != cam["id"]]
            raise
        with self._lock:
            self._save()
        return session.status()

    def remove_camera(self, cam_id: str) -> bool:
        with self._lock:
            before = len(self._cameras)
            self._cameras = [c for c in self._cameras if c["id"] != cam_id]
            session = self._sessions.pop(cam_id, None)
            removed = len(self._cameras) != before
            if removed:
                self._save()
        if session is not None:
            session.stop()
        return removed

    def _start_session(self, cam: dict[str, Any]) -> CameraSession:
        self._require_factories()
        stream = self.stream_factory(cam)
        pipeline = self.session_factory(cam["id"], cam["name"])
        if hasattr(pipeline, "blur_faces"):
            pipeline.blur_faces = self.blur_faces
        session = CameraSession(
            cam,
            stream,
            pipeline,
            tile_fps=self.tile_fps,
            tile_max_width=self.tile_max_width,
            jpeg_quality=self.jpeg_quality,
            stale_after_seconds=self.stale_after_seconds,
        )
        session.start()
        with self._lock:
            old = self._sessions.pop(cam["id"], None)
            self._sessions[cam["id"]] = session
        if old is not None:
            old.stop()
        return session

    def _save(self) -> None:
        if self.persist is None:
            return
        try:
            self.persist(list(self._cameras))
        except Exception as e:
            logger.error("[MultiCamera] cannot save camera list: %s", e)

    def get_session(self, cam_id: str) -> Optional[CameraSession]:
        with self._lock:
            return self._sessions.get(cam_id)

    def list_status(self) -> list[dict[str, Any]]:
        with self._lock:
            cameras = list(self._cameras)
            sessions = dict(self._sessions)
        out = []
        for cam in cameras:
            session = sessions.get(cam["id"])
            if session is not None:
                out.append(session.status())
            else:
                out.append({**public_camera_entry(cam), "online": False, "tracked_persons": 0, "emergency": False})
        return out

    def webcam_indices_in_use(self) -> set[int]:
        with self._lock:
            return {int(c["device_index"]) for c in self._cameras if c["source_type"] == "webcam"}

    def set_blur(self, enabled: bool) -> None:
        with self._lock:
            self.blur_faces = bool(enabled)
            sessions = list(self._sessions.values())
        for session in sessions:
            session.set_blur(enabled)

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.stop()
