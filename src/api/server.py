
"""FastAPI Web Server exposing REST APIs and WebSockets for monitoring and alerts."""

import asyncio
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.pipeline.realtime_pipeline import RealtimePipeline
from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream, build_stream_url, mask_url, probe_stream, validate_stream_url
from src.camera.video_file import VideoFileStream
from src.camera.camera_settings import load_camera_config, save_user_camera_settings
from src.camera.devices import build_webcam_choices, list_webcam_devices
from src.camera.diagnostics import diagnose_stream_failure
from src.camera.discovery import discover_cameras
from src.pipeline.multi_camera import CameraConfigError, MultiCameraMonitor, default_stream_factory
from src.alerts.clip_recorder import find_clip_file
from src.alerts.event_logger import EventLogger, AlertEvent
from src.alerts.telegram import TelegramSettings
from src.api import telegram_routes
from src.auth.middleware import AuthMiddleware
from src.auth import routes as auth_routes
from src.update import routes as update_routes
from src.version import __version__
from src.utils.config import get_project_root, load_config
from src.utils.profiler import ResourceMonitor

# ---------------------------------------------------------------------------
# Security configuration (loaded once at import time)
# ---------------------------------------------------------------------------
_security_cfg: Dict[str, Any] = {}
try:
    _security_cfg = load_config("security.yaml") or {}
except Exception:
    pass

_cors_cfg = _security_cfg.get("cors", {}) or {}
_cors_origins = _cors_cfg.get("allowed_origins", ["http://127.0.0.1:8000", "http://localhost:8000"])

_rate_cfg = _security_cfg.get("rate_limit", {}) or {}
_rate_enabled = _rate_cfg.get("enabled", True)
_rate_login = _rate_cfg.get("login", "5/minute")
_rate_default = _rate_cfg.get("api_default", "60/minute")
_rate_stream = _rate_cfg.get("stream", "10/minute")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, enabled=_rate_enabled)

app = FastAPI(
    title="Fall and Stroke Warning System API",
    description="Real-time Computer Vision & Temporal Behavioral Anomaly Detection API",
    version=__version__,
)

# Rate limiting error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=429,
    content={"detail": "Quá nhiều yêu cầu. Vui lòng thử lại sau."},
))
app.add_middleware(SlowAPIMiddleware)

# Authentication middleware (must be added BEFORE CORS so it processes first)
app.add_middleware(AuthMiddleware)

# CORS: use configured origins instead of wildcard "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth API routes
app.include_router(auth_routes.router)

from fastapi.responses import JSONResponse

# Global runtime state
state_lock = threading.Lock()
pipeline_instance: Optional[RealtimePipeline] = None
camera_stream: Optional[Any] = None
is_running = False
# True when app_runner.py drives camera reads itself; the server must not start a second loop
external_ingestion_loop = False
latest_annotated_frame: Optional[bytes] = None
latest_frame_seq = 0  # Incremented on every newly encoded JPEG
active_ws_clients: List[WebSocket] = []
event_logger = EventLogger()
resource_monitor = ResourceMonitor()
# Extra cameras (multi-camera mode); the primary camera above stays the UI-switchable one
multi_camera_monitor: Optional[MultiCameraMonitor] = None

# Uvicorn's actual running event loop, captured at startup so background
# pipeline threads (which are not the asyncio loop thread) can schedule
# WebSocket sends onto it via run_coroutine_threadsafe.
server_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _apply_telegram_settings_to_pipeline(settings: TelegramSettings) -> None:
    alert_manager = getattr(pipeline_instance, "alert_manager", None)
    if alert_manager is not None:
        alert_manager.dispatcher.update_telegram_settings(settings)


app.include_router(telegram_routes.router)
telegram_routes.add_settings_listener(_apply_telegram_settings_to_pipeline)
app.include_router(update_routes.router)


@app.on_event("startup")
async def _capture_running_event_loop() -> None:
    global server_event_loop
    server_event_loop = asyncio.get_running_loop()
    update_routes.check_in_background()

current_camera_info: Dict[str, Any] = {
    "source_type": "webcam",
    "device_index": 0,
    "rtsp_url": None,
    "video_path": None,
    "name": "Webcam 0 (Mặc định)",
    "is_fallback": False,
}


class CameraStartRequest(BaseModel):
    source_type: str = "webcam"  # "webcam", "rtsp", "video"
    device_index: int = 0
    rtsp_url: Optional[str] = None
    video_path: Optional[str] = None


class CameraSwitchRequest(BaseModel):
    source_type: str = "webcam"  # "webcam", "rtsp" (any network / remote camera), "video"
    device_index: int = 0
    rtsp_url: Optional[str] = None
    # Optional separate credentials, percent-encoded into the URL (passwords with @ : # /)
    rtsp_username: Optional[str] = None
    rtsp_password: Optional[str] = None
    video_path: Optional[str] = None
    save_as_default: bool = False


class CameraTestRequest(BaseModel):
    rtsp_url: str
    rtsp_username: Optional[str] = None
    rtsp_password: Optional[str] = None


def _resolve_stream_url(url: Optional[str], username: Optional[str], password: Optional[str]) -> str:
    full = build_stream_url(url or "", username, password)
    error = validate_stream_url(full)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return full


def _unmask_stream_url(url: Optional[str]) -> Optional[str]:
    """If incoming URL has masked password :****@ and matches current camera, restore real password."""
    if not url or ":****@" not in url:
        return url
    with state_lock:
        current_url = current_camera_info.get("rtsp_url")
    if not current_url:
        return url
    import re
    m_new = re.search(r"://([^:/@]+):[*]+@([^/]+)", url)
    if not m_new:
        return url
    user, host = m_new.group(1), m_new.group(2)
    m_orig = re.search(r"://([^:/@]+):([^@]*)@([^/]+)", current_url)
    if m_orig and m_orig.group(1) == user and m_orig.group(3) == host:
        real_pass = m_orig.group(2)
        return url.replace(f"://{user}:****@{host}", f"://{user}:{real_pass}@{host}")
    return url


def _connection_status() -> Optional[Dict[str, Any]]:
    stream = camera_stream
    if stream is not None and hasattr(stream, "connection_status"):
        return stream.connection_status()
    return None


def probe_available_webcams() -> List[Dict[str, Any]]:
    """Webcams on the host, listed by name WITHOUT opening any of them.

    Opening each index to test it wakes a phone linked to Windows as a camera (Phone Link),
    so only the camera the user selects is ever opened.
    """
    active_idx = _primary_webcam_index()
    monitor = multi_camera_monitor
    extra = set(monitor.webcam_indices_in_use()) if monitor is not None else set()
    return build_webcam_choices(list_webcam_devices(), active_idx, extra)


def save_camera_config_to_yaml(cfg_dict: Dict[str, Any]) -> bool:
    """Save the default camera to the per-user settings file (%APPDATA%), never to configs/.

    The RTSP URL can contain the camera password; configs/camera.yaml is tracked by git
    and shipped in the installer, so it must keep only placeholder defaults.
    """
    try:
        save_user_camera_settings(cfg_dict)
        return True
    except Exception as e:
        print(f"[Error] Cannot save camera settings: {e}")
        return False


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Check API server health and GPU telemetry."""
    telemetry = resource_monitor.get_telemetry()
    return {
        "status": "healthy",
        "service": "fall-and-stroke-warning-system",
        "timestamp": time.time(),
        "telemetry": telemetry,
    }


@app.post("/api/system/shutdown")
def shutdown_system() -> Dict[str, Any]:
    """Graceful shutdown endpoint for background runners and remote stop."""
    import os, signal, threading
    def _kill():
        time.sleep(0.6)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return {"status": "shutting_down", "message": "Hệ thống đang dừng an toàn..."}


@app.get("/api/settings/autostart")
def get_autostart_status() -> Dict[str, Any]:
    """Check if Windows auto-start on boot is currently enabled."""
    import os
    from pathlib import Path
    appdata = os.getenv("APPDATA", "")
    shortcut = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "FallAndStrokeWarningSystem.lnk"
    return {"enabled": shortcut.exists(), "path": str(shortcut)}


@app.post("/api/settings/autostart")
def toggle_autostart(enable: bool = True) -> Dict[str, Any]:
    """Enable or disable Windows auto-start on boot."""
    import subprocess
    root = get_project_root()
    if enable:
        vbs_script = root / "scripts" / "cai_dat_startup.vbs"
        subprocess.run(["cscript.exe", "//nologo", str(vbs_script)], capture_output=True, text=True)
        return {"enabled": True, "message": "Đã BẬT tự động chạy ngầm mỗi khi mở máy tính!"}
    else:
        vbs_script = root / "scripts" / "huy_startup.vbs"
        subprocess.run(["cscript.exe", "//nologo", str(vbs_script)], capture_output=True, text=True)
        return {"enabled": False, "message": "Đã TẮT tự động khởi động cùng Windows!"}


@app.post("/api/settings/check-camera")
def check_camera_status() -> Dict[str, Any]:
    """Test access to the selected webcam only (never other indices, e.g. a linked phone)."""
    idx = _primary_webcam_index()
    with state_lock:
        streaming = is_running and camera_stream is not None and camera_stream.is_opened()
    if idx is not None and streaming:
        # Already capturing: opening it a second time would fail or steal the device
        return {"status": True, "message": f"Camera {idx} đang hoạt động bình thường.", "working_devices": [idx]}
    if idx is None:
        idx = int(load_camera_config().get("webcam", {}).get("device_index", 0))
    try:
        from src.camera.camera_permission import test_camera_access
        ok, msg, cams = test_camera_access(idx)
        return {"status": ok, "message": msg, "working_devices": cams}
    except Exception as e:
        return {"status": False, "message": f"Lỗi kiểm tra camera: {e}", "working_devices": []}


@app.post("/api/settings/open-camera-settings")
def open_windows_camera_settings() -> Dict[str, Any]:
    """Launch Windows Camera Privacy Settings."""
    import subprocess
    try:
        subprocess.Popen(["cmd", "/c", "start", "ms-settings:privacy-webcam"], shell=True)
        return {"success": True, "message": "Đã mở Windows Camera Privacy Settings."}
    except Exception as e:
        return {"success": False, "message": str(e)}



@app.get("/status")
def system_status() -> Dict[str, Any]:
    """Retrieve operational state of camera and AI inference pipeline."""
    with state_lock:
        running = is_running
        prof = getattr(pipeline_instance, "render_profiler", getattr(pipeline_instance, "profiler", None))
        fps = prof.fps if prof else 0.0
        tracks_count = 0
        if pipeline_instance:
            if hasattr(pipeline_instance, "shared_state"):
                tracks_count = len(pipeline_instance.shared_state.tracks)
            elif hasattr(pipeline_instance, "tracker") and hasattr(pipeline_instance.tracker, "_tracks"):
                tracks_count = len(pipeline_instance.tracker._tracks)
    extras = multi_camera_monitor.list_status() if multi_camera_monitor is not None else []

    return {
        "camera_active": running,
        "current_fps": round(fps, 1),
        "tracked_persons_count": tracks_count,
        "camera_connection": _connection_status(),  # network cameras only (state / reconnect attempts)
        "extra_cameras_online": sum(1 for c in extras if c.get("online")),
        "extra_cameras_total": len(extras),
        "timestamp": time.time(),
    }


@app.get("/camera/devices")
def get_camera_devices() -> List[Dict[str, Any]]:
    """Probe and return available webcam devices on the host."""
    return probe_available_webcams()


@app.get("/camera/info")
def get_camera_info() -> Dict[str, Any]:
    """Retrieve detailed information about current active camera stream."""
    with state_lock:
        active = is_running and camera_stream is not None and camera_stream.is_opened()
        fps = camera_stream.get_fps() if active and hasattr(camera_stream, "get_fps") else 0.0
        w = camera_stream.get_width() if active and hasattr(camera_stream, "get_width") else 0
        h = camera_stream.get_height() if active and hasattr(camera_stream, "get_height") else 0
        info = dict(current_camera_info)
        # Protect RTSP credentials: never expose raw passwords to the client
        if info.get("rtsp_url"):
            info["rtsp_url"] = mask_url(info["rtsp_url"])
        info.update({
            "is_active": active,
            "fps": round(fps, 1),
            "resolution": f"{w}x{h}" if w > 0 and h > 0 else "1280x720",
            "connection": _connection_status(),
        })
    return info


@app.post("/camera/switch")
def switch_camera(req: CameraSwitchRequest) -> Dict[str, Any]:
    """Dynamically switch camera source (Webcam, RTSP, Video) without breaking server."""
    global camera_stream, pipeline_instance, is_running, current_camera_info

    # Validate inputs
    if req.source_type == "rtsp":
        req.rtsp_url = _unmask_stream_url(req.rtsp_url)
        if not req.rtsp_url or not req.rtsp_url.strip():
            raise HTTPException(
                status_code=400,
                detail="Vui lòng nhập đường dẫn RTSP Stream URL hợp lệ (ví dụ: rtsp://admin:pass@192.168.1.100:554/stream)."
            )
        req.rtsp_url = _resolve_stream_url(req.rtsp_url, req.rtsp_username, req.rtsp_password)
        req.rtsp_username = req.rtsp_password = None  # now inside the URL; never echoed back separately
    elif req.source_type == "video":
        if not req.video_path or not req.video_path.strip():
            req.video_path = "data/videos/sample_adl_fall.mp4"
        req.video_path = req.video_path.strip()
    elif req.source_type == "webcam" and multi_camera_monitor is not None:
        if req.device_index in multi_camera_monitor.webcam_indices_in_use():
            raise HTTPException(
                status_code=400,
                detail=f"Webcam {req.device_index} đang được dùng làm camera thêm. Hãy xóa camera đó trước.",
            )

    with state_lock:
        # Release previous camera stream cleanly
        if camera_stream is not None:
            try:
                camera_stream.release()
            except Exception as e:
                print(f"[Warning] Error releasing previous camera: {e}")
            camera_stream = None
            time.sleep(0.25)

        new_stream = None
        friendly_name = ""

        try:
            if req.source_type == "webcam":
                cam_cfg = load_config("camera.yaml")
                api_pref = cam_cfg.get("webcam", {}).get("api_preference", "dshow")
                new_stream = WebcamStream(device_index=req.device_index, api_preference=api_pref)
                friendly_name = f"Webcam {req.device_index}"
            elif req.source_type == "rtsp":
                new_stream = RTSPStream.from_config(req.rtsp_url)
                friendly_name = f"IP Camera ({mask_url(req.rtsp_url)})"
            elif req.source_type == "video":
                new_stream = VideoFileStream(video_path=req.video_path, loop=True)
                friendly_name = f"Video ({Path(req.video_path).name})"
            else:
                raise HTTPException(status_code=400, detail=f"Loại nguồn không hỗ trợ: {req.source_type}")

            success = new_stream.start()
            if not success:
                new_stream.release()
                if req.source_type == "rtsp":
                    diag = diagnose_stream_failure(req.rtsp_url)
                    raise HTTPException(status_code=400, detail=f"{diag['message']} {diag['hint']}".strip())
                else:
                    # Fallback to synthetic if hardware webcam fails
                    from src.camera.synthetic import SyntheticCameraStream
                    new_stream = SyntheticCameraStream(width=1280, height=720, fps=30)
                    new_stream.start()
                    friendly_name += " [Giả lập Synthetic]"

            camera_stream = new_stream
            current_camera_info.update({
                "source_type": req.source_type,
                "device_index": req.device_index,
                "rtsp_url": req.rtsp_url,
                "video_path": req.video_path,
                "name": friendly_name,
                "is_fallback": "Synthetic" in friendly_name,
            })

            # Save as default if requested
            if req.save_as_default:
                save_camera_config_to_yaml(req.dict())

            # Ensure pipeline is initialized
            if pipeline_instance is None:
                try:
                    from src.pipeline.decoupled_pipeline import DecoupledPipeline
                    pipeline_instance = DecoupledPipeline(model_name="yolov8n-pose.pt", img_size=480)
                    pipeline_instance.start()
                except Exception:
                    pipeline_instance = RealtimePipeline()
                # Wire the alarm WebSocket broadcast so alerts from this
                # freshly created pipeline still reach the frontend chime.
                pipeline_instance.alert_manager.dispatcher.subscribe_websocket(broadcast_alert_event)
            elif hasattr(pipeline_instance, "start") and not getattr(pipeline_instance, "_running", True):
                # Resuming after /camera/stop, which stopped the AI worker thread
                pipeline_instance.start()

            # Ensure background worker is running (the desktop app runs its own ingestion loop)
            if not is_running:
                is_running = True
                if not external_ingestion_loop:
                    threading.Thread(target=_pipeline_worker_loop, daemon=True, name="PipelineWorker").start()

            res_str = f"{new_stream.get_width()}x{new_stream.get_height()}" if hasattr(new_stream, "get_width") else "1280x720"

            return {
                "status": "success",
                "message": f"Đã kết nối thành công với {friendly_name}!",
                "camera_info": {
                    "source_type": req.source_type,
                    "name": friendly_name,
                    "resolution": res_str,
                    "fps": round(new_stream.get_fps() if hasattr(new_stream, "get_fps") else 30.0, 1),
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            from src.camera.synthetic import SyntheticCameraStream
            fallback = SyntheticCameraStream(width=1280, height=720, fps=30)
            fallback.start()
            camera_stream = fallback
            current_camera_info.update({
                "source_type": "synthetic",
                "name": "Synthetic (Dự phòng lỗi)",
                "is_fallback": True,
            })
            raise HTTPException(status_code=500, detail=f"Lỗi khi chuyển đổi camera: {str(e)}")


@app.post("/camera/test")
def test_camera_connection(req: CameraTestRequest) -> Dict[str, Any]:
    """Try a network / remote camera URL (one frame) without switching the active camera."""
    url = _resolve_stream_url(req.rtsp_url, req.rtsp_username, req.rtsp_password)
    result = probe_stream(url)
    if not result.get("ok"):
        result.update(diagnose_stream_failure(url))  # specific reason instead of FFmpeg's "could not open"
    result["url"] = mask_url(url)
    return result


@app.get("/camera/discover")
def discover_network_cameras() -> Dict[str, Any]:
    """Scan the LAN this computer is on for IP cameras ("Dò camera trong mạng")."""
    return discover_cameras()


@app.post("/camera/save-config")
def save_camera_configuration(req: CameraSwitchRequest) -> Dict[str, Any]:
    """Save the selected camera setup as the per-user default (%APPDATA%)."""
    if req.source_type == "rtsp":
        req.rtsp_url = _unmask_stream_url(req.rtsp_url)
        req.rtsp_url = _resolve_stream_url(req.rtsp_url, req.rtsp_username, req.rtsp_password)
    success = save_camera_config_to_yaml(req.dict())
    if success:
        return {"status": "success", "message": "Đã lưu camera mặc định. Lần sau mở ứng dụng sẽ tự dùng camera này."}
    raise HTTPException(status_code=500, detail="Không thể lưu cấu hình camera mặc định")


@app.post("/camera/start")
def start_camera(req: CameraStartRequest) -> Dict[str, Any]:
    """Start or auto-reset the video ingestion and AI inference pipeline."""
    switch_req = CameraSwitchRequest(
        source_type=req.source_type,
        device_index=req.device_index,
        rtsp_url=req.rtsp_url,
        video_path=req.video_path,
    )
    return switch_camera(switch_req)


@app.post("/camera/reset")
def reset_camera(req: Optional[CameraStartRequest] = None) -> Dict[str, Any]:
    """Cleanly reset, release hardware lock, and restart camera pipeline."""
    return start_camera(req or CameraStartRequest())


@app.post("/camera/stop")
def stop_camera() -> Dict[str, Any]:
    """Stop the video pipeline and release camera resources."""
    global camera_stream, pipeline_instance, is_running

    with state_lock:
        if not is_running:
            return {"status": "not_running", "message": "Pipeline is not running."}

        is_running = False
        if hasattr(pipeline_instance, "stop"):
            try:
                pipeline_instance.stop()
            except Exception:
                pass
        if camera_stream:
            camera_stream.release()
            camera_stream = None

    return {"status": "stopped"}


@app.get("/api/camera/blur")
def get_face_blur() -> Dict[str, Any]:
    """Get current face blurring privacy status."""
    blur = False
    with state_lock:
        if pipeline_instance:
            if hasattr(pipeline_instance, "blur_faces"):
                blur = bool(pipeline_instance.blur_faces)
            elif hasattr(pipeline_instance, "pipeline") and hasattr(pipeline_instance.pipeline, "blur_faces"):
                blur = bool(pipeline_instance.pipeline.blur_faces)
    return {"status": "ok", "face_blur": blur}


@app.post("/api/camera/blur")
def set_face_blur(enabled: bool = True) -> Dict[str, Any]:
    """Toggle face blurring privacy mode."""
    with state_lock:
        if pipeline_instance:
            if hasattr(pipeline_instance, "blur_faces"):
                pipeline_instance.blur_faces = enabled
            elif hasattr(pipeline_instance, "pipeline") and hasattr(pipeline_instance.pipeline, "blur_faces"):
                pipeline_instance.pipeline.blur_faces = enabled
    if multi_camera_monitor is not None:
        multi_camera_monitor.set_blur(enabled)
    return {"status": "ok", "face_blur": enabled}


def _get_startup_shortcut_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "FallAndStrokeWarningSystem.lnk"
    return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "FallAndStrokeWarningSystem.lnk"


@app.get("/api/settings/autostart")
def get_autostart_status() -> Dict[str, Any]:
    """Check if auto-start shortcut exists in Windows Startup folder."""
    shortcut = _get_startup_shortcut_path()
    return {"enabled": shortcut.exists()}


@app.post("/api/settings/autostart")
def set_autostart(enable: bool = True) -> Dict[str, Any]:
    """Create or remove Windows startup shortcut for silent 24/7 surveillance."""
    import subprocess
    shortcut = _get_startup_shortcut_path()
    root_dir = get_project_root()
    target_exe = root_dir / "Fall_and_Stroke_Warning_System.exe"
    if not target_exe.exists():
        target_exe = root_dir / "2_Chay_He_Thong.bat"

    if enable:
        shortcut.parent.mkdir(parents=True, exist_ok=True)
        ps_script = (
            f"$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{str(shortcut)}'); "
            f"$s.TargetPath = '{str(target_exe)}'; "
            f"$s.WorkingDirectory = '{str(root_dir)}'; "
            f"$s.Description = 'Fall and Stroke Warning System'; "
            f"$s.Save()"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, timeout=5)
            return {"enabled": True, "message": "Đã bật tự động chạy ngầm cùng Windows!"}
        except Exception as e:
            return {"enabled": False, "message": f"Không thể tạo phím tắt: {e}"}
    else:
        if shortcut.exists():
            try:
                shortcut.unlink()
            except Exception as e:
                return {"enabled": True, "message": f"Không thể xóa phím tắt: {e}"}
        return {"enabled": False, "message": "Đã tắt tự động chạy ngầm cùng Windows!"}


@app.post("/api/settings/open-camera-settings")
def open_win_camera_settings() -> Dict[str, Any]:
    """Launch Windows Privacy & Security camera settings."""
    import subprocess
    try:
        subprocess.Popen(["cmd", "/c", "start", "ms-settings:privacy-webcam"], shell=True)
        return {"status": "ok", "message": "Đã mở Cài đặt Quyền Camera Windows."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/system/shutdown")
def shutdown_system() -> Dict[str, Any]:
    """Safely terminate monitoring server process."""
    def _delayed_exit():
        time.sleep(0.8)
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "ok", "message": "Hệ thống đang tắt an toàn..."}


def _pipeline_worker_loop() -> None:
    """Continuous processing loop running on a dedicated thread."""
    global latest_annotated_frame, is_running

    while is_running and camera_stream and camera_stream.is_opened():
        packet = camera_stream.read(timeout=0.5)
        if packet is None or packet.frame is None:
            time.sleep(0.005)
            continue

        if hasattr(pipeline_instance, "submit_frame"):
            pipeline_instance.submit_frame(packet)
            annotated_frame, _ = pipeline_instance.render_frame(packet.frame, packet.timestamp)
        else:
            result = pipeline_instance.process_frame(packet.frame, packet.timestamp)
            annotated_frame = result.annotated_frame

        publish_annotated_frame(annotated_frame)


# ---------------------------------------------------------------------------
# Background JPEG encoder: 720p JPEG encoding costs ~15-20 ms, so doing it inline
# capped the capture/render loop well below camera FPS. Producers now only hand
# over the newest frame; a single encoder thread (cv2 releases the GIL) encodes
# whatever is latest and silently drops stale frames.
# ---------------------------------------------------------------------------
_encode_cond = threading.Condition()
_pending_frame: Optional[Any] = None
_encoder_thread: Optional[threading.Thread] = None
_jpeg_quality = int(load_config("inference.yaml").get("display", {}).get("jpeg_quality", 80))


def publish_annotated_frame(frame: Any) -> None:
    """Non-blocking: queue the newest annotated BGR frame for the MJPEG web feed."""
    global _pending_frame, _encoder_thread
    with _encode_cond:
        _pending_frame = frame
        if _encoder_thread is None or not _encoder_thread.is_alive():
            _encoder_thread = threading.Thread(target=_jpeg_encoder_loop, daemon=True, name="JpegEncoder")
            _encoder_thread.start()
        _encode_cond.notify()


def _jpeg_encoder_loop() -> None:
    global _pending_frame, latest_annotated_frame, latest_frame_seq
    while True:
        with _encode_cond:
            while _pending_frame is None:
                _encode_cond.wait()
            frame = _pending_frame
            _pending_frame = None
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality])
        if ret:
            with state_lock:
                latest_annotated_frame = jpeg.tobytes()
                latest_frame_seq += 1



def _with_clip_status(evt: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(evt)
    out["clip_available"] = find_clip_file(evt.get("clip_path")) is not None
    return out


@app.get("/events")
def list_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent historical emergency alerts (clip_available: the fall clip can be played)."""
    return [_with_clip_status(e) for e in event_logger.get_recent_events(limit=limit)]


@app.get("/events/{event_id}/clip")
def get_event_clip(event_id: str):
    """Play the anonymized fall clip of an event (404 while still recording, or after expiry)."""
    for evt in event_logger.get_recent_events(limit=10**9):
        if evt.get("event_id") == event_id:
            clip = find_clip_file(evt.get("clip_path"))
            clips_root = (event_logger.log_dir / "clips").resolve()
            if clip is None or clips_root not in clip.resolve().parents:
                if evt.get("clip_path"):
                    raise HTTPException(status_code=404, detail="Đoạn video đang được ghi hoặc đã hết hạn lưu trữ (7 ngày).")
                raise HTTPException(status_code=404, detail="Sự kiện này không có đoạn video (chỉ lưu video khi té ngã).")
            media = "video/webm" if clip.suffix == ".webm" else "video/mp4"
            return FileResponse(str(clip), media_type=media)
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")


@app.get("/events/{event_id}")
def get_event(event_id: str) -> Dict[str, Any]:
    """Fetch details of a specific event."""
    events = event_logger.get_recent_events(limit=200)
    for evt in events:
        if evt.get("event_id") == event_id:
            return _with_clip_status(evt)
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")
@app.post("/events/clear")
@app.delete("/events")
def clear_all_events() -> Dict[str, Any]:
    """Clear all stored emergency alert events."""
    event_logger.clear_events()
    return {"status": "cleared", "message": "All alert events cleared."}


@app.delete("/events/{event_id}")
@app.post("/events/{event_id}/delete")
def delete_single_event(event_id: str) -> Dict[str, Any]:
    """Permanently delete a specific alert event by its ID."""
    deleted = event_logger.delete_event(event_id)
    return {"status": "success", "deleted": deleted, "event_id": event_id}

def mjpeg_generator():
    """Generator yielding multipart MJPEG video frames with resilient keepalive."""
    empty_wait = 0
    last_seq = -1
    last_sent = 0.0
    while is_running or empty_wait < 50:
        with state_lock:
            frame_bytes = latest_annotated_frame
            seq = latest_frame_seq

        now = time.time()
        # Send only new frames (no duplicate re-sends); resend ~1/s as keepalive if stalled
        if frame_bytes is not None and (seq != last_seq or now - last_sent > 1.0):
            empty_wait = 0
            last_seq = seq
            last_sent = now
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        elif frame_bytes is None:
            empty_wait += 1
        time.sleep(0.008)  # Poll fast enough to forward every frame at 60 FPS


@app.get("/stream/mjpeg")
def video_mjpeg_feed():
    """MJPEG Live Video Stream for HTML <img> embedding."""
    global is_running
    if not is_running:
        try:
            start_camera(CameraStartRequest())
        except Exception as e:
            print(f"[Warning] Auto-start camera in /stream/mjpeg failed: {e}")
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Multi-camera: extra cameras, each with its own AI session and MJPEG tile feed
# ---------------------------------------------------------------------------
class ExtraCameraRequest(BaseModel):
    name: Optional[str] = None
    source_type: str = "rtsp"  # "rtsp" (any network / remote camera), "webcam", "video"
    device_index: int = 0
    rtsp_url: Optional[str] = None
    rtsp_username: Optional[str] = None
    rtsp_password: Optional[str] = None
    video_path: Optional[str] = None


def init_multi_camera(pipeline: Any, camera_cfg: Optional[Dict[str, Any]] = None) -> MultiCameraMonitor:
    """Create the extra-camera monitor sharing `pipeline`'s models, and restore saved cameras.

    Called by main.py / app_runner.py once the primary pipeline exists.
    """
    global multi_camera_monitor
    camera_cfg = camera_cfg if camera_cfg is not None else load_camera_config()
    extra_cfg = camera_cfg.get("extra_cameras", {}) or {}
    monitor = MultiCameraMonitor(
        settings=extra_cfg,
        stream_factory=default_stream_factory(camera_cfg),
        jpeg_quality=_jpeg_quality,
    )
    monitor.attach_base_pipeline(pipeline)
    with state_lock:
        multi_camera_monitor = monitor
    for err in monitor.load_saved(extra_cfg.get("cameras") or []):
        print(f"[Warning] Camera thêm không khởi động được: {err}")
    return monitor


def _get_multi_camera_monitor() -> MultiCameraMonitor:
    if multi_camera_monitor is None:
        return init_multi_camera(pipeline_instance)
    return multi_camera_monitor


def _primary_webcam_index() -> Optional[int]:
    with state_lock:
        if current_camera_info.get("source_type") == "webcam":
            return int(current_camera_info.get("device_index") or 0)
    return None


@app.get("/api/cameras")
def list_cameras() -> Dict[str, Any]:
    """Primary camera + every extra camera with its live status (passwords masked)."""
    monitor = multi_camera_monitor
    with state_lock:
        primary = {
            "id": "main",
            "name": current_camera_info.get("name"),
            "source_type": current_camera_info.get("source_type"),
            "online": bool(is_running and camera_stream is not None and camera_stream.is_opened()),
        }
    return {
        "primary": primary,
        "cameras": monitor.list_status() if monitor is not None else [],
        "max_cameras": monitor.max_cameras if monitor is not None else 0,
    }


@app.post("/api/cameras")
def add_extra_camera(req: ExtraCameraRequest) -> Dict[str, Any]:
    """Add and start an extra camera; it is saved (%APPDATA%) and restored at next start."""
    entry = req.model_dump()
    if req.source_type == "rtsp":
        entry["rtsp_url"] = _resolve_stream_url(req.rtsp_url, req.rtsp_username, req.rtsp_password)
    entry.pop("rtsp_username", None)
    entry.pop("rtsp_password", None)
    try:
        status = _get_multi_camera_monitor().add_camera(entry, primary_webcam_index=_primary_webcam_index())
    except CameraConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "message": f"Đã thêm camera {status['name']}.", "camera": status}


@app.delete("/api/cameras/{cam_id}")
def remove_extra_camera(cam_id: str) -> Dict[str, Any]:
    """Stop an extra camera and forget it."""
    monitor = multi_camera_monitor
    if monitor is None or not monitor.remove_camera(cam_id):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy camera '{cam_id}'.")
    return {"status": "success", "camera_id": cam_id}


def _camera_mjpeg_generator(cam_id: str):
    last_seq = -1
    last_sent = 0.0
    while True:
        monitor = multi_camera_monitor
        session = monitor.get_session(cam_id) if monitor is not None else None
        if session is None:
            return  # camera removed: end the stream so the browser tile stops
        frame_bytes, seq = session.latest_jpeg()
        now = time.time()
        if frame_bytes is not None and (seq != last_seq or now - last_sent > 1.0):
            last_seq = seq
            last_sent = now
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        time.sleep(0.02)


@app.get("/api/cameras/{cam_id}/stream")
def extra_camera_mjpeg_feed(cam_id: str):
    """MJPEG feed of one extra camera (annotated, downscaled tile)."""
    monitor = multi_camera_monitor
    if monitor is None or monitor.get_session(cam_id) is None:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' không hoạt động.")
    return StreamingResponse(_camera_mjpeg_generator(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")


def broadcast_alert_event(event: AlertEvent) -> None:
    """Send alert JSON to all connected WebSocket clients.

    Called synchronously from the AI pipeline's own thread (not the uvicorn
    event loop thread), so sends must be scheduled onto the captured
    `server_event_loop` via run_coroutine_threadsafe rather than relying on
    asyncio.get_event_loop(), which has no loop bound to that thread and
    would silently drop the alert.
    """
    from dataclasses import asdict
    payload = json.dumps(asdict(event))

    loop = server_event_loop
    if loop is None or not loop.is_running():
        print("[Warning] Server event loop not ready — alert WebSocket push skipped")
        return

    for ws in list(active_ws_clients):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(payload), loop)
        except Exception as e:
            print(f"[Warning] Failed to schedule alert WebSocket send: {e}")


@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """WebSocket stream pushing real-time emergency events."""
    from src.auth.middleware import verify_session_token
    from src.auth.request_guard import is_allowed_host, is_cross_site, is_local_request

    # Browsers always send Origin on WebSocket handshakes: refuse other sites (cross-site
    # WebSocket hijacking through the localhost bypass) and rebinding host names.
    if not is_allowed_host(websocket) or is_cross_site(websocket):
        await websocket.close(code=4003, reason="Forbidden origin")
        return
    if _security_cfg.get("auth", {}).get("enabled", True):
        allow_local = _security_cfg.get("auth", {}).get("allow_localhost_unauthenticated", True)
        if not (allow_local and is_local_request(websocket)):
            token = websocket.query_params.get("token") or websocket.cookies.get("access_token")
            if not token:
                await websocket.close(code=4001, reason="Unauthorized")
                return
            if verify_session_token(token) is None:
                await websocket.close(code=4001, reason="Invalid token")
                return

    await websocket.accept()
    active_ws_clients.append(websocket)
    try:
        while True:
            # Keep-alive ping
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_ws_clients:
            active_ws_clients.remove(websocket)


# ==============================================================================
# STATIC FRONTEND MOUNTING & SERVING
# ==============================================================================
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dir = get_project_root() / "frontend"
if not frontend_dir.exists():
    internal_frontend = get_project_root() / "_internal" / "frontend"
    if internal_frontend.exists():
        frontend_dir = internal_frontend

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend_index():
        """Serve dashboard HTML homepage."""
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/styles.css")
    def serve_frontend_css():
        """Serve dashboard stylesheet."""
        return FileResponse(str(frontend_dir / "styles.css"))

    @app.get("/app.js")
    def serve_frontend_js():
        """Serve dashboard controller script."""
        return FileResponse(str(frontend_dir / "app.js"))

    @app.get("/favicon.ico")
    def serve_frontend_favicon():
        """Serve browser tab favicon icon."""
        ico_path = frontend_dir / "favicon.ico"
        if ico_path.exists():
            return FileResponse(str(ico_path))
        return FileResponse(str(frontend_dir / "icon.png"))

    @app.get("/icon.png")
    def serve_frontend_icon_png():
        """Serve PNG tab icon."""
        return FileResponse(str(frontend_dir / "icon.png"))

    @app.get("/logo.png")
    def serve_frontend_logo_png():
        """Serve header branding app logo."""
        return FileResponse(str(frontend_dir / "logo.png"))

    @app.get("/apple-touch-icon.png")
    def serve_frontend_apple_icon():
        """Serve apple touch icon."""
        return FileResponse(str(frontend_dir / "apple-touch-icon.png"))

    @app.get("/huong-dan.html")
    @app.get("/guide")
    def serve_frontend_guide():
        """Serve bilingual user guide page."""
        guide_path = frontend_dir / "huong-dan.html"
        if not guide_path.exists():
            guide_path = get_project_root() / "landing" / "huong-dan.html"
        if guide_path.exists():
            return FileResponse(str(guide_path))
        return FileResponse(str(frontend_dir / "index.html"))

