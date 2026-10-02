"""FALL AND STROKE WARNING SYSTEM — NATIVE DESKTOP APP RUNNER
================================================================================
Entrypoint tối ưu hóa cho PyInstaller đóng gói ứng dụng Windows độc lập:
1. Chạy hoàn toàn không hiện cửa sổ dòng lệnh (no-console / windowed).
2. Tự động chuyển hướng log an toàn vào app_log.txt (tránh crash NoneType stdout).
3. Hỗ trợ multiprocessing.freeze_support() chuẩn Windows.
4. Tự động mở Web Dashboard dưới dạng Native App Window (Microsoft Edge / Chrome --app mode).
5. Quản lý vòng đời ứng dụng: khi người dùng đóng cửa sổ app, hệ thống tự động
   dừng Camera & AI an toàn và thoát hoàn toàn.
================================================================================
"""

import os
import re
import sys
import time
import shutil
import ctypes
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional

# 1. Chuyển hướng stdout / stderr an toàn khi chạy chế độ no-console
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
LOG_FILE_PATH = APP_DIR / "app_log.txt"

class SafeLogWriter:
    """Writer ghi đồng thời ra file log và console nếu có, tương thích chuẩn IO stream."""
    def __init__(self, filepath: Path, original_stream=None):
        self.filepath = filepath
        self.original_stream = original_stream
        self.encoding = "utf-8"

    def write(self, message: str) -> None:
        if not message:
            return
        try:
            with open(self.filepath, "a", encoding="utf-8", errors="replace") as f:
                f.write(message)
        except Exception:
            pass
        if self.original_stream is not None:
            try:
                self.original_stream.write(message)
            except Exception:
                pass

    def flush(self) -> None:
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        if self.original_stream is not None and hasattr(self.original_stream, "isatty"):
            try:
                return bool(self.original_stream.isatty())
            except Exception:
                pass
        return False

    def fileno(self):
        if self.original_stream is not None and hasattr(self.original_stream, "fileno"):
            try:
                return self.original_stream.fileno()
            except Exception:
                pass
        raise OSError("SafeLogWriter does not support fileno")

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

# Nếu sys.stdout là None (PyInstaller noconsole), gán sang SafeLogWriter
if sys.stdout is None or sys.stderr is None:
    safe_writer = SafeLogWriter(LOG_FILE_PATH)
    sys.stdout = safe_writer
    sys.stderr = safe_writer
else:
    # Vẫn ghi lại log vào file để tiện theo dõi chẩn đoán
    sys.stdout = SafeLogWriter(LOG_FILE_PATH, sys.stdout)
    sys.stderr = SafeLogWriter(LOG_FILE_PATH, sys.stderr)

# Đảm bảo đường dẫn gốc của project có trong sys.path
sys.path.insert(0, str(APP_DIR))

# Đảm bảo multiprocessing hoạt động chính xác sau khi đóng gói PyInstaller
import multiprocessing
multiprocessing.freeze_support()


def show_native_message_box(title: str, message: str, icon_type: str = "info") -> None:
    """Hiển thị hộp thoại thông báo Windows Native không cần thư viện ngoài."""
    try:
        flags = 0x00040000  # MB_TOPMOST
        if icon_type == "error":
            flags |= 0x00000010  # MB_ICONERROR
        elif icon_type == "warning":
            flags |= 0x00000030  # MB_ICONWARNING
        else:
            flags |= 0x00000040  # MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(0, message, title, flags)
    except Exception:
        pass


def find_system_browser() -> Optional[str]:
    """Tìm đường dẫn trình duyệt Chromium hỗ trợ chế độ App-Mode (--app)."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path_str in candidates:
        p = Path(path_str)
        if p.exists() and p.is_file():
            return str(p)

    # Thử qua PATH
    which_edge = shutil.which("msedge")
    if which_edge:
        return which_edge
    which_chrome = shutil.which("chrome")
    if which_chrome:
        return which_chrome
    return None


def launch_native_app_window(target_url: str, window_title: str = "Fall and Stroke Warning System") -> Optional[subprocess.Popen]:
    """Khởi chạy giao diện dưới dạng Native App Window độc lập không có thanh URL."""
    browser_exe = find_system_browser()
    profile_dir = Path(tempfile.gettempdir()) / "FallAndStrokeAppProfile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    if browser_exe:
        print(f"[*] Khởi chạy Native App Window qua: {browser_exe}")
        cmd = [
            browser_exe,
            f"--app={target_url}",
            f"--user-data-dir={str(profile_dir)}",
            "--window-size=1440,900",
            "--no-first-run",
            "--no-default-browser-check",
            f"--app-id=fall_and_stroke_monitor",
        ]
        try:
            proc = subprocess.Popen(cmd)
            return proc
        except Exception as e:
            print(f"[Cảnh báo] Không thể khởi chạy App Mode qua trình duyệt: {e}")

    # Fallback nếu không có Edge/Chrome
    import webbrowser
    print(f"[*] Mở qua trình duyệt mặc định: {target_url}")
    webbrowser.open(target_url)
    return None


def wait_for_server_ready(url: str, timeout_seconds: float = 8.0) -> bool:
    """Thăm dò kiểm tra server đã sẵn sàng nhận kết nối hay chưa."""
    import urllib.request
    start_t = time.time()
    while time.time() - start_t < timeout_seconds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False





def run_app():
    """Hàm chạy chính toàn bộ ứng dụng."""
    print("=" * 80)
    print("    FALL AND STROKE WARNING SYSTEM — STANDALONE DESKTOP APPLICATION")
    print("=" * 80)
    print(f"[*] Thư mục ứng dụng: {APP_DIR}")
    print(f"[*] Log hệ thống ghi tại: {LOG_FILE_PATH}")

    try:
        import main as core_main
        import src.api.server as srv
    except Exception as e:
        import traceback
        err_msg = f"Lỗi nạp thư viện hệ thống:\n{traceback.format_exc()}"
        print(err_msg)
        show_native_message_box("Lỗi Khởi Động Ứng Dụng", err_msg, "error")
        sys.exit(1)

    # 1. Đọc cấu hình hệ thống
    try:
        configs = core_main.load_system_configs()
    except Exception as e:
        err_msg = f"Không tìm thấy hoặc lỗi đọc file cấu hình tại {APP_DIR / 'configs'}:\n{e}"
        print(err_msg)
        show_native_message_box("Lỗi Cấu Hình", err_msg, "error")
        sys.exit(1)

    port = 8000
    dashboard_url = f"http://localhost:{port}"

    # 2. Khởi động Web API Server trên background thread (với log_config=None để tương thích hoàn toàn môi trường no-console)
    print("[*] Đang khởi động Web Server API...")
    def _run_custom_uvicorn():
        import uvicorn
        uvicorn.run(srv.app, host="0.0.0.0", port=port, log_level="warning", log_config=None)
    threading.Thread(target=_run_custom_uvicorn, daemon=True, name="WebServerThread").start()

    # 3. Khởi tạo Pipeline AI
    print("[*] Đang khởi động AI Engine...")
    decoupled_pipe = None
    try:
        from src.pipeline.decoupled_pipeline import DecoupledPipeline
        model_name = configs["model"].get("pose", {}).get("model_name", "models/pose/rtmo-s.onnx")
        img_size = configs["model"].get("pose", {}).get("img_size", 640)
        device = configs["inference"].get("runtime", {}).get("device", "cuda")
        half = configs["inference"].get("runtime", {}).get("use_fp16", True)

        decoupled_pipe = DecoupledPipeline(
            model_name=model_name,
            img_size=img_size,
            device=device,
            half=half,
        )
        decoupled_pipe.start()
        pipeline = decoupled_pipe
    except Exception as e:
        print(f"[Cảnh báo] Lỗi khởi động Decoupled Pipeline, thử chuyển sang Pipeline tiêu chuẩn: {e}")
        try:
            pipeline = core_main.initialize_ai_pipeline(blur_faces=False)
        except Exception as e2:
            import traceback
            err_msg = f"Lỗi khởi tạo AI Pipeline:\n{traceback.format_exc()}"
            print(err_msg)
            show_native_message_box("Lỗi AI Engine", err_msg, "error")
            sys.exit(1)

    # Đồng bộ Alert Dispatcher và Pipeline với Web Server
    try:
        pipeline.alert_manager.dispatcher.subscribe_websocket(srv.broadcast_alert_event)
        with srv.state_lock:
            srv.pipeline_instance = pipeline
            srv.external_ingestion_loop = True  # vòng lặp ở bước 7 tự đọc camera
    except Exception as e:
        print(f"[Warning] Đồng bộ WebSocket dispatcher: {e}")

    # 4. Khởi tạo Camera Stream: camera người dùng đã "LƯU MẶC ĐỊNH" (%APPDATA%), mặc định Webcam 0
    cam_cfg = configs.get("camera", {})
    source_type = cam_cfg.get("source_type", "webcam")
    if source_type not in ("webcam", "rtsp", "video"):
        source_type = "webcam"
    device_index = int(cam_cfg.get("webcam", {}).get("device_index", 0))
    rtsp_url = cam_cfg.get("rtsp", {}).get("url") if source_type == "rtsp" else None
    video_path = cam_cfg.get("video", {}).get("path") if source_type == "video" else None
    if source_type == "rtsp":
        cam_name = "RTSP (" + re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", rtsp_url or "") + ")"
    elif source_type == "video":
        cam_name = f"Video: {Path(video_path or '').name}"
    else:
        cam_name = f"Webcam {device_index}"
    try:
        stream = core_main.initialize_camera_stream(
            source_type=source_type,
            device_index=device_index,
            rtsp_url=rtsp_url,
            video_path=video_path,
            camera_cfg=cam_cfg,
        )
        with srv.state_lock:
            srv.camera_stream = stream
            srv.is_running = True
            srv.current_camera_info.update({
                "source_type": source_type,
                "device_index": device_index,
                "rtsp_url": rtsp_url,
                "video_path": video_path,
                "name": cam_name,
            })
    except Exception as e:
        print(f"[Warning] Lỗi khởi tạo camera thực tế, chuyển sang Synthetic: {e}")
        from src.camera.synthetic import SyntheticCameraStream
        stream = SyntheticCameraStream(width=1280, height=720, fps=30)
        stream.start()
        with srv.state_lock:
            srv.camera_stream = stream
            srv.is_running = True

    # 5. Hàm dừng toàn bộ hệ thống
    stop_event = threading.Event()

    def stop_all():
        stop_event.set()
        with srv.state_lock:
            srv.is_running = False
            if srv.camera_stream:
                try:
                    srv.camera_stream.release()
                except Exception:
                    pass
        if decoupled_pipe is not None:
            try:
                decoupled_pipe.stop()
            except Exception:
                pass
        print("[*] Hệ thống đã dừng thành công.")

    # 6. Chờ server sẵn sàng và mở Native App Window
    def open_window_async():
        server_ready = wait_for_server_ready(dashboard_url, timeout_seconds=8.0)
        if server_ready:
            print(f"[+] Server sẵn sàng. Đang mở Native App Window...")
        else:
            print(f"[!] Server mất nhiều thời gian hơn bình thường, vẫn mở ứng dụng...")
        
        launch_native_app_window(dashboard_url)

    threading.Thread(target=open_window_async, daemon=True, name="WindowLauncher").start()

    # 7. Vòng lặp xử lý video AI liên tục (Main Ingestion Loop)
    print("\n[*] Vòng lặp giám sát thời gian thực đang chạy...")
    import cv2

    active_stream = stream
    try:
        while not stop_event.is_set():
            # "DỪNG STREAM" chỉ tạm dừng camera (is_running = False): chờ người dùng bật lại.
            # Nút TẮT HỆ THỐNG gọi /api/system/shutdown, endpoint đó tự thoát tiến trình.
            with srv.state_lock:
                paused = not srv.is_running
                if srv.camera_stream is not None:
                    active_stream = srv.camera_stream
            if paused:
                active_stream = None
                time.sleep(0.05)
                continue

            if active_stream is None or not active_stream.is_opened():
                time.sleep(0.02)
                continue

            packet = active_stream.read(timeout=0.5)
            if packet is None or packet.frame is None:
                time.sleep(0.002)
                continue

            if decoupled_pipe is not None:
                decoupled_pipe.submit_frame(packet)
                annotated_frame, alerts = decoupled_pipe.render_frame(packet.frame, packet.timestamp)
            else:
                result = pipeline.process_frame(frame=packet.frame, timestamp=packet.timestamp)
                annotated_frame = result.annotated_frame
                alerts = result.alerts

            # Đẩy ảnh sang MJPEG Web Stream (mã hóa JPEG trên luồng nền)
            srv.publish_annotated_frame(annotated_frame)

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        import traceback
        err_msg = f"Lỗi không mong muốn trong vòng lặp AI:\n{traceback.format_exc()}"
        print(err_msg)
        show_native_message_box("Lỗi Giám Sát AI", err_msg, "error")
    finally:
        stop_all()
        print("[*] Kết thúc phiên ứng dụng an toàn.\n")
        os._exit(0)


if __name__ == "__main__":
    run_app()
