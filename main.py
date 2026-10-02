"""MAIN ENTRYPOINT — ELDERLY AI MONITOR SYSTEM
================================================================================
Hệ thống Giám sát Camera & Phát hiện Hành vi Bất thường Thời gian thực.
Chỉ cần chạy:
    python main.py

Hệ thống sẽ tự động:
1. Kiểm tra môi trường (Python 3.11, CUDA/GPU, RAM).
2. Tải toàn bộ cấu hình từ thư mục configs/.
3. Khởi động Web API server (FastAPI + WebSockets + MJPEG feed tại http://localhost:8000).
4. Khởi tạo Pipeline AI:
   - Ingestion: Camera/Webcam/RTSP/Video
   - Perception: YOLOv8 Person Detector
   - Tracking: ByteTrack Multi-Object Tracker
   - Pose: YOLOv8-Pose (17 COCO Keypoints & Normalization)
   - Temporal: Sliding Window Sequence Buffer (T=30)
   - Dual-AI: Spatial-Temporal Transformer + Pose Autoencoder Anomaly Scorer
   - Kinematics: Tốc độ rơi Vy, gia tốc Ay, góc nghiêng thân người, chỉ số bất động
   - Risk Engine: Hợp nhất đa tín hiệu & Máy trạng thái Debounce/Hysteresis
   - Alerts: Ghi log sự kiện mã hóa, làm mờ mặt (Face blurring), thông báo khẩn cấp
5. Mở cửa sổ trực quan hóa OpenCV thời gian thực (GUI HUD, BBox, Skeleton, Risk Gauge).
================================================================================
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import cv2

# Đảm bảo đường dẫn gốc của project có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import các module cốt lõi của hệ thống
from src.utils.config import load_config, get_project_root
from src.utils.profiler import ResourceMonitor
from src.camera.stream import CameraStream, FramePacket
from src.camera.camera_settings import load_camera_config
from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream
from src.camera.video_file import VideoFileStream
from src.pipeline.realtime_pipeline import RealtimePipeline, PipelineFrameResult
from src.alerts.event_logger import AlertEvent


def print_banner() -> None:
    """In banner hệ thống và thông điệp an toàn y tế."""
    banner = """
================================================================================
   FALL AND STROKE WARNING SYSTEM — REAL-TIME VIDEO INTELLIGENCE SYSTEM
================================================================================
[*] Core Pipeline:
    Camera -> YOLOv8 Detect -> ByteTrack -> YOLO-Pose -> ST-Transformer
    -> Pose Autoencoder -> Kinematic Reasoner -> Debounced State Machine -> Alert
[*] Safety Notice:
    Hệ thống KHÔNG đưa ra chẩn đoán y khoa tự động. Mọi cảnh báo phát ra dưới dạng:
    "Possible medical emergency / abnormal behavior detected. Please check the person."
================================================================================
"""
    print(banner)


def check_system_environment() -> Dict[str, Any]:
    """Kiểm tra Python version, GPU CUDA và bộ nhớ hệ thống."""
    print("[1/5] Kiểm tra môi trường hệ thống...")
    py_ver = sys.version_split = sys.version.split()[0]
    print(f"  • Python Version  : {py_ver} (Target: Python 3.11)")

    monitor = ResourceMonitor()
    telemetry = monitor.get_telemetry()
    has_gpu = telemetry.get("gpu_available", False)

    if has_gpu:
        print(f"  • GPU Acceleration: ENABLED ({telemetry.get('gpu_device_name')})")
        print(f"  • VRAM Allocated  : {telemetry.get('gpu_vram_allocated_mb', 0)} MB")
    else:
        print("  • GPU Acceleration: CPU Mode (CUDA không khả dụng, pipeline chạy trên CPU)")

    print(f"  • System RAM Used : {telemetry.get('ram_used_gb', 0)} GB ({telemetry.get('system_ram_percent', 0)}%)")
    return telemetry


def load_system_configs() -> Dict[str, Any]:
    """Tải toàn bộ tệp cấu hình YAML độc lập."""
    print("[2/5] Đọc cấu hình hệ thống từ configs/...")
    model_cfg = load_config("model.yaml")
    infer_cfg = load_config("inference.yaml")
    thresholds_cfg = load_config("thresholds.yaml")
    camera_cfg = load_camera_config()  # configs/camera.yaml + camera saved by the user (%APPDATA%)
    print("  • configs/model.yaml       : [OK]")
    print("  • configs/inference.yaml   : [OK]")
    print("  • configs/thresholds.yaml  : [OK]")
    print("  • configs/camera.yaml      : [OK]")
    return {
        "model": model_cfg,
        "inference": infer_cfg,
        "thresholds": thresholds_cfg,
        "camera": camera_cfg,
    }


def initialize_camera_stream(
    source_type: str = "webcam",
    device_index: int = 0,
    rtsp_url: Optional[str] = None,
    video_path: Optional[str] = None,
    camera_cfg: Optional[Dict[str, Any]] = None,
) -> CameraStream:
    """Khởi tạo nguồn camera stream phù hợp (Webcam, RTSP hoặc File)."""
    print(f"[3/5] Khởi tạo luồng video (Nguồn: {source_type.upper()})...")
    camera_cfg = camera_cfg or {}

    if source_type == "webcam":
        w_idx = device_index if device_index is not None else camera_cfg.get("webcam", {}).get("device_index", 0)
        
        # Kiểm tra quyền và tính khả dụng của Camera trên Windows
        try:
            from src.camera.camera_permission import test_camera_access, request_windows_camera_permission_dialog
            ok, msg, working_devices = test_camera_access(w_idx)
            if not ok:
                print(f"  [Cảnh báo Camera] {msg}")
                request_windows_camera_permission_dialog()
            elif w_idx not in working_devices and len(working_devices) > 0:
                print(f"  [Thông báo Camera] Camera {w_idx} bận. Tự động chuyển sang Camera {working_devices[0]}.")
                w_idx = working_devices[0]
        except Exception as e:
            print(f"  [Notice] Bỏ qua kiểm tra quyền camera: {e}")

        width = camera_cfg.get("webcam", {}).get("width", 1280)
        height = camera_cfg.get("webcam", {}).get("height", 720)
        fps = camera_cfg.get("webcam", {}).get("fps", 30)
        api_pref = camera_cfg.get("webcam", {}).get("api_preference", "dshow")
        stream = WebcamStream(device_index=w_idx, width=width, height=height, fps=fps, api_preference=api_pref)

    elif source_type == "rtsp":
        url = rtsp_url or camera_cfg.get("rtsp", {}).get("url")
        if not url:
            raise ValueError("RTSP URL chưa được cung cấp. Vui lòng kiểm tra configs/camera.yaml hoặc cờ --rtsp-url")
        stream = RTSPStream.from_config(url, camera_cfg.get("rtsp", {}))
        # A remote camera may still be offline at boot (router / 4G restarting): keep
        # retrying in the background instead of silently falling back to a fake stream.
        stream.allow_offline_start = True

    elif source_type == "video":
        vpath = video_path or camera_cfg.get("video", {}).get("path")
        if not vpath or not Path(vpath).exists():
            print(f"  [Notice] File video '{vpath}' không tìm thấy. Sử dụng webcam mặc định index 0.")
            api_pref = camera_cfg.get("webcam", {}).get("api_preference", "dshow")
            stream = WebcamStream(device_index=0, api_preference=api_pref)
        else:
            stream = VideoFileStream(video_path=vpath, loop=True)
    else:
        raise ValueError(f"Nguồn camera không hợp lệ: {source_type}")

    if not stream.start():
        print("  [Warning] Không thể mở camera trực tiếp. Khởi tạo SyntheticCameraStream giả lập để kiểm thử.")
        from src.camera.synthetic import SyntheticCameraStream
        stream = SyntheticCameraStream(width=1280, height=720, fps=30)
        stream.start()

    print(f"  • Video Ingestion Worker: [READY] (Target FPS: {stream.get_fps():.0f})")
    return stream


def start_background_web_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Khởi động FastAPI server chạy ngầm trên background daemon thread."""
    try:
        import uvicorn
        from src.api.server import app

        def run_server():
            uvicorn.run(app, host=host, port=port, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True, name="WebServerThread")
        server_thread.start()
        print(f"  • Web API & Dashboard    : [ONLINE] -> http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        print(f"  • Swagger API Docs       : http://localhost:{port}/docs")
    except Exception as e:
        print(f"  [Notice] Không thể khởi động web server: {e}")


from src.camera.multi_camera import MultiCameraManager


def initialize_ai_pipeline(blur_faces: bool = False) -> RealtimePipeline:
    """Khởi tạo toàn bộ chuỗi suy luận AI đa tầng."""
    print("[4/5] Nạp các mô hình Deep Learning & Temporal Pipeline...")
    pipeline = RealtimePipeline(blur_faces=blur_faces)
    print("  • YOLOv8 Person Detector : [LOADED] (Hỗ trợ thay đổi model qua configs/model.yaml)")
    print("  • ByteTrack Multi-Tracker: [LOADED] (Giữ ID ổn định, chống nhảy track)")
    print("  • YOLOv8-Pose (17 Joints): [LOADED]")
    print("  • Spatial-Temporal Model : [LOADED] (Nhận diện chuỗi theo thời gian)")
    print("  • Pose Anomaly Scorer    : [LOADED] (Phát hiện dị biệt)")
    print("  • Risk Engine & FSM      : [LOADED] (Debounce chống báo động giả)")
    print(f"  • Face Blurring Realtime : [{'BẬT' if blur_faces else 'TẮT — Mặt hiển thị rõ nét'}]")
    return pipeline


def main() -> None:
    """HÀM CHÍNH — KHỞI CHẠY TOÀN BỘ HỆ THỐNG."""
    parser = argparse.ArgumentParser(description="Fall and Stroke Warning System - Single Master Runner")
    parser.add_argument("--source", type=str, choices=["webcam", "rtsp", "video", "multi"], default="webcam", help="Nguồn video đầu vào (webcam, rtsp, video, hoặc multi cho nhiều camera)")
    parser.add_argument("--device-index", type=int, default=0, help="Device index nếu dùng webcam")
    parser.add_argument("--rtsp-url", type=str, default=None, help="RTSP Stream URL")
    parser.add_argument("--video-path", type=str, default=None, help="Đường dẫn tệp video MP4/AVI")
    parser.add_argument("--blur-faces", action="store_true", default=False, help="Bật làm mờ khuôn mặt (mặc định TẮT để theo dõi rõ nét)")
    parser.add_argument("--gui", action="store_true", default=False, help="Mở thêm cửa sổ desktop OpenCV (mặc định TẮT, ưu tiên hiển thị trên Web Dashboard http://localhost:8000)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy chế độ headless không hiển thị cửa sổ OpenCV")
    parser.add_argument("--no-server", action="store_true", help="Không khởi động FastAPI web server")
    parser.add_argument("--api-port", type=int, default=8000, help="Port cho FastAPI server")
    parser.add_argument("--fast", action="store_true", default=False, help="Chế độ siêu nhanh Nano yolov8n-pose @ 480px")
    parser.add_argument("--sequential", action="store_true", help="Chạy chế độ tuần tự đơn luồng cũ thay vì Decoupled 60 FPS")
    args = parser.parse_args()

    print_banner()

    # BƯỚC 1: Kiểm tra phần cứng và môi trường
    check_system_environment()

    # BƯỚC 2: Tải cấu hình
    configs = load_system_configs()

    # BƯỚC 3: Khởi động Web Server (tùy chọn)
    if not args.no_server:
        start_background_web_server(port=args.api_port)

    # BƯỚC 4: Khởi tạo Pipeline AI (Decoupled High-FPS hoặc Tuần tự)
    decoupled_pipe = None
    if not args.sequential:
        print("[4/5] Khởi tạo DECOUPLED PIPELINE (Tách rời Render Thread 60 FPS & AI Worker)...")
        model_name = "yolov8n-pose.pt" if args.fast else configs["model"].get("pose", {}).get("model_name", "yolov8n-pose.pt")
        img_size = 480 if args.fast else configs["model"].get("pose", {}).get("img_size", 640)
        print(f"  • Single-Pass Model: {model_name} ({img_size}px) — IoU Matching (>=0.70)")
        print("  • Cascade AI Gate  : Kinematic Trigger + Ground Proximity + Hold-on 45 frames")
        print("  • Skip Compensation: Translation Offset (Khử trôi lệch xương)")
        print("  • Keypoint Filter  : One Euro Filter (Khử 100% rung lắc)")
        print("  • Render Target    : 30 - 60 FPS mượt mà (Non-blocking display)")

        from src.pipeline.decoupled_pipeline import DecoupledPipeline
        decoupled_pipe = DecoupledPipeline(
            model_name=model_name,
            img_size=img_size,
            device=configs["inference"].get("runtime", {}).get("device", "cuda"),
            half=configs["inference"].get("runtime", {}).get("use_fp16", True),
        )
        decoupled_pipe.start()
        pipeline = decoupled_pipe
    else:
        pipeline = initialize_ai_pipeline(blur_faces=args.blur_faces)

    # Đồng bộ Alert Dispatcher và Pipeline với Web Dashboard Server
    if not args.no_server:
        try:
            import src.api.server as srv
            pipeline.alert_manager.dispatcher.subscribe_websocket(srv.broadcast_alert_event)
            with srv.state_lock:
                srv.pipeline_instance = pipeline
        except Exception as e:
            print(f"[Warning] Không thể đồng bộ WebSocket dispatcher: {e}")


    # Cấu hình hiển thị (Mặc định Web-First: tắt cửa sổ desktop OpenCV trừ khi truyền --gui)
    display_cfg = configs["inference"].get("display", {})
    window_name = display_cfg.get("window_name", "Fall and Stroke Warning System - Live HUD Stream")
    show_gui = args.gui and not args.no_gui

    # BƯỚC 5: Xử lý chạy đơn camera hoặc đa camera (Multi-Camera Scalability)
    if args.source == "multi":
        print("[5/5] Khởi động chế độ GIÁM SÁT ĐA CAMERA (MULTI-CAMERA GRID)...")
        cam_list = configs["camera"].get("multi_camera", {}).get("cameras", [])
        multi_mgr = MultiCameraManager(camera_configs=cam_list)
        print(f"  • Đã kết nối {len(multi_mgr.streams)} camera đang hoạt động.")
        print("  • Nhấn [q] hoặc [ESC] trên cửa sổ hình ảnh để dừng lại.\n")
        print("-" * 80)

        try:
            while len(multi_mgr.streams) > 0:
                packets_dict = multi_mgr.read_all(timeout=0.2)
                annotated_frames = {}

                for cam_id, packet in packets_dict.items():
                    res = pipeline.process_frame(packet.frame, packet.timestamp)
                    annotated_frames[cam_id] = res.annotated_frame

                    for alert in res.alerts:
                        print(f"\n[CẢNH BÁO TỪ {packet.source_name}]")
                        print(f"  • Đối tượng ID   : {alert.person_id}")
                        print(f"  • Điểm rủi ro    : {alert.risk_score:.2f}")
                        print(f"  • Hành động chính: {alert.evidence.get('primary_action')}")
                        print(f"  • Thông báo      : {alert.message}")
                        print("-" * 80)

                if show_gui and annotated_frames:
                    grid_view = multi_mgr.create_grid_display(annotated_frames)
                    cv2.imshow("Elderly AI Monitor - Multi-Camera Surveillance Grid", grid_view)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break
        finally:
            multi_mgr.stop_all()
            if show_gui:
                cv2.destroyAllWindows()
            print("[Success] Hệ thống Multi-Camera đã dừng an toàn.\n")
            return

    # Chế độ Đơn Camera (Single Camera Realtime Pipeline)
    stream = initialize_camera_stream(
        source_type=args.source,
        device_index=args.device_index,
        rtsp_url=args.rtsp_url,
        video_path=args.video_path,
        camera_cfg=configs["camera"],
    )

    if not args.no_server:
        try:
            import src.api.server as srv
            with srv.state_lock:
                srv.camera_stream = stream
                srv.is_running = True
                src_name = f"Webcam {args.device_index}" if args.source == "webcam" else (f"RTSP Stream" if args.source == "rtsp" else "Video File")
                srv.current_camera_info.update({
                    "source_type": args.source,
                    "device_index": args.device_index,
                    "rtsp_url": args.rtsp_url,
                    "video_path": args.video_path,
                    "name": src_name,
                })
        except Exception:
            pass

    print("\n[5/5] Hệ thống đang hoạt động và giám sát liên tục (CHẾ ĐỘ WEB DASHBOARD).")
    print(f"      Truy cập màn hình theo dõi & điều khiển trực tiếp tại: http://localhost:{args.api_port}")
    if show_gui:
        print("      Cửa sổ desktop OpenCV đang mở. Nhấn [q] hoặc [ESC] để dừng.")
    else:
        print("      Đang chạy chế độ Web-First mượt mà. Nhấn [Ctrl+C] trong terminal để dừng.")
    print("-" * 80)

    try:
        active_stream = stream
        while True:
            # Tham chiếu động tới active stream từ server để hỗ trợ Web UI đổi camera tức thì
            if not args.no_server:
                try:
                    import src.api.server as srv
                    with srv.state_lock:
                        if not srv.is_running:
                            break
                        if srv.camera_stream is not None:
                            active_stream = srv.camera_stream
                except Exception:
                    pass

            if active_stream is None or not active_stream.is_opened():
                time.sleep(0.02)
                continue

            packet: Optional[FramePacket] = active_stream.read(timeout=0.5)
            if packet is None or packet.frame is None:
                time.sleep(0.002)
                continue

            if decoupled_pipe is not None:
                # 1. Đẩy frame sang luồng AI Worker phi đồng bộ
                decoupled_pipe.submit_frame(packet)

                # 2. Render ngay lập tức trên luồng hiển thị (2ms latency -> 60+ FPS)
                annotated_frame, alerts = decoupled_pipe.render_frame(packet.frame, packet.timestamp)
            else:
                # Chế độ tuần tự cũ
                result: PipelineFrameResult = pipeline.process_frame(
                    frame=packet.frame,
                    timestamp=packet.timestamp,
                )
                annotated_frame = result.annotated_frame
                alerts = result.alerts

            # Cập nhật MJPEG Web Stream (mã hóa JPEG chạy trên luồng nền, không chặn vòng lặp)
            if not args.no_server:
                try:
                    import src.api.server as srv
                    srv.publish_annotated_frame(annotated_frame)
                except Exception:
                    pass

            # Xử lý cảnh báo
            for alert in alerts:
                print(f"\n[CẢNH BÁO KHẨN CẤP ĐƯỢC PHÁT HIỆN]")
                print(f"  • Đối tượng ID   : {alert.person_id}")
                print(f"  • Điểm rủi ro    : {alert.risk_score:.2f}")
                print(f"  • Hành động chính: {alert.evidence.get('primary_action')}")
                print(f"  • Thông báo      : {alert.message}")
                print(f"  • Bằng chứng     : {alert.evidence}")
                if alert.snapshot_path:
                    print(f"  • Ảnh chụp lưu tại: {alert.snapshot_path}")
                print("-" * 80)

            # Hiển thị giao diện đồ họa OpenCV trực tiếp (Mặt không bị làm mờ)
            if show_gui:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("\n[Info] Nhận tín hiệu dừng từ bàn phím. Đang tắt hệ thống an toàn...")
                    break

    except KeyboardInterrupt:
        print("\n[Info] Nhận tín hiệu KeyboardInterrupt (Ctrl+C). Đang tắt hệ thống...")
    finally:
        if decoupled_pipe is not None:
            decoupled_pipe.stop()
        if not args.no_server:
            try:
                import src.api.server as srv
                with srv.state_lock:
                    if srv.camera_stream:
                        srv.camera_stream.release()
                    srv.is_running = False
            except Exception:
                pass
        else:
            stream.release()
        if show_gui:
            cv2.destroyAllWindows()
        print("[Success] Hệ thống đã dừng thành công. Tạm biệt!\n")


if __name__ == "__main__":
    main()
