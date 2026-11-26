"""Command-line script for real-time camera monitoring and behavior inference."""

import argparse
import sys
import time
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream
from src.camera.video_file import VideoFileStream
from src.pipeline.realtime_pipeline import RealtimePipeline
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Elderly AI Monitor - Realtime Inference Runner")
    parser.add_argument("--source", type=str, choices=["webcam", "rtsp", "video"], default="webcam", help="Input stream source")
    parser.add_argument("--device-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--rtsp-url", type=str, default=None, help="RTSP stream URL")
    parser.add_argument("--video-path", type=str, default=None, help="Path to MP4/AVI video file")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--loop", action="store_true", help="Loop video playback if source is video")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--disable-motion-gating", action="store_true", help="Disable Tier-0 Motion Gating (forces AI inference on every frame)")
    args = parser.parse_args()


    # Load configuration
    cam_cfg = load_config("camera.yaml")
    infer_cfg = load_config("inference.yaml")

    # Select Camera
    if args.source == "webcam":
        print(f"[Info] Starting Webcam on index {args.device_index}...")
        stream = WebcamStream(device_index=args.device_index)
    elif args.source == "rtsp":
        url = args.rtsp_url or cam_cfg.get("rtsp", {}).get("url")
        print(f"[Info] Connecting to RTSP stream: {url}...")
        stream = RTSPStream(url=url)
    elif args.source == "video":
        vpath = args.video_path or cam_cfg.get("video", {}).get("path")
        print(f"[Info] Opening Video file: {vpath}...")
        stream = VideoFileStream(video_path=vpath, loop=args.loop)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    if not stream.start():
        print("[Error] Failed to open video source. Exiting.")
        sys.exit(1)

    # Initialize Pipeline
    print("[Info] Initializing Realtime AI Pipeline...")
    from src.detection.motion_gater import MotionGater
    motion_gater = MotionGater(enabled=not args.disable_motion_gating)
    pipeline = RealtimePipeline(motion_gater=motion_gater)
    print(f"[Info] Pipeline initialized successfully (Tier-0 Motion Gating: {'DISABLED' if args.disable_motion_gating else 'ENABLED'}). Monitoring active...")


    window_name = infer_cfg.get("display", {}).get("window_name", "Elderly AI Monitor")

    try:
        while stream.is_opened():
            packet = stream.read(timeout=1.0)
            if packet is None:
                continue

            # Process frame
            result = pipeline.process_frame(packet.frame, packet.timestamp)

            # Check for newly triggered emergency alerts
            for alert in result.alerts:
                print(f"[ALERT TRIGGERED] Track {alert.person_id} | Risk: {alert.risk_score:.2f} | Evidence: {alert.evidence}")

            # Display GUI
            if not args.no_display:
                cv2.imshow(window_name, result.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):  # ESC or 'q' to quit
                    print("[Info] Termination key pressed.")
                    break

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("[Info] Shutdown complete.")


if __name__ == "__main__":
    main()
