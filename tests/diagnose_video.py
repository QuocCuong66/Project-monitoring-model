"""Diagnostic script to test pipeline detection on snaptik video."""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.realtime_pipeline import RealtimePipeline
from src.utils.config import load_config

def main():
    video_path = PROJECT_ROOT / "tests" / "snaptik.vn_7674987674838551821.mp4"
    if not video_path.exists():
        video_path = Path(r"d:\Downloads\snaptik.vn_7674987674838551821.mp4")

    print(f"Testing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {total_frames/fps:.1f}s")

    pipeline = RealtimePipeline(single_pass=True, blur_faces=False)

    frame_idx = 0
    alerts_all = []
    max_risk_seen = 0.0
    states_seen = set()
    actions_seen = {}

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp = frame_idx / fps

        result = pipeline.process_frame(frame, timestamp)

        if result.alerts:
            for alert in result.alerts:
                alerts_all.append((frame_idx, timestamp, alert))
                print(f"\n>>> [ALERT TRIGGERED at frame {frame_idx}, {timestamp:.2f}s] <<<", flush=True)
                print(f"    Track ID: {alert.person_id}, Risk: {alert.risk_score:.3f}", flush=True)
                print(f"    Action: {alert.evidence.get('primary_action')}", flush=True)
                print(f"    Message: {alert.message}", flush=True)
                print(f"    Evidence: {alert.evidence}\n", flush=True)

        if not result.assessments and frame_idx % 30 == 0:
            print(f"[Frame {frame_idx:4d} | {timestamp:5.2f}s] Active tracks: {len(result.tracks)}, Poses: {len(result.poses)}, Bypassed: {result.telemetry.get('ai_bypassed', False)}", flush=True)

        for ass in result.assessments:
            states_seen.add(ass.state)
            if ass.risk_score > max_risk_seen:
                max_risk_seen = ass.risk_score
            act = ass.evidence.get("primary_action", "unknown")
            actions_seen[act] = actions_seen.get(act, 0) + 1

            if ass.risk_score > 0.4 or ass.state != "NORMAL" or frame_idx % 30 == 0:
                print(f"[Frame {frame_idx:4d} | {timestamp:5.2f}s] Track {ass.track_id} | State: {ass.state:10s} | Risk: {ass.risk_score:.3f} | Action: {act:12s} | Angle: {ass.evidence.get('torso_angle_deg', 0):.1f} | AR: {ass.evidence.get('aspect_ratio', 0):.2f} | Vy: {ass.evidence.get('vertical_velocity', 0):.2f} | Immob: {ass.evidence.get('immobility_index', 0):.2f}", flush=True)

    cap.release()
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"Processing finished in {elapsed:.2f}s ({total_frames/elapsed:.1f} FPS)")
    print(f"Total alerts triggered: {len(alerts_all)}")
    print(f"Max risk score observed: {max_risk_seen:.3f}")
    print(f"States observed: {states_seen}")
    print(f"Actions distribution: {actions_seen}")
    print("="*60)

if __name__ == "__main__":
    main()
