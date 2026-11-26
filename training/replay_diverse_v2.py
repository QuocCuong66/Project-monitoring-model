"""Replay cached held-out benign recordings through action/anomaly/risk/debounce.

This is a pose replay with a fixed single-person identity, not a live camera test.
Unknown/missing intervals break history; unrelated clips are never concatenated.
"""

from collections import defaultdict, deque
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.action.action_classifier import ActionClassifier  # noqa: E402
from src.anomaly.anomaly_score import ReconstructionAnomalyScorer  # noqa: E402
from src.pose.keypoints import normalize_keypoints, compute_torso_angle  # noqa: E402
from src.risk.risk_engine import RiskEngine  # noqa: E402
from src.risk.threshold_manager import ThresholdManager  # noqa: E402
from src.temporal.sequence_buffer import TrackSnapshot  # noqa: E402
from src.temporal.sequence_features import build_model_sequence  # noqa: E402
from src.temporal.temporal_features import TemporalFeatureExtractor  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT/"configs/diverse_v2.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT/"configs/model.yaml").read_text(encoding="utf-8"))
    output = ROOT/cfg["training"]["output"]
    report = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    if report["selected"] is None:
        raise ValueError("Replay only after validation selection is complete")
    models = {"baseline": ROOT/cfg["integration"]["baseline_dir"]/"active.onnx", "candidate": output.with_suffix(".onnx")}
    temporal = model_cfg["temporal"]
    classifiers = {name: ActionClassifier(weights_path=path, architecture=temporal["architecture"],
                                          device="cuda", normal_activity_policy=temporal.get("normal_activity_policy"))
                   for name, path in models.items()}
    a = model_cfg["anomaly"]
    anomaly = ReconstructionAnomalyScorer(weights_path=ROOT/a["weights_path"], threshold=a["threshold"],
                                          steepness=a["steepness"], device="cuda")
    groups: dict[str, dict] = defaultdict(dict)
    for name in cfg["sources"]:
        if name == "oops":
            continue
        rows = json.loads((ROOT/cfg["dataset"]["processed_dir"]/f"{name}.windows.json").read_text(encoding="utf-8"))
        for row in rows:
            if row["split"] == 2 and row["normal_recording"]:
                groups[f"{name}:{row['group']}:{row['camera']}"][row["pose_file"]] = {**row, "source": name}
    results = []
    fps, length = cfg["preparation"]["target_fps"], temporal["window_size"]
    for group, unique in sorted(groups.items()):
        histories = deque(maxlen=length)
        engines = {name: RiskEngine(ThresholdManager(), camera_id="diverse_v2_replay") for name in models}
        alerts = {name: [] for name in models}
        previous, block_start, observed, max_block, resets, evaluated, tick = None, None, 0., 0., 0, 0, 0
        for row in sorted(unique.values(), key=lambda r: r["start"]):
            pose_file = ROOT/row["pose_file"]
            video_dir = ROOT/cfg["dataset"]["extracted_dir"]/row["source"]
            video = video_dir/(pose_file.stem+Path(row["member"]).suffix)
            capture = cv2.VideoCapture(str(video))
            frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            capture.release()
            with np.load(pose_file, allow_pickle=False) as pack:
                track, boxes, times = pack["pose"], pack["bbox"], pack["timestamps"]
                if "phase_cropped" not in pack:
                    offset = row["media_offset_seconds"]
                    mask = (times >= offset) & (times < offset+row["end"]-row["start"])
                    track, boxes, times = track[mask], boxes[mask], times[mask]-offset
                for kp, box, relative in zip(track, boxes, times):
                    now = float(relative+row["start"])
                    if previous is not None and now <= previous:
                        continue
                    if (kp[:, 2] >= cfg["preparation"]["keypoint_confidence"]).sum() < cfg["preparation"]["min_visible_joints"]:
                        continue
                    if previous is None or now-previous > 2/fps:
                        histories.clear()
                        engines = {name: RiskEngine(ThresholdManager(), camera_id="diverse_v2_replay") for name in models}
                        block_start = now
                        resets += 1
                    else:
                        observed += now-previous
                    previous = now
                    max_block = max(max_block, now-block_start)
                    snapshot = TrackSnapshot(now, tuple(box), kp, normalize_keypoints(kp), compute_torso_angle(kp))
                    histories.append(snapshot)
                    tick += 1
                    if len(histories) < max(10, length//2) or tick % temporal["stride"]:
                        continue
                    kin = TemporalFeatureExtractor(fps=fps).extract(list(histories))
                    sequence = build_model_sequence(np.stack([s.keypoints for s in histories]), length)
                    anom = anomaly.score(sequence)
                    for name, classifier in classifiers.items():
                        prediction = classifier.predict(sequence, kin)
                        assessment = engines[name].assess(1, prediction, anom, kin, now, snapshot, frame_size)
                        if assessment.should_alert:
                            alerts[name].append({"timestamp": now, "streams": assessment.alert_streams})
                    evaluated += 1
        results.append({"group": group, "observed_seconds": observed, "max_continuous_seconds": max_block,
                        "history_blocks": resets, "evaluated_ticks": evaluated, "alerts": alerts})
        print(group, "seconds", round(observed, 2), "alerts", {k: len(v) for k, v in alerts.items()}, flush=True)
    total = sum(r["observed_seconds"] for r in results)
    summary = {"observed_minutes": total/60, "recordings": len(results),
               "max_continuous_seconds": max((r["max_continuous_seconds"] for r in results), default=0),
               "alerts": {name: sum(len(r["alerts"][name]) for r in results) for name in models},
               "protocol": "Held-out benign recordings, fixed person identity, no camera gating; missing/unknown intervals reset history",
               "limitation": "Short labeled clips; no overnight sleep or field false-alert-rate claim", "recording_results": results}
    output.with_suffix(".replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "recording_results"}), flush=True)


if __name__ == "__main__":
    main()
