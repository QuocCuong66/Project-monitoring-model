"""Only risk events are stored, faces are anonymized, and old records expire."""

import json
import os
import time
from pathlib import Path

import numpy as np

from src.alerts.event_logger import EventLogger
from src.utils.privacy import anonymize_people


def _person() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[0] = (320, 110, 0.9)   # nose
    kp[1] = (310, 100, 0.9)   # eyes
    kp[2] = (330, 100, 0.9)
    kp[3] = (300, 105, 0.9)   # ears
    kp[4] = (340, 105, 0.9)
    return frame, kp, (270.0, 70.0, 370.0, 460.0)


def test_head_is_pixelated_and_source_untouched() -> None:
    frame, kp, bbox = _person()
    original = frame.copy()
    out = anonymize_people(frame, [(kp, bbox)])
    np.testing.assert_array_equal(frame, original)            # caller's frame not modified
    face_before = original[95:115, 305:335].reshape(-1, 3)
    face_after = out[95:115, 305:335].reshape(-1, 3)
    # Mosaic: hundreds of distinct pixel colours collapse into a handful of flat blocks
    assert len(np.unique(face_after, axis=0)) < len(np.unique(face_before, axis=0)) * 0.05
    np.testing.assert_array_equal(out[300:460, 270:370], original[300:460, 270:370])  # body kept


def test_head_falls_back_to_box_top_without_keypoints() -> None:
    frame, _, bbox = _person()
    out = anonymize_people(frame, [(None, bbox)])
    assert not np.array_equal(out[72:90, 272:368], frame[72:90, 272:368])


def test_snapshots_can_be_disabled(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path, retention_days=7, save_snapshots=False)
    event = logger.log_event(person_id=1, risk_score=0.9, evidence={}, frame=np.zeros((10, 10, 3), np.uint8))
    assert event.snapshot_path is None
    assert list((tmp_path / "snapshots").glob("*.jpg")) == []


def test_expired_events_and_snapshots_are_purged(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path, retention_days=7, save_snapshots=True)
    old_img = tmp_path / "snapshots" / "evt_old.jpg"
    old_img.write_bytes(b"x")
    ten_days_ago = time.time() - 10 * 86400
    os.utime(old_img, (ten_days_ago, ten_days_ago))
    with open(logger.jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": "old", "timestamp": "2000-01-01T00:00:00Z"}) + "\n")

    logger.log_event(person_id=2, risk_score=0.9, evidence={})

    assert not old_img.exists()
    ids = [json.loads(line)["event_id"] for line in logger.jsonl_file.read_text(encoding="utf-8").splitlines()]
    assert "old" not in ids and len(ids) == 1


def test_whole_frame_pixelated_when_alerted_person_undetected() -> None:
    from src.utils.privacy import anonymize_alert_snapshot
    frame, kp, bbox = _person()
    drifted_track = (0.0, 0.0, 120.0, 200.0)  # tracker box far from anyone detected
    out = anonymize_alert_snapshot(frame, detections=[(kp, bbox)], tracks=[], alerted_bbox=drifted_track)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) < len(np.unique(frame.reshape(-1, 3), axis=0)) * 0.01
    kept = anonymize_alert_snapshot(frame, detections=[(kp, bbox)], tracks=[], alerted_bbox=bbox)
    np.testing.assert_array_equal(kept[300:460, 270:370], frame[300:460, 270:370])
