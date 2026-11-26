"""Fall clips: only falls are recorded, faces are pixelated, only recent clips are kept."""

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.alerts.alert_manager import AlertManager
from src.alerts.clip_recorder import EventClipRecorder, find_clip_file
from src.alerts.event_logger import EventLogger
from src.risk.risk_engine import RiskAssessment


def _cfg(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True, "pre_seconds": 1.0, "post_seconds": 1.0, "fps": 10.0, "max_width": 320,
        "retention_days": 7, "max_clips": 3, "fall_actions": ["falling", "lying", "immobile"],
        "maintenance_interval_s": 3600,
    }
    cfg.update(overrides)
    return cfg


def _feed(recorder: EventClipRecorder, start: float, seconds: float, people: list = ()) -> float:
    rng = np.random.default_rng(1)
    t = start
    for _ in range(int(seconds * 10)):
        recorder.add_frame(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), t, list(people))
        t += 0.1
    return t


def _wait_for(path_fn, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = path_fn()
        if found is not None:
            return found
        time.sleep(0.05)
    return None


class _SilentDispatcher:
    """Never reaches the user's real Telegram / webhook settings."""

    def dispatch(self, event: Any) -> None:
        pass


def _assessment(action: str) -> RiskAssessment:
    return RiskAssessment(track_id=1, risk_score=0.9, state="ALERT_SENT", state_changed=True,
                          should_alert=True, evidence={"primary_action": action})


def test_clip_has_pre_and_post_roll(tmp_path: Path) -> None:
    rec = EventClipRecorder(clips_dir=tmp_path, cfg=_cfg())
    t = _feed(rec, 1000.0, 2.0)
    rel = rec.start_clip(person_id=1, timestamp=t)
    _feed(rec, t, 1.5)
    clip = _wait_for(lambda: find_clip_file(str(tmp_path / Path(rel).name)))
    assert clip is not None
    cap = cv2.VideoCapture(str(clip))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert 15 <= n <= 22  # ~1 s before + ~1 s after at 10 fps


def test_clip_frames_are_anonymized(tmp_path: Path) -> None:
    rec = EventClipRecorder(clips_dir=tmp_path, cfg=_cfg())
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    bbox = (100.0, 20.0, 200.0, 230.0)
    rec.add_frame(frame, 1.0, [(None, bbox)])
    buffered = rec._preroll[-1]
    head = buffered[22:60, 102:198].reshape(-1, 3)
    assert len(np.unique(head, axis=0)) < len(np.unique(frame[22:60, 102:198].reshape(-1, 3), axis=0)) * 0.05


def test_only_falls_get_a_clip(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path, retention_days=7, save_snapshots=False)
    rec = EventClipRecorder(clips_dir=tmp_path / "clips", cfg=_cfg())
    manager = AlertManager(logger=logger, dispatcher=_SilentDispatcher(), clip_recorder=rec)
    stroke = manager.process_assessment(_assessment("abnormal_movement"), timestamp=100.0)
    manager.reset_cooldown(1)
    fall = manager.process_assessment(_assessment("falling"), timestamp=200.0)
    assert stroke is not None and stroke.clip_path is None
    assert fall is not None and fall.clip_path is not None


def test_only_recent_clips_kept_and_old_ones_expire(tmp_path: Path) -> None:
    rec = EventClipRecorder(clips_dir=tmp_path, cfg=_cfg(max_clips=3))
    now = time.time()
    for i in range(5):
        f = tmp_path / f"clip_{i}.webm"
        f.write_bytes(b"x")
        os.utime(f, (now - 100 + i, now - 100 + i))
    old = tmp_path / "clip_old.webm"
    old.write_bytes(b"x")
    eight_days_ago = now - 8 * 86400
    os.utime(old, (eight_days_ago, eight_days_ago))

    rec.purge_expired(now=now)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip_2.webm", "clip_3.webm", "clip_4.webm"]


def test_deleting_event_removes_its_clip(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path, retention_days=7, save_snapshots=False)
    clip = tmp_path / "clips" / "clip_x.webm"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")
    event = logger.log_event(person_id=3, risk_score=0.9, evidence={}, clip_path=str(clip))
    assert logger.delete_event(event.event_id)
    assert not clip.exists()
