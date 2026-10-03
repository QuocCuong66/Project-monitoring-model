"""Structured event logger for auditing medical emergency alerts."""

import json
import re
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from src.utils.config import get_project_root, load_config


@dataclass
class AlertEvent:
    """Represents a validated emergency event."""
    event_id: str
    event_type: str
    person_id: int
    timestamp: str
    risk_score: float
    evidence: Dict[str, Any]
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None  # anonymized fall clip; the file appears a few seconds after the alert
    message: str = "Possible medical emergency / abnormal behavior detected. Please check the person."
    camera_id: Optional[str] = None  # which camera raised it (several cameras share one event log)
    camera_name: Optional[str] = None


# Every camera's AI worker appends to the same events.jsonl, and purge / delete rewrite it:
# one process-wide lock keeps those writes from interleaving or losing records.
_JSONL_LOCK = threading.RLock()


def safe_file_tag(value: str) -> str:
    """Camera id reduced to characters that are safe inside a file name / event id."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")[:32]


class EventLogger:
    """Manages local, encrypted or anonymized JSON event logging."""

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        retention_days: Optional[float] = None,
        save_snapshots: Optional[bool] = None,
    ) -> None:
        if log_dir is None:
            self.log_dir = get_project_root() / "data" / "processed" / "events"
        else:
            self.log_dir = Path(log_dir)

        privacy_cfg: Dict[str, Any] = {}
        try:
            privacy_cfg = (load_config("inference.yaml") or {}).get("privacy", {}) or {}
        except Exception:
            pass
        self.retention_days = float(retention_days if retention_days is not None else privacy_cfg.get("retention_days", 7))
        self.save_snapshots = bool(save_snapshots if save_snapshots is not None else privacy_cfg.get("save_snapshots", True))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.log_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_file = self.log_dir / "events.jsonl"
        self.purge_expired()

    def purge_expired(self, now: Optional[float] = None) -> int:
        """Delete snapshots and event records older than retention_days. Returns files removed."""
        if self.retention_days <= 0:
            return 0
        cutoff = (now if now is not None else time.time()) - self.retention_days * 86400.0
        removed = 0
        for img in self.snapshots_dir.glob("*.jpg"):
            try:
                if img.stat().st_mtime < cutoff:
                    img.unlink()
                    removed += 1
            except OSError:
                pass
        with _JSONL_LOCK:
            if self.jsonl_file.exists():
                kept: List[str] = []
                with open(self.jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            ts = json.loads(line)["timestamp"].rstrip("Z")
                            if datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp() < cutoff:
                                continue
                        except (ValueError, KeyError, TypeError):
                            pass
                        kept.append(line)
                with open(self.jsonl_file, "w", encoding="utf-8") as f:
                    f.writelines(kept)
        return removed

    def log_event(
        self,
        person_id: int,
        risk_score: float,
        evidence: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
        clip_path: Optional[str] = None,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
    ) -> AlertEvent:
        """Create, record, and persist a new emergency alert event.

        Args:
            person_id: Track ID of the person (unique per camera only).
            risk_score: Calculated risk severity score.
            evidence: Dictionary of kinematic and action signals.
            frame: Optional BGR evidence frame. Callers must pass an already anonymized
                frame (src.utils.privacy.anonymize_people); it is saved only when
                privacy.save_snapshots is enabled.
            clip_path: Evidence clip being recorded for this event (falls only), if any.
            camera_id: Camera that raised the alert; part of the event id so two cameras
                tracking "person 1" in the same second never collide.
            camera_name: Human-readable room / camera name shown in the UI and Telegram.

        Returns:
            AlertEvent dataclass instance.
        """
        now = datetime.utcnow()
        timestamp_str = now.isoformat() + "Z"
        cam_tag = f"_{safe_file_tag(camera_id)}" if camera_id else ""
        event_id = f"evt_{now.strftime('%Y%m%d_%H%M%S')}{cam_tag}_trk{person_id}"

        snapshot_rel_path = None
        if frame is not None and self.save_snapshots:
            filename = f"{event_id}.jpg"
            save_path = self.snapshots_dir / filename
            cv2.imwrite(str(save_path), frame)
            snapshot_rel_path = str(save_path.relative_to(get_project_root()))

        event = AlertEvent(
            event_id=event_id,
            event_type="possible_medical_emergency",
            person_id=person_id,
            timestamp=timestamp_str,
            risk_score=round(risk_score, 3),
            evidence=evidence,
            snapshot_path=snapshot_rel_path,
            clip_path=clip_path,
            camera_id=camera_id,
            camera_name=camera_name,
        )

        # Append to JSONL log
        with _JSONL_LOCK:
            with open(self.jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")

        self.purge_expired()
        return event

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the most recent logged events."""
        if not self.jsonl_file.exists():
            return []

        events = []
        with open(self.jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events[-limit:]

    def clear_events(self) -> None:
        """Clear all historical logged events, their snapshots and clips, and reset JSONL log."""
        with _JSONL_LOCK:
            for evt in self.get_recent_events(limit=10**9):
                self._remove_media(evt)
            if self.jsonl_file.exists():
                try:
                    self.jsonl_file.unlink()
                except Exception:
                    with open(self.jsonl_file, "w", encoding="utf-8") as f:
                        f.truncate(0)

    def delete_event(self, event_id: str) -> bool:
        """Permanently remove a single event by event_id from storage and disk."""
        with _JSONL_LOCK:
            if not self.jsonl_file.exists():
                return False

            remaining_events = []
            found = False
            with open(self.jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("event_id") == event_id:
                            found = True
                            self._remove_media(data)
                            continue
                        remaining_events.append(data)
                    except json.JSONDecodeError:
                        continue

            if found:
                with open(self.jsonl_file, "w", encoding="utf-8") as f:
                    for ev in remaining_events:
                        f.write(json.dumps(ev) + "\n")
            return found

    @staticmethod
    def _remove_media(event: Dict[str, Any]) -> None:
        """Delete the snapshot and clip files that belong to an event record."""
        from src.alerts.clip_recorder import find_clip_file

        files = []
        if event.get("snapshot_path"):
            files.append(get_project_root() / event["snapshot_path"])
        clip = find_clip_file(event.get("clip_path"))
        if clip is not None:
            files.append(clip)
        for path in files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
