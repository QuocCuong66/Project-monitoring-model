"""Short evidence clips around fall alerts, with faces pixelated and bounded retention.

Only falls are kept: a clip is written when an alert fires for one of the configured
fall actions, covering a few seconds before and after the alert. Nothing else is ever
written to disk. Clips older than retention_days are deleted, and at most max_clips of the
most recent ones are kept.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from src.alerts.event_logger import safe_file_tag
from src.utils.config import get_project_root, load_config
from src.utils.privacy import anonymize_people

Person = tuple[Optional[np.ndarray], tuple[float, float, float, float]]

CLIP_SUFFIXES = (".webm", ".mp4")


def find_clip_file(clip_path: Optional[str]) -> Optional[Path]:
    """Locate a finished clip on disk (its extension depends on which codec was available)."""
    if not clip_path:
        return None
    base = Path(clip_path)
    if not base.is_absolute():
        base = get_project_root() / base
    for suffix in CLIP_SUFFIXES:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


@dataclass
class _Recording:
    path: Path
    end_time: float
    frames: list[np.ndarray] = field(default_factory=list)


class EventClipRecorder:
    """Keeps an anonymized pre-roll in memory and writes a clip when a fall alert fires."""

    def __init__(
        self,
        clips_dir: Optional[Union[str, Path]] = None,
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        if cfg is None:
            try:
                cfg = (load_config("inference.yaml") or {}).get("event_clips", {}) or {}
            except Exception:
                cfg = {}
        self.enabled = bool(cfg.get("enabled", True))
        self.pre_seconds = float(cfg.get("pre_seconds", 5.0))
        self.post_seconds = float(cfg.get("post_seconds", 5.0))
        self.fps = max(1.0, float(cfg.get("fps", 10.0)))
        self.max_width = int(cfg.get("max_width", 640))
        self.retention_days = float(cfg.get("retention_days", 7))
        self.max_clips = int(cfg.get("max_clips", 30))
        self.fall_actions = {str(a) for a in cfg.get("fall_actions", ["falling", "lying", "immobile"])}
        self.codecs: list[tuple[str, str]] = [
            (str(c[0]), str(c[1])) for c in cfg.get("codecs", [["VP80", ".webm"], ["mp4v", ".mp4"]])
        ]
        self.maintenance_interval_s = float(cfg.get("maintenance_interval_s", 3600.0))

        self.clips_dir = Path(clips_dir) if clips_dir is not None else (
            get_project_root() / "data" / "processed" / "events" / "clips"
        )
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(round(self.pre_seconds * self.fps))))
        self._recordings: list[_Recording] = []
        self._last_sample_ts = 0.0
        self._last_maintenance = 0.0
        self.purge_expired()

    def is_fall(self, evidence: dict[str, Any]) -> bool:
        return str(evidence.get("primary_action", "")) in self.fall_actions

    def add_frame(self, frame: np.ndarray, timestamp: float, people: list[Person]) -> None:
        """Feed a camera frame (subsampled to the clip fps; heads pixelated before buffering)."""
        if not self.enabled or frame is None:
            return
        if timestamp - self._last_sample_ts < 0.9 / self.fps:
            return
        self._last_sample_ts = timestamp

        small = anonymize_people(frame, people)
        h, w = small.shape[:2]
        if w > self.max_width:
            small = cv2.resize(small, (self.max_width, int(round(h * self.max_width / w))), interpolation=cv2.INTER_AREA)

        finished: list[_Recording] = []
        with self._lock:
            self._preroll.append(small)
            for rec in self._recordings:
                rec.frames.append(small)
            finished = [r for r in self._recordings if timestamp >= r.end_time]
            self._recordings = [r for r in self._recordings if timestamp < r.end_time]
        for rec in finished:
            threading.Thread(target=self._write_clip, args=(rec,), daemon=True, name="ClipWriter").start()

        if timestamp - self._last_maintenance >= self.maintenance_interval_s:
            self._last_maintenance = timestamp
            self.purge_expired()

    def start_clip(self, person_id: int, timestamp: float, camera_id: Optional[str] = None) -> Optional[str]:
        """Begin a clip for an alert at `timestamp`. Returns its path relative to the project root.

        The file appears once post_seconds of footage have been collected.
        """
        if not self.enabled:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = self.codecs[0][1] if self.codecs else ".webm"
        cam_tag = f"_{safe_file_tag(camera_id)}" if camera_id else ""
        path = self.clips_dir / f"clip_{stamp}{cam_tag}_trk{person_id}{ext}"
        with self._lock:
            self._recordings.append(_Recording(path=path, end_time=timestamp + self.post_seconds, frames=list(self._preroll)))
        try:
            return str(path.relative_to(get_project_root()))
        except ValueError:
            return str(path)

    def _write_clip(self, rec: _Recording) -> None:
        if not rec.frames:
            return
        h, w = rec.frames[0].shape[:2]
        for fourcc, ext in self.codecs:
            target = rec.path.with_suffix(ext)
            tmp = target.with_name(target.stem + ".part" + ext)
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*fourcc), self.fps, (w, h))
            if not writer.isOpened():
                writer.release()
                tmp.unlink(missing_ok=True)
                continue
            for f in rec.frames:
                writer.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
            writer.release()
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(target)  # atomic: readers never see a half-written clip
                break
            tmp.unlink(missing_ok=True)
        self.purge_expired()

    def purge_expired(self, now: Optional[float] = None) -> int:
        """Delete clips older than retention_days, then all but the max_clips most recent."""
        now = now if now is not None else time.time()
        clips = sorted(
            (p for p in self.clips_dir.iterdir() if p.suffix in CLIP_SUFFIXES and ".part" not in p.name),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        cutoff = now - self.retention_days * 86400.0 if self.retention_days > 0 else None
        removed = 0
        for i, clip in enumerate(clips):
            expired = cutoff is not None and clip.stat().st_mtime < cutoff
            over_cap = self.max_clips > 0 and i >= self.max_clips
            if expired or over_cap:
                try:
                    clip.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def flush(self) -> None:
        """Write any clip still being collected (e.g. on shutdown) synchronously."""
        with self._lock:
            pending, self._recordings = self._recordings, []
        for rec in pending:
            self._write_clip(rec)
