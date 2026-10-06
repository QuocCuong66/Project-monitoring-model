"""Alert Manager orchestrating event cooldowns, deduplication, and dispatch."""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.alerts.clip_recorder import EventClipRecorder, Person
from src.alerts.event_logger import EventLogger, AlertEvent
from src.alerts.notification import NotificationDispatcher
from src.risk.risk_engine import EVENT_FALL, EVENT_STROKE, RiskAssessment

# Old-style assessments (no alert_streams) are routed by their action label
_STROKE_ACTIONS = {"abnormal_movement", "stumbling"}
from src.risk.threshold_manager import ThresholdManager


class AlertManager:
    """Manages alert rate-limiting, cooldown timers, and notifications."""

    def __init__(
        self,
        thresholds: Optional[ThresholdManager] = None,
        logger: Optional[EventLogger] = None,
        dispatcher: Optional[NotificationDispatcher] = None,
        clip_recorder: Optional[EventClipRecorder] = None,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
    ) -> None:
        # Cooldowns, the clip pre-roll and track ids belong to one camera, so every camera
        # gets its own AlertManager; the event log and dispatcher may be shared between them.
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.thresholds = thresholds or ThresholdManager()
        self.logger = logger or EventLogger()
        self.dispatcher = dispatcher or NotificationDispatcher()
        self.clip_recorder = clip_recorder if clip_recorder is not None else EventClipRecorder(
            clips_dir=self.logger.log_dir / "clips"
        )
        # Cooldown per (track, stream): a fall alarm must not silence a stroke-sign alarm, or vice versa
        self._last_alert_timestamps: Dict[Tuple[int, str], float] = {}
        self._last_maintenance = time.time()

    def record_frame(self, frame: np.ndarray, timestamp: float, people: List[Person]) -> None:
        """Feed every camera frame: buffers the clip pre-roll and runs hourly retention cleanup."""
        self.clip_recorder.add_frame(frame, timestamp, people)
        if timestamp - self._last_maintenance >= self.clip_recorder.maintenance_interval_s:
            self._last_maintenance = timestamp
            self.logger.purge_expired()

    @staticmethod
    def _streams(assessment: RiskAssessment) -> List[str]:
        if assessment.alert_streams:
            return list(assessment.alert_streams)
        if not assessment.should_alert:
            return []
        action = str(assessment.evidence.get("primary_action", ""))
        return [EVENT_STROKE if action in _STROKE_ACTIONS else EVENT_FALL]

    def process(
        self,
        assessment: RiskAssessment,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[AlertEvent]:
        """Dispatch one alert per stream (fall / stroke warning) that fired, each with its own cooldown.

        Args:
            assessment: RiskAssessment produced by RiskEngine.
            frame: Optional (already anonymized) frame for snapshot evidence.
            timestamp: Current timestamp in seconds.

        Returns:
            The AlertEvents dispatched on this call (usually none, sometimes one, rarely two).
        """
        curr_time = timestamp if timestamp is not None else time.time()
        tid = assessment.track_id
        events: List[AlertEvent] = []
        for stream in self._streams(assessment):
            key = (tid, stream)
            if (curr_time - self._last_alert_timestamps.get(key, float("-inf"))) < self.thresholds.cooldown_seconds:
                continue  # within this stream's cooldown: suppress the duplicate notification
            self._last_alert_timestamps[key] = curr_time

            # 1. Evidence clip for falls only, log the event and save the anonymized snapshot
            clip_path = None
            if stream == EVENT_FALL and self.clip_recorder is not None:
                clip_path = self.clip_recorder.start_clip(tid, curr_time, camera_id=self.camera_id)
            event = self.logger.log_event(
                person_id=tid,
                risk_score=assessment.stream_scores.get(stream, assessment.risk_score),
                evidence={**assessment.evidence, "event_stream": stream},
                frame=frame,
                clip_path=clip_path,
                camera_id=self.camera_id,
                camera_name=self.camera_name,
                event_type=stream,
            )

            # 2. Dispatch notifications across external channels
            self.dispatcher.dispatch(event)
            events.append(event)
        return events

    def process_assessment(
        self,
        assessment: RiskAssessment,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        """Single-event form of process(): the first alert dispatched, or None."""
        events = self.process(assessment, frame=frame, timestamp=timestamp)
        return events[0] if events else None

    def reset_cooldown(self, track_id: int) -> None:
        """Clear the cooldowns of every stream for a specific track ID."""
        for key in [k for k in self._last_alert_timestamps if k[0] == track_id]:
            self._last_alert_timestamps.pop(key, None)
