"""Alert Manager orchestrating event cooldowns, deduplication, and dispatch."""

import time
from typing import Dict, List, Optional
import numpy as np
from src.alerts.clip_recorder import EventClipRecorder, Person
from src.alerts.event_logger import EventLogger, AlertEvent
from src.alerts.notification import NotificationDispatcher
from src.risk.risk_engine import RiskAssessment
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
        self._last_alert_timestamps: Dict[int, float] = {}
        self._last_maintenance = time.time()

    def record_frame(self, frame: np.ndarray, timestamp: float, people: List[Person]) -> None:
        """Feed every camera frame: buffers the clip pre-roll and runs hourly retention cleanup."""
        self.clip_recorder.add_frame(frame, timestamp, people)
        if timestamp - self._last_maintenance >= self.clip_recorder.maintenance_interval_s:
            self._last_maintenance = timestamp
            self.logger.purge_expired()

    def process_assessment(
        self,
        assessment: RiskAssessment,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        """Check if assessment requires alert dispatch, respecting cooldown.

        Args:
            assessment: RiskAssessment produced by RiskEngine.
            frame: Optional frame for snapshot evidence.
            timestamp: Current timestamp in seconds.

        Returns:
            AlertEvent if dispatched, else None.
        """
        if not assessment.should_alert:
            return None

        curr_time = timestamp if timestamp is not None else time.time()
        tid = assessment.track_id

        # Cooldown check
        last_alert = self._last_alert_timestamps.get(tid, 0.0)
        if (curr_time - last_alert) < self.thresholds.cooldown_seconds:
            # Within cooldown, suppress duplicate notification
            return None

        self._last_alert_timestamps[tid] = curr_time

        # 1. Start the evidence clip (falls only), log event and save anonymized snapshot
        clip_path = None
        if self.clip_recorder is not None and self.clip_recorder.is_fall(assessment.evidence):
            clip_path = self.clip_recorder.start_clip(tid, curr_time, camera_id=self.camera_id)
        event = self.logger.log_event(
            person_id=tid,
            risk_score=assessment.risk_score,
            evidence=assessment.evidence,
            frame=frame,
            clip_path=clip_path,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
        )

        # 2. Dispatch notifications across external channels
        self.dispatcher.dispatch(event)

        return event

    def reset_cooldown(self, track_id: int) -> None:
        """Clear cooldown for a specific track ID."""
        self._last_alert_timestamps.pop(track_id, None)
