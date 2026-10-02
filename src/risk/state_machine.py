"""Debounced Event State Machine with Hysteresis for False Positive suppression."""

from enum import Enum
from typing import Optional, Tuple
from src.risk.threshold_manager import ThresholdManager


class MonitorState(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ABNORMAL = "ABNORMAL"
    HIGH_RISK = "HIGH_RISK"
    ALERT_SENT = "ALERT_SENT"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"


class EventStateMachine:
    """Manages the risk state transitions for a single tracked individual."""

    def __init__(self, thresholds: ThresholdManager, initial_time: float = 0.0) -> None:
        self.thresholds = thresholds
        self.state: MonitorState = MonitorState.NORMAL
        self.state_entry_time: float = initial_time
        self.normal_recovery_time: float = 0.0
        self.last_alert_time: float = 0.0
        # Start of the current unbroken run of scores at/above each threshold. Escalation needs
        # the risk to stay high for the whole debounce window: time_in_state alone let a state
        # held while the score had dropped (e.g. frozen by a "walking" label) alert on one spike.
        self._abnormal_since: Optional[float] = None
        self._high_since: Optional[float] = None

    def _sustained(self, since: Optional[float], timestamp: float, duration: float) -> bool:
        return since is not None and (timestamp - since) >= duration

    def update(
        self,
        risk_score: float,
        action: str,
        timestamp: float,
    ) -> Tuple[MonitorState, bool]:
        """Update state machine with latest risk score and behavior.

        Args:
            risk_score: Current combined risk score in [0.0, 1.0].
            action: Primary detected action label.
            timestamp: Current timestamp in seconds.

        Returns:
            Tuple of (new_state, state_changed_boolean).
        """
        old_state = self.state
        time_in_state = timestamp - self.state_entry_time
        if risk_score >= self.thresholds.thresh_abnormal:
            self._abnormal_since = timestamp if self._abnormal_since is None else self._abnormal_since
        else:
            self._abnormal_since = None
        if risk_score >= self.thresholds.thresh_high_risk:
            self._high_since = timestamp if self._high_since is None else self._high_since
        else:
            self._high_since = None

        # 1. Recovery Check: if person is walking, standing, or getting up with low risk
        if action in ["getting_up", "standing", "walking"] and risk_score < self.thresholds.thresh_suspicious:
            if self.state == MonitorState.NORMAL:
                return MonitorState.NORMAL, False

            # If person was only in SUSPICIOUS, immediately drop back to NORMAL without recovery wait
            if self.state == MonitorState.SUSPICIOUS:
                self.state = MonitorState.NORMAL
                self.state_entry_time = timestamp
                self.normal_recovery_time = 0.0
                return self.state, True

            if self.normal_recovery_time == 0.0:
                self.normal_recovery_time = timestamp
            elif (timestamp - self.normal_recovery_time) >= self.thresholds.min_recovery_duration:
                # Reset back to NORMAL after sustaining recovery behavior
                self.state = MonitorState.NORMAL
                self.state_entry_time = timestamp
                self.normal_recovery_time = 0.0
                return self.state, True

            # In the process of recovering from ABNORMAL / HIGH_RISK, maintain current state
            return self.state, False
        else:
            self.normal_recovery_time = 0.0

        # 2. Forward Escalation Logic
        if self.state == MonitorState.NORMAL:
            if risk_score >= self.thresholds.thresh_high_risk:
                # Acute high-severity event: direct jump to ABNORMAL for rapid temporal debouncing
                self.state = MonitorState.ABNORMAL
                self.state_entry_time = timestamp
            elif risk_score >= self.thresholds.thresh_suspicious:
                self.state = MonitorState.SUSPICIOUS
                self.state_entry_time = timestamp

        elif self.state == MonitorState.SUSPICIOUS:
            if risk_score >= self.thresholds.thresh_abnormal:
                if time_in_state >= self.thresholds.min_duration_suspicious:
                    self.state = MonitorState.ABNORMAL
                    self.state_entry_time = timestamp
            elif risk_score < self.thresholds.thresh_suspicious * 0.8:
                # Drop back to normal if risk recedes
                self.state = MonitorState.NORMAL
                self.state_entry_time = timestamp

        elif self.state == MonitorState.ABNORMAL:
            if risk_score >= self.thresholds.thresh_high_risk:
                if time_in_state >= self.thresholds.min_duration_abnormal and self._sustained(
                    self._high_since, timestamp, self.thresholds.min_duration_abnormal
                ):
                    self.state = MonitorState.HIGH_RISK
                    self.state_entry_time = timestamp
            elif risk_score < self.thresholds.thresh_suspicious:
                self.state = MonitorState.SUSPICIOUS
                self.state_entry_time = timestamp

        elif self.state == MonitorState.HIGH_RISK:
            # Confirmed emergency state! Ready for alert dispatch
            # Require that risk remains elevated throughout the debounce confirmation time
            if risk_score >= self.thresholds.thresh_abnormal:
                if time_in_state >= self.thresholds.min_duration_high_risk and self._sustained(
                    self._abnormal_since, timestamp, self.thresholds.min_duration_high_risk
                ):
                    self.state = MonitorState.ALERT_SENT
                    self.state_entry_time = timestamp
                    self.last_alert_time = timestamp
            elif risk_score < self.thresholds.thresh_suspicious:
                self.state = MonitorState.ABNORMAL
                self.state_entry_time = timestamp

        elif self.state == MonitorState.ALERT_SENT:
            # Move to waiting for confirmation while alert cooldown is active
            if time_in_state >= 2.0:
                self.state = MonitorState.WAITING_FOR_CONFIRMATION
                self.state_entry_time = timestamp

        elif self.state == MonitorState.WAITING_FOR_CONFIRMATION:
            # Stay in waiting unless autonomous recovery or alert cooldown expires
            pass

        state_changed = (self.state != old_state)
        return self.state, state_changed
