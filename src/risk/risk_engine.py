"""Contextual Risk Engine: two separate streams, falls and stroke / acute-illness warning signs.

* Fall stream   - a drop to the floor and what follows (lying still, unable to get up). Evidence: drop
                  kinematics, the action model's falling / lying / immobile classes, immobility, and the
                  lying context (sleeping, phone use and turning over are normal).
* Stroke stream - visible warning signs: staggering, clutching head / chest, one-sided movement,
                  unsteadiness right before a fall, collapsing from a seated position.

Each stream has its own score and its own debounced state machine, so each raises its own alert type
("fall" / "stroke_warning") with its own cooldown. Neither stream ever diagnoses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.temporal_features import KinematicFeatures
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.convulsion import ConvulsionConfig
from src.temporal.lying_context import NOT_LYING, LyingContext, LyingContextAnalyzer, LyingContextConfig
from src.temporal.stroke_signs import StrokeSignAnalyzer, StrokeSignConfig, StrokeSigns
from src.risk.threshold_manager import ThresholdManager
from src.risk.state_machine import MonitorState, EventStateMachine

EVENT_FALL = "fall"
EVENT_STROKE = "stroke_warning"
STREAMS = (EVENT_FALL, EVENT_STROKE)

# Action-model classes that are evidence for the fall stream (staggering and clutching head / chest
# belong to the stroke stream and reach it through StrokeSignAnalyzer, sustained over seconds)
FALL_ACTIONS = ("falling", "immobile", "lying")

# Label shown for a person outside the monitoring scope (lying down without a fall from standing)
SAFE_LABEL = "safe"

_STATE_RANK = {s.value: i for i, s in enumerate([
    MonitorState.NORMAL, MonitorState.SUSPICIOUS, MonitorState.ABNORMAL,
    MonitorState.WAITING_FOR_CONFIRMATION, MonitorState.HIGH_RISK, MonitorState.ALERT_SENT,
])}


@dataclass
class RiskAssessment:
    """Consolidated outcome of the multi-signal risk evaluation.

    risk_score / state / should_alert summarise both streams (the more severe one) for display;
    stream_scores / stream_states / alert_streams hold each stream separately.
    """
    track_id: int
    risk_score: float
    state: str
    state_changed: bool
    should_alert: bool
    evidence: Dict[str, Any]
    stream_scores: Dict[str, float] = field(default_factory=dict)
    stream_states: Dict[str, str] = field(default_factory=dict)
    alert_streams: List[str] = field(default_factory=list)   # streams whose alarm fires on this tick


def display_action(evidence: Dict[str, Any]) -> str:
    """Label drawn on the person card: safe (out of scope), lying context, latest stroke sign, or action."""
    if evidence.get("monitoring_safe"):
        return SAFE_LABEL
    signs = evidence.get("stroke_signs") or []
    return str(evidence.get("lying_context") or (signs[-1] if signs else None)
               or evidence.get("primary_action", "detected"))


class RiskEngine:
    """Multi-signal decision engine evaluating risk severity and dispatching alerts."""

    def __init__(self, thresholds: Optional[ThresholdManager] = None, camera_id: str = "main") -> None:
        self.thresholds = thresholds or ThresholdManager()
        lying_cfg = LyingContextConfig.from_dict(self.thresholds.lying_context)
        self.lying_analyzer: Optional[LyingContextAnalyzer] = (
            LyingContextAnalyzer(lying_cfg, camera_id=camera_id,
                                 convulsion=ConvulsionConfig.from_dict(self.thresholds.convulsion))
            if lying_cfg.enabled else None
        )
        stroke_cfg = StrokeSignConfig.from_dict(self.thresholds.stroke_signs)
        self.stroke_analyzer: Optional[StrokeSignAnalyzer] = (
            StrokeSignAnalyzer(stroke_cfg) if stroke_cfg.enabled else None
        )
        self._state_machines: Dict[int, EventStateMachine] = {}          # fall stream
        self._stroke_machines: Dict[int, EventStateMachine] = {}         # stroke stream
        self.assessments: Dict[int, RiskAssessment] = {}
        self._inactive_counts: Dict[int, int] = {}

    # ------------------------------------------------------------------ fall stream
    def _fall_score(
        self,
        action_pred: ActionPrediction,
        anomaly_res: AnomalyResult,
        kinematics: KinematicFeatures,
        context: Optional[LyingContext],
    ) -> float:
        w_act = self.thresholds.w_action
        w_anom = self.thresholds.w_anomaly
        w_kin = self.thresholds.w_kinematics
        w_immob = self.thresholds.w_immobility

        # 1. Action signal: only the fall-type classes
        probs = action_pred.probabilities or {}
        fall_prob = probs.get("falling", 0.0) + probs.get("immobile", 0.0) + 0.75 * probs.get("lying", 0.0)
        fall_sev = action_pred.severity_score if action_pred.primary_action in FALL_ACTIONS else 0.0
        s_action = float(min(1.0, max(fall_prob, fall_sev)))

        # 2. Anomaly signal
        s_anomaly = anomaly_res.anomaly_score

        # 3. Kinematics signal: drop severity
        s_kin = kinematics.drop_severity_score

        # 4. Immobility signal, only when down / lying
        # Close to the camera the bbox holds only the upper body (wide, hips guessed): not "down"
        # Bending / squatting / kneeling: the legs hold the body up, so it is not "down" either
        is_down = getattr(kinematics, "lower_body_visible", True) and not getattr(kinematics, "feet_supported", False) and (
            kinematics.aspect_ratio_curr > 0.90 or kinematics.torso_angle_curr > 50.0
        )
        s_immob = kinematics.immobility_index if is_down else 0.0

        fused_score = w_act * s_action + w_anom * s_anomaly + w_kin * s_kin + w_immob * s_immob

        # 5. Synergistic escalation
        # A) Acute fall: physical drop AND the action model mutually confirm each other
        synergy_min = self.thresholds.synergy_min_signal
        if s_kin >= synergy_min and s_action >= synergy_min:
            fused_score = max(fused_score, 0.50 * s_action + 0.50 * s_kin)
        # B) Prolonged immobility on the floor
        if is_down and s_immob >= self.thresholds.immobility_synergy_min:
            fused_score = max(fused_score, 0.45 * s_action + 0.40 * s_immob + 0.15 * s_anomaly)

        # 6. Lying context: sleeping, phone use and turning over are normal; failing to get up or lying
        # still on the floor are fall-stream warnings
        if context is not None and self.lying_analyzer is not None:
            # Turning over while already lying is often labelled falling / stumbling. It is not a fall
            # unless the body ended up clearly lower (rolled off the bed) outside a known bed / sofa.
            rolled_in_place = context.in_rest_zone or not context.level_drop
            calm_actions = ("lying", "immobile") + (("falling", "stumbling") if rolled_in_place else ())
            if context.benign and action_pred.primary_action in calm_actions:
                cap = self.lying_analyzer.cfg.benign_action_cap
                fused_score = w_act * min(s_action, cap) + w_anom * s_anomaly + w_kin * s_kin
            if context.concern > 0.0:
                fused_score = max(fused_score, context.concern)

        return float(np.clip(fused_score, 0.0, 1.0))

    # ------------------------------------------------------------------ monitoring scope
    def _safe_by_scope(self, context: Optional[LyingContext]) -> bool:
        """Lying-is-safe scope: anyone not lying (standing, walking, sitting) is monitored; someone lying is
        safe unless they got there by falling / collapsing or are convulsing.

        Safe covers a person already lying when tracking began and one who lay down deliberately (sat first,
        lowered themselves slowly, kept scrolling / turning over after flopping down). A sudden drop (onset
        reason in monitoring_scope.fall_onset_reasons) and any convulsion stay fully monitored.
        """
        if not self.thresholds.safe_when_lying or context is None or context.label == NOT_LYING:
            return False
        if context.convulsing or context.convulsed:
            return False
        fell = context.onset == "sudden" and (
            context.onset_reason in self.thresholds.fall_onset_reasons or context.onset_reason == "inherited"
        )
        return not fell

    # ------------------------------------------------------------------ main API
    def assess(
        self,
        track_id: int,
        action_pred: ActionPrediction,
        anomaly_res: AnomalyResult,
        kinematics: KinematicFeatures,
        timestamp: float,
        snapshot: Optional[TrackSnapshot] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> RiskAssessment:
        """Update both streams for a tracked person.

        Args:
            track_id: Identity of the person.
            action_pred: Supervised action classification result.
            anomaly_res: Unsupervised anomaly score result.
            kinematics: Kinematic feature indicators.
            timestamp: Current timestamp.
            snapshot: Latest pose snapshot (enables lying context and the stroke warning-sign stream).
            frame_size: (width, height) of the camera frame, for bed / sofa rest zones.

        Returns:
            RiskAssessment with each stream's score / state and the alarms firing on this tick.
        """
        # Stroke signs first: unsteadiness right before going down decides how the lying began
        signs: Optional[StrokeSigns] = None
        if self.stroke_analyzer is not None and snapshot is not None:
            signs = self.stroke_analyzer.update(track_id, snapshot, kinematics, timestamp, action_pred.probabilities)

        context: Optional[LyingContext] = None
        if self.lying_analyzer is not None and snapshot is not None:
            context = self.lying_analyzer.update(
                track_id, snapshot, kinematics, timestamp, frame_size,
                unsteady_recently=bool(signs and signs.unsteady_recently),
            )
        safe = self._safe_by_scope(context)
        if context is not None and not safe and self.stroke_analyzer is not None and context.stroke_signs:
            for sign in context.stroke_signs:
                self.stroke_analyzer.note(track_id, sign, timestamp)
            signs = self.stroke_analyzer.evaluate(track_id, timestamp, signs.instability if signs else 0.0,
                                                  signs.staggering_now if signs else False)

        if safe:
            fall_score = stroke_score = 0.0
        else:
            fall_score = self._fall_score(action_pred, anomaly_res, kinematics, context)
            stroke_score = signs.risk if signs is not None else 0.0

        self._inactive_counts[track_id] = 0
        fall_sm = self._state_machines.setdefault(track_id, EventStateMachine(self.thresholds, initial_time=timestamp))
        stroke_sm = self._stroke_machines.setdefault(track_id, EventStateMachine(self.thresholds, initial_time=timestamp))
        # Out of scope counts as recovered: an earlier state winds down like a person back on their feet
        fall_action = "standing" if safe else action_pred.primary_action
        fall_state, fall_changed = fall_sm.update(fall_score, fall_action, timestamp)
        # The stroke machine ignores the action label (a "walking" label must not wipe out a staggering
        # warning); once every sign has faded it recovers like a person back on their feet
        recovery = "standing" if stroke_score < self.thresholds.thresh_suspicious else ""
        stroke_state, stroke_changed = stroke_sm.update(stroke_score, recovery, timestamp)

        alert_streams = [
            stream for stream, state, changed in (
                (EVENT_FALL, fall_state, fall_changed), (EVENT_STROKE, stroke_state, stroke_changed),
            ) if state == MonitorState.ALERT_SENT and changed
        ]
        worst_state = max((fall_state.value, stroke_state.value), key=lambda s: _STATE_RANK.get(s, 0))
        primary_stream = EVENT_STROKE if stroke_score > fall_score else EVENT_FALL

        evidence: Dict[str, Any] = {
            "primary_action": action_pred.primary_action,
            "action_confidence": round(action_pred.confidence, 3),
            "emergency_probability": round(action_pred.emergency_prob, 3),
            "anomaly_score": round(anomaly_res.anomaly_score, 3),
            "reconstruction_error": round(anomaly_res.reconstruction_error, 4),
            "drop_severity": round(kinematics.drop_severity_score, 3),
            "immobility_index": round(kinematics.immobility_index, 3),
            "torso_angle_deg": round(kinematics.torso_angle_curr, 1),
            "aspect_ratio": round(kinematics.aspect_ratio_curr, 2),
            "vertical_velocity": round(kinematics.vertical_velocity, 2),
            "fall_risk": round(fall_score, 3),
            "stroke_risk": round(stroke_score, 3),
            "fall_state": fall_state.value,
            "stroke_state": stroke_state.value,
            "primary_stream": primary_stream,
        }
        if safe:
            evidence["monitoring_safe"] = True
        if signs is not None:
            evidence["gait_instability"] = round(signs.instability, 2)
            if signs.recent_signs:
                evidence["stroke_signs"] = list(signs.recent_signs)
        if context is not None and context.label != "not_lying":
            evidence.update({
                "lying_context": context.label,
                "lying_onset": context.onset,
                "lying_onset_reason": context.onset_reason,
                "lying_seconds": round(context.lying_seconds, 1),
                "in_rest_zone": context.in_rest_zone,
                "warning_signs": list(context.warnings),
            })
        if context is not None and (context.convulsing or context.convulsed):
            evidence["convulsion"] = "ongoing" if context.convulsing else "ended"

        assessment = RiskAssessment(
            track_id=track_id,
            risk_score=max(fall_score, stroke_score),
            state=worst_state,
            state_changed=fall_changed or stroke_changed,
            should_alert=bool(alert_streams),
            evidence=evidence,
            stream_scores={EVENT_FALL: fall_score, EVENT_STROKE: stroke_score},
            stream_states={EVENT_FALL: fall_state.value, EVENT_STROKE: stroke_state.value},
            alert_streams=alert_streams,
        )
        self.assessments[track_id] = assessment
        return assessment

    def cleanup_inactive(self, active_track_ids: Set[int]) -> None:
        """Evict state machines for tracks that have left the scene, with a grace period."""
        if self.lying_analyzer is not None:
            self.lying_analyzer.set_active(active_track_ids)
        for tid in list(self._state_machines.keys()):
            if tid not in active_track_ids:
                self._inactive_counts[tid] = self._inactive_counts.get(tid, 0) + 1
                if self._inactive_counts[tid] > 45:  # ~1.5s - 3s grace buffer
                    remaining = set(self._state_machines) - {tid}
                    if self.lying_analyzer is not None:
                        self.lying_analyzer.cleanup_inactive(remaining)
                    if self.stroke_analyzer is not None:
                        self.stroke_analyzer.cleanup_inactive(remaining)
                    self._state_machines.pop(tid, None)
                    self._stroke_machines.pop(tid, None)
                    self.assessments.pop(tid, None)
                    self._inactive_counts.pop(tid, None)
            else:
                self._inactive_counts[tid] = 0
