"""Moving close to the camera or bending briefly must not alarm; a real sustained fall still must."""

import numpy as np

from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.risk.risk_engine import RiskEngine
from src.risk.state_machine import EventStateMachine, MonitorState
from src.risk.threshold_manager import ThresholdManager
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import KinematicFeatures, TemporalFeatureExtractor

FPS = 15.0
DT = 1.0 / FPS


def _snapshot(t: float, bbox: tuple[float, float, float, float], hip_y: float, hip_conf: float,
              shoulder_y: float, torso_angle: float) -> TrackSnapshot:
    kp = np.zeros((17, 3), dtype=np.float32)
    cx = (bbox[0] + bbox[2]) / 2.0
    kp[5] = (cx - 40, shoulder_y, 0.9)
    kp[6] = (cx + 40, shoulder_y, 0.9)
    kp[11] = (cx - 25, hip_y, hip_conf)
    kp[12] = (cx + 25, hip_y, hip_conf)
    return TrackSnapshot(t, bbox, kp, kp.copy(), torso_angle)


def test_close_up_arm_movement_is_not_a_drop() -> None:
    # Seated person filling the frame: wide upper-body bbox, hips guessed at the frame bottom with low
    # confidence, bbox jumping up and down as the arms move.
    rng = np.random.default_rng(0)
    snaps = []
    for i in range(30):
        top = 40.0 + rng.uniform(-60, 60)
        snaps.append(_snapshot(i * DT, (100.0, top, 1100.0, 720.0), hip_y=700.0 + rng.uniform(-80, 80),
                               hip_conf=0.3, shoulder_y=400.0, torso_angle=float(rng.uniform(40, 80))))
    kin = TemporalFeatureExtractor(fps=FPS, min_hip_confidence=0.5).extract(snaps)
    assert not kin.lower_body_visible
    assert kin.drop_severity_score == 0.0
    assert kin.vertical_velocity == 0.0


def test_real_fall_with_visible_hips_still_scores_a_drop() -> None:
    snaps = []
    for i in range(30):
        t = i * DT
        if i < 20:   # standing upright
            snaps.append(_snapshot(t, (500.0, 100.0, 650.0, 600.0), 380.0, 0.9, 200.0, 5.0))
        else:        # sudden drop to the floor, lying horizontally
            k = min(1.0, (i - 19) / 3.0)
            snaps.append(_snapshot(t, (400.0, 100.0 + 380.0 * k, 800.0, 600.0), 380.0 + 190.0 * k, 0.9,
                                   200.0 + 340.0 * k, 5.0 + 80.0 * k))
    kin = TemporalFeatureExtractor(fps=FPS, min_hip_confidence=0.5).extract(snaps)
    assert kin.lower_body_visible
    assert kin.drop_severity_score >= 0.5


def test_close_up_wide_bbox_is_not_counted_as_lying_still() -> None:
    engine = RiskEngine(ThresholdManager())
    kin = KinematicFeatures(
        vertical_velocity=0.0, vertical_acceleration=0.0, aspect_ratio_curr=1.6, aspect_ratio_change=0.0,
        torso_angle_curr=70.0, torso_angle_velocity=0.0, immobility_index=1.0, drop_severity_score=0.0,
        lower_body_visible=False,
    )
    action = ActionPrediction("lying", 0.6, {"lying": 0.6}, emergency_prob=0.45, severity_score=0.5)
    anomaly = AnomalyResult(0.3, False, 0.1, 0.22)
    risk = engine.assess(1, action, anomaly, kin, timestamp=0.0).risk_score
    assert risk < ThresholdManager().thresh_suspicious


def _run(sm: EventStateMachine, t: float, seconds: float, risk: float, action: str) -> tuple[float, list[str]]:
    states = []
    for _ in range(int(round(seconds / DT))):
        state, _ = sm.update(risk_score=risk, action=action, timestamp=t)
        states.append(state.value)
        t += DT
    return t, states


def test_shipped_config_ignores_two_second_high_risk_burst() -> None:
    sm = EventStateMachine(ThresholdManager(), initial_time=0.0)
    t, states = _run(sm, 0.0, 2.0, 0.95, "falling")      # e.g. a quick bend misread as a fall
    t, more = _run(sm, t, 3.0, 0.2, "standing")
    assert MonitorState.ALERT_SENT.value not in states + more


def test_shipped_config_alerts_on_sustained_fall() -> None:
    sm = EventStateMachine(ThresholdManager(), initial_time=0.0)
    _, states = _run(sm, 0.0, 4.0, 0.95, "falling")
    assert MonitorState.ALERT_SENT.value in states


def test_shipped_config_needs_at_least_two_seconds_to_alert() -> None:
    th = ThresholdManager()
    assert th.min_duration_abnormal + th.min_duration_high_risk >= 2.0
