"""Alerts need the risk to stay high for the whole debounce window, not one spike."""

from src.risk.state_machine import EventStateMachine, MonitorState
from src.risk.threshold_manager import ThresholdManager

CFG = {
    "state_machine": {
        "thresh_suspicious": 0.50, "thresh_abnormal": 0.68, "thresh_high_risk": 0.80,
        "min_duration_suspicious": 0.3, "min_duration_abnormal": 0.6,
        "min_duration_high_risk": 0.6, "min_recovery_duration": 2.5,
    },
}
DT = 1.0 / 15.0


def _run(sm: EventStateMachine, t: float, seconds: float, risk: float, action: str) -> tuple[float, list[str]]:
    states = []
    for _ in range(int(round(seconds / DT))):
        state, _ = sm.update(risk_score=risk, action=action, timestamp=t)
        states.append(state.value)
        t += DT
    return t, states


def test_sustained_fall_risk_alerts() -> None:
    sm = EventStateMachine(ThresholdManager(CFG), initial_time=0.0)
    _, states = _run(sm, 0.0, 2.0, 0.9, "falling")
    assert MonitorState.ALERT_SENT.value in states


def test_single_spike_after_walking_does_not_alert() -> None:
    sm = EventStateMachine(ThresholdManager(CFG), initial_time=0.0)
    t, states = _run(sm, 0.0, 1.0, 0.9, "falling")          # reaches HIGH_RISK
    assert states[-1] == MonitorState.HIGH_RISK.value
    t, _ = _run(sm, t, 1.0, 0.1, "walking")                  # walking label holds HIGH_RISK
    _, states = _run(sm, t, DT * 2, 0.85, "falling")          # one brief spike
    assert MonitorState.ALERT_SENT.value not in states


def test_flickering_risk_does_not_alert() -> None:
    sm = EventStateMachine(ThresholdManager(CFG), initial_time=0.0)
    t, states = 0.0, []
    for _ in range(20):                                       # 0.13 s high / 0.13 s low, 2.7 s total
        t, s1 = _run(sm, t, DT * 2, 0.9, "falling")
        t, s2 = _run(sm, t, DT * 2, 0.6, "bending")
        states += s1 + s2
    assert MonitorState.ALERT_SENT.value not in states
