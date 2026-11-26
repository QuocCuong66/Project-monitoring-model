"""Two separate alert streams (fall / stroke warning signs), and deliberate lying-down vs staggering then falling."""

import math
from pathlib import Path
from typing import Any

from src.alerts.alert_manager import AlertManager
from src.alerts.event_logger import EventLogger
from src.risk.risk_engine import EVENT_FALL, EVENT_STROKE, RiskAssessment
from tests.test_lying_context import LYING, SEATED, UPRIGHT, Scene, _kp

TORSO_PX = 120.0   # Scene draws normalized skeletons at 120 px per torso length


class StreamScene(Scene):
    """Scene that also records every alarm per stream."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.fired: list[str] = []
        self.last = None

    def step(self, *a: Any, **k: Any) -> None:
        super().step(*a, **k)
        ass = self.engine.assessments[k.get("tid", 1)]
        self.fired += ass.alert_streams
        self.last = ass

    def onset(self, tid: int = 1) -> tuple[str, str]:
        ep = self.engine.lying_analyzer._tracks[tid].episode
        return ep.onset, ep.reason

    # --- movements ---------------------------------------------------------
    def stand_still(self, seconds: float) -> None:
        self.run(seconds, lambda _: self.step(_kp(UPRIGHT, noise=0.01, rng=self.rng), 5.0, "standing", immob=0.6))

    def walk(self, seconds: float, sway: float = 0.04, speed: float = 150.0, curve: bool = False,
             x0: float = 640.0) -> None:
        start = self.t

        def f(_: float) -> None:
            e = self.t - start
            # curve: turning around furniture at ~1.2 rad/s, a tight path for a 2 s gait window
            x = x0 + (250.0 * math.sin(1.2 * e) if curve else speed * e) + sway * TORSO_PX * math.sin(2 * math.pi * 1.0 * e)
            self.step(_kp(UPRIGHT, noise=0.01, rng=self.rng), 5.0 + 2.0 * math.sin(2 * math.pi * 1.0 * e), "walking",
                      immob=0.1, center=(x, 500.0))
        self.run(seconds, f)

    def stagger(self, seconds: float) -> None:
        start = self.t

        def f(_: float) -> None:
            e = self.t - start
            x = 600.0 + 0.45 * TORSO_PX * math.sin(2 * math.pi * 1.2 * e)          # zig-zag, no progress
            self.step(_kp(UPRIGHT, noise=0.02, rng=self.rng), 12.0 + 10.0 * math.sin(2 * math.pi * 0.9 * e),
                      "stumbling", immob=0.1, center=(x, 500.0))
        self.run(seconds, f)

    def lower_slowly(self, seconds: float = 2.5) -> None:
        def f(e: float) -> None:
            k = min(1.0, e / seconds)
            # the skeleton rotates gradually from upright to lying
            pose = {j: (UPRIGHT[j][0] * (1 - k) + LYING[j][0] * k, UPRIGHT[j][1] * (1 - k) + LYING[j][1] * k)
                    for j in UPRIGHT}
            self.step(_kp(pose, noise=0.01, rng=self.rng), 5.0 + 80.0 * k,
                      "lying" if k > 0.6 else "bending", vy=0.4, immob=0.3)
        self.run(seconds, f)

    def scroll_phone(self, seconds: float) -> None:
        def f(e: float) -> None:
            w = 0.12 * math.sin(e * 6.0)
            self.step(_kp(LYING, {10: (-0.95 + w, -0.05)}, noise=0.01, rng=self.rng), 85.0, "lying", immob=0.8)
        self.run(seconds, f)


# --- deliberate lying down ---------------------------------------------------

def test_stop_stand_still_then_lower_slowly_is_deliberate() -> None:
    s = StreamScene()
    s.walk(3.0)
    s.stand_still(2.0)
    s.lower_slowly(2.5)
    s.lie_still(60.0)
    assert s.onset() == ("controlled", "paused_then_lowered")
    assert not s.fired


def test_sitting_first_then_lying_back_is_deliberate() -> None:
    s = StreamScene()
    s.walk(3.0)
    s.lie_down_calmly()
    s.lie_still(60.0)
    assert s.onset() == ("controlled", "sat_down_first")
    assert not s.fired


def test_walking_on_after_a_pause_then_going_down_is_not_deliberate() -> None:
    s = StreamScene()
    s.stand_still(2.0)
    s.walk(1.5)                                    # walks on after the pause ...
    s.lower_slowly(1.8)                            # ... and goes down (trip): no deliberate cue left
    s.lie_still(15.0)
    assert s.onset()[0] == "sudden"
    assert EVENT_FALL in s.fired


# --- staggering then falling -------------------------------------------------

def test_staggering_then_slow_fall_is_a_fall_and_a_stroke_sign() -> None:
    s = StreamScene()
    s.stagger(4.0)
    s.lower_slowly(2.6)                            # slower than "upright_to_floor": only the stagger tells
    s.lie_still(15.0)
    assert s.onset() == ("sudden", "unsteady_before")
    assert EVENT_FALL in s.fired
    assert "unsteady_before_fall" in s.last.evidence["stroke_signs"]
    assert EVENT_STROKE in s.fired                 # staggering + unsteady fall = two signs


def test_same_slow_descent_without_staggering_is_not_a_fall() -> None:
    s = StreamScene()
    s.walk(3.0)
    s.stand_still(2.0)
    s.lower_slowly(2.6)
    s.lie_still(15.0)
    assert s.onset()[0] == "controlled"
    assert not s.fired


# --- stroke stream on its own --------------------------------------------------

def test_staggering_alone_is_a_warning_not_an_alarm() -> None:
    s = StreamScene()
    s.stagger(12.0)
    assert "staggering" in s.last.evidence["stroke_signs"]
    assert s.last.stream_scores[EVENT_STROKE] >= 0.76            # shown as ABNORMAL
    assert not s.fired


def test_normal_walking_straight_or_in_a_curve_is_not_staggering() -> None:
    for curve in (False, True):
        s = StreamScene()
        s.walk(12.0, curve=curve)
        assert s.last.evidence["gait_instability"] < 0.5, curve
        assert "stroke_signs" not in s.last.evidence, curve
        assert not s.fired


def test_clutching_head_plus_staggering_raises_a_stroke_alarm_only() -> None:
    s = StreamScene()
    s.run(6.0, lambda _: s.step(_kp(UPRIGHT, noise=0.01, rng=s.rng), 8.0, "abnormal_movement", immob=0.5))
    assert "abnormal_gesture" in s.last.evidence["stroke_signs"]
    assert not s.fired                             # one sign: warning only
    s.stagger(10.0)
    assert s.fired.count(EVENT_STROKE) == 1        # two signs: stroke alarm ...
    assert EVENT_FALL not in s.fired               # ... and no fall alarm


def test_collapse_from_sitting_alarms_both_streams() -> None:
    s = StreamScene()
    s.run(3.0, lambda _: s.step(_kp(SEATED, noise=0.01, rng=s.rng), 5.0, "sitting", immob=0.5))

    def slump(e: float) -> None:
        k = min(1.0, e / 0.5)
        s.step(_kp(SEATED if k < 0.5 else LYING, noise=0.01, rng=s.rng), 5.0 + 80.0 * k, "falling", vy=3.0,
               drop=0.9, immob=0.0)
    s.run(0.6, slump)
    s.lie_still(15.0)
    assert s.onset() == ("sudden", "collapse_from_sitting")
    assert EVENT_STROKE in s.fired and EVENT_FALL in s.fired


def test_lying_down_and_using_phone_never_feeds_the_stroke_stream() -> None:
    s = StreamScene()
    s.walk(3.0)
    s.lie_down_calmly()
    s.scroll_phone(60.0)
    assert max(s.risks[-600:]) < 0.60
    assert not s.fired


# --- alert manager: one event type and cooldown per stream ----------------------

class _Silent:
    def dispatch(self, event: Any) -> None:
        pass


def _assessment(streams: list[str]) -> RiskAssessment:
    return RiskAssessment(track_id=1, risk_score=0.93, state="ALERT_SENT", state_changed=True,
                          should_alert=bool(streams), evidence={"primary_action": "falling"},
                          stream_scores={EVENT_FALL: 0.93, EVENT_STROKE: 0.92}, alert_streams=streams)


def test_each_stream_gets_its_own_event_type_and_cooldown(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path, retention_days=7, save_snapshots=False)
    manager = AlertManager(logger=logger, dispatcher=_Silent(), clip_recorder=None)
    fall = manager.process(_assessment([EVENT_FALL]), timestamp=100.0)
    assert [e.event_type for e in fall] == [EVENT_FALL]
    assert manager.process(_assessment([EVENT_FALL]), timestamp=110.0) == []          # fall cooling down
    stroke = manager.process(_assessment([EVENT_STROKE]), timestamp=110.0)            # stroke is not
    assert [e.event_type for e in stroke] == [EVENT_STROKE]
    assert stroke[0].event_id != fall[0].event_id
    assert {e["event_type"] for e in logger.get_recent_events()} == {EVENT_FALL, EVENT_STROKE}


def test_gait_sway_ignores_smooth_paths_but_not_zig_zags() -> None:
    import numpy as np
    from src.temporal.stroke_signs import _oscillation, _reversals
    t = np.arange(0.0, 2.0, 1.0 / 15.0)
    curve = 250.0 * np.sin(1.2 * (t + 1.0)) / TORSO_PX            # turning around furniture
    zigzag = 0.45 * np.sin(2 * np.pi * 1.2 * t)                    # staggering
    assert _oscillation(t, curve) < 0.12 and _reversals(t, curve, 0.3) < 2
    assert _oscillation(t, zigzag) > 0.25 and _reversals(t, zigzag, 0.3) >= 2
