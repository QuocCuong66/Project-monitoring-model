"""Lying-is-safe scope: anyone not lying (standing, walking, sitting) is monitored; someone lying is shown as
safe unless they fell / collapsed or are convulsing."""

import math

from src.risk.risk_engine import SAFE_LABEL, display_action
from src.risk.threshold_manager import ThresholdManager
from tests.test_fall_vs_stroke_streams import StreamScene
from tests.test_lying_context import LYING, SEATED, UPRIGHT, _kp


class ScopeScene(StreamScene):
    """Scene with the shipped monitoring scope (lying is safe), recording the displayed label."""

    def __init__(self) -> None:
        super().__init__(safe_when_lying=True)
        self.shown: list[str] = []

    def step(self, *a, **k) -> None:
        super().step(*a, **k)
        self.shown.append(display_action(self.last.evidence))

    def sit(self, seconds: float) -> None:
        self.run(seconds, lambda _: self.step(_kp(SEATED, noise=0.01, rng=self.rng), 5.0, "sitting", immob=0.5))


def test_shipped_config_shows_lying_as_safe_and_monitors_sitting_collapses() -> None:
    th = ThresholdManager()
    assert th.safe_when_lying
    assert {"collapse_from_sitting", "convulsion", "fast_drop", "upright_to_floor"} <= set(th.fall_onset_reasons)


def test_person_already_lying_when_tracking_starts_is_safe() -> None:
    s = ScopeScene()
    s.lie_still(60.0)
    assert not s.fired
    assert s.shown[-1] == SAFE_LABEL
    assert max(s.risks) == 0.0


def test_lying_down_deliberately_is_safe() -> None:
    s = ScopeScene()
    s.walk(3.0)
    s.lie_down_calmly()
    s.lie_still(60.0)
    assert not s.fired
    assert s.shown[-1] == SAFE_LABEL
    assert max(s.risks[-900:]) == 0.0


def test_one_sided_movement_while_lying_is_safe_in_this_scope() -> None:
    s = ScopeScene()
    s.walk(3.0)
    s.lie_down_calmly()

    def left_side_only(e: float) -> None:
        a = 0.5 * math.sin(e * 3.0)
        off = {7: (a, a), 9: (a * 1.5, a), 13: (a, -a), 15: (a * 1.2, -a)}
        s.step(_kp(LYING, off, noise=0.01, rng=s.rng), 85.0, "lying", immob=0.3)
    s.run(20.0, left_side_only)
    assert not s.fired
    assert max(s.risks[-60:]) == 0.0


def test_sitting_person_is_monitored_and_collapse_from_sitting_alarms() -> None:
    s = ScopeScene()
    s.sit(3.0)
    assert SAFE_LABEL not in s.shown

    def slump(e: float) -> None:
        k = min(1.0, e / 0.5)
        s.step(_kp(SEATED if k < 0.5 else LYING, noise=0.01, rng=s.rng), 5.0 + 80.0 * k, "falling", vy=3.0,
               drop=0.9, immob=0.0)
    s.run(0.6, slump)
    s.lie_still(15.0)
    assert s.onset() == ("sudden", "collapse_from_sitting")
    assert "fall" in s.fired
    assert SAFE_LABEL not in s.shown


def test_fall_from_standing_still_alarms() -> None:
    s = ScopeScene()
    s.stand(3.0)
    s.collapse()
    s.lie_still(10.0)
    assert "fall" in s.fired
    assert SAFE_LABEL not in s.shown


def test_slow_trip_straight_from_standing_still_alarms() -> None:
    s = ScopeScene()
    s.stand(3.0)

    def slip(e: float) -> None:
        k = min(1.0, e / 1.2)
        s.step(_kp(UPRIGHT if k < 0.5 else LYING, noise=0.01, rng=s.rng), 5.0 + 80.0 * k, "lying", vy=0.9,
               immob=0.0)
    s.run(1.2, slip)
    s.lie_still(10.0)
    assert s.onset() == ("sudden", "upright_to_floor")
    assert "fall" in s.fired


def test_new_track_id_mid_fall_keeps_the_standing_history() -> None:
    s = ScopeScene()
    s.stand(3.0)
    s.engine.cleanup_inactive({1})
    s.step(_kp(UPRIGHT, noise=0.01, rng=s.rng), 40.0, "lying", vy=0.9, immob=0.0, tid=1)
    s.engine.cleanup_inactive({2})                 # tracker swapped the id while the person went down

    # Slow trip (no fast hip drop): only the standing history of the old id shows it was a fall
    s.run(0.5, lambda _: s.step(_kp(LYING, noise=0.01, rng=s.rng), 85.0, "lying", vy=0.9, immob=0.0, tid=2))
    s.run(10.0, lambda _: s.step(_kp(LYING, noise=0.01, rng=s.rng), 85.0, "immobile", tid=2))
    assert s.onset(tid=2) == ("sudden", "upright_to_floor")
    assert "fall" in s.fired
    assert SAFE_LABEL not in s.shown


def test_alert_state_winds_down_once_the_person_is_out_of_scope() -> None:
    s = ScopeScene()
    s.stand(3.0)
    s.collapse()
    s.lie_still(10.0)
    assert "fall" in s.fired
    s.scroll_phone(40.0)                           # awake and scrolling: rescued as a normal lie-down
    assert s.shown[-1] == SAFE_LABEL
    assert s.states[-1] == "NORMAL"
