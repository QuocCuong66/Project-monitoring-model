"""Lying normally (sleep, phone, turning over) must not alarm; collapse and warning signs must stand out."""

import math
from typing import Callable, Optional

import numpy as np

from src.action.action_classifier import ActionPrediction
from src.anomaly.anomaly_detector import AnomalyResult
from src.risk.risk_engine import RiskEngine
from src.risk.threshold_manager import ThresholdManager
from src.temporal.lying_context import (
    ASYMMETRIC, COLLAPSED, REPOSITIONING, RESTING, STRUGGLING, USING_PHONE,
)
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import KinematicFeatures

DT = 1.0 / 15.0
FRAME = (1280, 720)

# Normalized COCO-17 skeletons (hip centre = origin, torso length = 1)
UPRIGHT = {0: (0, -1.4), 1: (-0.1, -1.5), 2: (0.1, -1.5), 3: (-0.2, -1.45), 4: (0.2, -1.45),
           5: (-0.3, -1.0), 6: (0.3, -1.0), 7: (-0.4, -0.5), 8: (0.4, -0.5), 9: (-0.4, 0.0), 10: (0.4, 0.0),
           11: (-0.15, 0.0), 12: (0.15, 0.0), 13: (-0.15, 0.9), 14: (0.15, 0.9), 15: (-0.15, 1.8), 16: (0.15, 1.8)}
LYING = {j: (y, x) for j, (x, y) in UPRIGHT.items()}   # rotated 90 degrees: head to the left
SEATED = {**UPRIGHT, 13: (-0.2, 0.15), 14: (0.2, 0.15), 15: (-0.2, 1.0), 16: (0.2, 1.0)}


def _kp(base: dict[int, tuple[float, float]], offsets: Optional[dict[int, tuple[float, float]]] = None,
        noise: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    kp = np.zeros((17, 3), dtype=np.float32)
    for j, (x, y) in base.items():
        dx, dy = (offsets or {}).get(j, (0.0, 0.0))
        kp[j] = (x + dx, y + dy, 0.9)
    if noise and rng is not None:
        kp[:, :2] += rng.normal(0, noise, (17, 2))
    return kp


class Scene:
    """Feeds one person through a RiskEngine and records states / contexts."""

    def __init__(self, rest_zones: Optional[dict] = None, safe_when_lying: bool = False) -> None:
        th = ThresholdManager()
        # These scenarios cover the full lying analysis; the standing-falls-only scope has its own tests
        th.safe_when_lying = safe_when_lying
        if rest_zones is not None:
            th.lying_context = {**th.lying_context, "rest_zones": rest_zones}
        self.engine = RiskEngine(th)
        self.t = 0.0
        self.rng = np.random.default_rng(1)
        self.states: list[str] = []
        self.labels: list[str] = []
        self.risks: list[float] = []
        self.warnings: set[str] = set()

    def step(self, kp: np.ndarray, angle: float, action: str, tid: int = 1, vy: float = 0.0,
             drop: float = 0.0, immob: float = 1.0, center: tuple[float, float] = (640.0, 500.0)) -> None:
        raw = kp.copy()
        raw[:, :2] = kp[:, :2] * 120.0 + np.array(center)
        # bbox around the drawn skeleton (taller standing than seated, wide when lying), as a detector gives
        x1, y1 = raw[:, :2].min(axis=0) - 20.0
        x2, y2 = raw[:, :2].max(axis=0) + 20.0
        bbox = (float(x1), float(y1), float(x2), float(y2))
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        snap = TrackSnapshot(self.t, bbox, raw, kp, angle)
        kin = KinematicFeatures(vy, 0.0, w / h, 0.0, angle, 0.0, immob, drop, True)
        emergency = {"immobile": 0.98, "lying": 0.75, "falling": 0.95}.get(action, 0.0)
        severity = {"immobile": 0.98, "lying": 0.6, "falling": 0.95}.get(action, 0.0)
        pred = ActionPrediction(action, 0.8, {action: 0.8}, emergency, severity)
        ass = self.engine.assess(tid, pred, AnomalyResult(0.3, False, 0.1, 0.22), kin, self.t,
                                 snapshot=snap, frame_size=FRAME)
        self.states.append(ass.state)
        self.labels.append(ass.evidence.get("lying_context", "not_lying"))
        self.risks.append(ass.risk_score)
        self.warnings.update(ass.evidence.get("warning_signs", []))
        self.t += DT

    def run(self, seconds: float, fn: Callable[[float], None]) -> None:
        start = self.t
        while self.t - start < seconds:
            fn(self.t - start)

    def stand(self, seconds: float) -> None:
        self.run(seconds, lambda _: self.step(_kp(UPRIGHT, noise=0.01, rng=self.rng), 5.0, "standing", immob=0.3))

    def lie_down_calmly(self) -> None:          # sit on the bed edge, then lie back over ~1.5 s
        self.run(2.0, lambda _: self.step(_kp(SEATED, noise=0.01, rng=self.rng), 5.0, "sitting", immob=0.4))

        def f(e: float) -> None:
            k = min(1.0, e / 1.5)
            self.step(_kp(SEATED if k < 0.5 else LYING, noise=0.01, rng=self.rng), 5.0 + 80.0 * k,
                      "sitting" if k < 0.6 else "lying", vy=0.4, immob=0.2)
        self.run(1.5, f)

    def collapse(self) -> None:                 # upright -> floor in 0.4 s
        def f(e: float) -> None:
            k = min(1.0, e / 0.4)
            self.step(_kp(LYING, noise=0.01, rng=self.rng), 5.0 + 80.0 * k, "falling", vy=4.5, drop=0.9, immob=0.0)
        self.run(0.6, f)

    def lie_still(self, seconds: float, tid: int = 1) -> None:
        self.run(seconds, lambda _: self.step(_kp(LYING, noise=0.01, rng=self.rng), 85.0, "immobile", tid=tid))

    def alerted(self) -> bool:
        return "ALERT_SENT" in self.states


def test_sleeping_after_lying_down_calmly_does_not_alarm() -> None:
    s = Scene()
    s.stand(3.0)
    s.lie_down_calmly()
    s.lie_still(120.0)
    assert not s.alerted()
    assert s.labels[-1] == RESTING
    assert max(s.risks[-1500:]) < ThresholdManager().thresh_suspicious


def test_sudden_collapse_then_stillness_alarms() -> None:
    s = Scene()
    s.stand(3.0)
    s.collapse()
    s.lie_still(10.0)
    assert s.alerted()
    assert COLLAPSED in s.labels


def test_phone_use_while_lying_is_normal() -> None:
    s = Scene()
    s.stand(3.0)
    s.lie_down_calmly()
    phone = {10: (-0.95, -0.05)}                  # right wrist moved up to the face (LYING head at x=-1.4)

    def scroll(e: float) -> None:
        wiggle = 0.12 * math.sin(e * 6.0)
        off = {10: (phone[10][0] + wiggle, phone[10][1] + wiggle / 2), 8: (-0.4, 0.2)}
        s.step(_kp(LYING, off, noise=0.01, rng=s.rng), 85.0, "lying", immob=0.8)
    s.run(60.0, scroll)
    assert not s.alerted()
    assert s.labels[-1] == USING_PHONE


def test_turning_over_repeatedly_is_normal() -> None:
    s = Scene()
    s.stand(3.0)
    s.lie_down_calmly()
    flipped = {j: (x, -y) for j, (x, y) in LYING.items()}

    def roll(e: float) -> None:
        phase = e % 10.0
        if phase < 1.5:                           # 1.5 s turning over, every 10 s
            k = phase / 1.5
            base = {j: (x, LYING[j][1] * (1 - 2 * k)) for j, (x, _) in LYING.items()}
            s.step(_kp(base, noise=0.02, rng=s.rng), 85.0, "lying", immob=0.1)
        else:
            s.step(_kp(flipped if int(e // 10) % 2 == 0 else LYING, noise=0.01, rng=s.rng), 85.0, "immobile")
    s.run(90.0, roll)
    assert not s.alerted()
    assert REPOSITIONING in s.labels
    assert s.labels[-1] == RESTING


def test_repeated_failed_sit_up_attempts_after_collapse_alarm() -> None:
    s = Scene()
    s.stand(3.0)
    s.collapse()

    def attempts(e: float) -> None:
        phase = e % 5.0
        if 2.0 <= phase < 3.6:                    # half up for 1.6 s, then back down
            s.step(_kp(UPRIGHT, noise=0.02, rng=s.rng), 45.0, "getting_up", immob=0.2)
        else:
            s.step(_kp(LYING, noise=0.01, rng=s.rng), 85.0, "lying", immob=0.6)
    s.run(20.0, attempts)
    assert STRUGGLING in s.warnings
    assert s.alerted()


def test_one_sided_movement_is_flagged_but_alone_does_not_alarm() -> None:
    s = Scene()
    s.stand(3.0)
    s.lie_down_calmly()

    def left_side_only(e: float) -> None:
        a = 0.5 * math.sin(e * 3.0)
        off = {7: (a, a), 9: (a * 1.5, a), 13: (a, -a), 15: (a * 1.2, -a)}   # left arm + leg move
        s.step(_kp(LYING, off, noise=0.01, rng=s.rng), 85.0, "lying", immob=0.3)
    s.run(20.0, left_side_only)
    assert ASYMMETRIC in s.warnings
    assert max(s.risks[-60:]) >= ThresholdManager().thresh_abnormal   # shown as a warning on the dashboard
    assert not s.alerted()                                            # but no alarm on its own


def test_slow_fall_straight_from_standing_is_sudden() -> None:
    s = Scene()
    s.stand(3.0)

    def slip(e: float) -> None:                   # 1.2 s, slow hips: only "no sitting first" gives it away
        k = min(1.0, e / 1.2)
        s.step(_kp(UPRIGHT if k < 0.5 else LYING, noise=0.01, rng=s.rng), 5.0 + 80.0 * k, "lying", vy=0.9,
               immob=0.0)
    s.run(1.2, slip)
    s.lie_still(10.0)
    assert s.engine.lying_analyzer._tracks[1].episode.onset == "sudden"
    assert s.alerted()


def test_flopping_onto_a_configured_bed_then_scrolling_is_normal() -> None:
    bed = {"main": [[[0.3, 0.5], [0.8, 0.5], [0.8, 0.9], [0.3, 0.9]]]}
    s = Scene(rest_zones=bed)
    s.stand(3.0)
    s.collapse()                                  # drops straight onto the bed

    def scroll(e: float) -> None:
        wiggle = 0.12 * math.sin(e * 6.0)
        s.step(_kp(LYING, {10: (-0.95 + wiggle, -0.05)}, noise=0.01, rng=s.rng), 85.0, "lying", immob=0.8)
    s.run(30.0, scroll)
    assert s.labels[-1] == USING_PHONE
    assert max(s.risks[-150:]) < ThresholdManager().thresh_suspicious


def test_unknown_onset_on_a_configured_bed_is_rest() -> None:
    bed = {"main": [[[0.3, 0.5], [0.8, 0.5], [0.8, 0.9], [0.3, 0.9]]]}
    s = Scene(rest_zones=bed)
    s.lie_still(60.0)                             # system started while the person already lay in bed
    assert not s.alerted()
    assert s.labels[-1] == RESTING


def test_new_track_id_after_collapse_keeps_the_sudden_onset() -> None:
    s = Scene()
    s.stand(3.0)
    s.collapse()
    s.lie_still(1.0, tid=1)
    s.engine.cleanup_inactive(set())              # tracker lost the person ...
    for _ in range(50):
        s.engine.cleanup_inactive(set())
    s.lie_still(10.0, tid=2)                      # ... and re-identified them lying at the same spot
    assert s.alerted()
