"""Picking something up, turning over in bed and flopping down to scroll must not alarm."""

import math
from types import SimpleNamespace

import numpy as np

from src.action.action_classifier import ACTION_CLASSES, ActionClassifier
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import TemporalFeatureExtractor
from tests.test_lying_context import LYING, UPRIGHT, Scene, _kp

DT = 1.0 / 15.0


def _bend_snapshot(t: float, k: float) -> TrackSnapshot:
    """Standing (k=0) -> bent right over to the floor (k=1); feet stay planted under the hips."""
    kp = np.zeros((17, 3), dtype=np.float32)
    hip = np.array([600.0, 400.0 + 60.0 * k])                 # hips sink a little (knees bend)
    lean = math.radians(5.0 + 80.0 * k)
    shoulder = hip + 150.0 * np.array([math.sin(lean), -math.cos(lean)])
    for j, p in {5: shoulder + (0, -10), 6: shoulder + (0, 10), 0: shoulder + 60.0 * np.array([math.sin(lean), -math.cos(lean)]),
                 11: hip + (-15, 0), 12: hip + (15, 0), 13: (585.0, 520.0), 14: (615.0, 520.0),
                 15: (585.0, 640.0), 16: (615.0, 640.0), 9: shoulder + (40, 120 * k), 10: shoulder + (40, 120 * k)}.items():
        kp[j] = (p[0], p[1], 0.9)
    xs, ys = kp[[0, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16], 0], kp[[0, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16], 1]
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    vec = (shoulder - hip)
    angle = float(np.degrees(np.arctan2(abs(vec[0]), -vec[1])))
    return TrackSnapshot(t, bbox, kp, kp.copy(), angle)


def test_bending_to_pick_up_is_not_lying_or_a_drop() -> None:
    snaps = [_bend_snapshot(i * DT, 0.0) for i in range(15)]
    snaps += [_bend_snapshot((15 + i) * DT, min(1.0, (i + 1) / 6.0)) for i in range(15)]   # fast 0.4 s bend
    kin = TemporalFeatureExtractor(fps=15.0, min_hip_confidence=0.5).extract(snaps)
    assert kin.torso_angle_curr > 60.0 and kin.aspect_ratio_curr > 1.0     # would look like lying
    assert kin.feet_supported
    assert kin.drop_severity_score == 0.0


def test_standing_upright_is_not_flagged_as_supported_bending() -> None:
    snaps = [_bend_snapshot(i * DT, 0.0) for i in range(20)]
    kin = TemporalFeatureExtractor(fps=15.0, min_hip_confidence=0.5).extract(snaps)
    assert not kin.feet_supported


def test_classifier_reads_supported_lean_as_bending() -> None:
    clf = ActionClassifier(weights_path=None)
    kin = SimpleNamespace(drop_severity_score=0.0, torso_angle_curr=80.0, aspect_ratio_curr=1.3,
                          immobility_index=0.0, vertical_velocity=2.5, lower_body_visible=True, feet_supported=True)
    probs = np.full(len(ACTION_CLASSES), 0.01, dtype=np.float32)
    probs[ACTION_CLASSES.index("falling")] = 0.8
    fused = clf._fuse_kinematics_with_probs(probs / probs.sum(), kin)
    assert ACTION_CLASSES[int(np.argmax(fused))] == "bending"
    assert ACTION_CLASSES[int(np.argmax(clf._heuristic_predict(np.zeros((30, 17, 3)), kin)))] == "bending"


def test_turning_over_labelled_falling_without_a_bed_zone_does_not_alarm() -> None:
    s = Scene()
    s.stand(3.0)
    s.lie_down_calmly()
    flipped = {j: (x, -y) for j, (x, y) in LYING.items()}

    def roll(e: float) -> None:
        phase = e % 10.0
        if phase < 4.5:                           # 4.5 s of tossing and turning, labelled "falling"
            s.step(_kp(flipped if int(phase) % 2 else LYING, noise=0.05, rng=s.rng), 85.0, "falling",
                   vy=1.5, drop=0.8, immob=0.0)
        else:
            s.step(_kp(flipped, noise=0.01, rng=s.rng), 85.0, "immobile")
    start = len(s.risks)
    s.run(60.0, roll)
    assert not s.alerted()
    assert max(s.risks[start:]) < 0.76            # never even shown as ABNORMAL


def test_rolling_off_the_bed_is_not_excused_as_turning_over() -> None:
    s = Scene()
    s.stand(3.0)
    s.lie_down_calmly()
    s.lie_still(5.0)

    def fall_off(e: float) -> None:                # hips end up 1.5 torso lengths lower (bed -> floor)
        k = min(1.0, e / 0.8)
        s.step(_kp(LYING, noise=0.01, rng=s.rng), 85.0, "falling" if e < 2.0 else "immobile", vy=3.0 * (1 - k),
               drop=0.9 if e < 2.0 else 0.0, immob=0.0 if e < 2.0 else 1.0, center=(640.0, 500.0 + 180.0 * k))
    s.run(10.0, fall_off)
    ctx = s.engine.lying_analyzer
    assert ctx is not None
    assert max(s.risks[-140:]) >= 0.88


def test_flopping_down_without_bed_zone_then_scrolling_is_rescued() -> None:
    s = Scene()
    s.stand(3.0)
    s.collapse()                                  # flops onto a bed nobody configured

    def scroll(e: float) -> None:
        wiggle = 0.12 * math.sin(e * 6.0)
        s.step(_kp(LYING, {10: (-0.95 + wiggle, -0.05)}, noise=0.01, rng=s.rng), 85.0, "lying", immob=0.8)
    s.run(40.0, scroll)
    assert s.labels[-1] == "using_phone"
    assert max(s.risks[-150:]) < 0.60
