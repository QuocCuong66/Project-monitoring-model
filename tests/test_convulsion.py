"""Convulsion (rhythmic jerking) vs deliberate lying down and normal movement."""

import math
from typing import Callable, Optional

import numpy as np

from src.risk.risk_engine import EVENT_FALL, EVENT_STROKE, SAFE_LABEL
from src.temporal.convulsion import CONVULSION, ConvulsionConfig, ConvulsionTracker, analyze_window
from src.temporal.lying_context import AFTER_CONVULSION
from tests.test_lying_context import DT, LYING, SEATED, UPRIGHT, _kp
from tests.test_lying_safe_scope import ScopeScene

JERK_JOINTS = (0, 7, 9, 8, 10, 13, 15, 14, 16)   # head, both arms, both legs


def _jerks(e: float, hz: float = 3.0, amp: float = 0.15,
           joints: tuple[int, ...] = JERK_JOINTS) -> dict[int, tuple[float, float]]:
    """Clonic-like jerks: a sharp beat (fundamental + harmonic), each joint slightly out of phase."""
    out = {}
    for i, j in enumerate(joints):
        ph = 2 * math.pi * hz * e + 0.4 * i
        v = amp * (math.sin(ph) + 0.35 * math.sin(2 * ph))
        out[j] = (0.6 * v, v)
    return out


def _window(fn: Callable[[float], dict[int, tuple[float, float]]], base: dict, seconds: float = 4.0,
            noise: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, seconds, DT)
    kps = np.stack([_kp(base, fn(t), noise=noise, rng=rng) for t in times])
    return times, kps


# --- spectral rhythm: what counts as convulsive jerking ------------------------

def test_whole_body_rhythmic_jerking_is_detected() -> None:
    r = analyze_window(*_window(lambda t: _jerks(t), LYING), ConvulsionConfig())
    assert r.positive
    assert 2.5 <= r.frequency <= 3.5


def test_keypoint_jitter_is_not_jerking() -> None:
    for seed in range(5):
        assert not analyze_window(*_window(lambda t: {}, LYING, noise=0.05, seed=seed), ConvulsionConfig()).positive


def test_walking_arm_and_leg_swing_is_too_slow() -> None:
    assert not analyze_window(*_window(lambda t: _jerks(t, hz=1.0, amp=0.3), UPRIGHT), ConvulsionConfig()).positive


def test_clapping_moves_the_arms_only() -> None:
    clap = lambda t: _jerks(t, hz=3.0, amp=0.2, joints=(7, 9, 8, 10))
    assert not analyze_window(*_window(clap, SEATED), ConvulsionConfig()).positive


def test_phone_scrolling_moves_one_wrist_only() -> None:
    scroll = lambda t: _jerks(t, hz=2.5, amp=0.1, joints=(10,))
    assert not analyze_window(*_window(scroll, LYING), ConvulsionConfig()).positive


def test_too_low_sampling_rate_is_never_positive() -> None:
    times, kps = _window(lambda t: _jerks(t), LYING)
    assert not analyze_window(times[::3], kps[::3], ConvulsionConfig()).positive   # 5 Hz


# --- scenes through the risk engine ------------------------------------------------

def _convulse(s: ScopeScene, seconds: float, base: dict, angle: float, hz: float = 3.0) -> None:
    start = s.t
    s.run(seconds, lambda _: s.step(_kp(base, _jerks(s.t - start, hz=hz), noise=0.02, rng=s.rng), angle,
                                    "lying" if base is LYING else "sitting", immob=0.0))


def test_lying_down_deliberately_then_convulsing_is_not_safe() -> None:
    s = ScopeScene()
    s.walk(3.0)
    s.lie_down_calmly()
    s.lie_still(10.0)
    assert s.shown[-1] == SAFE_LABEL
    _convulse(s, 20.0, LYING, 85.0)
    assert s.shown[-1] == CONVULSION
    assert EVENT_STROKE in s.fired                       # acute-illness warning, one sign is enough
    assert s.last.evidence["convulsion"] == "ongoing"
    s.lie_still(20.0)                                    # jerking stopped, still lying
    assert s.shown[-1] == AFTER_CONVULSION               # never goes back to "safe" in this episode
    assert SAFE_LABEL not in s.shown[-300:]


def test_convulsing_then_slumping_down_is_not_a_deliberate_lie_down() -> None:
    s = ScopeScene()
    s.sit(3.0)
    _convulse(s, 3.0, SEATED, 5.0)

    def slump(e: float) -> None:                          # slow (2 s) slide down while still jerking
        k = min(1.0, e / 2.0)
        pose = {j: (SEATED[j][0] * (1 - k) + LYING[j][0] * k, SEATED[j][1] * (1 - k) + LYING[j][1] * k)
                for j in SEATED}
        s.step(_kp(pose, _jerks(e), noise=0.02, rng=s.rng), 5.0 + 80.0 * k, "lying", vy=0.4, immob=0.0)
    s.run(2.0, slump)
    _convulse(s, 10.0, LYING, 85.0)
    s.lie_still(15.0)
    assert s.onset() == ("sudden", "convulsion")
    assert EVENT_FALL in s.fired and EVENT_STROKE in s.fired
    assert SAFE_LABEL not in s.shown


def test_same_slow_slide_down_without_jerking_is_deliberate() -> None:
    s = ScopeScene()
    s.sit(6.0)

    def slide(e: float) -> None:
        k = min(1.0, e / 2.0)
        pose = {j: (SEATED[j][0] * (1 - k) + LYING[j][0] * k, SEATED[j][1] * (1 - k) + LYING[j][1] * k)
                for j in SEATED}
        s.step(_kp(pose, noise=0.01, rng=s.rng), 5.0 + 80.0 * k, "lying", vy=0.4, immob=0.2)
    s.run(2.0, slide)
    s.lie_still(30.0)
    assert s.onset()[0] == "controlled"
    assert not s.fired
    assert s.shown[-1] == SAFE_LABEL


def test_convulsing_while_seated_is_monitored() -> None:
    s = ScopeScene()
    s.sit(3.0)
    _convulse(s, 15.0, SEATED, 5.0)
    assert CONVULSION in s.shown
    assert EVENT_STROKE in s.fired


def test_turning_over_while_lying_is_not_a_convulsion() -> None:
    s = ScopeScene()
    s.walk(3.0)
    s.lie_down_calmly()
    flipped = {j: (x, -y) for j, (x, y) in LYING.items()}

    def roll(e: float) -> None:
        phase = e % 10.0
        if phase < 4.5:                                   # tossing and turning, noisy keypoints
            s.step(_kp(flipped if int(phase) % 2 else LYING, noise=0.05, rng=s.rng), 85.0, "falling",
                   vy=1.5, drop=0.8, immob=0.0)
        else:
            s.step(_kp(flipped, noise=0.01, rng=s.rng), 85.0, "immobile")
    s.run(60.0, roll)
    assert CONVULSION not in s.shown
    assert not s.fired
    assert s.shown[-1] == SAFE_LABEL


def test_heavy_keypoint_jitter_while_lying_is_not_a_convulsion() -> None:
    s = ScopeScene()
    s.walk(3.0)
    s.lie_down_calmly()
    s.run(60.0, lambda _: s.step(_kp(LYING, noise=0.10, rng=s.rng), 85.0, "lying", immob=0.5))
    assert CONVULSION not in s.shown
    assert not s.fired


def test_scattered_jitter_hits_never_add_up_to_a_convulsion() -> None:
    """Heavy jitter (0.12 torso lengths) gives ~14% rhythmic-looking windows; with 1 s gaps tolerated they
    chain into 5 s runs, but only ~60% of their windows are positive (real jerking: ~100%)."""
    cfg = ConvulsionConfig()
    n = int(cfg.window_seconds / DT)
    chained = 0
    for seed in (0, 2, 5):
        rng = np.random.default_rng(seed)
        times = np.arange(0.0, 60.0, DT)
        kps = np.stack([_kp(LYING, noise=0.12, rng=rng) for _ in times])
        tracker = ConvulsionTracker()
        for i in range(8, len(times)):
            tracker.update(times[i], analyze_window(times[max(0, i - n + 1):i + 1], kps[max(0, i - n + 1):i + 1], cfg), cfg)
            assert not tracker.convulsing(times[i], cfg)
            chained += tracker.run_seconds() >= cfg.min_seconds
    assert chained                                        # the gap-tolerant runs did form: the share rule decided
