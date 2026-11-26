"""Stroke / acute-illness warning-sign stream, kept separate from the fall stream.

Visible signs only, never a diagnosis:

* staggering           - unsteady gait: the hips sway sideways and the torso tilt oscillates for seconds
                         (smooth paths such as walking across the room or along a curve are removed first)
* abnormal_gesture     - clutching the head / chest or vomiting, as recognised by the action model and held
                         for seconds (NTU RGB+D medical classes -> "abnormal_movement")
* asymmetric_movement  - while lying, only one side of the body moves (reported by the lying analyzer)
* unsteady_before_fall - the person was staggering just before going down
* collapse_from_sitting- a fast drop to lying from a seated position (not a trip while walking)
* convulsion           - rhythmic jerking of several body parts for seconds, in any posture (reported by
                         the lying analyzer, see convulsion.py)

One sign alone only raises the stroke risk to a warning level shown on the dashboard. Two different
signs within a couple of minutes, a collapse from sitting, or a convulsion reach the alarm level.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.temporal.convulsion import CONVULSION
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import KinematicFeatures

STAGGERING = "staggering"
ABNORMAL_GESTURE = "abnormal_gesture"
ASYMMETRIC_MOVEMENT = "asymmetric_movement"
UNSTEADY_BEFORE_FALL = "unsteady_before_fall"
COLLAPSE_FROM_SITTING = "collapse_from_sitting"

DEFAULT_SIGN_RISK: dict[str, float] = {
    STAGGERING: 0.80,
    ABNORMAL_GESTURE: 0.82,
    ASYMMETRIC_MOVEMENT: 0.80,
    UNSTEADY_BEFORE_FALL: 0.82,
    COLLAPSE_FROM_SITTING: 0.92,
    CONVULSION: 0.92,
}


@dataclass
class StrokeSignConfig:
    """Thresholds for the stroke warning-sign stream (configs/thresholds.yaml -> stroke_signs)."""
    enabled: bool = True
    gait_window_seconds: float = 2.0          # sway / tilt window (2 s: turning around furniture stays smooth)
    min_upright_seconds: float = 1.2          # need this much upright history to judge the gait
    upright_torso_angle: float = 30.0
    sway_min: float = 0.12                    # hip sideways oscillation (torso lengths) still normal
    sway_full: float = 0.35                   # ... clearly staggering
    tilt_min: float = 6.0                     # torso tilt oscillation (deg) still normal
    tilt_full: float = 15.0
    min_reversals: int = 2                    # sideways / tilt direction changes needed within the window
    sway_min_speed: float = 0.3               # torso lengths / s for a sideways direction change to count
    tilt_min_speed: float = 15.0              # deg / s for a tilt direction change to count
    stagger_min_score: float = 0.5            # gait instability needed to call it staggering...
    stagger_model_min_prob: float = 0.30      # ...when the action model also sees "stumbling"
    stagger_kinematic_alone: float = 0.8      # ...or this much instability without the model agreeing
    stagger_min_seconds: float = 3.0          # sustained staggering = a warning sign
    pre_fall_seconds: float = 5.0             # look back this far for unsteadiness before a fall...
    pre_fall_min_stagger_seconds: float = 1.0 # ...and require this much staggering in it (a slip is shorter)
    gesture_min_prob: float = 0.60            # mean "abnormal_movement" probability over the window
    gesture_window_seconds: float = 2.0
    gesture_min_seconds: float = 4.0
    sign_hold_seconds: float = 10.0           # a sign keeps raising the risk this long after last seen
    combo_window_seconds: float = 120.0       # different signs within this window add up
    combo_min_signs: int = 2
    combo_risk: float = 0.92
    sign_risk: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGN_RISK))
    history_seconds: float = 30.0

    @classmethod
    def from_dict(cls, cfg: Optional[dict[str, Any]]) -> "StrokeSignConfig":
        cfg = dict(cfg or {})
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__ and k != "sign_risk"}
        out = cls(**known)
        out.sign_risk = {**DEFAULT_SIGN_RISK, **{str(k): float(v) for k, v in (cfg.get("sign_risk") or {}).items()}}
        return out


@dataclass
class StrokeSigns:
    """Stroke-stream evaluation for one person at one moment."""
    risk: float                       # [0, 1] stroke-stream risk (fed to its own state machine)
    active_signs: list[str]           # signs seen within sign_hold_seconds
    recent_signs: list[str]           # distinct signs within combo_window_seconds
    instability: float                # current gait instability [0, 1]
    staggering_now: bool
    unsteady_recently: bool           # staggering at some point in the last pre_fall_seconds


@dataclass
class _GaitSample:
    t: float
    upright: bool
    hip_x: Optional[float]            # raw image px
    torso_px: Optional[float]
    tilt: float
    stumbling_prob: float
    gesture_prob: float
    staggering: bool = False


@dataclass
class _TrackSigns:
    samples: deque = field(default_factory=deque)
    stagger_since: Optional[float] = None
    gesture_since: Optional[float] = None
    first_seen: dict[str, float] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)


def _reversals(t: np.ndarray, v: np.ndarray, min_speed: float) -> int:
    """How often v changes direction (speed above min_speed on both sides): a zig-zag, not a turn."""
    if len(v) < 4:
        return 0
    speed = np.convolve(np.diff(v) / np.maximum(1e-3, np.diff(t)), np.ones(3) / 3.0, mode="valid")
    signs = [np.sign(s) for s in speed if abs(s) >= min_speed]
    return int(sum(1 for a, b in zip(signs, signs[1:]) if a != b))


def _oscillation(t: np.ndarray, v: np.ndarray) -> float:
    """Std of v around a quadratic trend: walking straight or along a curve is not sway."""
    if len(v) < 6 or float(np.ptp(t)) <= 0:
        return 0.0
    tc = t - t.mean()
    coef = np.polyfit(tc, v, 2)
    return float(np.std(v - np.polyval(coef, tc)))


class StrokeSignAnalyzer:
    """Per-camera warning-sign tracker; one instance lives in each RiskEngine."""

    def __init__(self, config: Optional[StrokeSignConfig] = None) -> None:
        self.cfg = config or StrokeSignConfig()
        self._tracks: dict[int, _TrackSigns] = {}

    # ------------------------------------------------------------------ gait
    def _gait_instability(self, samples: list[_GaitSample]) -> float:
        cfg = self.cfg
        up = [s for s in samples if s.upright and s.hip_x is not None and s.torso_px]
        if len(up) < 8 or up[-1].t - up[0].t < cfg.min_upright_seconds:
            return 0.0
        t = np.array([s.t for s in up])
        torso = float(np.median([s.torso_px for s in up]))
        x = np.array([s.hip_x for s in up]) / max(1.0, torso)
        tilt = np.array([s.tilt for s in up])
        # Staggering zig-zags: starting to walk, turning or bending changes direction once at most
        sway = _oscillation(t, x) if _reversals(t, x, cfg.sway_min_speed) >= cfg.min_reversals else 0.0
        wobble = _oscillation(t, tilt) if _reversals(t, tilt, cfg.tilt_min_speed) >= cfg.min_reversals else 0.0
        sway_score = (sway - cfg.sway_min) / max(1e-6, cfg.sway_full - cfg.sway_min)
        tilt_score = (wobble - cfg.tilt_min) / max(1e-6, cfg.tilt_full - cfg.tilt_min)
        return float(np.clip(max(sway_score, tilt_score), 0.0, 1.0))

    # ------------------------------------------------------------------ API
    def update(
        self,
        track_id: int,
        snapshot: TrackSnapshot,
        kinematics: KinematicFeatures,
        timestamp: float,
        action_probabilities: Optional[dict[str, float]] = None,
    ) -> StrokeSigns:
        """Record one pose sample, update gait / gesture signs and return the current evaluation."""
        cfg = self.cfg
        st = self._tracks.setdefault(track_id, _TrackSigns())
        probs = action_probabilities or {}

        kp = snapshot.keypoints
        hips = [kp[i, :2] for i in (11, 12) if kp[i, 2] >= 0.3]
        shoulders = [kp[i, :2] for i in (5, 6) if kp[i, 2] >= 0.3]
        hip_x = float(np.mean(hips, axis=0)[0]) if hips else None
        torso_px = (float(np.linalg.norm(np.mean(shoulders, axis=0) - np.mean(hips, axis=0)))
                    if hips and shoulders else None)
        upright = (getattr(kinematics, "lower_body_visible", True) and not getattr(kinematics, "feet_supported", False)
                   and kinematics.torso_angle_curr < cfg.upright_torso_angle)
        sample = _GaitSample(timestamp, upright, hip_x, torso_px if torso_px and torso_px >= 1.0 else None,
                             float(kinematics.torso_angle_curr), float(probs.get("stumbling", 0.0)),
                             float(probs.get("abnormal_movement", 0.0)))
        st.samples.append(sample)
        while st.samples and timestamp - st.samples[0].t > cfg.history_seconds:
            st.samples.popleft()

        # Staggering: unsteady gait, confirmed by the model or strong enough on its own, held for seconds
        window = [s for s in st.samples if timestamp - s.t <= cfg.gait_window_seconds]
        instability = self._gait_instability(window)
        stumble = float(np.mean([s.stumbling_prob for s in window])) if window else 0.0
        staggering_now = instability >= cfg.stagger_min_score and (
            stumble >= cfg.stagger_model_min_prob or instability >= cfg.stagger_kinematic_alone
        )
        sample.staggering = staggering_now
        if staggering_now:
            st.stagger_since = st.stagger_since if st.stagger_since is not None else timestamp
            if timestamp - st.stagger_since >= cfg.stagger_min_seconds:
                self.note(track_id, STAGGERING, timestamp)
        else:
            st.stagger_since = None

        # Clutching head / chest, vomiting: the model's "abnormal_movement", sustained, not lying down
        g_window = [s for s in st.samples if timestamp - s.t <= cfg.gesture_window_seconds]
        gesture = float(np.mean([s.gesture_prob for s in g_window])) if g_window else 0.0
        not_lying = kinematics.torso_angle_curr < 60.0 or getattr(kinematics, "feet_supported", False)
        if gesture >= cfg.gesture_min_prob and not_lying:
            st.gesture_since = st.gesture_since if st.gesture_since is not None else timestamp
            if timestamp - st.gesture_since >= cfg.gesture_min_seconds:
                self.note(track_id, ABNORMAL_GESTURE, timestamp)
        else:
            st.gesture_since = None

        return self.evaluate(track_id, timestamp, instability, staggering_now)

    def note(self, track_id: int, sign: str, timestamp: float) -> None:
        """Record a sign observed by another analyzer (e.g. one-sided movement while lying)."""
        st = self._tracks.setdefault(track_id, _TrackSigns())
        st.first_seen.setdefault(sign, timestamp)
        st.last_seen[sign] = timestamp

    def evaluate(self, track_id: int, timestamp: float, instability: float = 0.0,
                 staggering_now: bool = False) -> StrokeSigns:
        cfg = self.cfg
        st = self._tracks.setdefault(track_id, _TrackSigns())
        for sign in [s for s, t in st.last_seen.items() if timestamp - t > cfg.combo_window_seconds]:
            st.last_seen.pop(sign, None)
            st.first_seen.pop(sign, None)
        active = sorted(s for s, t in st.last_seen.items() if timestamp - t <= cfg.sign_hold_seconds)
        recent = sorted(st.last_seen)
        risk = max((cfg.sign_risk.get(s, 0.0) for s in active), default=0.0)
        if active and len(recent) >= cfg.combo_min_signs:
            risk = max(risk, cfg.combo_risk)
        pre_fall = [s for s in st.samples if timestamp - s.t <= cfg.pre_fall_seconds]
        stagger_time = sum(b.t - a.t for a, b in zip(pre_fall, pre_fall[1:]) if b.staggering)
        unsteady = stagger_time >= cfg.pre_fall_min_stagger_seconds
        return StrokeSigns(risk, active, recent, instability, staggering_now, unsteady)

    def cleanup_inactive(self, active_track_ids: set[int]) -> None:
        for tid in [t for t in self._tracks if t not in active_track_ids]:
            self._tracks.pop(tid, None)
