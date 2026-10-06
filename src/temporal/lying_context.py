"""Lying-down context: tell normal lying (sleeping, phone use, turning over) from visible warning signs.

A person who is horizontal and still is not an emergency by itself: they may be asleep. What matters
is how they got down and what they do while down:

* onset     - controlled (sat / knelt first, or stopped, stood still and lowered themselves slowly)
              vs sudden (rhythmic jerking / convulsing just before, staggering just before, a fast hip
              drop, or upright to lying within seconds without any of the deliberate cues).
              onset_reason says which rule decided.
* activity  - phone use (hand held near the head, moving), turning over (short whole-body bursts that
              settle again), resting / sleeping (little motion after a controlled lie-down)
* warnings  - repeated failed attempts to sit up, movement confined to one side of the body for many
              seconds, staying still on the floor (outside a configured bed / sofa zone) for a long time,
              convulsing (rhythmic jerking of several body parts, see convulsion.py) at any time

This is a decision-support heuristic on 2D skeletons. It never diagnoses: a stroke during sleep with no
visible movement looks exactly like sleep and cannot be told apart from video.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.temporal.convulsion import CONVULSION, ConvulsionConfig, ConvulsionTracker, analyze_window
from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import KinematicFeatures

NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16
L_WRIST, R_WRIST = 9, 10
LEFT_LIMBS = (7, 9, 13, 15)    # elbow, wrist, knee, ankle
RIGHT_LIMBS = (8, 10, 14, 16)

# Labels (ASCII: also drawn on the video by OpenCV)
NOT_LYING = "not_lying"
LYING = "lying"
RESTING = "resting"
USING_PHONE = "using_phone"
REPOSITIONING = "repositioning"
STRUGGLING = "struggling"
ASYMMETRIC = "asymmetric_movement"
STILL_ON_FLOOR = "still_on_floor"
COLLAPSED = "collapsed"
CONVULSING = CONVULSION
AFTER_CONVULSION = "after_convulsion"


@dataclass
class LyingContextConfig:
    """Thresholds for the lying context analyzer (configs/thresholds.yaml -> lying_context)."""
    enabled: bool = True
    horizontal_torso_angle: float = 60.0      # deg: torso this far from vertical = lying
    horizontal_aspect_ratio: float = 1.05     # bbox width / height: wider = lying
    upright_torso_angle: float = 30.0         # deg: below = standing / sitting upright...
    upright_aspect_ratio: float = 0.8         # ...with a narrow bbox (a sideways fall widens it first)
    standing_to_lying_seconds: float = 2.0    # standing (legs straight) -> lying this fast, no sitting = sudden
    seated_min_seconds: float = 0.8           # sat / knelt at least this long before lying = deliberate
    standing_thigh_length: float = 0.6        # knee below hip by >= this (torso lengths) = legs straight
    standing_shin_length: float = 0.5         # ankle below knee by >= this = not kneeling
    knee_above_hip_tolerance: float = 0.15    # knees higher than this above the hips: camera from above
    sudden_onset_seconds: float = 0.6         # legs not visible: torso 30 -> 60 deg faster than this = sudden
    sudden_onset_velocity: float = 1.2        # peak hip drop (body heights / s) = sudden
    activity_window_seconds: float = 10.0
    motion_noise_floor: float = 0.02          # joint std (torso lengths) treated as keypoint jitter
    phone_hand_to_head: float = 0.9           # wrist-to-nose distance / torso length
    phone_min_fraction: float = 0.6           # share of the window with a hand at the head
    phone_min_wrist_motion: float = 0.04      # hand must move (scrolling / typing), not rest
    reposition_motion: float = 0.25           # whole-body motion burst = turning over
    reposition_max_burst_seconds: float = 5.0 # longer continuous vigorous motion is not turning over
    struggle_window_seconds: float = 30.0
    struggle_min_attempts: int = 2            # lying -> half up -> lying again without standing
    struggle_min_raise_seconds: float = 1.0   # a half-up phase shorter than this is a pose flicker
    asymmetry_min_motion: float = 0.15        # moving side must really move
    asymmetry_max_ratio: float = 0.25         # still side motion / moving side motion
    asymmetry_min_seconds: float = 8.0
    floor_still_seconds: float = 600.0        # still outside rest zones this long (zones configured)
    inherit_seconds: float = 30.0             # re-identified track inherits a nearby recent episode
    inherit_distance: float = 0.25            # ... within this share of the frame
    benign_action_cap: float = 0.30           # max action evidence while lying normally
    warning_risk: float = 0.80                # warning sign alone: shown as ABNORMAL, no alarm
    escalated_warning_risk: float = 0.92      # warning sign + sudden onset / floor: can alarm
    history_seconds: float = 60.0
    roll_off_drop: float = 1.0                # hips this much lower (torso lengths) than when lying began
    sudden_onset_rescue_seconds: float = 15.0 # sustained phone use / turning over after a drop = awake
    pause_min_seconds: float = 1.0            # stood still at least this long before going down...
    pause_max_motion: float = 0.06            # ...with whole-body motion (torso lengths) below this...
    pause_max_tilt_change: float = 5.0        # ...and the torso angle steady (deg)...
    pause_max_translation: float = 0.10       # ...and not moving across the frame (walking is no pause)...
    deliberate_min_descent_seconds: float = 1.0  # ...then took at least this long to get down = deliberate
    pause_to_descent_max_seconds: float = 0.7 # lowering must start within this of the pause ending
    rest_seated_seconds: float = 2.0          # sat this long: unsteadiness before it no longer counts
    seated_max_translation: float = 0.15      # a seated person does not move across the frame (torso lengths)
    seated_max_height_ratio: float = 0.85     # ... and is clearly shorter than when upright before
    rest_zones: dict[str, list[list[list[float]]]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[dict[str, Any]]) -> "LyingContextConfig":
        cfg = dict(cfg or {})
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        out = cls(**known)
        out.rest_zones = {str(k): [list(map(list, poly)) for poly in (v or [])]
                          for k, v in (cfg.get("rest_zones") or {}).items()}
        return out


@dataclass
class LyingContext:
    """What the person is doing while lying, plus how it should change the risk."""
    label: str
    onset: str                      # "controlled" | "sudden" | "unknown"
    in_rest_zone: bool
    lying_seconds: float
    benign: bool                    # normal lying: immobility must not raise the risk
    concern: float                  # [0, 1] risk floor from visible warning signs
    warnings: list[str] = field(default_factory=list)
    level_drop: bool = False        # body ended up clearly lower than where it lay down (off the bed)
    onset_reason: str = ""          # which rule decided the onset (see LyingContextAnalyzer._decide_onset)
    stroke_signs: list[str] = field(default_factory=list)  # signs for the separate stroke stream
    convulsing: bool = False        # rhythmic jerking going on right now
    convulsed: bool = False         # this lying episode began with or included a convulsion


@dataclass
class _Sample:
    t: float
    horizontal: bool
    upright: bool
    raised: bool                    # half up (sitting-up attempt) while not upright
    vy: float
    angle: float                    # torso angle (deg)
    legs: str                       # "standing" | "seated" | "unknown" (upright torso only)
    kp: np.ndarray                  # (17, 3) normalized keypoints
    center: tuple[float, float]     # normalized frame position
    phone_hand: bool
    hip_y: Optional[float] = None   # raw image hip height (px)
    hip_x: Optional[float] = None
    bbox_h: float = 0.0
    torso_px: Optional[float] = None


@dataclass
class _Episode:
    start: float
    onset: str
    benign: bool = False
    last_motion: float = 0.0
    hip_baseline: Optional[float] = None    # image hip height (px) while settling into the lying posture
    torso_px: Optional[float] = None
    voluntary_since: Optional[float] = None # start of the current run of phone use / turning over
    reason: str = ""
    convulsed: bool = False


@dataclass
class _TrackState:
    samples: deque = field(default_factory=deque)
    episode: Optional[_Episode] = None
    last_upright: Optional[float] = None
    peak_vy_since_upright: float = 0.0
    last_standing: Optional[float] = None
    seated_start: Optional[float] = None
    seated_since_standing: bool = False
    seated_seconds: float = 0.0             # longest sitting / kneeling run since last standing
    convulsion: ConvulsionTracker = field(default_factory=ConvulsionTracker)


def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1:
            inside = not inside
    return inside


class LyingContextAnalyzer:
    """Per-camera analyzer keeping a long (~1 min) per-track history of compact pose samples."""

    def __init__(self, config: Optional[LyingContextConfig] = None, camera_id: str = "main",
                 convulsion: Optional[ConvulsionConfig] = None) -> None:
        self.cfg = config or LyingContextConfig()
        self.convulsion_cfg = convulsion or ConvulsionConfig()
        self.camera_id = camera_id
        self._tracks: dict[int, _TrackState] = {}
        # Ended / interrupted episodes: (time, center, onset, benign). A pose change often makes the
        # tracker issue a new id, which must not erase that the person fell (or lay down calmly).
        self._recent_episodes: list[tuple[float, tuple[float, float], str, bool]] = []
        # Track ids the tracker currently reports (None until set_active is called): a track missing
        # from it is lost, and its history can be handed to a new id appearing at the same spot
        self._active: Optional[set[int]] = None

    # ------------------------------------------------------------------ helpers
    def set_active(self, active_track_ids: set[int]) -> None:
        self._active = set(active_track_ids)

    def _handoff(self, center: tuple[float, float], now: float) -> Optional[_TrackState]:
        """History of a just-lost track last seen at this spot (the tracker re-issued the id mid-fall)."""
        if self._active is None:
            return None
        best: Optional[tuple[float, int]] = None
        for tid, st in self._tracks.items():
            if tid in self._active or not st.samples or now - st.samples[-1].t > self.cfg.inherit_seconds:
                continue
            c = st.samples[-1].center
            d = float(np.hypot(c[0] - center[0], c[1] - center[1]))
            if d <= self.cfg.inherit_distance and (best is None or d < best[0]):
                best = (d, tid)
        return self._tracks.pop(best[1]) if best else None

    @property
    def rest_zones(self) -> list[list[list[float]]]:
        return self.cfg.rest_zones.get(self.camera_id, [])

    def _in_rest_zone(self, center: tuple[float, float]) -> bool:
        return any(_point_in_polygon(center[0], center[1], poly) for poly in self.rest_zones if len(poly) >= 3)

    def _phone_hand(self, kp: np.ndarray) -> bool:
        if kp[NOSE, 2] < 0.3:
            return False
        sh = [kp[i, :2] for i in (L_SH, R_SH) if kp[i, 2] >= 0.3]
        hp = [kp[i, :2] for i in (L_HIP, R_HIP) if kp[i, 2] >= 0.3]
        if not sh or not hp:
            return False
        torso = float(np.linalg.norm(np.mean(sh, axis=0) - np.mean(hp, axis=0)))
        if torso < 1e-3:
            return False
        for w in (L_WRIST, R_WRIST):
            if kp[w, 2] >= 0.3 and np.linalg.norm(kp[w, :2] - kp[NOSE, :2]) <= self.cfg.phone_hand_to_head * torso:
                return True
        return False

    def _legs(self, kp: np.ndarray) -> str:
        """Standing (thigh and shin vertical) vs seated / kneeling, from the normalized skeleton."""
        def mid_y(a: int, b: int) -> Optional[float]:
            ys = [kp[i, 1] for i in (a, b) if kp[i, 2] >= 0.3]
            return float(np.mean(ys)) if ys else None
        hip, knee, ankle = mid_y(L_HIP, R_HIP), mid_y(L_KNEE, R_KNEE), mid_y(L_ANKLE, R_ANKLE)
        if hip is None or knee is None:
            return "unknown"
        if knee < hip - self.cfg.knee_above_hip_tolerance:
            return "unknown"           # knees above the hips: overhead / fisheye view, legs foreshortened
        if knee - hip < self.cfg.standing_thigh_length:
            return "seated"
        if ankle is not None and ankle - knee < self.cfg.standing_shin_length:
            return "seated"                                            # kneeling
        return "standing"

    @staticmethod
    def _hip_and_torso(kp: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Raw image hip (y, x) and torso length in px."""
        hips = [kp[i, :2] for i in (L_HIP, R_HIP) if kp[i, 2] >= 0.3]
        shoulders = [kp[i, :2] for i in (L_SH, R_SH) if kp[i, 2] >= 0.3]
        if not hips:
            return None, None, None
        hip = np.mean(hips, axis=0)
        torso = float(np.linalg.norm(np.mean(shoulders, axis=0) - hip)) if shoulders else None
        return float(hip[1]), float(hip[0]), (torso if torso and torso >= 1.0 else None)

    @staticmethod
    def _translation(samples: list[_Sample]) -> Optional[float]:
        """How much the hips moved across the frame (std, torso lengths); None when unknown."""
        pts = [(s.hip_x, s.hip_y, s.torso_px) for s in samples if s.hip_x is not None and s.torso_px]
        if len(pts) < 3:
            return None
        torso = float(np.median([p[2] for p in pts]))
        xy = np.array([(p[0], p[1]) for p in pts]) / max(1.0, torso)
        return float(np.mean(np.std(xy, axis=0)))

    @staticmethod
    def _joint_motion(samples: list[_Sample], joints: tuple[int, ...]) -> Optional[float]:
        pts = np.array([s.kp[list(joints)] for s in samples])        # (N, J, 3)
        conf_ok = np.mean(pts[:, :, 2] >= 0.3, axis=0) >= 0.7          # joints visible most of the time
        if not np.any(conf_ok):
            return None
        return float(np.mean(np.std(pts[:, conf_ok, :2], axis=0)))

    def _count_sit_up_attempts(self, samples: list[_Sample]) -> int:
        """Half-up phases (>= struggle_min_raise_seconds) that start and end lying down."""
        runs: list[tuple[str, float, float]] = []                     # (posture, start, end)
        for s in samples:
            p = "h" if s.horizontal else ("u" if s.upright else "r")
            if runs and runs[-1][0] == p:
                runs[-1] = (p, runs[-1][1], s.t)
            else:
                runs.append((p, s.t, s.t))
        return sum(
            1 for i in range(1, len(runs) - 1)
            if runs[i][0] == "r" and runs[i - 1][0] == "h" and runs[i + 1][0] == "h"
            and runs[i][2] - runs[i][1] >= self.cfg.struggle_min_raise_seconds
        )

    def _plausibly_seated(self, st: _TrackState, sample: _Sample) -> bool:
        """Leg geometry says seated; is it believable? From a high camera standing legs look short.

        A seated person stays in place, and is clearly shorter than when they were seen upright with
        legs that did not look seated (standing / unreadable) earlier in the history.
        """
        cfg = self.cfg
        last_second = [s for s in st.samples if sample.t - s.t <= 1.0]
        moved = self._translation(last_second)
        if moved is not None and moved > cfg.seated_max_translation:
            return False                                           # walking, not sitting
        upright_heights = [s.bbox_h for s in st.samples if s.upright and s.legs != "seated" and s.bbox_h > 0]
        if len(upright_heights) >= 5 and sample.bbox_h > cfg.seated_max_height_ratio * max(upright_heights):
            return False                                           # as tall as when standing
        return True

    def _pause_end_before(self, st: _TrackState, now: float) -> Optional[float]:
        """End time of the last still, upright pause (>= pause_min_seconds) in the recent history."""
        cfg = self.cfg
        recent = [s for s in st.samples if now - s.t <= cfg.history_seconds]
        pause_end: Optional[float] = None
        run: list[_Sample] = []
        for s in recent:
            if not s.upright:
                run = []
                continue
            run.append(s)
            while len(run) > 1 and run[-1].t - run[1].t >= cfg.pause_min_seconds:
                run = run[1:]                                   # keep a ~pause_min_seconds window
            if run[-1].t - run[0].t >= cfg.pause_min_seconds and len(run) >= 5:
                motion = self._joint_motion(run, tuple(range(17)))
                tilt_change = max(r.angle for r in run) - min(r.angle for r in run)
                moved = self._translation(run)
                if (motion is not None and motion <= cfg.pause_max_motion and tilt_change <= cfg.pause_max_tilt_change
                        and (moved is None or moved <= cfg.pause_max_translation)):
                    pause_end = s.t
        return pause_end

    def _decide_onset(self, st: _TrackState, now: float, unsteady_recently: bool) -> tuple[str, str]:
        """How did the person get down? Returns (onset, reason).

        The action label is not used: lying back onto a bed is often labelled "falling".
        """
        cfg = self.cfg
        if self.convulsion_cfg.enabled and st.convulsion.jerked_recently(now, self.convulsion_cfg):
            return "sudden", "convulsion"                  # jerking rhythmically, then went down
        fast_drop = st.peak_vy_since_upright >= cfg.sudden_onset_velocity
        if st.seated_since_standing and fast_drop:
            return "sudden", "collapse_from_sitting"
        if unsteady_recently and st.seated_seconds < cfg.rest_seated_seconds:
            return "sudden", "unsteady_before"            # staggered, then went down
        if st.seated_since_standing:
            return "controlled", "sat_down_first"
        if fast_drop:
            return "sudden", "fast_drop"
        if st.last_upright is not None:
            descent = now - st.last_upright
            pause_end = self._pause_end_before(st, now)
            # Stopped, stood still, then took a while to get down: deliberate (fainting while standing
            # still is a fast hip drop and was caught above)
            # Walking on after the pause (then tripping) does not count: the lowering must start right
            # after the pause
            if (pause_end is not None
                    and st.last_upright - pause_end <= cfg.pause_to_descent_max_seconds
                    and now - pause_end >= cfg.deliberate_min_descent_seconds):
                return "controlled", "paused_then_lowered"
            if descent <= cfg.standing_to_lying_seconds:
                # Upright -> lying within seconds and no deliberate cue: a possible fall (leg posture is
                # often unreadable, e.g. from an overhead camera)
                return "sudden", "upright_to_floor"
            return "controlled", "slow_transition"
        return "unknown", "unknown"

    def _window(self, st: _TrackState, now: float, seconds: float) -> list[_Sample]:
        return [s for s in st.samples if now - s.t <= seconds]

    def _inherit(self, now: float, center: tuple[float, float]) -> Optional[tuple[str, bool]]:
        best = None
        for t, c, onset, benign in self._recent_episodes:
            d = float(np.hypot(c[0] - center[0], c[1] - center[1]))
            if now - t <= self.cfg.inherit_seconds and d <= self.cfg.inherit_distance:
                if best is None or d < best[0]:
                    best = (d, onset, benign)
        return (best[1], best[2]) if best else None

    def _remember(self, st: _TrackState, now: float) -> None:
        if st.episode is not None and st.samples:
            self._recent_episodes.append((now, st.samples[-1].center, st.episode.onset, st.episode.benign))
        self._recent_episodes = [e for e in self._recent_episodes if now - e[0] <= self.cfg.inherit_seconds]

    # ------------------------------------------------------------------ main API
    def update(
        self,
        track_id: int,
        snapshot: TrackSnapshot,
        kinematics: KinematicFeatures,
        timestamp: float,
        frame_size: Optional[tuple[int, int]] = None,
        unsteady_recently: bool = False,
    ) -> LyingContext:
        """Update one person's lying context.

        unsteady_recently: the stroke-sign stream saw staggering in the last few seconds; going down
        right after that is a fall (or collapse), never a deliberate lie-down.
        """
        cfg = self.cfg
        x1, y1, x2, y2 = snapshot.bbox
        w, h = frame_size if frame_size else (max(1.0, x2), max(1.0, y2))
        center = (float((x1 + x2) / 2.0 / w), float((y1 + y2) / 2.0 / h))
        st = self._tracks.get(track_id)
        if st is None:
            st = self._handoff(center, timestamp) or _TrackState()
            self._tracks[track_id] = st

        visible = getattr(kinematics, "lower_body_visible", True)
        angle = kinematics.torso_angle_curr
        horizontal = visible and not getattr(kinematics, "feet_supported", False) and (angle >= cfg.horizontal_torso_angle or kinematics.aspect_ratio_curr >= cfg.horizontal_aspect_ratio)
        upright = (visible and not horizontal and angle < cfg.upright_torso_angle
                   and kinematics.aspect_ratio_curr < cfg.upright_aspect_ratio)
        raised = visible and not horizontal and not upright

        legs = (self._legs(snapshot.normalized_keypoints)
                if visible and not horizontal and angle < cfg.upright_torso_angle else "unknown")
        hip_y, hip_x, torso_px = self._hip_and_torso(snapshot.keypoints)
        sample = _Sample(timestamp, horizontal, upright, raised, kinematics.vertical_velocity, angle, legs,
                         np.asarray(snapshot.normalized_keypoints, dtype=np.float32).copy(), center,
                         self._phone_hand(snapshot.normalized_keypoints), hip_y=hip_y, hip_x=hip_x,
                         torso_px=torso_px, bbox_h=float(y2 - y1))
        st.samples.append(sample)
        while st.samples and timestamp - st.samples[0].t > cfg.history_seconds:
            st.samples.popleft()

        # Convulsion: rhythmic jerking of several body parts, in any posture
        convulsing = False
        ccfg = self.convulsion_cfg
        if ccfg.enabled:
            win = self._window(st, timestamp, ccfg.window_seconds)
            if len(win) >= 8:
                rhythm = analyze_window(np.array([s.t for s in win]), np.stack([s.kp for s in win]), ccfg)
                st.convulsion.update(timestamp, rhythm, ccfg)
            convulsing = st.convulsion.convulsing(timestamp, ccfg)

        # Onset bookkeeping: last time standing, whether the person sat / knelt since, last upright
        # time and the fastest hip drop on the way down
        if legs == "standing":
            st.last_standing = timestamp
            st.seated_start = None
            st.seated_since_standing = False
            st.seated_seconds = 0.0
        elif legs == "seated" and self._plausibly_seated(st, sample):
            st.seated_start = timestamp if st.seated_start is None else st.seated_start
            st.seated_seconds = max(st.seated_seconds, timestamp - st.seated_start)
            if st.seated_seconds >= cfg.seated_min_seconds:
                st.seated_since_standing = True
        else:
            st.seated_start = None
        if upright:
            if st.episode is not None:
                self._remember(st, timestamp)
            st.episode = None
            st.last_upright = timestamp
            st.peak_vy_since_upright = 0.0
        else:
            st.peak_vy_since_upright = max(st.peak_vy_since_upright, kinematics.vertical_velocity)

        if not horizontal:
            if st.episode is None:
                return LyingContext(NOT_LYING, "unknown", self._in_rest_zone(center), 0.0, False, 0.0,
                                    stroke_signs=[CONVULSION] if convulsing else [], convulsing=convulsing)
            # Half up during an episode (sitting-up attempt): keep the episode, evaluate below
        elif st.episode is None:
            onset, reason = self._decide_onset(st, timestamp, unsteady_recently)
            inherited_benign = False
            if onset == "unknown":
                inherited = self._inherit(timestamp, center)
                if inherited:
                    onset, inherited_benign = inherited
                    reason = "inherited"
            st.episode = _Episode(start=timestamp, onset=onset, benign=inherited_benign, last_motion=timestamp,
                                  reason=reason, convulsed=reason == "convulsion")

        ep = st.episode
        assert ep is not None
        ep.convulsed = ep.convulsed or convulsing
        in_zone = self._in_rest_zone(center)
        lying_s = timestamp - ep.start

        # Where the body lay down (first ~2 s of the episode): rolling off a bed shows as hips ending lower
        if horizontal and sample.hip_y is not None and sample.torso_px is not None:
            if ep.hip_baseline is None or lying_s <= 2.0:
                ep.hip_baseline = sample.hip_y if ep.hip_baseline is None else min(ep.hip_baseline, sample.hip_y)
                ep.torso_px = sample.torso_px if ep.torso_px is None else max(ep.torso_px, sample.torso_px)
        level_drop = bool(
            ep.hip_baseline is not None and ep.torso_px and sample.hip_y is not None
            and (sample.hip_y - ep.hip_baseline) / ep.torso_px >= cfg.roll_off_drop
        )

        # ---- activity over the recent window
        win = self._window(st, timestamp, cfg.activity_window_seconds)
        left = self._joint_motion(win, LEFT_LIMBS) if len(win) >= 5 else None
        right = self._joint_motion(win, RIGHT_LIMBS) if len(win) >= 5 else None
        # Phone use is often one-handed: the busier wrist counts
        wrist_motions = [self._joint_motion(win, (w,)) for w in (L_WRIST, R_WRIST)] if len(win) >= 5 else []
        wrists = max((m for m in wrist_motions if m is not None), default=None)
        phone_frac = float(np.mean([s.phone_hand for s in win])) if win else 0.0

        using_phone = (phone_frac >= cfg.phone_min_fraction and wrists is not None
                       and wrists >= cfg.phone_min_wrist_motion)

        # Turning over: whole-body motion in short bursts (seconds) that settle again
        burst = self._window(st, timestamp, 2.0)
        burst_motion = self._joint_motion(burst, tuple(range(17))) if len(burst) >= 5 else None
        moving_now = burst_motion is not None and burst_motion >= cfg.reposition_motion
        long_vigorous = False
        if moving_now:
            long_win = self._window(st, timestamp, cfg.reposition_max_burst_seconds)
            chunks = [long_win[i:i + 8] for i in range(0, len(long_win), 8)]
            chunk_motion = [self._joint_motion(c, tuple(range(17))) for c in chunks if len(c) >= 5]
            long_vigorous = (len(chunk_motion) >= 3 and all(m is not None and m >= cfg.reposition_motion for m in chunk_motion)
                             and timestamp - ep.start >= cfg.reposition_max_burst_seconds)
        repositioning = moving_now and not long_vigorous

        if using_phone or repositioning or (burst_motion is not None and burst_motion > cfg.motion_noise_floor * 3):
            ep.last_motion = timestamp

        # ---- warning signs
        warnings: list[str] = []
        struggle_win = [s for s in self._window(st, timestamp, cfg.struggle_window_seconds) if s.t >= ep.start]
        if (self._count_sit_up_attempts(struggle_win) >= cfg.struggle_min_attempts
                and not any(s.upright for s in struggle_win)):
            warnings.append(STRUGGLING)

        asym_win = self._window(st, timestamp, cfg.asymmetry_min_seconds)
        if (left is not None and right is not None and not using_phone and len(asym_win) >= 5
                and asym_win[-1].t - asym_win[0].t >= cfg.asymmetry_min_seconds * 0.9):
            l_m = self._joint_motion(asym_win, LEFT_LIMBS)
            r_m = self._joint_motion(asym_win, RIGHT_LIMBS)
            if l_m is not None and r_m is not None:
                hi, lo = max(l_m, r_m), min(l_m, r_m)
                if hi >= cfg.asymmetry_min_motion and lo <= cfg.asymmetry_max_ratio * hi:
                    warnings.append(ASYMMETRIC)

        zones_configured = bool(self.rest_zones)
        still_s = timestamp - ep.last_motion
        if zones_configured and not in_zone and horizontal and still_s >= cfg.floor_still_seconds:
            warnings.append(STILL_ON_FLOOR)

        # ---- normal lying?
        # Lying down calmly, or moving deliberately after an unseen onset, is normal. A sudden drop
        # outside a configured bed / sofa stays a possible fall even if the person moves afterwards.
        voluntary = using_phone or repositioning
        ep.voluntary_since = (ep.voluntary_since or timestamp) if voluntary else None
        if ep.onset == "controlled" or (ep.onset == "unknown" and voluntary):
            ep.benign = True
        # Flopping down quickly (e.g. onto a bed nobody configured) and then scrolling or turning over
        # for a long while: an awake, responsive person, not someone lying still after a fall.
        if (ep.onset == "sudden" and ep.voluntary_since is not None
                and timestamp - ep.voluntary_since >= cfg.sudden_onset_rescue_seconds):
            ep.onset = "controlled"
            ep.reason = "awake_after_drop"
            ep.benign = True
        # A configured bed / sofa is a normal place to lie, however the person got there (flopping onto
        # the bed looks like a sudden drop). Warning signs still apply there.
        benign = (ep.benign and ep.onset != "sudden") or in_zone
        benign = benign and not warnings and not long_vigorous and not ep.convulsed

        # ---- risk floor from warnings
        # Fall stream: cannot get up / lying still on the floor. One-sided movement is a stroke sign and
        # is reported to the stroke stream instead.
        fall_warnings = [w for w in warnings if w in (STRUGGLING, STILL_ON_FLOOR)]
        concern = 0.0
        if fall_warnings:
            escalated = ep.onset == "sudden" or STILL_ON_FLOOR in warnings or (zones_configured and not in_zone)
            concern = cfg.escalated_warning_risk if escalated else cfg.warning_risk
        stroke_signs = [ASYMMETRIC] if ASYMMETRIC in warnings else []
        if convulsing:
            stroke_signs.append(CONVULSION)
        if ep.reason == "collapse_from_sitting":
            stroke_signs.append("collapse_from_sitting")
        elif ep.reason == "unsteady_before":
            stroke_signs.append("unsteady_before_fall")

        if convulsing:
            label = CONVULSING
        elif warnings:
            label = warnings[0]
        elif ep.convulsed:
            label = AFTER_CONVULSION
        elif ep.onset == "sudden" and not benign and still_s >= 3.0:
            label = COLLAPSED
        elif using_phone:
            label = USING_PHONE
        elif repositioning:
            label = REPOSITIONING
        elif benign:
            label = RESTING
        else:
            label = LYING

        return LyingContext(label, ep.onset, in_zone, lying_s, benign, concern, warnings, level_drop,
                            ep.reason, stroke_signs, convulsing=convulsing, convulsed=ep.convulsed)

    def cleanup_inactive(self, active_track_ids: set[int], timestamp: Optional[float] = None) -> None:
        for tid in list(self._tracks):
            if tid not in active_track_ids:
                st = self._tracks.pop(tid)
                if st.samples:
                    self._remember(st, timestamp if timestamp is not None else st.samples[-1].t)
