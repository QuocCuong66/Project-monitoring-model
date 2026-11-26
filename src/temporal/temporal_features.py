"""Kinematic feature extraction from temporal skeletal and bounding box trajectories."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from src.temporal.sequence_buffer import TrackSnapshot
from src.utils.config import load_config


@dataclass
class KinematicFeatures:
    """Quantitative physical kinematics extracted over a temporal sequence."""
    vertical_velocity: float       # Downward velocity of hip center (norm units / sec)
    vertical_acceleration: float   # Downward acceleration (Ay)
    aspect_ratio_curr: float       # Current Width / Height ratio (>1.2 = lying horizontal)
    aspect_ratio_change: float     # Rate of change from vertical to horizontal
    torso_angle_curr: float        # Current angle relative to vertical [0..90]
    torso_angle_velocity: float    # Angular rate of change (deg / sec)
    immobility_index: float        # [0.0 = high motion, 1.0 = completely stationary]
    drop_severity_score: float     # [0.0 = none, 1.0 = severe sudden drop]
    # False when the hips are not confidently detected (person close to the camera / cropped by the
    # frame edge): the bbox then covers only the upper body, so aspect ratio and torso angle say nothing
    # about lying down and the bbox-centre "hip" jumps with arm movements.
    lower_body_visible: bool = True
    # True when the legs stand vertically under the hips although the torso leans far over: bending to
    # pick something up, squatting, kneeling or crawling. The body is held up by the legs, not lying.
    feet_supported: bool = False


class TemporalFeatureExtractor:
    """Extracts physics-based kinematic indicators from temporal snapshots."""

    def __init__(
        self,
        fps: float = 15.0,
        min_hip_confidence: Optional[float] = None,
        supported_leg_angle: Optional[float] = None,
        supported_min_leg_drop: Optional[float] = None,
    ) -> None:
        self.fps = fps
        try:
            kin_cfg = load_config("thresholds.yaml").get("kinematics", {}) or {}
        except Exception:
            kin_cfg = {}
        self.min_hip_confidence = float(
            min_hip_confidence if min_hip_confidence is not None else kin_cfg.get("min_hip_confidence", 0.5)
        )
        # Thigh or whole leg within this many degrees of vertical = the legs carry the body
        self.supported_leg_angle = float(
            supported_leg_angle if supported_leg_angle is not None else kin_cfg.get("supported_leg_angle", 40.0)
        )
        # Only while the torso leans at least this far: upright people have vertical legs too
        self.supported_min_torso_angle = float(kin_cfg.get("supported_min_torso_angle", 45.0))
        # ... and the knee / ankle at least this far (torso lengths) below the hip
        self.supported_min_leg_drop = float(
            supported_min_leg_drop if supported_min_leg_drop is not None
            else kin_cfg.get("supported_min_leg_drop", 0.4)
        )

    def _feet_supported(self, kp: np.ndarray) -> bool:
        """Legs vertical under the hips (bend / squat / kneel / crawl) instead of lying along the torso."""
        c = self.min_hip_confidence

        def mid(a: int, b: int) -> Optional[np.ndarray]:
            pts = [kp[i, :2] for i in (a, b) if kp[i, 2] >= c]
            return np.mean(pts, axis=0) if pts else None

        hip, shoulder = mid(11, 12), mid(5, 6)
        if hip is None or shoulder is None:
            return False
        torso = float(np.linalg.norm(shoulder - hip))
        if torso < 1.0:
            return False
        for joint in (mid(13, 14), mid(15, 16)):                   # thigh, then the whole leg
            if joint is None:
                continue
            leg = joint - hip
            drop = float(leg[1]) / torso                            # +y is down
            angle = float(np.degrees(np.arctan2(abs(leg[0]), max(1e-6, leg[1])))) if leg[1] > 0 else 180.0
            if drop >= self.supported_min_leg_drop and angle <= self.supported_leg_angle:
                return True
        return False

    def _hip_y(self, snapshot: TrackSnapshot) -> Optional[float]:
        """Mid-hip height in image coords, or None when neither hip is confidently detected."""
        kp = snapshot.keypoints
        c = self.min_hip_confidence
        if kp[11, 2] >= c and kp[12, 2] >= c:
            return float((kp[11, 1] + kp[12, 1]) / 2.0)
        if kp[11, 2] >= c:
            return float(kp[11, 1])
        if kp[12, 2] >= c:
            return float(kp[12, 1])
        return None

    def extract(self, snapshots: List[TrackSnapshot]) -> KinematicFeatures:
        """Extract kinematic features from snapshot history.

        Args:
            snapshots: Sequential history of snapshots for a single track.

        Returns:
            KinematicFeatures dataclass instance.
        """
        if len(snapshots) < 2:
            return KinematicFeatures(
                vertical_velocity=0.0,
                vertical_acceleration=0.0,
                aspect_ratio_curr=0.5,
                aspect_ratio_change=0.0,
                torso_angle_curr=0.0,
                torso_angle_velocity=0.0,
                immobility_index=0.0,
                drop_severity_score=0.0,
            )

        n = len(snapshots)
        curr = snapshots[-1]
        prev = snapshots[-2]
        dt = max(1e-3, curr.timestamp - prev.timestamp)

        # 1. Bounding box aspect ratio
        curr_bw = max(1.0, curr.bbox[2] - curr.bbox[0])
        curr_bh = max(1.0, curr.bbox[3] - curr.bbox[1])
        aspect_ratio_curr = curr_bw / curr_bh

        # Aspect ratio change over last ~1 second (min(n, int(self.fps)))
        hist_idx = max(0, n - int(self.fps))
        past = snapshots[hist_idx]
        past_bw = max(1.0, past.bbox[2] - past.bbox[0])
        past_bh = max(1.0, past.bbox[3] - past.bbox[1])
        past_ar = past_bw / past_bh
        aspect_ratio_change = aspect_ratio_curr - past_ar

        # 2. Vertical velocity and acceleration of Hip Center (in image coordinates: +y is downward)
        # Only frames with confidently detected hips count. The old bbox-centre fallback turned arm
        # movements of a person close to the camera into "fast drops".
        hip_y_series = [self._hip_y(s) for s in snapshots]
        lower_body_visible = hip_y_series[-1] is not None

        # Velocity and Acceleration: computed across the window to capture rapid drop peaks
        height_ref = max(30.0, curr_bh)
        v_series = []
        v_times = []
        for i in range(1, len(hip_y_series)):
            if hip_y_series[i] is None or hip_y_series[i - 1] is None:
                continue
            frame_dt = max(1e-3, snapshots[i].timestamp - snapshots[i-1].timestamp)
            v = (hip_y_series[i] - hip_y_series[i-1]) / (height_ref * frame_dt)
            v_series.append(v)
            v_times.append(snapshots[i].timestamp)

        curr_v_y = v_series[-1] if v_series else 0.0
        peak_v_y = max(v_series) if v_series else 0.0

        a_series = []
        for i in range(1, len(v_series)):
            frame_dt = max(1e-3, v_times[i] - v_times[i-1])
            a = (v_series[i] - v_series[i-1]) / frame_dt
            a_series.append(a)

        curr_a_y = a_series[-1] if a_series else 0.0
        peak_a_y = max(a_series) if a_series else 0.0

        # 3. Torso angle and angular velocity
        torso_angle_curr = curr.torso_angle
        torso_angle_velocity = (curr.torso_angle - past.torso_angle) / max(1e-3, curr.timestamp - past.timestamp)

        # 4. Immobility Index: measure lack of movement across recent frames (last 2-5s)
        window_pts = [s.normalized_keypoints[:, :2] for s in snapshots[-min(n, int(self.fps * 3)):]]
        if len(window_pts) >= 3:
            pts_arr = np.array(window_pts)  # (N, 17, 2)
            motion_std = float(np.mean(np.std(pts_arr, axis=0)))
            immobility_index = float(np.clip(1.0 - (motion_std / 0.08), 0.0, 1.0))
        else:
            immobility_index = 0.0

        # 5. Drop Severity Score: requires both horizontal posture AND dynamic drop evidence in window
        posture_horiz_factor = float(np.clip((torso_angle_curr - 30.0) / 45.0, 0.0, 1.0))
        aspect_ratio_factor = float(np.clip((aspect_ratio_curr - 0.7) / 0.8, 0.0, 1.0))
        if not lower_body_visible:
            # Upper-body-only bbox is wide and the torso angle uses guessed hips: no lying evidence
            posture_horiz_factor = 0.0
            aspect_ratio_factor = 0.0
        feet_supported = (lower_body_visible and torso_angle_curr >= self.supported_min_torso_angle
                          and self._feet_supported(curr.keypoints))
        if feet_supported:
            # Bending / squatting: torso far over and the box wide, but the legs still hold the body up
            posture_horiz_factor = 0.0
            aspect_ratio_factor = 0.0
        is_horizontal = max(posture_horiz_factor, aspect_ratio_factor)

        # Dynamic evidence: peak downward velocity or aspect ratio collapse during the window
        drop_motion_factor = float(np.clip(peak_v_y / 1.5, 0.0, 1.0))
        transition_factor = float(np.clip(aspect_ratio_change / 0.8, 0.0, 1.0)) if lower_body_visible else 0.0
        dynamic_drop_evidence = max(drop_motion_factor, transition_factor)

        if dynamic_drop_evidence < 0.15 or is_horizontal < 0.30:
            # Person is resting/sleeping without any drop motion, or still upright
            drop_severity = 0.0
        else:
            drop_severity = float(np.clip(
                0.40 * is_horizontal +
                0.35 * dynamic_drop_evidence +
                0.25 * (1.0 if peak_a_y > 1.0 else 0.0),
                0.0, 1.0
            ))

        return KinematicFeatures(
            vertical_velocity=float(peak_v_y),
            vertical_acceleration=float(peak_a_y),
            aspect_ratio_curr=float(aspect_ratio_curr),
            aspect_ratio_change=float(aspect_ratio_change),
            torso_angle_curr=float(torso_angle_curr),
            torso_angle_velocity=float(torso_angle_velocity),
            immobility_index=float(immobility_index),
            drop_severity_score=drop_severity,
            lower_body_visible=lower_body_visible,
            feet_supported=feet_supported,
        )
