"""Cascade / Gated Inference Kinematic Trigger.

Tier 1 of Cascade AI: Ultra-fast NumPy-based biomechanical filtering (~0 ms).
Protects against the 'Dead Gate' trap by evaluating:
1. Hard Falls: High vertical drop velocity (Vy), acceleration (Ay), and torso angular velocity.
2. Slow Slide / Collapse: Ground Proximity (hip/shoulder dropping near ground_plane_y baseline).
3. Z-Axis Falls & Contortion: Delta Area collapse (|Delta Area / Area_base| > 0.35) and height collapse.
4. Floor Horizontal Postures: Aspect ratio W/H > 1.15 or torso angle > 50 deg.
5. Gate Hold-on Timer: Keeps gate locked OPEN for at least 45 frames (~3.0s) once triggered.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.temporal.sequence_buffer import TrackSnapshot
from src.temporal.temporal_features import KinematicFeatures
from src.action.action_classifier import ActionPrediction, ACTION_SEVERITY


@dataclass
class GatedTriggerResult:
    """Outcome of Tier 1 Kinematic Gating evaluation."""
    should_run_ai: bool
    trigger_reason: str
    hold_on_frames_remaining: int
    heuristic_prediction: Optional[ActionPrediction] = None


class KinematicGatedTrigger:
    """Tier 1 Biomechanical Gating Filter with Anti-Dead-Gate Protections."""

    def __init__(
        self,
        vy_thresh: float = 1.10,
        ay_thresh: float = 0.75,
        tilt_vel_thresh: float = 28.0,
        aspect_ratio_thresh: float = 1.15,
        torso_angle_thresh: float = 50.0,
        area_change_thresh: float = 0.32,
        ground_proximity_thresh: float = 0.22,  # Fraction of baseline standing height above floor
        hold_on_frames: int = 45,  # 45 frames ~ 3.0s at 15 FPS
    ) -> None:
        self.vy_thresh = vy_thresh
        self.ay_thresh = ay_thresh
        self.tilt_vel_thresh = tilt_vel_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.torso_angle_thresh = torso_angle_thresh
        self.area_change_thresh = area_change_thresh
        self.ground_proximity_thresh = ground_proximity_thresh
        self.hold_on_frames = hold_on_frames

        # Per-track tracking state
        self._hold_on_counters: Dict[int, int] = {}
        self._ground_plane_y: Dict[int, float] = {}       # Maximum bottom y observed when upright
        self._baseline_standing_h: Dict[int, float] = {}   # Baseline upright height
        self._baseline_area: Dict[int, float] = {}         # Baseline upright bbox area

    def update_track_baselines(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        torso_angle: float,
    ) -> None:
        """Continuously update ground plane and standing height baselines during normal posture."""
        bw = max(1.0, bbox[2] - bbox[0])
        bh = max(1.0, bbox[3] - bbox[1])
        ar = bw / bh
        area = bw * bh
        bottom_y = bbox[3]

        # Consider upright if aspect ratio < 0.65 and torso angle < 30 deg
        if ar < 0.65 and torso_angle < 30.0:
            if track_id not in self._ground_plane_y:
                self._ground_plane_y[track_id] = bottom_y
                self._baseline_standing_h[track_id] = bh
                self._baseline_area[track_id] = area
            else:
                # Floor level is the lowest contact point (max y in image space)
                self._ground_plane_y[track_id] = max(self._ground_plane_y[track_id], bottom_y)
                self._baseline_standing_h[track_id] = max(self._baseline_standing_h[track_id], bh)
                # Running average for baseline area
                self._baseline_area[track_id] = 0.90 * self._baseline_area[track_id] + 0.10 * area

    def evaluate(
        self,
        track_id: int,
        snapshots: List[TrackSnapshot],
        kinematics: KinematicFeatures,
        current_state: str = "NORMAL",
    ) -> GatedTriggerResult:
        """Evaluate whether Tier 2 Deep AI inference is required for this track.

        Returns:
            GatedTriggerResult with should_run_ai boolean and rationale.
        """
        if not snapshots:
            return GatedTriggerResult(
                should_run_ai=False,
                trigger_reason="no_snapshots",
                hold_on_frames_remaining=0,
                heuristic_prediction=self._heuristic_normal_action(0.0, 0.5),
            )

        curr = snapshots[-1]
        self.update_track_baselines(track_id, curr.bbox, curr.torso_angle)

        curr_bw = max(1.0, curr.bbox[2] - curr.bbox[0])
        curr_bh = max(1.0, curr.bbox[3] - curr.bbox[1])
        curr_ar = curr_bw / curr_bh
        curr_area = curr_bw * curr_bh

        base_h = self._baseline_standing_h.get(track_id, curr_bh)
        base_area = self._baseline_area.get(track_id, curr_area)
        ground_y = self._ground_plane_y.get(track_id, curr.bbox[3])

        # Current hip height in image coords (+y is down)
        hip_y = None
        if curr.keypoints is not None:
            kp = curr.keypoints
            if kp[11, 2] > 0.2 and kp[12, 2] > 0.2:
                hip_y = (kp[11, 1] + kp[12, 1]) / 2.0
            elif kp[11, 2] > 0.2:
                hip_y = kp[11, 1]
            elif kp[12, 2] > 0.2:
                hip_y = kp[12, 1]

        triggers: List[str] = []

        # 1. HARD FALL TRIGGER: Dynamic velocity or acceleration spike
        if kinematics.vertical_velocity >= self.vy_thresh:
            triggers.append(f"high_vy({kinematics.vertical_velocity:.2f})")
        if kinematics.vertical_acceleration >= self.ay_thresh:
            triggers.append(f"high_ay({kinematics.vertical_acceleration:.2f})")
        if abs(kinematics.torso_angle_velocity) >= self.tilt_vel_thresh:
            triggers.append(f"rapid_tilt({kinematics.torso_angle_velocity:.1f}deg/s)")

        # 2. SLOW SLIDE / COLLAPSE TRIGGER (Ground Proximity):
        # Hip dropped close to ground plane or height collapsed to < 45% of upright baseline
        if hip_y is not None:
            hip_distance_to_floor = ground_y - hip_y
            if hip_distance_to_floor <= (self.ground_proximity_thresh * base_h):
                triggers.append(f"ground_proximity(hip_dist={hip_distance_to_floor:.1f}px)")
        if curr_bh <= (0.45 * base_h):
            triggers.append(f"vertical_collapse(h_ratio={curr_bh/base_h:.2f})")

        # 3. Z-AXIS FALL & CONTORTION TRIGGER:
        # Frontal/backward fall towards/away from camera causes significant bbox area changes while curling up
        area_ratio_change = abs(curr_area - base_area) / max(1.0, base_area)
        if area_ratio_change >= self.area_change_thresh and curr_bh <= (0.58 * base_h):
            triggers.append(f"z_axis_collapse(dArea={area_ratio_change:.2f})")

        # 4. FLOOR HORIZONTAL POSTURE TRIGGER:
        if curr_ar >= self.aspect_ratio_thresh:
            triggers.append(f"horizontal_ar({curr_ar:.2f})")
        if curr.torso_angle >= self.torso_angle_thresh:
            triggers.append(f"high_torso_angle({curr.torso_angle:.1f}deg)")

        # 5. SUSTAINED RISK STATE (FSM Hysteresis):
        # If the person is already under investigation or potential fall, keep Gate open
        if current_state in ["SUSPICIOUS", "ABNORMAL", "HIGH_RISK", "ALERT_SENT", "INVESTIGATING", "POTENTIAL_FALL"]:
            triggers.append(f"state_hysteresis({current_state})")

        # Check Gate Hold-on Timer
        remaining_hold_on = self._hold_on_counters.get(track_id, 0)

        if triggers:
            # Refresh hold-on timer back to full duration
            self._hold_on_counters[track_id] = self.hold_on_frames
            reason = " + ".join(triggers)
            return GatedTriggerResult(
                should_run_ai=True,
                trigger_reason=reason,
                hold_on_frames_remaining=self.hold_on_frames,
                heuristic_prediction=None,
            )

        # No new triggers fired. Check if hold-on timer is still active
        if remaining_hold_on > 0:
            remaining_hold_on -= 1
            self._hold_on_counters[track_id] = remaining_hold_on
            return GatedTriggerResult(
                should_run_ai=True,
                trigger_reason=f"gate_hold_on({remaining_hold_on}_frames_left)",
                hold_on_frames_remaining=remaining_hold_on,
                heuristic_prediction=None,
            )

        # Gate CLOSED: person is walking, standing, or sitting stably
        heuristic_pred = self._heuristic_normal_action(curr.torso_angle, curr_ar)
        return GatedTriggerResult(
            should_run_ai=False,
            trigger_reason="gate_closed_normal_motion",
            hold_on_frames_remaining=0,
            heuristic_prediction=heuristic_pred,
        )

    def cleanup_inactive(self, active_track_ids: set) -> None:
        """Remove state for tracks that are no longer active."""
        dead_ids = [tid for tid in self._hold_on_counters if tid not in active_track_ids]
        for tid in dead_ids:
            self._hold_on_counters.pop(tid, None)
            self._ground_plane_y.pop(tid, None)
            self._baseline_standing_h.pop(tid, None)
            self._baseline_area.pop(tid, None)

    def _heuristic_normal_action(self, torso_angle: float, aspect_ratio: float) -> ActionPrediction:
        """Fast ~0 ms heuristic for regular upright/seated behavior."""
        if torso_angle < 18.0 and aspect_ratio < 0.55:
            action = "standing"
        elif torso_angle < 28.0:
            action = "walking"
        elif torso_angle < 45.0 or (0.55 <= aspect_ratio < 0.95):
            action = "sitting"
        else:
            action = "bending"

        probs = {
            "walking": 0.05,
            "standing": 0.05,
            "sitting": 0.05,
            "lying": 0.01,
            "bending": 0.01,
            "falling": 0.00,
            "getting_up": 0.01,
            "stumbling": 0.01,
            "abnormal_movement": 0.01,
            "immobile": 0.00,
        }
        probs[action] = 0.85

        return ActionPrediction(
            primary_action=action,
            confidence=0.85,
            probabilities=probs,
            emergency_prob=0.01,
            severity_score=float(ACTION_SEVERITY.get(action, 0.0)),
        )
