"""Visualizer for rendering tracking, skeleton keypoints, risk scores, and alert UI."""

from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

# COCO Keypoint Skeleton Connections
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # Limbs to hips
    (5, 11), (6, 12), (5, 6),                           # Torso box
    (5, 7), (7, 9), (6, 8), (8, 10),                    # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6) # Head/Face
]

# Color constants (BGR)
COLOR_NORMAL = (0, 200, 0)        # Green
COLOR_SUSPICIOUS = (0, 215, 255)   # Amber / Yellow
COLOR_ABNORMAL = (0, 140, 255)     # Orange
COLOR_HIGH_RISK = (0, 0, 240)      # Red
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 20, 20)


def _blend_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blend a filled rectangle in place, touching only the ROI (not a full-frame copy)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    solid = np.empty_like(roi)
    solid[:] = color
    cv2.addWeighted(solid, alpha, roi, 1.0 - alpha, 0, roi)


class PipelineVisualizer:
    """Renders overlays on video frames for real-time monitoring display."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.show_bbox = self.config.get("show_bbox", True)
        self.show_skeleton = self.config.get("show_skeleton", True)
        self.show_track_id = self.config.get("show_track_id", True)
        self.show_risk_gauge = self.config.get("show_risk_gauge", True)
        self.show_fps_meter = self.config.get("show_fps_meter", True)

    def get_state_color(self, state: str) -> Tuple[int, int, int]:
        """Map state name to BGR color."""
        mapping = {
            "NORMAL": COLOR_NORMAL,
            "SUSPICIOUS": COLOR_SUSPICIOUS,
            "ABNORMAL": COLOR_ABNORMAL,
            "HIGH_RISK": COLOR_HIGH_RISK,
            "ALERT_SENT": COLOR_HIGH_RISK,
            "WAITING_FOR_CONFIRMATION": COLOR_HIGH_RISK,
        }
        return mapping.get(state.upper(), COLOR_NORMAL)

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int] = COLOR_NORMAL,
        conf_thresh: float = 0.35,
    ) -> np.ndarray:
        """Draw COCO 17-keypoints and skeleton links.

        Args:
            frame: BGR image.
            keypoints: Array of shape (17, 3) (x, y, conf).
            color: Link and point color.
            conf_thresh: Minimum confidence to draw point.
        """
        # Draw skeleton limbs
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                x1, y1, c1 = keypoints[p1_idx]
                x2, y2, c2 = keypoints[p2_idx]
                if c1 >= conf_thresh and c2 >= conf_thresh:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw joints
        for i, (x, y, c) in enumerate(keypoints):
            if c >= conf_thresh:
                pt = (int(x), int(y))
                cv2.circle(frame, pt, 4, COLOR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)

        return frame

    def draw_person_card(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        state: str,
        action: str,
        risk_score: float,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Draw bounding box, state badge, action label, and mini risk meter."""
        color = self.get_state_color(state)
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw bounding box (Thick glowing red border when emergency/fall detected)
        is_emergency = (state in ["HIGH_RISK", "ALERT_SENT"]) or (risk_score >= 0.7)
        thickness = 4 if is_emergency else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if is_emergency:
            cv2.rectangle(frame, (max(0, x1 - 3), max(0, y1 - 3)), (min(frame.shape[1] - 1, x2 + 3), min(frame.shape[0] - 1, y2 + 3)), (0, 0, 255), 2, cv2.LINE_AA)

        # Draw top label badge
        label = f"ID:{track_id} | {action.upper()} | {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_y1 = max(0, y1 - th - 10)
        badge_y2 = y1
        badge_x2 = min(frame.shape[1], x1 + tw + 12)

        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        text_color = (0, 0, 0) if state in ["SUSPICIOUS"] else COLOR_WHITE
        cv2.putText(
            frame,
            label,
            (x1 + 6, badge_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

        # Draw mini risk bar right below badge or below bbox
        bar_w = max(60, x2 - x1)
        bar_h = 6
        bar_x1 = x1
        bar_y1 = y2 + 4
        bar_x2 = bar_x1 + bar_w
        bar_y2 = bar_y1 + bar_h

        if bar_y2 < frame.shape[0]:
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            fill_w = int(bar_w * min(1.0, max(0.0, risk_score)))
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), color, -1)

        # If HIGH_RISK or ALERT_SENT, draw warning banner across top of frame
        if state in ["HIGH_RISK", "ALERT_SENT"]:
            self._draw_alert_banner(frame, track_id, risk_score)

        return frame

    def _draw_alert_banner(self, frame: np.ndarray, track_id: int, risk_score: float) -> None:
        """Draw high visibility alert banner on emergency state."""
        h, w = frame.shape[:2]
        banner_h = 44
        _blend_rect(frame, 0, 0, w, banner_h, (0, 0, 180), 0.85)

        msg = f"EMERGENCY ALERT: Possible medical emergency detected (ID:{track_id}, Risk: {risk_score:.2f})!"
        cv2.putText(
            frame,
            msg,
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        active_tracks: int,
        system_status: str = "ONLINE",
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Draw heads-up display (FPS, active people, status)."""
        h, w = frame.shape[:2]
        hud_w, hud_h = 260, 95
        hud_x, hud_y = w - hud_w - 10, 10

        _blend_rect(frame, hud_x, hud_y, hud_x + hud_w, hud_y + hud_h, COLOR_DARK_BG, 0.75)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 80, 80), 1)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Status: {system_status[:16]}",
            (hud_x + 10, hud_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Monitored Persons: {active_tracks}",
            (hud_x + 10, hud_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # Tier-0 Motion Gating indicator
        if telemetry and "tier0_status" in telemetry:
            t0_status = telemetry["tier0_status"]
            t0_ratio = telemetry.get("tier0_motion_ratio", 0.0) * 100.0
            if t0_status == "IDLE":
                t0_str = f"Tier-0: IDLE (AI OFF, 10-15 FPS)"
                t0_color = (255, 200, 0)  # Cyan/Sky
            elif t0_status == "COOLDOWN":
                t0_str = f"Tier-0: COOLDOWN ({t0_ratio:.1f}%)"
                t0_color = (0, 200, 255)  # Orange
            else:
                t0_str = f"Tier-0: ACTIVE ({t0_ratio:.1f}%)"
                t0_color = (0, 255, 120)  # Green

            cv2.putText(
                frame,
                t0_str,
                (hud_x + 10, hud_y + 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                t0_color,
                1,
                cv2.LINE_AA,
            )

        if telemetry and "gpu_vram_allocated_mb" in telemetry:
            vram_mb = telemetry["gpu_vram_allocated_mb"]
            cv2.putText(
                frame,
                f"GPU VRAM: {vram_mb:.0f} MB",
                (hud_x + 10, hud_y + 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw_emergency_frame_border(self, frame: np.ndarray) -> np.ndarray:
        """Draw an emergency flashing red border around the entire camera frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 240), 6, cv2.LINE_AA)
        cv2.rectangle(frame, (8, 8), (w - 9, h - 9), (0, 0, 255), 2, cv2.LINE_AA)
        return frame
