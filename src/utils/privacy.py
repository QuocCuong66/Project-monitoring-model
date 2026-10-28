"""Privacy preservation and face anonymization utility."""

from typing import Optional, List, Tuple
import cv2
import numpy as np


class FaceBlurrer:
    """Anonymizes faces in video frames using detected keypoints or bounding boxes."""

    def __init__(self, kernel_size: int = 31, sigma: float = 15.0) -> None:
        """Initialize FaceBlurrer.

        Args:
            kernel_size: Gaussian blur kernel size (must be odd positive integer).
            sigma: Gaussian standard deviation.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma = sigma

    def blur_keypoints_face(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        head_kp_indices: Optional[List[int]] = None,
        padding_ratio: float = 0.4,
    ) -> np.ndarray:
        """Blur face region inferred from facial keypoints (Nose, Eyes, Ears: 0..4 in COCO).

        Args:
            frame: Input BGR image (H, W, 3).
            keypoints: Array of shape (17, 3) where each row is (x, y, conf).
            head_kp_indices: Indices for head keypoints. Defaults to [0, 1, 2, 3, 4].
            padding_ratio: Expansion around head keypoints bounding box.

        Returns:
            Frame with blurred face region.
        """
        if head_kp_indices is None:
            head_kp_indices = [0, 1, 2, 3, 4]  # Nose, L-Eye, R-Eye, L-Ear, R-Ear

        h, w = frame.shape[:2]
        valid_pts = []
        for idx in head_kp_indices:
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > 0.3 and 0 <= x < w and 0 <= y < h:
                    valid_pts.append((x, y))

        if not valid_pts:
            return frame

        pts_np = np.array(valid_pts, dtype=np.float32)
        min_x, min_y = np.min(pts_np, axis=0)
        max_x, max_y = np.max(pts_np, axis=0)

        # Apply padding
        box_w = max_x - min_x
        box_h = max_y - min_y
        pad_x = max(box_w * padding_ratio, 15.0)
        pad_y = max(box_h * padding_ratio, 15.0)

        x1 = max(0, int(min_x - pad_x))
        y1 = max(0, int(min_y - pad_y))
        x2 = min(w, int(max_x + pad_x))
        y2 = min(h, int(max_y + pad_y))

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), self.sigma)
            frame[y1:y2, x1:x2] = blurred_roi

        return frame

    def blur_bbox_head(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        head_height_ratio: float = 0.22,
    ) -> np.ndarray:
        """Fallback blurring when keypoints are occluded: blurs top portion of person bbox."""
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox
        x1 = max(0, int(bx1))
        y1 = max(0, int(by1))
        x2 = min(w, int(bx2))
        head_h = (by2 - by1) * head_height_ratio
        y2 = min(h, int(by1 + head_h))

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), self.sigma)
            frame[y1:y2, x1:x2] = blurred_roi

        return frame


HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # COCO: nose, eyes, ears


def _head_boxes(
    keypoints: Optional[np.ndarray],
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, ...],
    kp_conf: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Regions that may show a face: around the facial keypoints and the top of the person box.

    Both are returned (over-masking is preferred to leaking a face): keypoints miss heads
    seen from behind, and the box top misses heads of people lying sideways.
    """
    h, w = frame_shape[:2]
    bx1, by1, bx2, by2 = bbox
    person_h = max(1.0, by2 - by1)
    person_w = max(1.0, bx2 - bx1)
    boxes: List[Tuple[float, float, float, float]] = []
    if keypoints is not None and len(keypoints) >= 5:
        pts = np.array([keypoints[i, :2] for i in HEAD_KEYPOINTS if keypoints[i, 2] > kp_conf], dtype=np.float32)
        if len(pts):
            cx, cy = pts.mean(axis=0)
            spread = float(np.max(np.ptp(pts, axis=0))) if len(pts) > 1 else 0.0
            # Head ~ 1/7 of body height; cap so a bad keypoint cannot smear across the frame
            half = float(np.clip(spread * 0.9, max(person_h, person_w) * 0.09, max(person_h, person_w) * 0.2))
            half = max(half, 12.0)
            boxes.append((cx - half, cy - half * 1.3, cx + half, cy + half))
        # Head seen from behind / from above: no facial keypoints, but it sits on or just above
        # the shoulders (overlapping them in top-down ceiling and doorbell views)
        shoulders = [keypoints[i, :2] for i in (5, 6) if keypoints[i, 2] > kp_conf]
        if shoulders:
            sx, sy = np.mean(shoulders, axis=0)
            width = float(np.linalg.norm(shoulders[0] - shoulders[1])) if len(shoulders) == 2 else 0.0
            half = max(width * 0.9, max(person_h, person_w) * 0.09, 12.0)
            boxes.append((sx - half, sy - half * 2.0, sx + half, sy + half * 0.6))
    boxes.append((bx1, by1, bx2, by1 + person_h * 0.25))
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def anonymize_people(
    frame: np.ndarray,
    people: List[Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]],
    block: int = 6,
) -> np.ndarray:
    """Return a copy of the frame with every person's head pixelated beyond recognition.

    Args:
        frame: BGR image.
        people: (keypoints (17, 3) or None, bbox x1y1x2y2) per person. Pass the raw detections
            of this exact frame as well as tracks: tracker boxes and smoothed keypoints lag.
        block: Pixels per side of the coarse mosaic (smaller = coarser = stronger).
    """
    out = frame.copy()
    for keypoints, bbox in people:
        for x1, y1, x2, y2 in _head_boxes(keypoints, bbox, out.shape):
            roi = out[y1:y2, x1:x2]
            small = cv2.resize(roi, (block, block), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def anonymize_alert_snapshot(
    frame: np.ndarray,
    detections: List[Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]],
    tracks: List[Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]],
    alerted_bbox: Tuple[float, float, float, float],
    min_iou: float = 0.1,
) -> np.ndarray:
    """Evidence snapshot that never shows a recognisable face.

    Heads of every detected / tracked person are pixelated. When the alerted person is not
    found among this frame's raw detections (occluded, cut off, mid-fall motion blur), the
    tracker box may have drifted and their head cannot be located, so the whole frame is
    pixelated instead: the scene stays readable but nobody is identifiable.
    """
    if not any(_iou(bbox, alerted_bbox) >= min_iou for _, bbox in detections):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return anonymize_people(frame, detections + tracks)
