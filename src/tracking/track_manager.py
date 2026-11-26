"""ByteTrack-based multi-object track manager implementation."""

from collections import deque
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    # Imported once at module load: a lazy import inside linear_assignment()
    # stalled the very first tracker update by ~0.8 s.
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from src.detection.detector import Detection
from src.tracking.tracker import Track, Tracker


def bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return (inter_area / union_area) if union_area > 1e-6 else 0.0


def bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU between (N, 4) and (M, 4) xyxy boxes -> (N, M).

    Replaces the O(N*M) Python double loop so association cost stays flat
    when many people are in view.
    """
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-6, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy or Hungarian matching for cost matrix."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs = zip(row_ind.tolist(), col_ind.tolist())
    else:
        # Greedy fallback: lowest cost first, each row/col used once
        pairs = []
        used_r, used_c = set(), set()
        for flat in np.argsort(cost_matrix, axis=None):
            r, c = (int(v) for v in np.unravel_index(flat, cost_matrix.shape))
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)

    matches = [(r, c) for r, c in pairs if cost_matrix[r, c] <= thresh]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_a = [r for r in range(cost_matrix.shape[0]) if r not in matched_r]
    unmatched_b = [c for c in range(cost_matrix.shape[1]) if c not in matched_c]
    return matches, unmatched_a, unmatched_b


class KalmanBoxTracker:
    """Simplified Bounding Box Kalman Filter."""

    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]) -> None:
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        # State: [cx, cy, s, r, vx, vy, vs] where s = area, r = aspect ratio
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        s = w * h
        r = w / h

        self.state = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float32)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history: deque[Tuple[float, float, float, float]] = deque([bbox], maxlen=120)

    def predict(self) -> Tuple[float, float, float, float]:
        """Advance the state vector and return the predicted bounding box."""
        self.state[0] += self.state[4]
        self.state[1] += self.state[5]
        self.state[2] += self.state[6]
        self.age += 1
        self.time_since_update += 1

        w = np.sqrt(max(1.0, self.state[2] * self.state[3]))
        h = self.state[2] / max(1e-4, w)
        cx, cy = self.state[0], self.state[1]
        predicted_bbox = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        return predicted_bbox

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        """Update the state vector with observed bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        s = w * h
        r = w / h

        # Simple alpha-beta smoothing filter
        alpha = 0.7
        self.state[4] = 0.8 * self.state[4] + 0.2 * (cx - self.state[0])
        self.state[5] = 0.8 * self.state[5] + 0.2 * (cy - self.state[1])
        self.state[0] = (1 - alpha) * self.state[0] + alpha * cx
        self.state[1] = (1 - alpha) * self.state[1] + alpha * cy
        self.state[2] = (1 - alpha) * self.state[2] + alpha * s
        self.state[3] = (1 - alpha) * self.state[3] + alpha * r

        self.history.append(bbox)

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return the current bounding box estimate."""
        w = np.sqrt(max(1.0, self.state[2] * self.state[3]))
        h = self.state[2] / max(1e-4, w)
        cx, cy = self.state[0], self.state[1]
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class ByteTrackManager(Tracker):
    """ByteTrack implementation: Two-stage association with high & low score detections."""

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.60,
        match_thresh: float = 0.70,
        max_lost_frames: int = 45,
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames

        self._trackers: Dict[int, KalmanBoxTracker] = {}
        self._tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        self._trackers.clear()
        self._tracks.clear()
        KalmanBoxTracker.count = 0

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        # 1. Split detections into high confidence and low confidence
        dets_high: List[Detection] = []
        dets_low: List[Detection] = []

        for d in detections:
            if d.confidence >= self.track_high_thresh:
                dets_high.append(d)
            elif d.confidence >= self.track_low_thresh:
                dets_low.append(d)

        # 2. Predict positions for all existing trackers
        track_ids = list(self._trackers.keys())
        predicted_boxes = [self._trackers[tid].predict() for tid in track_ids]

        # 3. First Association: Match High Confidence Detections with Existing Tracks
        matched_tracks_1: Dict[int, Detection] = {}
        unmatched_dets_high: List[Detection] = []
        unmatched_track_ids: List[int] = track_ids[:]

        if track_ids and dets_high:
            cost_matrix = 1.0 - bbox_iou_matrix(
                np.array(predicted_boxes, dtype=np.float32),
                np.array([d.bbox for d in dets_high], dtype=np.float32),
            )

            matches, u_tracks, u_dets = linear_assignment(cost_matrix, thresh=self.match_thresh)
            for t_idx, d_idx in matches:
                matched_tracks_1[track_ids[t_idx]] = dets_high[d_idx]
            unmatched_track_ids = [track_ids[i] for i in u_tracks]
            unmatched_dets_high = [dets_high[j] for j in u_dets]
        else:
            unmatched_dets_high = dets_high[:]

        # 4. Second Association: Match Low Confidence Detections with Remaining Tracks
        matched_tracks_2: Dict[int, Detection] = {}
        remaining_unmatched_tracks: List[int] = unmatched_track_ids[:]

        if unmatched_track_ids and dets_low:
            cost_matrix_low = 1.0 - bbox_iou_matrix(
                np.array([self._trackers[tid].get_state() for tid in unmatched_track_ids], dtype=np.float32),
                np.array([d.bbox for d in dets_low], dtype=np.float32),
            )

            matches_2, u_tracks_2, _ = linear_assignment(cost_matrix_low, thresh=0.6)
            for t_idx, d_idx in matches_2:
                matched_tracks_2[unmatched_track_ids[t_idx]] = dets_low[d_idx]
            remaining_unmatched_tracks = [unmatched_track_ids[i] for i in u_tracks_2]

        # 4.5. Third Association (Fall Protection): Center-proximity matching for collapsing bodies
        matched_tracks_3: Dict[int, Detection] = {}
        if remaining_unmatched_tracks and unmatched_dets_high:
            for tid in list(remaining_unmatched_tracks):
                p_box = self._trackers[tid].get_state()
                p_cx = (p_box[0] + p_box[2]) / 2.0
                p_cy = (p_box[1] + p_box[3]) / 2.0
                max_dim = max(p_box[2] - p_box[0], p_box[3] - p_box[1])

                best_idx = -1
                min_dist = float("inf")
                for j, d in enumerate(unmatched_dets_high):
                    d_cx = (d.bbox[0] + d.bbox[2]) / 2.0
                    d_cy = (d.bbox[1] + d.bbox[3]) / 2.0
                    dist = np.hypot(p_cx - d_cx, p_cy - d_cy)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = j

                # Relaxed matching distance during vertical-to-horizontal collapse (up to 1.6x height)
                if best_idx >= 0 and min_dist <= (max_dim * 1.6):
                    matched_tracks_3[tid] = unmatched_dets_high[best_idx]
                    remaining_unmatched_tracks.remove(tid)
                    unmatched_dets_high.pop(best_idx)

        # 5. Update matched trackers
        all_matches = {**matched_tracks_1, **matched_tracks_2, **matched_tracks_3}
        for tid, det in all_matches.items():
            self._trackers[tid].update(det.bbox)
            if tid not in self._tracks:
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    last_seen=timestamp,
                )
            tr = self._tracks[tid]
            tr.bbox = det.bbox
            tr.confidence = det.confidence
            tr.last_seen = timestamp
            tr.time_since_update = 0
            tr.bbox_history.append(det.bbox)
            tr.trajectory.append(tr.bottom_center)
            tr.state = "tracked"

            # Update baseline upright height if standing (aspect ratio < 0.6)
            if tr.aspect_ratio < 0.6:
                if tr.standing_height_baseline is None:
                    tr.standing_height_baseline = tr.height
                else:
                    tr.standing_height_baseline = max(tr.standing_height_baseline, tr.height)

        # 6. Initialize New Tracks from unmatched high-confidence detections
        for det in unmatched_dets_high:
            if det.confidence >= self.new_track_thresh:
                new_tracker = KalmanBoxTracker(det.bbox)
                tid = new_tracker.id
                self._trackers[tid] = new_tracker
                new_track = Track(
                    track_id=tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    last_seen=timestamp,
                )
                new_track.bbox_history.append(det.bbox)
                new_track.trajectory.append(new_track.bottom_center)
                if new_track.aspect_ratio < 0.6:
                    new_track.standing_height_baseline = new_track.height
                self._tracks[tid] = new_track

        # 7. Remove stale tracks
        active_tracks: List[Track] = []
        dead_track_ids: List[int] = []

        for tid, tracker in self._trackers.items():
            if tid in remaining_unmatched_tracks:
                # predict() already advanced tracker.time_since_update for this frame;
                # incrementing it again here halved the effective lost-track buffer.
                if tid in self._tracks:
                    self._tracks[tid].time_since_update += 1
                    self._tracks[tid].state = "lost"

            if tracker.time_since_update > self.max_lost_frames:
                dead_track_ids.append(tid)
            elif tid in self._tracks and self._tracks[tid].state == "tracked":
                active_tracks.append(self._tracks[tid])

        for tid in dead_track_ids:
            self._trackers.pop(tid, None)
            self._tracks.pop(tid, None)

        return active_tracks
