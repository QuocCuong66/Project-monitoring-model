"""COCO-style keypoint (OKS) and bounding-box (IoU) detection metrics.

Independent re-implementation (no pycocotools dependency, since pycocotools has
no prebuilt wheel for Python 3.14 on Windows at the time this was written).
Matching strategy is a per-image greedy assignment by descending prediction
confidence, which is a documented simplification versus pycocotools:

- No area-range breakdown (small/medium/large AP) is computed.
- `iscrowd` ground-truth regions are not given special treatment.

AP is computed with COCO's 101-point recall interpolation, so results should
be close to but not bit-identical with pycocotools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Official COCO per-keypoint sigmas, in the 17-point COCO order:
# nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow,
# l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle
COCO_KEYPOINT_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
], dtype=np.float64)

COCO_OKS_THRESHOLDS = np.linspace(0.50, 0.95, 10)  # 0.50, 0.55, ..., 0.95


def compute_oks(
    pred_kpts: np.ndarray,
    gt_kpts: np.ndarray,
    gt_area: float,
    sigmas: np.ndarray = COCO_KEYPOINT_SIGMAS,
) -> float:
    """Object Keypoint Similarity between one prediction and one GT instance.

    pred_kpts, gt_kpts: (K, 3) arrays of (x, y, v). GT v==0 keypoints are
    excluded from the score (unlabeled). gt_area is the GT instance's
    segmentation/bbox area in pixels^2 (COCO's object scale term).
    """
    if gt_area <= 0:
        return 0.0
    visible = gt_kpts[:, 2] > 0
    if not np.any(visible):
        return 0.0
    dx = pred_kpts[:, 0] - gt_kpts[:, 0]
    dy = pred_kpts[:, 1] - gt_kpts[:, 1]
    d_sq = dx ** 2 + dy ** 2
    k = 2 * sigmas
    e = d_sq / ((k ** 2) * gt_area * 2)
    return float(np.sum(np.exp(-e)[visible]) / np.sum(visible))


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Instance:
    image_id: int
    keypoints: Optional[np.ndarray] = None  # (17, 3): x, y, v
    bbox: Optional[Sequence[float]] = None  # x1, y1, x2, y2
    area: float = 0.0
    score: float = 1.0  # predictions only; ignored for GT


@dataclass
class ImageMatchResult:
    image_id: int
    tp: int
    fp: int
    fn: int
    matched_scores: List[float] = field(default_factory=list)
    matched_flags: List[int] = field(default_factory=list)  # 1=TP, 0=FP, aligned to matched_scores


def match_image_keypoints(
    preds: List[Instance], gts: List[Instance], oks_threshold: float
) -> ImageMatchResult:
    """Greedy match: sort predictions by score desc, assign each to the
    highest-OKS unmatched GT above `oks_threshold`."""
    preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
    gt_used = [False] * len(gts)
    tp = fp = 0
    scores: List[float] = []
    flags: List[int] = []

    for pred in preds_sorted:
        best_oks, best_idx = 0.0, -1
        for i, gt in enumerate(gts):
            if gt_used[i]:
                continue
            oks = compute_oks(pred.keypoints, gt.keypoints, gt.area)
            if oks > best_oks:
                best_oks, best_idx = oks, i
        scores.append(pred.score)
        if best_idx >= 0 and best_oks >= oks_threshold:
            gt_used[best_idx] = True
            tp += 1
            flags.append(1)
        else:
            fp += 1
            flags.append(0)

    fn = gt_used.count(False)
    return ImageMatchResult(
        image_id=preds[0].image_id if preds else (gts[0].image_id if gts else -1),
        tp=tp, fp=fp, fn=fn, matched_scores=scores, matched_flags=flags,
    )


def _ap_from_matches(scores: np.ndarray, flags: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP given per-detection (score, is_tp) pairs."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    flags = flags[order]
    tp_cum = np.cumsum(flags)
    fp_cum = np.cumsum(1 - flags)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Monotonic precision envelope (precision[i] = max precision at recall >= r_i)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall_levels = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_levels:
        idxs = np.where(recall >= r)[0]
        p = precision[idxs].max() if len(idxs) else 0.0
        ap += p
    return float(ap / 101.0)


@dataclass
class DetectionMetrics:
    ap: float             # mean AP over OKS/IoU thresholds 0.50:0.05:0.95
    ap50: float
    ap75: float
    precision: float      # at the single operating point provided (e.g. conf >= threshold)
    recall: float
    tp: int
    fp: int
    fn: int
    n_images: int


def evaluate_keypoints(
    predictions_by_image: dict,
    ground_truth_by_image: dict,
    score_threshold: float = 0.3,
) -> DetectionMetrics:
    """Evaluate pose predictions against COCO-format keypoint ground truth.

    predictions_by_image / ground_truth_by_image: {image_id: List[Instance]}
    Every image_id present in ground_truth_by_image is scored, even if the
    predictions dict has no entry for it (counts as all-FN for that image).
    """
    image_ids = sorted(ground_truth_by_image.keys())

    ap_per_threshold = []
    for oks_thr in COCO_OKS_THRESHOLDS:
        all_scores, all_flags, n_gt_total = [], [], 0
        for img_id in image_ids:
            gts = ground_truth_by_image[img_id]
            preds = predictions_by_image.get(img_id, [])
            n_gt_total += len(gts)
            result = match_image_keypoints(preds, gts, oks_thr)
            all_scores.extend(result.matched_scores)
            all_flags.extend(result.matched_flags)
        ap = _ap_from_matches(np.array(all_scores), np.array(all_flags), n_gt_total)
        ap_per_threshold.append(ap)

    ap_per_threshold = np.array(ap_per_threshold)
    ap_mean = float(np.nanmean(ap_per_threshold)) if not np.all(np.isnan(ap_per_threshold)) else float("nan")
    ap50 = float(ap_per_threshold[0])
    ap75 = float(ap_per_threshold[5])

    # Precision/Recall at a single operating point: keep predictions with
    # score >= score_threshold, match at OKS >= 0.50.
    tp = fp = fn = 0
    for img_id in image_ids:
        gts = ground_truth_by_image[img_id]
        preds = [p for p in predictions_by_image.get(img_id, []) if p.score >= score_threshold]
        result = match_image_keypoints(preds, gts, oks_threshold=0.50)
        tp += result.tp
        fp += result.fp
        fn += result.fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return DetectionMetrics(
        ap=ap_mean, ap50=ap50, ap75=ap75,
        precision=precision, recall=recall,
        tp=tp, fp=fp, fn=fn, n_images=len(image_ids),
    )
