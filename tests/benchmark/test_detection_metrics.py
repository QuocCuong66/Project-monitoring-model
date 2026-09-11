"""Unit tests for benchmark/metrics/detection.py (OKS + AP), verified by hand."""
import math

import numpy as np
import pytest

from benchmark.metrics.detection import (
    COCO_KEYPOINT_SIGMAS,
    Instance,
    compute_iou,
    compute_oks,
    evaluate_keypoints,
)


def test_compute_oks_perfect_match_is_one():
    kpts = np.zeros((17, 3))
    kpts[:, 0] = np.arange(17) * 10.0
    kpts[:, 1] = np.arange(17) * 5.0
    kpts[:, 2] = 2.0  # all visible
    assert compute_oks(kpts, kpts, gt_area=1000.0) == pytest.approx(1.0)


def test_compute_oks_matches_hand_calculation():
    # Single visible keypoint (nose, sigma=0.026), displaced by 5px, area=400.
    sigma = COCO_KEYPOINT_SIGMAS[0]
    pred = np.zeros((17, 3))
    gt = np.zeros((17, 3))
    pred[0] = [5.0, 0.0, 2.0]
    gt[0] = [0.0, 0.0, 2.0]
    area = 400.0
    expected = math.exp(-(5.0 ** 2) / ((2 * sigma) ** 2 * area * 2))
    assert compute_oks(pred, gt, gt_area=area) == pytest.approx(expected, rel=1e-6)


def test_compute_oks_ignores_invisible_keypoints():
    pred = np.zeros((17, 3))
    gt = np.zeros((17, 3))
    gt[:, 2] = 0.0  # nothing labeled visible
    assert compute_oks(pred, gt, gt_area=100.0) == 0.0


def test_compute_oks_zero_area_is_zero():
    kpts = np.ones((17, 3))
    assert compute_oks(kpts, kpts, gt_area=0.0) == 0.0


def test_compute_iou_known_boxes():
    # Two 10x10 boxes overlapping in a 5x5 corner => intersection=25, union=175
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert compute_iou(a, b) == pytest.approx(25 / 175)


def test_compute_iou_no_overlap():
    assert compute_iou((0, 0, 1, 1), (10, 10, 11, 11)) == 0.0


def test_compute_iou_identical_boxes_is_one():
    box = (2, 3, 20, 25)
    assert compute_iou(box, box) == pytest.approx(1.0)


def _make_person(cx: float, cy: float) -> np.ndarray:
    kpts = np.zeros((17, 3))
    kpts[:, 0] = cx + np.arange(17) * 2.0
    kpts[:, 1] = cy + np.arange(17) * 2.0
    kpts[:, 2] = 2.0
    return kpts


def test_evaluate_keypoints_perfect_predictions_gives_ap_one():
    gt_kpts = _make_person(100, 100)
    gt = {1: [Instance(image_id=1, keypoints=gt_kpts, area=5000.0)]}
    pred = {1: [Instance(image_id=1, keypoints=gt_kpts.copy(), area=5000.0, score=0.99)]}

    metrics = evaluate_keypoints(pred, gt, score_threshold=0.3)

    assert metrics.ap == pytest.approx(1.0)
    assert metrics.ap50 == pytest.approx(1.0)
    assert metrics.ap75 == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.tp == 1 and metrics.fp == 0 and metrics.fn == 0


def test_evaluate_keypoints_missing_prediction_is_false_negative():
    gt_kpts = _make_person(100, 100)
    gt = {1: [Instance(image_id=1, keypoints=gt_kpts, area=5000.0)]}
    pred = {1: []}  # detector missed the person entirely

    metrics = evaluate_keypoints(pred, gt, score_threshold=0.3)

    assert metrics.ap == pytest.approx(0.0)
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.tp == 0 and metrics.fn == 1


def test_evaluate_keypoints_spurious_prediction_is_false_positive():
    gt = {1: []}  # no person in this image
    ghost_kpts = _make_person(300, 300)
    pred = {1: [Instance(image_id=1, keypoints=ghost_kpts, area=5000.0, score=0.9)]}

    metrics = evaluate_keypoints(pred, gt, score_threshold=0.3)

    assert metrics.fp == 1
    assert metrics.tp == 0
    # No GT instances at all => AP is undefined (nan) for this image, but
    # precision at the operating point must reflect the false alarm.
    assert metrics.precision == pytest.approx(0.0)


def test_evaluate_keypoints_low_confidence_prediction_excluded_from_precision():
    gt_kpts = _make_person(100, 100)
    gt = {1: [Instance(image_id=1, keypoints=gt_kpts, area=5000.0)]}
    pred = {1: [Instance(image_id=1, keypoints=gt_kpts.copy(), area=5000.0, score=0.1)]}

    metrics = evaluate_keypoints(pred, gt, score_threshold=0.3)

    # Below score_threshold => not counted at the operating point => FN.
    assert metrics.tp == 0 and metrics.fn == 1
    # But AP (which sweeps all confidence levels) still credits the perfect match.
    assert metrics.ap == pytest.approx(1.0)
