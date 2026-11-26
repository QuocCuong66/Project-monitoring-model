"""Unit tests for the window-level skeleton normalization shared by training and inference."""

import numpy as np

from src.temporal.sequence_features import build_model_sequence


def _standing(hip_x: float, hip_y: float, torso: float) -> np.ndarray:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[:, 0] = hip_x
    kp[:, 1] = hip_y - torso  # upper body around the shoulders
    kp[[11, 12], 1] = hip_y
    kp[11, 0], kp[12, 0] = hip_x - 0.1 * torso, hip_x + 0.1 * torso
    kp[5, 0], kp[6, 0] = hip_x - 0.16 * torso, hip_x + 0.16 * torso
    kp[[15, 16], 1] = hip_y + 2 * torso
    return kp


def test_output_shape_and_front_padding() -> None:
    raw = np.stack([_standing(100, 200, 50)] * 5)
    seq = build_model_sequence(raw, 30)
    assert seq.shape == (30, 17, 3)
    np.testing.assert_allclose(seq[0], seq[24])


def test_keeps_whole_body_drop_unlike_per_frame_centering() -> None:
    # Hip falls by two torso lengths across the window: the drop must remain visible.
    frames = [_standing(100, 200 + i * (100 / 29), 50) for i in range(30)]
    seq = build_model_sequence(np.stack(frames), 30)
    hip_start = seq[0, [11, 12], 1].mean()
    hip_end = seq[-1, [11, 12], 1].mean()
    assert hip_start == 0.0
    np.testing.assert_allclose(hip_end, 2.0, atol=1e-4)


def test_scale_and_translation_invariant() -> None:
    a = np.stack([_standing(100, 200, 50)] * 30)
    b = np.stack([_standing(640, 90, 120)] * 30)
    np.testing.assert_allclose(build_model_sequence(a, 30), build_model_sequence(b, 30), atol=1e-5)


def test_low_confidence_joints_zeroed() -> None:
    raw = np.stack([_standing(100, 200, 50)] * 30)
    raw[:, 0, 2] = 0.05
    seq = build_model_sequence(raw, 30)
    assert np.all(seq[:, 0] == 0.0)


def test_empty_history_returns_zeros() -> None:
    assert np.all(build_model_sequence(np.zeros((0, 17, 3)), 30) == 0.0)
