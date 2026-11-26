"""3D benchmark preprocessing preserves depth and recording identities."""

import numpy as np
import pytest

from training.train_ntu3d import normalize_3d, subject_id


def skeleton() -> np.ndarray:
    x = np.ones((1, 8, 25, 3), dtype=np.float32)
    x[..., 2] = 3.0
    x[:, :, 20, 1] += 0.5
    return x


def test_ntu_actor_id_is_parsed_without_mixing_camera_or_setup() -> None:
    assert subject_id("S017C003P025R002A043") == 25
    with pytest.raises(ValueError):
        subject_id("not-an-ntu-clip")


def test_3d_translation_and_scale_normalization() -> None:
    a = skeleton()
    b = a * 2 + np.array([10, 2, 5])
    np.testing.assert_allclose(normalize_3d(a, 30, 5, 0.05), normalize_3d(b, 30, 5, 0.05), atol=1e-5)


def test_depth_motion_is_preserved_and_not_treated_as_confidence() -> None:
    x = skeleton()
    x[0, :, :, 2] += np.arange(8)[:, None] * 0.1
    result = normalize_3d(x, 30, 5, 0.05)
    assert result.shape == (30, 75)
    # Last root z differs from its first value by 0.7 m / 0.5 m torso length.
    assert result[-1, 2] == pytest.approx(1.4)


def test_missing_body_is_rejected_and_17_joint_input_is_not_silently_used() -> None:
    assert normalize_3d(np.zeros((1, 8, 25, 3)), 30, 5, 0.05) is None
    assert normalize_3d(np.zeros((0, 8, 25, 3)), 30, 5, 0.05) is None
    with pytest.raises(ValueError, match="25 joints"):
        normalize_3d(np.zeros((1, 8, 17, 3)), 30, 5, 0.05)
