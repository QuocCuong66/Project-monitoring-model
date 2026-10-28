"""Kinematic fusion must not drown upright stroke / acute-illness warning signs."""

from types import SimpleNamespace

import numpy as np

from src.action.action_classifier import ACTION_CLASSES, ActionClassifier

UPRIGHT = SimpleNamespace(drop_severity_score=0.0, torso_angle_curr=5.0, aspect_ratio_curr=0.4,
                          immobility_index=0.0, vertical_velocity=0.0)


def _probs(**kw: float) -> np.ndarray:
    p = np.full(len(ACTION_CLASSES), 0.01, dtype=np.float32)
    for name, v in kw.items():
        p[ACTION_CLASSES.index(name)] = v
    return p / p.sum()


def test_upright_prior_keeps_clutching_head_dominant() -> None:
    clf = ActionClassifier(weights_path=None)
    fused = clf._fuse_kinematics_with_probs(_probs(abnormal_movement=0.85), UPRIGHT)
    assert ACTION_CLASSES[int(np.argmax(fused))] == "abnormal_movement"
    assert fused[ACTION_CLASSES.index("abnormal_movement")] > 0.7


def test_upright_prior_still_applies_to_ordinary_standing() -> None:
    clf = ActionClassifier(weights_path=None)
    fused = clf._fuse_kinematics_with_probs(_probs(bending=0.4, standing=0.3), UPRIGHT)
    assert ACTION_CLASSES[int(np.argmax(fused))] == "standing"
