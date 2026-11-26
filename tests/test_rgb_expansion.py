"""Deployment regression checks for mixed-source labels, retention gates and benign fusion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.action.action_classifier import ACTION_CLASSES, ActionClassifier
from training.prepare_rgb_expansion import clip_windows, normal_mask, ucf_identity
from training.train_rgb_expansion import ProbabilityBlend, blend_probabilities, qualifies
from training.promote_rgb_expansion import add_policy
from src.risk.risk_engine import RiskEngine
from src.risk.threshold_manager import ThresholdManager
from src.temporal.temporal_features import KinematicFeatures
from src.anomaly.anomaly_detector import AnomalyResult
from src.temporal.lying_context import LyingContext, COLLAPSED, USING_PHONE
from src.action.action_classifier import ActionPrediction

POLICY = {"enabled": True, "min_probability": 0.9, "max_drop_severity": 0.35, "max_downward_velocity": 1.0}


def classifier(policy: dict | None, probs: np.ndarray) -> ActionClassifier:
    clf = ActionClassifier(device="cpu", normal_activity_policy=policy)
    clf.has_weights = clf.is_onnx = True
    clf.ort_session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input")],
                                      run=lambda *args: [np.log(probs[None])])
    return clf


def test_benign_lying_prediction_is_not_replaced_by_static_posture_prior() -> None:
    probs = np.full(10, 0.005, np.float32)
    probs[3], probs[1], probs[4] = 0.7, 0.2, 0.065
    probs /= probs.sum()
    kin = SimpleNamespace(drop_severity_score=0.0, vertical_velocity=0.0, torso_angle_curr=85,
                          aspect_ratio_curr=1.4, immobility_index=1.0)
    x = np.zeros((30, 17, 3), np.float32)
    assert classifier(None, probs).predict(x, kin).primary_action == "immobile"
    result = classifier(POLICY, probs).predict(x, kin)
    assert result.primary_action == "lying"
    assert result.probabilities["immobile"] < 0.01


def test_strong_drop_still_uses_physical_fall_evidence() -> None:
    probs = np.full(10, 0.005, np.float32)
    probs[3], probs[1] = 0.7, 0.26
    probs /= probs.sum()
    kin = SimpleNamespace(drop_severity_score=0.9, vertical_velocity=3.0, torso_angle_curr=85,
                          aspect_ratio_curr=1.4, immobility_index=0.0)
    clf = classifier(POLICY, probs)
    assert not clf._trust_normal_activity(probs, kin)
    assert clf.predict(np.zeros((30, 17, 3), np.float32), kin).primary_action == "falling"


def test_benign_model_evidence_does_not_escalate_to_an_alarm_when_context_is_missing() -> None:
    probs = np.full(10, 0.005, np.float32)
    probs[3], probs[1], probs[4] = 0.7, 0.2, 0.065
    probs /= probs.sum()
    kin = KinematicFeatures(0.0, 0.0, 1.4, 0.0, 85.0, 0.0, 1.0, 0.0)
    x = np.zeros((30, 17, 3), np.float32)
    old_pred = classifier(None, probs).predict(x, kin)
    new_pred = classifier(POLICY, probs).predict(x, kin)
    old, new = RiskEngine(ThresholdManager()), RiskEngine(ThresholdManager())
    old_alert = new_alert = False
    anomaly = AnomalyResult(0.3, False, 0.1, 0.22)
    for tick in range(100):
        old_alert |= old.assess(1, old_pred, anomaly, kin, tick / 15).should_alert
        new_alert |= new.assess(1, new_pred, anomaly, kin, tick / 15).should_alert
    assert old_alert
    assert not new_alert


def test_policy_does_not_override_distress_or_unknown_kinematics() -> None:
    probs = np.full(10, 0.01, np.float32); probs[8] = 0.91
    clf = classifier(POLICY, probs)
    assert not clf._trust_normal_activity(probs, None)
    assert not clf._trust_normal_activity(probs, SimpleNamespace(drop_severity_score=0.0, vertical_velocity=0.0))


def test_ucf_groups_and_safe_masks_preserve_meaning() -> None:
    assert ucf_identity("v_PushUps_g03_c04") == ("PushUps", "PushUps_g03")
    mask = normal_mask(["lying", "bending"])
    assert not mask[ACTION_CLASSES.index("falling")]
    assert mask.sum() == 2


def test_resampling_25fps_is_15hz_and_windows_are_bounded() -> None:
    raw = np.zeros((101, 17, 3), np.float32)
    raw[:, :, 0] = np.arange(101)[:, None]
    cfg = {"target_fps": 15, "window_size": 30, "windows_per_clip": 3}
    windows = clip_windows(raw, cfg, 25)
    assert len(windows) == 3
    assert all(len(w) == 30 for w in windows)
    assert windows[-1][-1, 0, 0] == 100
    assert np.diff(windows[-1][:, 0, 0]).max() <= 2


def test_probability_blend_exports_standard_logits() -> None:
    class Constant(torch.nn.Module):
        def __init__(self, logits: list[float]) -> None:
            super().__init__(); self.register_buffer("logits", torch.tensor(logits))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.logits.expand(len(x), -1)
    a, b = Constant([2.0, 0.0]), Constant([0.0, 2.0])
    x = torch.zeros(3, 1)
    output = ProbabilityBlend(a, b, 0.25)(x).softmax(-1)
    torch.testing.assert_close(output, 0.75 * a(x).softmax(-1) + 0.25 * b(x).softmax(-1))


def test_posture_calibration_does_not_swap_lying_into_immobility() -> None:
    a = np.full((2, 10), 0.01, dtype=np.float32)
    b = np.full((2, 10), 0.01, dtype=np.float32)
    a[:,3], a[:,9] = 0.7, 0.2
    b[:,3], b[:,9] = 0.01, 0.89
    a /= a.sum(1,keepdims=True); b /= b.sum(1,keepdims=True)
    p = blend_probabilities(a,b,0.5,True)
    np.testing.assert_allclose(p[:,3]/p[:,9], a[:,3]/a[:,9])
    np.testing.assert_allclose(p.sum(1), 1)
    assert np.all(p.argmax(1) == 3)


def test_observed_fall_then_stillness_is_not_erased_by_benign_posture_model() -> None:
    engine = RiskEngine(ThresholdManager())
    kin = KinematicFeatures(0.0,0.0,1.4,0.0,85.0,0.0,1.0,0.0)
    probs = np.full(10,0.005,np.float32);probs[3]=0.955;probs/=probs.sum()
    pred = classifier(POLICY,probs).predict(np.zeros((30,17,3),np.float32),kin)
    anomaly = AnomalyResult(0.3,False,0.1,0.22)
    fallen = LyingContext(COLLAPSED,"sudden",False,5.0,False,0.0,onset_reason="fast_drop")
    assert engine._fall_score(pred,anomaly,kin,fallen) >= engine.thresholds.thresh_high_risk
    awake = LyingContext(USING_PHONE,"sudden",False,5.0,False,0.0,onset_reason="fast_drop")
    assert engine._fall_score(pred,anomaly,kin,awake) < engine.thresholds.thresh_high_risk


def test_real_drop_history_still_alarms_when_model_calls_postfall_pose_lying(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_lying_safe_scope import ScopeScene
    scene = ScopeScene()
    original = scene.engine.assess
    def assess(track_id: int, action: ActionPrediction, *args: object, **kwargs: object) -> object:
        if action.primary_action == "immobile":
            action = ActionPrediction("lying",0.99,{"lying":0.99,"immobile":0.01},0.7525,0.6)
        return original(track_id,action,*args,**kwargs)
    monkeypatch.setattr(scene.engine,"assess",assess)
    scene.stand(3.0)
    scene.collapse()
    scene.lie_still(10.0)
    assert "fall" in scene.fired


def test_retention_gate_rejects_lost_falls_even_when_normal_fpr_improves() -> None:
    baseline = {"legacy": {"fall_recall": 0.98, "emergency_recall": 0.85, "normal_fpr": 0.12, "macro_f1": 0.8},
                "new_normal_fpr": 0.3, "upfall_impact_recall": 0.6}
    candidate = {"legacy": {**baseline["legacy"], "fall_recall": 0.9, "normal_fpr": 0.01},
                 "new_normal_fpr": 0.01, "upfall_impact_recall": 0.8}
    limits = {"max_fall_recall_drop": 0.01, "max_emergency_recall_drop": 0.02,
              "max_normal_fpr_increase": 0.0, "max_macro_f1_drop": 0.01, "max_upfall_recall_drop": 0.0}
    assert not qualifies(candidate, baseline, limits)


def test_integration_policy_preserves_other_configuration_and_comments() -> None:
    original = "# preserve this\npose:\n  architecture: rtmo_s\ntemporal:\n  weights_path: model.onnx\nanomaly:\n  threshold: 0.3\n"
    result = add_policy(original, POLICY)
    before, after = yaml.safe_load(original), yaml.safe_load(result)
    assert after["pose"] == before["pose"]
    assert after["anomaly"] == before["anomaly"]
    assert after["temporal"]["normal_activity_policy"] == POLICY
    assert "# preserve this" in result
