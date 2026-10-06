"""Supervised action classification for 10 elderly behavior categories."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import torch
    from src.temporal.temporal_model import SpatialTemporalTransformer, TCNSequenceClassifier
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    SpatialTemporalTransformer = None
    TCNSequenceClassifier = None
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

from src.utils.config import resolve_model_path
from src.utils.ort_providers import dml_provider

ACTION_CLASSES: List[str] = [
    "walking",            # 0
    "standing",           # 1
    "sitting",            # 2
    "lying",              # 3
    "bending",            # 4
    "falling",            # 5
    "getting_up",         # 6
    "stumbling",          # 7
    "abnormal_movement",  # 8
    "immobile",           # 9
]

ACTION_SEVERITY: Dict[str, float] = {
    "walking": 0.0,
    "standing": 0.0,
    "sitting": 0.0,
    "getting_up": 0.0,
    "bending": 0.15,
    "lying": 0.60,
    "stumbling": 0.65,
    "abnormal_movement": 0.80,
    "falling": 0.95,
    "immobile": 0.98,
}


@dataclass
class ActionPrediction:
    """Predicted action probabilities and severity."""
    primary_action: str
    confidence: float
    probabilities: Dict[str, float]
    emergency_prob: float  # Multi-class emergency probability (falling, immobile, lying, etc.)
    severity_score: float


class ActionClassifier:
    """Classifies temporal skeleton sequences into one of 10 human action classes."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        num_classes: int = 10,
        architecture: str = "tcn",
    ) -> None:
        self.num_classes = num_classes
        self.architecture = architecture.lower()
        self.has_weights = False
        self.ort_session = None
        self.is_onnx = False
        self.model = None

        resolved_weights = None
        if weights_path:
            p = resolve_model_path(weights_path)
            if p and p.exists():
                resolved_weights = p
            else:
                p_onnx = resolve_model_path(str(weights_path).replace(".pt", ".onnx"))
                if p_onnx and p_onnx.exists():
                    resolved_weights = p_onnx

        # 1. Try ONNX Runtime first if weights are .onnx
        if resolved_weights and str(resolved_weights).endswith(".onnx") and ORT_AVAILABLE:
            try:
                providers = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
                available = ort.get_available_providers()
                actual_providers = [dml_provider() if p == "DmlExecutionProvider" else p for p in providers if p in available]
                sess_options = ort.SessionOptions()
                if actual_providers[0] == "CPUExecutionProvider":
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                self.ort_session = ort.InferenceSession(str(resolved_weights), sess_options=sess_options, providers=actual_providers)
                self.is_onnx = True
                self.has_weights = True
            except Exception as e:
                print(f"[Warning] Failed to load ONNX action weights from {resolved_weights}: {e}")

        # 2. If not ONNX, fallback to PyTorch if available
        if not self.has_weights and TORCH_AVAILABLE:
            dev_str = device if torch.cuda.is_available() and device == "cuda" else "cpu"
            self.device = torch.device(dev_str)
            if self.architecture == "tcn" and TCNSequenceClassifier is not None:
                self.model = TCNSequenceClassifier(input_dim=51, num_classes=num_classes)
            elif SpatialTemporalTransformer is not None:
                self.model = SpatialTemporalTransformer(input_dim=51, num_classes=num_classes)

            if resolved_weights and Path(resolved_weights).exists():
                try:
                    ckpt = torch.load(resolved_weights, map_location=self.device, weights_only=False)
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        self.model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        self.model.load_state_dict(ckpt)
                    self.has_weights = True
                except Exception as e:
                    print(f"[Warning] Failed to load action weights from {resolved_weights}: {e}")

            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()

    def predict(self, sequence: np.ndarray, kinematics: Optional[Any] = None) -> ActionPrediction:
        """Predict action for a normalized sequence of shape (T, 17, 3), fused with kinematics.

        Args:
            sequence: Array of shape (T, 17, 3) where each row is (x, y, conf).
            kinematics: Optional KinematicFeatures providing physical ground-truth indicators.

        Returns:
            ActionPrediction instance.
        """
        probs = None
        # If model weights exist, run forward pass
        if self.has_weights:
            try:
                if self.is_onnx and self.ort_session is not None:
                    inp = sequence[np.newaxis, ...].astype(np.float32)
                    input_name = self.ort_session.get_inputs()[0].name
                    logits = self.ort_session.run(None, {input_name: inp})[0]
                    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    probs = (exp_l / np.sum(exp_l, axis=-1, keepdims=True))[0]
                elif TORCH_AVAILABLE and self.model is not None:
                    with torch.no_grad():
                        tensor_seq = torch.from_numpy(sequence).unsqueeze(0).to(self.device)  # (1, T, 17, 3)
                        logits = self.model(tensor_seq)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            except Exception:
                probs = None

        if probs is not None:
            conf = float(np.max(probs))
            # If the neural net is unconfident (e.g. uniform ~0.10 across classes),
            # blend with or defer to biomechanical kinematic estimation
            if conf < 0.35:
                heuristic_probs = self._heuristic_predict(sequence, kinematics)
                probs = 0.85 * heuristic_probs + 0.15 * probs
            elif kinematics is not None:
                probs = self._fuse_kinematics_with_probs(probs, kinematics)
        else:
            probs = self._heuristic_predict(sequence, kinematics)

        # Normalize
        s = float(np.sum(probs))
        probs = (probs / s) if s > 0 else (np.ones(self.num_classes, dtype=np.float32) / self.num_classes)

        top_idx = int(np.argmax(probs))
        primary = ACTION_CLASSES[top_idx]
        conf = float(probs[top_idx])

        prob_dict = {ACTION_CLASSES[i]: float(probs[i]) for i in range(len(ACTION_CLASSES))}
        emergency_prob = float(
            prob_dict.get("falling", 0.0) * 1.0 +
            prob_dict.get("immobile", 0.0) * 1.0 +
            prob_dict.get("lying", 0.0) * 0.75 +
            prob_dict.get("abnormal_movement", 0.0) * 0.70 +
            prob_dict.get("stumbling", 0.0) * 0.50
        )
        severity = float(ACTION_SEVERITY.get(primary, 0.0))

        return ActionPrediction(
            primary_action=primary,
            confidence=conf,
            probabilities=prob_dict,
            emergency_prob=min(1.0, emergency_prob),
            severity_score=severity,
        )

    def _fuse_kinematics_with_probs(self, probs: np.ndarray, kinematics: Any) -> np.ndarray:
        """Physically calibrate model probabilities using biomechanical ground-truth."""
        fused = probs.copy()
        drop_sev = getattr(kinematics, "drop_severity_score", 0.0)
        torso_angle = getattr(kinematics, "torso_angle_curr", 0.0)
        ar = getattr(kinematics, "aspect_ratio_curr", 0.5)
        if not getattr(kinematics, "lower_body_visible", True):
            ar = 0.5  # upper-body-only bbox: its width says nothing about lying down
        immob = getattr(kinematics, "immobility_index", 0.0)
        vy = getattr(kinematics, "vertical_velocity", 0.0)

        # 0. Legs still carry the body (picking something up, squatting, kneeling): bending, not a fall
        if getattr(kinematics, "feet_supported", False):
            fused[4] = max(fused[4], 0.70)         # bending
            fused[2] = max(fused[2], 0.15)         # sitting / squatting
            for idx in (3, 5, 9):                  # lying, falling, immobile
                fused[idx] = min(fused[idx], 0.02)
            s = float(np.sum(fused))
            return (fused / s) if s > 0 else probs

        # 1. Sudden Drop / Fall Event Confirmation
        if drop_sev >= 0.50 or (vy >= 2.0 and torso_angle >= 50.0):
            fall_boost = min(0.95, 0.45 + 0.50 * drop_sev)
            fused[5] = max(fused[5], fall_boost)   # falling
            fused[7] = max(fused[7], 0.35)         # stumbling
            fused[4] = min(fused[4], 0.10)         # suppress bending

        # 2. Horizontal Body Posture on Floor
        if torso_angle >= 60.0 or ar >= 1.05:
            if immob >= 0.50:
                fused[9] = max(fused[9], 0.88)     # immobile
                fused[3] = max(fused[3], 0.30)     # lying
                fused[4] = min(fused[4], 0.05)     # suppress bending
            else:
                fused[3] = max(fused[3], 0.78)     # lying
                if drop_sev >= 0.35:
                    fused[5] = max(fused[5], 0.65) # falling
                fused[4] = min(fused[4], 0.08)

        # 3. Upright Normal Posture (Standing / Walking)
        # Staggering and clutching head / chest (stroke & acute illness warning signs) happen
        # while upright, so the upright prior is skipped when the model is confident about them.
        elif torso_angle < 25.0 and ar < 0.60 and drop_sev < 0.15 and probs[7] + probs[8] < 0.5:
            fused[1] = max(fused[1], 0.65)         # standing
            fused[0] = max(fused[0], 0.25)         # walking
            fused[5] = min(fused[5], 0.02)         # clear fall
            fused[3] = min(fused[3], 0.02)         # clear lying

        # 4. Standard Upright Bending (e.g. picking up item from floor, AR stays upright < 0.85)
        elif 25.0 <= torso_angle < 55.0 and ar < 0.85 and drop_sev < 0.15:
            fused[4] = max(fused[4], 0.65)         # bending
            fused[2] = max(fused[2], 0.20)         # sitting
            fused[5] = min(fused[5], 0.02)         # no fall

        s = float(np.sum(fused))
        return (fused / s) if s > 0 else probs

    def _heuristic_predict(self, sequence: np.ndarray, kinematics: Optional[Any] = None) -> np.ndarray:
        """Physical heuristic classifier based on skeleton posture & vertical movement."""
        probs = np.zeros(self.num_classes, dtype=np.float32)

        # If kinematics is available, prioritize direct physical measurements
        if kinematics is not None:
            torso_angle = getattr(kinematics, "torso_angle_curr", 0.0)
            ar = getattr(kinematics, "aspect_ratio_curr", 0.5)
            if not getattr(kinematics, "lower_body_visible", True):
                ar = 0.5  # upper-body-only bbox: its width says nothing about lying down
            drop_sev = getattr(kinematics, "drop_severity_score", 0.0)
            immob = getattr(kinematics, "immobility_index", 0.0)
            vy = getattr(kinematics, "vertical_velocity", 0.0)

            if getattr(kinematics, "feet_supported", False):
                probs[4] = 0.75  # bending (legs still carry the body)
                probs[2] = 0.15  # squatting / sitting
                probs[1] = 0.10
            elif drop_sev >= 0.50 or (vy >= 2.0 and torso_angle >= 50.0):
                probs[5] = 0.85  # falling
                probs[7] = 0.10  # stumbling
                probs[3] = 0.05  # lying
            elif torso_angle >= 60.0 or ar >= 1.05:
                if immob >= 0.50:
                    probs[9] = 0.85  # immobile
                    probs[3] = 0.12  # lying
                else:
                    probs[3] = 0.75  # lying
                    probs[5] = 0.15 if drop_sev >= 0.3 else 0.05
                    probs[9] = 0.10
            elif 25.0 <= torso_angle < 55.0 and ar < 0.85 and drop_sev < 0.15:
                probs[4] = 0.70  # bending
                probs[2] = 0.20  # sitting
                probs[0] = 0.10  # walking
            elif torso_angle < 25.0 and ar < 0.60:
                probs[1] = 0.70  # standing
                probs[0] = 0.25  # walking
                probs[2] = 0.05
            else:
                probs[0] = 0.40  # walking
                probs[1] = 0.40  # standing
                probs[2] = 0.20
            return probs

        # Fallback when kinematics is None: calculate from raw sequence
        last_frame = sequence[-1]
        coords = last_frame[:, :2]
        confs = last_frame[:, 2]

        valid = confs > 0.2
        if np.any(valid):
            pts = coords[valid]
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            h = max(0.1, max_y - min_y)
            w = max(0.1, max_x - min_x)
            ar = w / h
        else:
            ar = 0.5

        motion_var = float(np.mean(np.var(sequence[:, :, :2], axis=0)))

        # Approximate torso angle from shoulders and hips if available
        if confs[5] > 0.2 and confs[6] > 0.2 and confs[11] > 0.2 and confs[12] > 0.2:
            mid_sh = (coords[5] + coords[6]) / 2.0
            mid_hip = (coords[11] + coords[12]) / 2.0
            dy = mid_hip[1] - mid_sh[1]
            dx = mid_hip[0] - mid_sh[0]
            approx_angle = float(np.degrees(np.abs(np.arctan2(dx, dy))))
        else:
            approx_angle = 0.0

        if ar >= 1.05 or approx_angle >= 55.0:
            if motion_var < 0.008:
                probs[9] = 0.85  # immobile
                probs[3] = 0.10  # lying
            else:
                probs[3] = 0.70  # lying
                probs[5] = 0.20  # falling
        elif ar < 0.60 and approx_angle < 25.0:
            probs[1] = 0.60  # standing
            probs[0] = 0.35  # walking
        elif approx_angle < 50.0 and ar < 0.85:
            probs[4] = 0.65  # bending
            probs[2] = 0.25  # sitting
        else:
            probs[0] = 0.40
            probs[1] = 0.40
            probs[4] = 0.20

        s = float(np.sum(probs))
        return probs / s if s > 0 else np.ones(self.num_classes, dtype=np.float32) / self.num_classes
