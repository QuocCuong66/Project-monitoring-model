"""Threshold and configuration manager for risk engine and state machine."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from src.utils.config import load_config


class ThresholdManager:
    """Provides validated risk weights, state machine thresholds, and alert cooldown settings."""

    def __init__(self, config_or_path: Optional[Union[Dict[str, Any], str, Path]] = None) -> None:
        if isinstance(config_or_path, dict):
            cfg = config_or_path
        elif isinstance(config_or_path, (str, Path)):
            from src.utils.config import load_yaml
            cfg = load_yaml(config_or_path)
        else:
            try:
                cfg = load_config("thresholds.yaml")
            except Exception:
                cfg = {}

        # Risk weights (normalized to sum to 1.0)
        weights = cfg.get("risk_weights", {})
        w_act = float(weights.get("action_prob", 0.40))
        w_anom = float(weights.get("anomaly_score", 0.25))
        w_kin = float(weights.get("kinematics", 0.20))
        w_immob = float(weights.get("immobility", 0.15))

        total_w = w_act + w_anom + w_kin + w_immob
        if total_w > 0:
            self.w_action = w_act / total_w
            self.w_anomaly = w_anom / total_w
            self.w_kinematics = w_kin / total_w
            self.w_immobility = w_immob / total_w
        else:
            self.w_action = 0.40
            self.w_anomaly = 0.25
            self.w_kinematics = 0.20
            self.w_immobility = 0.15

        # State machine thresholds
        sm = cfg.get("state_machine", {})
        self.thresh_suspicious = float(sm.get("thresh_suspicious", 0.55))
        self.thresh_abnormal = float(sm.get("thresh_abnormal", 0.72))
        self.thresh_high_risk = float(sm.get("thresh_high_risk", 0.85))

        # Multi-signal escalation: both the physical drop and the action model must reach this level
        # before their average can override the weighted score (a single noisy signal cannot).
        self.synergy_min_signal = float(sm.get("synergy_min_signal", 0.55))
        # Immobility index a person on the floor must reach before the "lying still" escalation applies
        self.immobility_synergy_min = float(sm.get("immobility_synergy_min", 0.60))

        # Temporal debounce durations (in seconds)
        self.min_duration_suspicious = float(sm.get("min_duration_suspicious", 0.5))
        self.min_duration_abnormal = float(sm.get("min_duration_abnormal", 1.5))
        self.min_duration_high_risk = float(sm.get("min_duration_high_risk", 2.5))
        self.min_recovery_duration = float(sm.get("min_recovery_duration", 3.0))

        # Monitoring scope: anyone not lying (standing, walking, sitting) is monitored; anyone lying without
        # a fall / collapse / convulsion (already lying when first seen, lay down deliberately) is shown as safe
        scope = cfg.get("monitoring_scope", {}) or {}
        self.safe_when_lying = bool(scope.get("safe_when_lying", False))
        self.fall_onset_reasons: tuple[str, ...] = tuple(
            str(r) for r in scope.get("fall_onset_reasons", (
                "fast_drop", "upright_to_floor", "unsteady_before", "collapse_from_sitting", "convulsion"))
        )

        # Lying context (sleep / phone / turning over vs warning signs); parsed by LyingContextConfig
        self.lying_context: Dict[str, Any] = dict(cfg.get("lying_context", {}) or {})
        # Convulsion (rhythmic jerking) detector; parsed by ConvulsionConfig
        self.convulsion: Dict[str, Any] = dict(cfg.get("convulsion", {}) or {})
        # Stroke / acute-illness warning-sign stream; parsed by StrokeSignConfig
        self.stroke_signs: Dict[str, Any] = dict(cfg.get("stroke_signs", {}) or {})

        # Alerts & Cooldown
        al = cfg.get("alerts", {})
        self.cooldown_seconds = float(al.get("cooldown_seconds", 60.0))
        self.non_diagnostic_message = str(al.get(
            "non_diagnostic_message",
            "Possible medical emergency / abnormal behavior detected. Please check the person."
        ))
