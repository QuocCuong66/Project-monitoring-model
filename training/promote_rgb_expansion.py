"""Install a validated COCO-17 ONNX model with a rollback copy and runtime smoke check."""

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.action.action_classifier import ACTION_CLASSES, ActionClassifier  # noqa: E402
from training.train_rgb_expansion import ProbabilityBlend, load_model, qualifies  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_policy(text: str, policy: dict) -> str:
    cfg = yaml.safe_load(text)
    if cfg["temporal"].get("normal_activity_policy") is not None:
        if cfg["temporal"]["normal_activity_policy"] != policy:
            raise ValueError("Existing normal activity policy differs; do not overwrite it implicitly")
        return text
    before, after = text.split("\nanomaly:", 1)
    block = yaml.safe_dump({"normal_activity_policy": policy}, sort_keys=False)
    indented = "\n".join("  " + line for line in block.rstrip().splitlines())
    return before.rstrip() + "\n" + indented + "\n\nanomaly:" + after


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/rgb_expansion.yaml").read_text(encoding="utf-8"))
    cfg, integration = config["training"], config["integration"]
    checkpoint = ROOT / cfg["output"]
    artifact = checkpoint.with_suffix(".onnx")
    report = json.loads(checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    selected = report["selected"]
    if selected is None or not qualifies(selected["validation"], report["baseline_validation"], cfg["monitor"]):
        raise ValueError("Candidate failed validation retention gates")
    if report["candidate_test"]["new_normal_fpr"] > report["baseline_test"]["new_normal_fpr"]:
        raise ValueError("Held-out normal activity regression: do not activate")
    if report["candidate_test"]["upfall_impact_recall"] < report["baseline_test"]["upfall_impact_recall"]:
        raise ValueError("Held-out impact recall regressed: do not activate")
    model_config = ROOT / "configs/model.yaml"
    original_config = model_config.read_text(encoding="utf-8")
    current = yaml.safe_load(original_config)
    if current["temporal"]["weights_path"] != integration["active_weights"]:
        raise ValueError("Active model path changed since experiment configuration")
    active = ROOT / integration["active_weights"]
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[1:] != [30, 17, 3] or session.get_outputs()[0].shape[-1] != len(ACTION_CLASSES):
        raise ValueError("Model is not live COCO-17 / 10-class compatible")
    with np.load(ROOT / "data/processed/rgb_expansion/ntu60.npz", allow_pickle=False) as data:
        sample = data["x"][np.flatnonzero(data["split"] == 1)[:16]]
    torch.set_num_threads(2)
    baseline = load_model(ROOT / cfg["baseline"], torch.device("cpu")).eval()
    candidate = load_model(checkpoint, torch.device("cpu")).eval()
    reference = ProbabilityBlend(baseline, candidate, selected["alpha"], cfg.get("preserve_posture_ratio", False)).eval()
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    with torch.no_grad():
        expected = reference(torch.from_numpy(sample)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    old_session = ort.InferenceSession(str(active), sess_options=options, providers=["CPUExecutionProvider"])
    old_logits = old_session.run(None, {old_session.get_inputs()[0].name: sample})[0]
    with torch.no_grad():
        np.testing.assert_allclose(old_logits, baseline(torch.from_numpy(sample)).numpy(), rtol=1e-4, atol=1e-5)
    backup = ROOT / integration["backup_dir"] / datetime.now(timezone.utc).strftime("rgb_%Y%m%dT%H%M%SZ")
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(active, backup / active.name)
    shutil.copy2(model_config, backup / model_config.name)
    old_sha = sha(active)
    # Close read-only preflight sessions before replacing a model file on Windows.
    del session, old_session
    staged = active.with_suffix(".onnx.new")
    try:
        shutil.copy2(artifact, staged)
        staged.replace(active)
        staged_config = model_config.with_suffix(".yaml.new")
        staged_config.write_text(add_policy(original_config, integration["normal_activity_policy"]), encoding="utf-8")
        staged_config.replace(model_config)
        runtime = ActionClassifier(weights_path=active, architecture=current["temporal"]["architecture"],
                                   device="cuda", normal_activity_policy=integration["normal_activity_policy"])
        if not runtime.has_weights or not runtime.is_onnx:
            raise RuntimeError("Runtime could not load the installed ONNX")
        device_logits = runtime.ort_session.run(None, {runtime.ort_session.get_inputs()[0].name: sample[:1]})[0]
        np.testing.assert_allclose(device_logits, actual[:1], rtol=1e-3, atol=1e-4)
        result = runtime.predict(sample[0])
        if not np.isfinite(list(result.probabilities.values())).all():
            raise RuntimeError("Runtime returned invalid probabilities")
        deployed = {"active": str(active), "sha256": sha(active), "previous_sha256": old_sha,
                    "backup": str(backup), "checkpoint": str(checkpoint), "selection": selected,
                    "normal_activity_policy": integration["normal_activity_policy"],
                    "providers": runtime.ort_session.get_providers(), "runtime_smoke_action": result.primary_action,
                    "classes": ACTION_CLASSES, "onnx_parity_max_error": float(np.abs(actual-expected).max()),
                    "runtime_provider_max_error": float(np.abs(device_logits-actual[:1]).max()),
                    "limitations": "Window metrics and limited fall datasets, not proof of overnight false-alarm performance"}
        checkpoint.with_suffix(".deployment.json").write_text(json.dumps(deployed, indent=2), encoding="utf-8")
        report["runtime_replaced"] = True
        report["deployment"] = deployed
        checkpoint.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(deployed, indent=2), flush=True)
    except Exception:
        shutil.copy2(backup / active.name, active)
        shutil.copy2(backup / model_config.name, model_config)
        raise


if __name__ == "__main__":
    main()
