"""Activate only an evaluated, runtime-compatible diverse-v2 candidate; rollback on failure."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.action.action_classifier import ACTION_CLASSES, ActionClassifier  # noqa: E402
from training.train_diverse_v2 import current_teacher, passes  # noqa: E402
from training.train_rgb_expansion import ProbabilityBlend, load_model  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def heldout_safe(candidate: dict, baseline: dict) -> bool:
    """The external multi-person OOPS sample is diagnostic, not a tuning target."""
    if candidate["old"]["new_normal_fpr"] > baseline["old"]["new_normal_fpr"]:
        return False
    if candidate["old"]["upfall_impact_recall"] < baseline["old"]["upfall_impact_recall"]:
        return False
    sources = [name for name in baseline["native"]["sources"] if name != "oops"]
    for key, direction in (("normal_fpr", -1), ("fall_recall", 1)):
        names = [n for n in sources if baseline["native"]["sources"][n][key] is not None]
        before = np.mean([baseline["native"]["sources"][n][key] for n in names])
        after = np.mean([candidate["native"]["sources"][n][key] for n in names])
        if direction*(after-before) < 0:
            return False
    return True


def main() -> None:
    config = yaml.safe_load((ROOT/"configs/diverse_v2.yaml").read_text(encoding="utf-8"))
    checkpoint = ROOT/config["training"]["output"]
    report_file = checkpoint.with_suffix(".metrics.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    selected = report["selected"]
    if selected is None or not passes(selected["validation"], report["baseline_validation"], config["training"]):
        raise ValueError("Validation gates failed; active model retained")
    if not heldout_safe(report["candidate_test"], report["baseline_test"]):
        raise ValueError("Held-out retention gates failed; active model retained")
    replay = json.loads(checkpoint.with_suffix(".replay.json").read_text(encoding="utf-8"))
    if replay["observed_minutes"] <= 0 or replay["alerts"]["candidate"] > replay["alerts"]["baseline"]:
        raise ValueError("Benign pose replay failed; active model retained")
    active = ROOT/config["integration"]["active_weights"]
    folder = ROOT/config["integration"]["baseline_dir"]
    if sha(active) != report["baseline_sha256"]:
        raise ValueError("Active model changed since baseline snapshot")
    artifact = checkpoint.with_suffix(".onnx")
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    cpu = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
    if cpu.get_inputs()[0].shape[1:] != [30, 17, 3] or cpu.get_outputs()[0].shape[-1] != len(ACTION_CLASSES):
        raise ValueError("Runtime interface mismatch")
    with np.load(ROOT/"data/processed/rgb_expansion/ntu60.npz", allow_pickle=False) as data:
        sample = data["x"][np.flatnonzero(data["split"] == 1)[:16]]
    torch.set_num_threads(2)
    device = torch.device("cpu")
    reference = ProbabilityBlend(current_teacher(folder, device), load_model(checkpoint, device),
                                 selected["alpha"], config["training"]["preserve_posture_ratio"]).eval()
    with torch.no_grad():
        expected = reference(torch.from_numpy(sample)).numpy()
    actual = cpu.run(None, {cpu.get_inputs()[0].name: sample})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    current = yaml.safe_load((ROOT/"configs/model.yaml").read_text(encoding="utf-8"))
    if current["temporal"]["weights_path"] != config["integration"]["active_weights"]:
        raise ValueError("Configured active model path changed")
    runtime = ActionClassifier(weights_path=artifact, architecture=current["temporal"]["architecture"],
                               device="cuda", normal_activity_policy=current["temporal"].get("normal_activity_policy"))
    if not runtime.has_weights or not runtime.is_onnx:
        raise RuntimeError("Candidate cannot load in runtime")
    session = runtime.ort_session
    if "DmlExecutionProvider" not in session.get_providers():
        raise RuntimeError("Expected GPU provider unavailable")
    device_logits = session.run(None, {session.get_inputs()[0].name: sample[:1]})[0]
    np.testing.assert_allclose(device_logits, actual[:1], atol=1e-4, rtol=1e-3)
    elapsed = []
    for index in range(65):
        start = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: sample[index % len(sample):index % len(sample)+1]})
        if index >= 5:
            elapsed.append((time.perf_counter()-start)*1000)
    p95 = float(np.percentile(elapsed, 95))
    if p95 > config["integration"]["max_model_p95_ms"]:
        raise RuntimeError(f"Candidate exceeds inference budget: {p95:.2f} ms")
    backup = ROOT/"models/checkpoints/backups"/datetime.now(timezone.utc).strftime("diverse_v2_%Y%m%dT%H%M%SZ")
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(active, backup/active.name)
    shutil.copy2(ROOT/"configs/model.yaml", backup/"model.yaml")
    staged = active.with_suffix(".onnx.new")
    try:
        shutil.copy2(artifact, staged)
        staged.replace(active)
        smoke = ActionClassifier(weights_path=active, architecture=current["temporal"]["architecture"],
                                 device="cuda", normal_activity_policy=current["temporal"].get("normal_activity_policy"))
        logits = smoke.ort_session.run(None, {smoke.ort_session.get_inputs()[0].name: sample[:1]})[0]
        np.testing.assert_allclose(logits, actual[:1], atol=1e-4, rtol=1e-3)
        deployment = {"active": str(active), "sha256": sha(active), "previous_sha256": report["baseline_sha256"],
                      "backup": str(backup), "selection": selected, "providers": session.get_providers(),
                      "onnx_parity_max_error": float(np.abs(actual-expected).max()), "p95_ms": p95,
                      "median_ms": float(np.median(elapsed)), "bytes": active.stat().st_size,
                      "limitations": "Window metrics; short clips cannot establish overnight sleep false-alarm rates."}
        checkpoint.with_suffix(".deployment.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")
        report["runtime_replaced"], report["deployment"] = True, deployment
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(deployment, indent=2), flush=True)
    except Exception:
        shutil.copy2(backup/active.name, active)
        raise


if __name__ == "__main__":
    main()
