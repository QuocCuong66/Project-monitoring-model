"""Recalibrate a trained checkpoint using per-posture validation gates, without retraining on test."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.train_rgb_expansion import (assess, blend_probabilities, export_selected,
                                         load_model, probabilities, qualifies, selection_score)  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/rgb_expansion.yaml").read_text(encoding="utf-8"))
    cfg = config["training"]
    torch.set_num_threads(cfg["cpu_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / cfg["output"]
    report = json.loads(out.with_suffix(".metrics.json").read_text())
    checkpoint = torch.load(out, map_location="cpu", weights_only=False)
    old, new = load_model(ROOT / cfg["baseline"], device), load_model(out, device)
    pieces = {}
    for path in (ROOT / "data/processed/rgb_expansion").glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            pieces[path.stem] = {k: data[k] for k in ("x", "allowed", "y", "split")}
    validation = {name: {k: v[d["split"] == 1] for k,v in d.items() if k != "split"} for name,d in pieces.items()}
    base = {name: probabilities(old, d["x"], device) for name,d in validation.items()}
    cand = {name: probabilities(new, d["x"], device) for name,d in validation.items()}
    baseline = assess(base, validation)
    choices, diagnostics = [], []
    for alpha in cfg["blend_grid"]:
        pred = {name: blend_probabilities(base[name], cand[name], alpha, True) for name in base}
        metrics = assess(pred, validation)
        accepted = qualifies(metrics, baseline, cfg["monitor"])
        diagnostics.append({"alpha": alpha, "accepted": accepted, "metrics": metrics})
        print(json.dumps(diagnostics[-1]), flush=True)
        if accepted:
            choices.append({"epoch": checkpoint["selection"]["epoch"], "alpha": alpha,
                            "score": selection_score(metrics), "validation": metrics})
    if not choices:
        raise RuntimeError("No posture-preserving blend passed validation gates")
    selected = max(choices, key=lambda item: item["score"])
    checkpoint["selection"] = selected
    checkpoint["config"] = config
    torch.save(checkpoint, out)
    report["initial_selection"] = report["selected"]
    report["selected"], report["baseline_validation"], report["config"] = selected, baseline, config
    report["posture_preserving_validation"] = diagnostics
    test = {name: {k: v[d["split"] == 2] for k,v in d.items() if k != "split"}
            for name,d in pieces.items() if np.any(d["split"] == 2)}
    a = {name: probabilities(old,d["x"],device) for name,d in test.items()}
    b = {name: probabilities(new,d["x"],device) for name,d in test.items()}
    p = {name: blend_probabilities(a[name],b[name],selected["alpha"],True) for name in a}
    report["baseline_test"], report["candidate_test"] = assess(a,test), assess(p,test)
    report["runtime_replaced"] = False
    report["development_note"] = "Posture-ratio calibration chosen on validation after a per-class development audit; source test sets remain excluded from gradient/selection, but have been inspected in this development session."
    out.with_suffix(".metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    export_selected(config)
    print(json.dumps({"selected":selected,"baseline_test":report["baseline_test"],"candidate_test":report["candidate_test"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
