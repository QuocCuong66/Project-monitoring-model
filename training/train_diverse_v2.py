"""Fine tune with new annotated phases, comparing against the frozen deployed ensemble.

Validation selects the epoch and blend; all subject-held-out tests are evaluated once
after selection. A failed validation gate leaves the deployed model untouched.
"""

import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from training.train_rgb_expansion import (ProbabilityBlend, load_model, probabilities,
    blend_probabilities, exact_metrics, assess, augment_normalized, EMERGENCY, FALL)  # noqa: E402
from training.train_urfd import partial_label_loss  # noqa: E402


def current_teacher(folder: Path, device: torch.device) -> nn.Module:
    deployment = json.loads((folder/"deployment.json").read_text(encoding="utf-8"))
    selected = deployment["selection"]
    return ProbabilityBlend(load_model(folder/"legacy.pt", device), load_model(folder/"student.pt", device),
                            selected["alpha"], True).to(device).eval()


def read_sets(config: dict) -> tuple[dict, dict]:
    old, native = {}, {}
    for folder, target in ((ROOT/"data/processed/rgb_expansion", old),
                           (ROOT/config["dataset"]["processed_dir"], native)):
        for path in sorted(folder.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                target[path.stem] = {key: data[key] for key in ("x", "allowed", "split", "y", "group")}
            if target is native:
                metadata = json.loads(path.with_suffix(".windows.json").read_text(encoding="utf-8"))
                if len(metadata) != len(target[path.stem]["x"]):
                    raise ValueError("Window metadata mismatch")
                target[path.stem]["source_label"] = np.array([r["label"] for r in metadata])
                target[path.stem]["normal_recording"] = np.array([r["normal_recording"] for r in metadata])
    if set(old) != {"ntu60", "ntu120", "ucf101", "hmdb51", "urfd", "upfall"}:
        raise ValueError("Incomplete old rehearsal sources")
    expected = {name for name, source in config["sources"].items() if not source.get("audit_only")}
    if set(native) != expected:
        raise ValueError("Prepare every frozen source before training")
    if "oops" in native and np.any(native["oops"]["split"] != 2):
        raise ValueError("OOPS must never enter training or validation")
    return old, native


def partition(pieces: dict, split: int) -> dict:
    return {name: {k: value[data["split"] == split] for k, value in data.items() if k != "split"}
            for name, data in pieces.items() if np.any(data["split"] == split)}


def native_metrics(predictions: dict[str, np.ndarray], sets: dict) -> dict:
    sources, normals, falls = {}, [], []
    for name, p in predictions.items():
        data = sets[name]
        label = data["source_label"]
        normal = np.isin(label, [0, 3, 4, 5, 7, 8]) | ((label == 6) & data["normal_recording"])
        fall = label == 1
        benign_down = np.isin(label, [5, 6]) & data["normal_recording"]
        predicted = p.argmax(1)
        result = {"windows": len(p), "normal_count": int(normal.sum()), "fall_count": int(fall.sum()),
                  "benign_down_count": int(benign_down.sum())}
        if "allowed" in data:
            result["allowed_label_accuracy"] = float(data["allowed"][np.arange(len(p)), predicted].mean())
            exact = data["y"] >= 0
            result["exact_label_accuracy"] = float((predicted[exact] == data["y"][exact]).mean()) if exact.any() else None
            lying = (label == 6) & data["normal_recording"]
            result["benign_lying_recall"] = float((predicted[lying] == ACTION_CLASSES.index("lying")).mean()) if lying.any() else None
        for key, mask, positive in (("normal_fpr", normal, np.isin(predicted, EMERGENCY)),
                                    ("fall_recall", fall, predicted == FALL),
                                    ("benign_down_fpr", benign_down, np.isin(predicted, EMERGENCY))):
            result[key] = float(positive[mask].mean()) if mask.any() else None
        sources[name] = result
        if normal.any():
            normals.append(result["normal_fpr"])
        if fall.any():
            falls.append(result["fall_recall"])
    return {"sources": sources, "normal_fpr": float(np.mean(normals)) if normals else None,
            "fall_recall": float(np.mean(falls)) if falls else None}


def passes(candidate: dict, baseline: dict, cfg: dict) -> bool:
    a, b = candidate["old"]["legacy"], baseline["old"]["legacy"]
    checks = [a[key] >= b[key]-cfg[f"max_old_{key}_drop"]
              for key in ("fall_recall", "lying_recall", "immobile_recall", "emergency_recall")]
    checks += [a["normal_fpr"] <= b["normal_fpr"]+cfg["max_old_normal_fpr_increase"],
               candidate["old"]["new_normal_fpr"] <= baseline["old"]["new_normal_fpr"],
               candidate["old"]["upfall_impact_recall"] >= baseline["old"]["upfall_impact_recall"],
               candidate["native"]["fall_recall"] >= baseline["native"]["fall_recall"]-cfg["max_native_fall_recall_drop"],
               candidate["native"]["normal_fpr"] <= baseline["native"]["normal_fpr"]*(1-cfg["min_native_normal_fpr_reduction"])]
    return all(checks)


def metrics(pred: dict, old: dict, native: dict) -> dict:
    return {"old": assess({k: pred["old:"+k] for k in old}, old),
            "native": native_metrics({k: pred["native:"+k] for k in native}, native)}


def export_model(model: nn.Module, output: Path) -> None:
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(model.cpu().eval(), torch.zeros(1, 30, 17, 3), str(output),
                          opset_version=17, input_names=["skeletal_sequence"], output_names=["action_logits"],
                          dynamic_axes={"skeletal_sequence": {0: "batch"}, "action_logits": {0: "batch"}}, dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def main() -> None:
    config = yaml.safe_load((ROOT/"configs/diverse_v2.yaml").read_text(encoding="utf-8"))
    cfg = config["training"]
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old, native = read_sets(config)
    old_val, native_val = partition(old, 1), partition(native, 1)
    validation = {**{"old:"+k: v for k, v in old_val.items()}, **{"native:"+k: v for k, v in native_val.items()}}
    folder = ROOT/config["integration"]["baseline_dir"]
    teacher = current_teacher(folder, device)
    # Verify the actual deployed ensemble, not merely an earlier .pt checkpoint.
    sample = old_val["ntu60"]["x"][:16]
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(folder/"active.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = teacher(torch.from_numpy(sample).to(device)).cpu().numpy()
    np.testing.assert_allclose(session.run(None, {session.get_inputs()[0].name: sample})[0], expected, rtol=1e-4, atol=1e-5)
    del session
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    candidate = load_model(folder/"student.pt", device)
    base = {name: probabilities(teacher, data["x"], device) for name, data in validation.items()}
    baseline = metrics(base, old_val, native_val)
    print("baseline", json.dumps(baseline), flush=True)
    xs, masks, weights, anchors = [], [], [], []
    for datasets, mass, is_old in ((partition(old, 0), cfg["old_training_fraction"], True),
                                   (partition(native, 0), 1-cfg["old_training_fraction"], False)):
        for name, data in datasets.items():
            labels = data["y"] if is_old else data["source_label"]
            unique, counts = np.unique(labels, return_counts=True)
            lookup = dict(zip(unique.tolist(), counts.tolist()))
            weights.extend(mass/len(datasets)/len(unique)/lookup[int(label)] for label in labels)
            xs.append(data["x"]); masks.append(data["allowed"]); anchors.extend([is_old]*len(labels))
    loader = DataLoader(TensorDataset(torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(masks)),
                                     torch.tensor(anchors, dtype=torch.bool)),
                        sampler=WeightedRandomSampler(torch.tensor(weights, dtype=torch.float64), cfg["draws_per_epoch"], replacement=True),
                        batch_size=cfg["batch_size"], num_workers=0)
    del xs, masks
    opt = torch.optim.AdamW(candidate.parameters(), lr=cfg["learning_rate"], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    output = ROOT/cfg["output"]
    best, history, stale = None, [], 0
    for epoch in range(1, cfg["epochs"]+1):
        candidate.train()
        total = 0.
        for x, allowed, anchor in loader:
            x, allowed, anchor = augment_normalized(x.to(device)), allowed.to(device), anchor.to(device)
            opt.zero_grad()
            logits = candidate(x)
            loss = partial_label_loss(logits, allowed)
            if anchor.any():
                with torch.no_grad():
                    targets = teacher(x[anchor]).softmax(-1)
                loss = loss + cfg["distillation_weight"]*nn.functional.kl_div(logits[anchor].log_softmax(-1), targets, reduction="batchmean")
            loss.backward()
            nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
            opt.step()
            total += loss.item()*len(x)
        scheduler.step()
        pred = {name: probabilities(candidate, data["x"], device) for name, data in validation.items()}
        candidates = []
        trials = []
        for alpha in cfg["blend_grid"]:
            blended = {name: blend_probabilities(base[name], value, alpha, cfg["preserve_posture_ratio"]) for name, value in pred.items()}
            result = metrics(blended, old_val, native_val)
            accepted = passes(result, baseline, cfg)
            trials.append({"alpha": alpha, "metrics": result, "passed": accepted})
            if accepted:
                score = 1-result["native"]["normal_fpr"] + result["native"]["fall_recall"] + result["old"]["legacy"]["macro_f1"]
                candidates.append((score, alpha, result))
        chosen = max(candidates, key=lambda row: row[0], default=None)
        history.append({"epoch": epoch, "loss": total/cfg["draws_per_epoch"], "trials": trials})
        print("epoch", epoch, "loss", round(history[-1]["loss"], 4), "selected_alpha", chosen[1] if chosen else None,
              "native", json.dumps(trials[2]["metrics"]["native"]), flush=True)
        if chosen is not None and (best is None or chosen[0] > best["score"]):
            best = {"score": chosen[0], "alpha": chosen[1], "validation": chosen[2], "epoch": epoch}
            torch.save({"model_state_dict": candidate.state_dict(), "classes": ACTION_CLASSES,
                        "selection": best, "config": config}, output)
            stale = 0
        else:
            stale += 1
        # Do not stop early before any candidate has passed retention gates.
        if best is not None and stale >= cfg["patience"]:
            break
    report = {"baseline_validation": baseline, "selected": best, "history": history,
              "config": config, "baseline_sha256": json.loads((folder/"snapshot.json").read_text())["sha256"],
              "training_counts": {name: int((d["split"] == 0).sum()) for name, d in {**old, **native}.items()},
              "runtime_replaced": False, "test_used_for_selection": False}
    if best is not None:
        candidate.load_state_dict(torch.load(output, map_location="cpu", weights_only=False)["model_state_dict"])
        old_test, native_test = partition(old, 2), partition(native, 2)
        tests = {**{"old:"+k: v for k, v in old_test.items()}, **{"native:"+k: v for k, v in native_test.items()}}
        test_base = {k: probabilities(teacher, d["x"], device) for k, d in tests.items()}
        test_new = {k: probabilities(candidate, d["x"], device) for k, d in tests.items()}
        mixed = {k: blend_probabilities(test_base[k], p, best["alpha"], cfg["preserve_posture_ratio"]) for k, p in test_new.items()}
        # Legacy has no untouched test partition; avoid presenting reused validation as test.
        report["baseline_test"] = metrics(test_base, old_test, native_test)
        report["candidate_test"] = metrics(mixed, old_test, native_test)
        export_model(ProbabilityBlend(teacher, candidate, best["alpha"], cfg["preserve_posture_ratio"]), output.with_suffix(".onnx"))
    else:
        # Preserve a trainable artifact for reproducibility even when it cannot replace runtime.
        torch.save({"model_state_dict": candidate.state_dict(), "classes": ACTION_CLASSES, "selection": None, "config": config}, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "config")}), flush=True)


if __name__ == "__main__":
    main()
