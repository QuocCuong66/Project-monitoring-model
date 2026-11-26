"""Fine-tune existing 10-class action model with coarse URFD labels and NTU replay.

Uses recording-disjoint train/val/test; no claim of subject independence. Test is
evaluated only after selecting checkpoint and threshold on validation. Runtime
weights are never replaced. Depth is archived but the RGB-pose model uses RGB.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from src.temporal.temporal_model import SpatialTemporalTransformer  # noqa: E402
from training.train_action import WindowDataset  # noqa: E402


def partial_label_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood of the set of source-supported classes."""
    if allowed.shape != logits.shape or not torch.all(allowed.any(dim=1)):
        raise ValueError("Every example needs at least one allowed class")
    return (torch.logsumexp(logits, dim=1)
            - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)).mean()


class PartialDataset(Dataset):
    def __init__(self, x: np.ndarray, allowed: np.ndarray, train: bool, seed: int) -> None:
        self.windows = WindowDataset(x, np.zeros(len(x), dtype=np.int64), train, seed)
        self.allowed = allowed

    def __len__(self) -> int:
        return len(self.allowed)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index][0], torch.from_numpy(self.allowed[index])


def binary_metrics(score: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = score >= threshold
    positive = target == 1
    tp, fp = int((pred & positive).sum()), int((pred & ~positive).sum())
    fn, tn = int((~pred & positive).sum()), int((~pred & ~positive).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "fpr": fp / max(1, fp + tn),
            "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def select_threshold(score: np.ndarray, target: np.ndarray, grid: list[float]) -> dict[str, Any]:
    return max((binary_metrics(score, target, t) for t in grid),
               key=lambda m: (m["f1"], m["recall"], -m["fpr"]))


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(WindowDataset(x, np.zeros(len(x), dtype=np.int64), False), batch_size=128)
    scores = []
    model.eval()
    with torch.no_grad():
        for windows, _ in loader:
            scores.extend(model(windows.to(device)).softmax(1)[:, ACTION_CLASSES.index("falling")].cpu().tolist())
    return np.asarray(scores)


def recording_metrics(score: np.ndarray, target: np.ndarray, group: np.ndarray, threshold: float) -> dict:
    falls, detected, normal, false_alarm = 0, 0, 0, 0
    for name in np.unique(group):
        mask = group == name
        if str(name).startswith("fall-"):
            falls += 1
            detected += int(np.any((score[mask] >= threshold) & (target[mask] == 1)))
        else:
            normal += 1
            false_alarm += int(np.any(score[mask] >= threshold))
    return {"fall_recordings": falls, "detected_during_annotated_transition_window": detected,
            "fall_recording_recall": detected / max(1, falls), "adl_recordings": normal,
            "adl_recordings_with_false_positive": false_alarm,
            "note": "Recording-level transition detection, not deployed risk-engine alarms or false alarms/hour"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/urfd.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg, seed = config["training"], config["preparation"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(ROOT / "data/processed/urfd_windows.npz", allow_pickle=False) as data:
        x, allowed, target, split, groups = (data[k] for k in ("x", "allowed", "event", "split", "group"))
    preparation = json.loads((ROOT / "data/processed/urfd_windows.json").read_text(encoding="utf-8"))
    for a in (0, 1, 2):
        for b in range(a + 1, 3):
            if set(groups[split == a]) & set(groups[split == b]):
                raise ValueError("Recording leakage between splits")
    tr, va, te = split == 0, split == 1, split == 2
    if not all(mask.any() for mask in (tr, va, te)) or not all(np.any(target[mask] == 1) for mask in (tr, va, te)):
        raise ValueError("Every split needs real fall-transition windows")
    with np.load(ROOT / cfg["replay_data"], allow_pickle=False) as replay:
        idx = []
        for cls in range(len(ACTION_CLASSES)):
            candidates = np.flatnonzero((replay["split"] == 0) & (replay["y"] == cls))
            idx.extend(rng.choice(candidates, min(len(candidates), cfg["replay_per_class"]), replace=False))
        replay_x, replay_y = replay["x"][idx], replay["y"][idx]
    train_x = np.concatenate([x[tr], replay_x])
    train_allowed = np.concatenate([allowed[tr], np.eye(len(ACTION_CLASSES), dtype=bool)[replay_y]])
    loader = DataLoader(PartialDataset(train_x, train_allowed, True, seed),
                        batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    baseline = torch.load(ROOT / cfg["baseline_weights"], map_location="cpu", weights_only=False)
    if baseline.get("classes") != ACTION_CLASSES:
        raise ValueError("Checkpoint class order mismatch")
    model = SpatialTemporalTransformer(input_dim=51, num_classes=len(ACTION_CLASSES)).to(device)
    model.load_state_dict(baseline["model_state_dict"])
    base_val = select_threshold(predict(model, x[va], device), target[va], cfg["threshold_grid"])
    best = base_val
    output = ROOT / cfg["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**baseline, "validation_urfd": best, "urfd_config": config}, output)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    history = []
    print(f"device={device}; URFD train/val/test={tr.sum()}/{va.sum()}/{te.sum()}; NTU replay={len(replay_x)}; baseline={base_val}", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for windows, mask in loader:
            opt.zero_grad()
            loss = partial_label_loss(model(windows.to(device)), mask.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(windows)
        metrics = select_threshold(predict(model, x[va], device), target[va], cfg["threshold_grid"])
        history.append({"epoch": epoch, "loss": total / len(train_x), **metrics})
        print(f"epoch {epoch:02d}: loss={total/len(train_x):.4f}; val={metrics}", flush=True)
        key = lambda m: (m["f1"], m["recall"], -m["fpr"])
        if key(metrics) > key(best):
            best = metrics
            torch.save({"model_state_dict": model.state_dict(), "classes": ACTION_CLASSES,
                        "validation_urfd": best, "epoch": epoch, "urfd_config": config}, output)
    # Test is untouched until all model and threshold selection is complete.
    model.load_state_dict(baseline["model_state_dict"])
    base_score = predict(model, x[te], device)
    selected = torch.load(output, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    candidate_score = predict(model, x[te], device)
    report = {"protocol": "recording-disjoint; not subject-disjoint", "baseline_validation": base_val,
              "preparation": preparation,
              "candidate_validation": best, "selected_epoch": selected.get("epoch", 0),
              "baseline_test": binary_metrics(base_score, target[te], base_val["threshold"]),
              "candidate_test": binary_metrics(candidate_score, target[te], best["threshold"]),
              "baseline_recordings": recording_metrics(base_score, target[te], groups[te], base_val["threshold"]),
              "candidate_recordings": recording_metrics(candidate_score, target[te], groups[te], best["threshold"]),
              "history": history, "config": config, "runtime_replaced": False,
              "limitations": ["Small recording-disjoint test; subjects may overlap", "Official depth posture labels indexed to paired cam0 RGB frames",
                              "RGB-only pose input; depth archived, not a model input", "Not evidence of sleep detection or clinical effectiveness"]}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output.with_suffix(".test_predictions.npz"), baseline=base_score,
                        candidate=candidate_score, target=target[te], group=groups[te])
    model.cpu().eval()
    torch.onnx.export(model, torch.zeros(1, x.shape[1], 17, 3), str(output.with_suffix(".onnx")),
                      opset_version=17, input_names=["skeletal_sequence"], output_names=["action_logits"],
                      dynamic_axes={"skeletal_sequence": {0: "batch"}, "action_logits": {0: "batch"}}, dynamo=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "config")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
