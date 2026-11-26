"""Train and select a deployable 17-joint model using validation-only retention gates."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from src.temporal.temporal_model import SpatialTemporalTransformer  # noqa: E402
from training.train_action import FLIP_PERM  # noqa: E402
from training.train_urfd import partial_label_loss  # noqa: E402

EMERGENCY = [ACTION_CLASSES.index(c) for c in ("falling", "stumbling", "abnormal_movement", "immobile")]
FALL = ACTION_CLASSES.index("falling")


class ProbabilityBlend(nn.Module):
    """Return logits whose softmax exactly matches a validated probability blend."""
    def __init__(self, baseline: nn.Module, candidate: nn.Module, alpha: float,
                 preserve_posture_ratio: bool = False) -> None:
        super().__init__()
        self.baseline, self.candidate, self.alpha = baseline, candidate, alpha
        self.preserve_posture_ratio = preserve_posture_ratio
        mask = torch.zeros(len(ACTION_CLASSES)); mask[[3, 9]] = 1
        self.register_buffer("posture_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 1.0 and not self.preserve_posture_ratio:
            return self.candidate(x)
        old = self.baseline(x).softmax(-1)
        p = (1 - self.alpha) * old + self.alpha * self.candidate(x).softmax(-1)
        if self.preserve_posture_ratio:
            original = old * self.posture_mask
            pair_mass = (p * self.posture_mask).sum(-1, keepdim=True)
            p = p * (1-self.posture_mask) + pair_mass * original / original.sum(-1, keepdim=True).clamp_min(1e-8)
        return p.clamp_min(1e-8).log()


def blend_probabilities(old: np.ndarray, new: np.ndarray, alpha: float, preserve: bool) -> np.ndarray:
    p = (1-alpha)*old + alpha*new
    if preserve:
        pair = [ACTION_CLASSES.index("lying"), ACTION_CLASSES.index("immobile")]
        p[:, pair] = p[:, pair].sum(1, keepdims=True) * old[:, pair] / np.maximum(1e-8, old[:, pair].sum(1, keepdims=True))
    return p


def probabilities(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            rows.append(model(torch.from_numpy(x[start:start + 512]).to(device)).softmax(-1).cpu().numpy())
    return np.concatenate(rows) if rows else np.empty((0, len(ACTION_CLASSES)), np.float32)


def exact_metrics(prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    cm = np.zeros((len(ACTION_CLASSES), len(ACTION_CLASSES)), dtype=np.int64)
    np.add.at(cm, (y, prob.argmax(1)), 1)
    tp = np.diag(cm).astype(float)
    precision, recall = tp / np.maximum(1, cm.sum(0)), tp / np.maximum(1, cm.sum(1))
    f1 = 2 * precision * recall / np.maximum(1e-9, precision + recall)
    em = np.isin(np.arange(len(ACTION_CLASSES)), EMERGENCY)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1[cm.sum(1) > 0].mean()),
            "fall_recall": float(recall[FALL]), "fall_precision": float(precision[FALL]),
            "lying_recall": float(recall[3]), "immobile_recall": float(recall[9]),
            "emergency_recall": float(cm[np.ix_(em, em)].sum() / max(1, cm[em].sum())),
            "normal_fpr": float(cm[np.ix_(~em, em)].sum() / max(1, cm[~em].sum()))}


def assess(predictions: dict[str, np.ndarray], sets: dict[str, dict]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "ntu60" in predictions:
        result["legacy"] = exact_metrics(predictions["ntu60"], sets["ntu60"]["y"])
    normal_rates = {}
    for name in ("ntu120", "ucf101", "hmdb51"):
        if name in predictions:
            normal_rates[name] = float(np.isin(predictions[name].argmax(1), EMERGENCY).mean())
    result["normal_sources_fpr"] = normal_rates
    result["new_normal_fpr"] = float(np.mean(list(normal_rates.values())))
    if "upfall" in predictions:
        result["upfall_impact_recall"] = float((predictions["upfall"].argmax(1) == FALL).mean())
    if "urfd" in predictions:
        p = predictions["urfd"].argmax(1) == FALL
        y = sets["urfd"]["allowed"][:, FALL]
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        result["urfd_transition_f1"] = float(2 * tp / max(1, 2 * tp + fp + fn))
    return result


def qualifies(candidate: dict, baseline: dict, limits: dict) -> bool:
    a, b = candidate["legacy"], baseline["legacy"]
    return (a["fall_recall"] >= b["fall_recall"] - limits["max_fall_recall_drop"]
            and a["emergency_recall"] >= b["emergency_recall"] - limits["max_emergency_recall_drop"]
            and a["normal_fpr"] <= b["normal_fpr"] + limits["max_normal_fpr_increase"]
            and a["macro_f1"] >= b["macro_f1"] - limits["max_macro_f1_drop"]
            and ("max_lying_recall_drop" not in limits or a["lying_recall"] >= b["lying_recall"] - limits["max_lying_recall_drop"])
            and ("max_immobile_recall_drop" not in limits or a["immobile_recall"] >= b["immobile_recall"] - limits["max_immobile_recall_drop"])
            and candidate["new_normal_fpr"] < baseline["new_normal_fpr"]
            and candidate["upfall_impact_recall"] >= baseline["upfall_impact_recall"] - limits["max_upfall_recall_drop"])


def selection_score(metrics: dict) -> float:
    return (metrics["legacy"]["macro_f1"] + 0.5 * (1 - metrics["new_normal_fpr"])
            + 0.1 * metrics["urfd_transition_f1"])


def augment_normalized(x: torch.Tensor) -> torch.Tensor:
    """GPU augmentation after shared normalization; preserve the meaning of confidence."""
    out = x.clone()
    flip = torch.rand(len(out), device=out.device) < 0.5
    out[flip] = out[flip][:, :, FLIP_PERM]
    out[flip, :, :, 0] *= -1
    noise = torch.randn_like(out[..., :2]) * 0.01
    out[..., :2] += noise * (out[..., 2:] > 0.2)
    drop = torch.rand_like(out[..., 2]) < 0.02
    out[drop] = 0
    return out


def load_model(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("classes") != ACTION_CLASSES:
        raise ValueError("10-class order mismatch")
    model = SpatialTemporalTransformer(input_dim=51, num_classes=len(ACTION_CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def export_selected(config: dict) -> None:
    cfg = config["training"]
    output = ROOT / cfg["output"]
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    baseline = load_model(ROOT / cfg["baseline"], torch.device("cpu"))
    candidate = load_model(output, torch.device("cpu"))
    model = ProbabilityBlend(baseline, candidate, checkpoint["selection"]["alpha"], cfg.get("preserve_posture_ratio", False)).eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        # Frozen teachers can activate a fused PyTorch-only encoder operator.
        # Export the equivalent standard attention graph supported by ONNX/DML.
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(model, torch.zeros(1, 30, 17, 3), str(output.with_suffix(".onnx")),
                          opset_version=17, input_names=["skeletal_sequence"], output_names=["action_logits"],
                          dynamic_axes={"skeletal_sequence": {0: "batch"}, "action_logits": {0: "batch"}}, dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    print("Exported", output.with_suffix(".onnx"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/rgb_expansion.yaml").read_text(encoding="utf-8"))
    cfg = config["training"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    if args.export_only:
        export_selected(config)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = ROOT / "data/processed/rgb_expansion"
    pieces = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            pieces[path.stem] = {k: data[k] for k in ("x", "allowed", "split", "y", "group")}
    if set(pieces) != set(cfg["source_weights"]):
        raise ValueError(f"Missing source pieces: {set(cfg['source_weights']) - set(pieces)}")
    validation = {name: {k: v[data["split"] == 1] for k, v in data.items() if k != "split"}
                  for name, data in pieces.items()}
    for name, data in validation.items():
        if not len(data["x"]):
            raise ValueError(f"Empty validation source {name}")
    train_x, train_masks, weights, anchors = [], [], [], []
    for name, data in pieces.items():
        mask = data["split"] == 0
        n = int(mask.sum())
        w = np.full(n, cfg["source_weights"][name] / n)
        if name == "ntu60":
            labels = data["y"][mask]
            counts = np.bincount(labels, minlength=len(ACTION_CLASSES))
            w = cfg["source_weights"][name] / len(ACTION_CLASSES) / counts[labels]
        train_x.append(data["x"][mask]); train_masks.append(data["allowed"][mask])
        weights.append(w); anchors.extend([name == "ntu60"] * n)
    x = torch.from_numpy(np.concatenate(train_x))
    allowed = torch.from_numpy(np.concatenate(train_masks))
    anchor = torch.tensor(anchors, dtype=torch.bool)
    del train_x, train_masks
    sampler = WeightedRandomSampler(torch.from_numpy(np.concatenate(weights)), cfg["epoch_samples"], replacement=True)
    loader = DataLoader(TensorDataset(x, allowed, anchor), batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    baseline = load_model(ROOT / cfg["baseline"], device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    candidate = load_model(ROOT / cfg["initialize"], device)
    base_predictions = {name: probabilities(baseline, data["x"], device) for name, data in validation.items()}
    base_metrics = assess(base_predictions, validation)
    print("baseline validation", json.dumps(base_metrics), flush=True)
    print("training windows", len(x), "draws/epoch", cfg["epoch_samples"], "device", device, flush=True)
    opt = torch.optim.AdamW(candidate.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    output = ROOT / cfg["output"]
    history, best, stale = [], None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        candidate.train()
        total = 0.0
        for windows, masks, is_anchor in loader:
            windows = augment_normalized(windows.to(device))
            masks, is_anchor = masks.to(device), is_anchor.to(device)
            opt.zero_grad()
            logits = candidate(windows)
            loss = partial_label_loss(logits, masks)
            if is_anchor.any():
                with torch.no_grad():
                    teacher = baseline(windows[is_anchor]).softmax(-1)
                loss = loss + cfg["distillation_weight"] * nn.functional.kl_div(
                    logits[is_anchor].log_softmax(-1), teacher, reduction="batchmean")
            loss.backward()
            nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(windows)
        scheduler.step()
        pred = {name: probabilities(candidate, data["x"], device) for name, data in validation.items()}
        accepted = []
        for alpha in cfg["blend_grid"]:
            mixed = {name: blend_probabilities(base_predictions[name], value, alpha, cfg.get("preserve_posture_ratio", False)) for name, value in pred.items()}
            metrics = assess(mixed, validation)
            if qualifies(metrics, base_metrics, cfg["monitor"]):
                accepted.append((selection_score(metrics), alpha, metrics))
        chosen = max(accepted, default=None, key=lambda item: item[0])
        history.append({"epoch": epoch, "loss": total / cfg["epoch_samples"], "raw_validation": assess(pred, validation),
                        "accepted_alpha": chosen[1] if chosen else None})
        print(f"epoch {epoch:02d} loss={history[-1]['loss']:.4f}; raw={json.dumps(history[-1]['raw_validation'])}; accepted_alpha={history[-1]['accepted_alpha']}", flush=True)
        if chosen is not None and (best is None or chosen[0] > best["score"]):
            best = {"score": chosen[0], "alpha": chosen[1], "validation": chosen[2], "epoch": epoch}
            torch.save({"model_state_dict": candidate.state_dict(), "classes": ACTION_CLASSES,
                        "selection": best, "config": config, "baseline_path": cfg["baseline"]}, output)
            stale = 0
        elif best is not None:
            stale += 1
        if best is not None and stale >= cfg["patience"]:
            break
    report = {"baseline_validation": base_metrics, "selected": best, "history": history, "config": config,
              "source_reports": {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")},
              "test_used_for_selection": False, "runtime_replaced": False}
    if best is None:
        output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError("No candidate passed retention gates; runtime remains unchanged")
    candidate.load_state_dict(torch.load(output, map_location="cpu", weights_only=False)["model_state_dict"])
    # Evaluate held-out source splits only after all validation-based selection.
    test_sets = {name: {k: v[data["split"] == 2] for k, v in data.items() if k != "split"}
                 for name, data in pieces.items() if np.any(data["split"] == 2)}
    test_base = {name: probabilities(baseline, data["x"], device) for name, data in test_sets.items()}
    test_candidate = {name: probabilities(candidate, data["x"], device) for name, data in test_sets.items()}
    mixed = {name: blend_probabilities(test_base[name], test_candidate[name], best["alpha"], cfg.get("preserve_posture_ratio", False)) for name in test_sets}
    report["baseline_test"] = assess(test_base, test_sets)
    report["candidate_test"] = assess(mixed, test_sets)
    report["test_counts"] = {name: len(data["x"]) for name, data in test_sets.items()}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    export_selected(config)
    print(json.dumps({"selected": best, "baseline_test": report["baseline_test"],
                      "candidate_test": report["candidate_test"], "test_counts": report["test_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
