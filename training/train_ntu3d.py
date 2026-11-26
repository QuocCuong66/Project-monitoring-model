"""Independent NTU 3D action benchmarks; never interpret z as pose confidence.

Keeps all original 60/120 action labels and 25 joints. Official cross-subject
test actors remain untouched; validation actors are held out from training.
NTU60 and NTU120 are trained separately because their recordings overlap.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.temporal.temporal_model import SpatialTemporalTransformer  # noqa: E402


def subject_id(name: str) -> int:
    match = re.fullmatch(r"S\d{3}C\d{3}P(\d{3})R\d{3}A\d{3}", name)
    if not match:
        raise ValueError(f"Invalid NTU recording identifier: {name}")
    return int(match[1])


def normalize_3d(keypoint: np.ndarray, frames: int, min_valid: int, min_scale: float) -> np.ndarray | None:
    """Resample one primary body; center once to retain whole-body displacement."""
    if keypoint.ndim != 4 or keypoint.shape[2:] != (25, 3):
        raise ValueError("Expected (people, time, 25 joints, xyz), not COCO confidence features")
    if keypoint.shape[0] == 0 or keypoint.shape[1] == 0:
        return None
    valid = np.isfinite(keypoint).all(-1) & (np.abs(keypoint).sum(-1) > 0)
    body = int(valid.sum((1, 2)).argmax())
    x, joint_valid = keypoint[body].astype(np.float32), valid[body]
    good = joint_valid[:, 0] & joint_valid[:, 20]
    if good.sum() < min_valid:
        return None
    scale = float(np.median(np.linalg.norm(x[good, 20] - x[good, 0], axis=-1)))
    if scale < min_scale:
        return None
    center = x[np.flatnonzero(good)[0], 0].copy()
    x = (x - center) / scale
    x[~joint_valid] = 0
    # The third coordinate stays physical depth; no confidence thresholding.
    index = np.rint(np.linspace(0, len(x) - 1, frames)).astype(int)
    return x[index].reshape(frames, 75)


def _prepare(name: str, source: dict, cfg: dict) -> Path:
    folder = ROOT / "data/processed/ntu3d" / name
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "metadata.json"
    if metadata.exists():
        return folder
    # Pickle is loaded only from the fixed, published MMAction2 source.
    with (ROOT / source["file"]).open("rb") as stream:
        data = pickle.load(stream)
    train_ids = set(data["split"]["xsub_train"])
    test_key = "xsub_val" if "xsub_val" in data["split"] else "xsub_test"
    test_ids = set(data["split"][test_key])
    if train_ids & test_ids:
        raise ValueError("Official split overlaps")
    train_subjects = sorted({subject_id(n) for n in train_ids})
    test_subjects = {subject_id(n) for n in test_ids}
    if set(train_subjects) & test_subjects:
        raise ValueError("Official cross-subject split contains overlapping actors")
    rng = np.random.default_rng(cfg["seed"])
    val_subjects = set(rng.choice(train_subjects, max(1, int(len(train_subjects) * cfg["validation_subject_fraction"])), replace=False).tolist())
    annotations = data["annotations"]
    n = len(annotations)
    features = np.lib.format.open_memmap(folder / "x.npy", mode="w+", dtype=np.float32,
                                        shape=(n, cfg["frames"], 75))
    labels, splits, subjects, names = [], [], [], []
    skipped = 0
    for i, ann in enumerate(annotations):
        clip = ann["frame_dir"]
        if clip not in train_ids and clip not in test_ids:
            skipped += 1
            continue
        label = int(ann["label"])
        if not 0 <= label < source["num_classes"]:
            raise ValueError("Class index outside source label range")
        seq = normalize_3d(ann["keypoint"], cfg["frames"], cfg["min_valid_frames"], cfg["min_torso_scale"])
        if seq is None:
            skipped += 1
            continue
        actor = subject_id(clip)
        features[len(labels)] = seq
        labels.append(label)
        splits.append(2 if clip in test_ids else 1 if actor in val_subjects else 0)
        subjects.append(actor)
        names.append(clip)
        if (i + 1) % 10000 == 0:
            print(f"{name} prepare {i+1}/{n}", flush=True)
    features.flush()
    np.savez_compressed(folder / "index.npz", y=np.asarray(labels, dtype=np.int64),
                        split=np.asarray(splits, dtype=np.int8), subject=np.asarray(subjects), group=np.asarray(names))
    info = {"dataset": name, "samples": len(labels), "allocated_rows": n, "skipped": skipped,
            "features": "25 joints xyz; 75 features, depth is not confidence", "frames": cfg["frames"],
            "split": "official cross-subject test; validation actors held out from official training actors",
            "validation_subjects": sorted(val_subjects), "official_test_subjects": sorted(test_subjects),
            "split_counts": {str(s): int((np.asarray(splits) == s).sum()) for s in (0, 1, 2)},
            "normalization": "first valid root origin, median spine length scale; uniform whole-clip sampling"}
    metadata.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)
    return folder


def prepare(name: str, source: dict, cfg: dict) -> Path:
    """Serialize cache creation when preprocessing and training overlap."""
    folder = ROOT / "data/processed/ntu3d" / name
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".prepare.lock"), timeout=7200):
        return _prepare(name, source, cfg)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> dict[str, Any]:
    cm = np.zeros((classes, classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device)).argmax(1).cpu().numpy()
            np.add.at(cm, (y.numpy(), prediction), 1)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(1, cm.sum(0))
    recall = tp / np.maximum(1, cm.sum(1))
    f1 = 2 * precision * recall / np.maximum(1e-9, precision + recall)
    return {"accuracy": float(tp.sum() / max(1, cm.sum())), "macro_f1": float(f1[cm.sum(1) > 0].mean()),
            "fall_A043": {"precision": float(precision[42]), "recall": float(recall[42]), "support": int(cm[42].sum())},
            "confusion": cm.tolist()}


def train(name: str, source: dict, config: dict) -> None:
    folder = prepare(name, source, config["preparation"])
    cfg = config["training"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = np.load(folder / "index.npz", allow_pickle=False)
    y, split, subjects = index["y"], index["split"], index["subject"]
    features = np.load(folder / "x.npy", mmap_mode="r")
    for a in (0, 1, 2):
        for b in range(a + 1, 3):
            if set(subjects[split == a]) & set(subjects[split == b]):
                raise ValueError("Subject leakage")
    loaders = []
    for s in (0, 1, 2):
        mask = split == s
        dataset = TensorDataset(torch.from_numpy(features[:len(y)][mask].copy()), torch.from_numpy(y[mask]))
        loaders.append(DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=s == 0, num_workers=0))
    model = SpatialTemporalTransformer(input_dim=75, num_classes=source["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    out = ROOT / cfg["checkpoint_dir"] / f"{name}_3d.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    print(f"{name}: device={device}; classes={source['num_classes']}; split counts={[len(l.dataset) for l in loaders]}", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for x, target in loaders[0]:
            x, target = x.to(device), target.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), target, label_smoothing=0.1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x)
        scheduler.step()
        metrics = evaluate(model, loaders[1], device, source["num_classes"])
        history.append({"epoch": epoch, "loss": total / len(loaders[0].dataset),
                        **{k: v for k, v in metrics.items() if k != "confusion"}})
        print(f"{name} epoch {epoch:02d}: loss={history[-1]['loss']:.4f}; val_acc={metrics['accuracy']:.4f}; val_F1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"model_state_dict": model.state_dict(), "input_dim": 75,
                        "num_classes": source["num_classes"], "epoch": epoch, "validation": metrics,
                        "input_semantics": "NTU 25-joint xyz, not compatible with live COCO-17", "config": config}, out)
    selected = torch.load(out, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    test = evaluate(model, loaders[2], device, source["num_classes"])
    report = {"dataset": name, "preparation": json.loads((folder / "metadata.json").read_text()),
              "selected_epoch": selected["epoch"], "validation": selected["validation"], "test": test,
              "history": history, "runtime_replaced": False,
              "note": "Initial 3D action benchmark; not a live RGB camera model or clinical validation"}
    out.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model.cpu().eval()
    torch.onnx.export(model, torch.zeros(1, config["preparation"]["frames"], 75), str(out.with_suffix(".onnx")),
                      opset_version=17, input_names=["skeleton_xyz"], output_names=["action_logits"],
                      dynamic_axes={"skeleton_xyz": {0: "batch"}, "action_logits": {0: "batch"}}, dynamo=False)
    print(f"{name} complete: {json.dumps({k:v for k,v in test.items() if k!='confusion'})}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ntu60", "ntu120"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/ntu3d.yaml").read_text(encoding="utf-8"))
    train(args.dataset, config["sources"][args.dataset], config)


if __name__ == "__main__":
    main()
