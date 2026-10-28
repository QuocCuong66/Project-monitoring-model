"""Train the Spatial-Temporal Transformer action model on NTU RGB+D 60 skeleton windows.

Prerequisite: ``python training/prepare_ntu.py`` (builds data/processed/ntu_windows.npz).

Augmentations act on raw pixel geometry before ``build_model_sequence`` (the exact live
preprocessing): horizontal flip, in-plane rotation, top-down vertical squash (ceiling /
doorbell cameras), time warping, occlusion and young-track front padding.

Evaluation uses the NTU cross-subject split (unseen people). The best checkpoint by
macro-F1 is saved to models/checkpoints/best_st_transformer.pt and exported to ONNX.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from src.temporal.sequence_features import build_model_sequence  # noqa: E402
from src.temporal.temporal_model import SpatialTemporalTransformer  # noqa: E402

FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
EMERGENCY = [ACTION_CLASSES.index(c) for c in ("falling", "stumbling", "abnormal_movement", "immobile")]


def augment(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = win.copy()
    t = len(w)
    valid = w[..., 2] > 0.2
    c = w[..., :2][valid].mean(0) if np.any(valid) else np.zeros(2, np.float32)
    pts = w[..., :2] - c
    if rng.random() < 0.5:
        pts[..., 0] *= -1
        pts = pts[:, FLIP_PERM]
        w[..., 2] = w[:, FLIP_PERM, 2]
    th = np.deg2rad(rng.uniform(-20, 20))
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    pts = pts @ rot.T
    pts[..., 1] *= rng.uniform(0.55, 1.05)   # high-mounted camera foreshortening
    pts[..., 0] *= rng.uniform(0.9, 1.1)
    w[..., :2] = pts + c
    if rng.random() < 0.5:                  # time warp (speed 0.75x .. 1.33x), end-anchored
        speed = rng.uniform(0.75, 1.33)
        idx = np.clip(np.round(t - 1 - np.arange(t)[::-1] * speed), 0, t - 1).astype(int)
        w = w[idx]
    if rng.random() < 0.2:                  # lower body hidden (furniture / railings)
        w[:, 11:, 2] = 0.0
    drop = rng.random((t, 17)) < 0.05       # flickering joints
    w[..., 2][drop] = 0.0
    w[..., 2] = np.clip(w[..., 2] * rng.uniform(0.8, 1.1), 0, 1)
    if rng.random() < 0.15:                 # young track: history shorter than the window
        k = int(rng.integers(6, t))
        w = np.concatenate([np.repeat(w[t - k:t - k + 1], t - k, 0), w[t - k:]], 0)
    return w


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool, seed: int = 0) -> None:
        self.x, self.y, self.train = x, y, train
        self.rng = np.random.default_rng(seed)
        self.t = x.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        win = self.x[i].astype(np.float32)
        if self.train:
            win = augment(win, self.rng)
        seq = build_model_sequence(win, self.t)
        if self.train:
            seq[..., :2] += self.rng.normal(0, 0.02, seq[..., :2].shape).astype(np.float32) * (seq[..., 2:] > 0)
        return torch.from_numpy(seq), torch.tensor(self.y[i])


def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device) -> dict:
    model.eval()
    n = len(ACTION_CLASSES)
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(dev)).argmax(1).cpu().numpy()
            for a, b in zip(y.numpy(), pred):
                cm[a, b] += 1
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(1, cm.sum(0))
    rec = tp / np.maximum(1, cm.sum(1))
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    present = cm.sum(1) > 0
    em = np.isin(np.arange(n), EMERGENCY)
    em_tp = cm[np.ix_(em, em)].sum()
    return {
        "acc": float(tp.sum() / max(1, cm.sum())),
        "macro_f1": float(f1[present].mean()),
        "per_class": {ACTION_CLASSES[i]: {"precision": round(prec[i], 3), "recall": round(rec[i], 3),
                                          "f1": round(f1[i], 3), "support": int(cm[i].sum())}
                      for i in range(n) if present[i]},
        "emergency_recall": float(em_tp / max(1, cm[em].sum())),
        "emergency_precision": float(em_tp / max(1, cm[:, em].sum())),
        "confusion": cm.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ST-Transformer on NTU RGB+D 60 windows")
    ap.add_argument("--data", default=str(ROOT / "data/processed/ntu_windows.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "models/checkpoints/best_st_transformer.pt"))
    args = ap.parse_args()

    d = np.load(args.data)
    x, y, split = d["x"], d["y"], d["split"]
    tr, va = split == 0, split == 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] train={tr.sum()} val={va.sum()} device={dev}")

    counts = np.bincount(y[tr], minlength=len(ACTION_CLASSES)).astype(float)
    weights = np.where(counts > 0, counts.sum() / np.maximum(1, counts) / len(ACTION_CLASSES), 0.0)
    weights = np.sqrt(weights)  # soften: rare medical classes up-weighted without dominating

    train_loader = DataLoader(WindowDataset(x[tr], y[tr], True), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(WindowDataset(x[va], y[va], False), batch_size=256, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    model = SpatialTemporalTransformer(input_dim=51, num_classes=len(ACTION_CLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader),
                                                pct_start=0.1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev), label_smoothing=0.1)

    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        m = evaluate(model, val_loader, dev)
        pc = m["per_class"]
        print(f"Epoch {epoch:02d} loss={tot / n:.3f} val_acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} "
              f"fall_R={pc['falling']['recall']:.2f} stagger_R={pc['stumbling']['recall']:.2f} "
              f"distress_R={pc['abnormal_movement']['recall']:.2f} emergency_R={m['emergency_recall']:.3f}", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save({"model_state_dict": model.state_dict(), "metrics": m, "classes": ACTION_CLASSES}, out)

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cpu()
    onnx_path = out.with_suffix(".onnx")
    torch.onnx.export(model, torch.zeros(1, x.shape[1], 17, 3), str(onnx_path), opset_version=17,
                      input_names=["skeletal_sequence"], output_names=["action_logits"],
                      dynamic_axes={"skeletal_sequence": {0: "batch"}, "action_logits": {0: "batch"}}, dynamo=False)
    report = out.with_name("st_transformer_metrics.json")
    report.write_text(json.dumps(ckpt["metrics"], indent=2))
    print(f"[Done] best macro-F1={best:.3f} -> {out}, {onnx_path}, {report}")


if __name__ == "__main__":
    main()
