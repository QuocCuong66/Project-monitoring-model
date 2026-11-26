"""Train the pose-sequence autoencoder on normal daily movement from NTU RGB+D 60.

Prerequisite: ``python training/prepare_ntu.py``. Uses the same ``build_model_sequence``
preprocessing as live inference. After training, the reconstruction-error threshold and
sigmoid steepness are calibrated on held-out (cross-subject) normal windows and written
to models/checkpoints/pose_autoencoder_calibration.json; copy them into
configs/model.yaml (anomaly.threshold / anomaly.steepness).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from src.anomaly.anomaly_score import PoseSequenceAutoencoder  # noqa: E402
from training.train_action import WindowDataset  # noqa: E402

NORMAL = [ACTION_CLASSES.index(c) for c in ("walking", "standing", "sitting", "bending", "getting_up")]
ABNORMAL = [ACTION_CLASSES.index(c) for c in ("falling", "stumbling", "abnormal_movement")]


def errors(model: nn.Module, loader: DataLoader, dev: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev)
            out.append(torch.mean(torch.abs(xb - model(xb)), dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    scores = np.concatenate([neg, pos])
    ranks = scores.argsort().argsort() + 1
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


class FixedBinAvgPool1d(nn.Module):
    """AdaptiveAvgPool1d unrolled for a known input length (identical bins and output).

    The TorchScript ONNX exporter rejects adaptive pooling when the output size does not
    divide the input size (15 -> 4 for T=30), so the bins are written out explicitly.
    """

    def __init__(self, in_len: int, out_len: int) -> None:
        super().__init__()
        self.bins = [((i * in_len) // out_len, -(-((i + 1) * in_len) // out_len)) for i in range(out_len)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, :, s:e].mean(dim=2, keepdim=True) for s, e in self.bins], dim=2)


def export_onnx(model: nn.Module, window: int, path: Path) -> None:
    """Static (1, T, 17, 3) export of the autoencoder."""
    model.eval().cpu()
    for i, layer in enumerate(model.encoder):
        if isinstance(layer, nn.AdaptiveAvgPool1d):
            model.encoder[i] = FixedBinAvgPool1d(window // 2, int(layer.output_size))  # after MaxPool1d(2)
    torch.onnx.export(model, torch.zeros(1, window, 17, 3), str(path), opset_version=17,
                      input_names=["skeletal_sequence"], output_names=["reconstruction"], dynamo=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train pose autoencoder on normal NTU movement")
    ap.add_argument("--data", default=str(ROOT / "data/processed/ntu_windows.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "models/checkpoints/best_pose_autoencoder.pt"))
    args = ap.parse_args()

    d = np.load(args.data)
    x, y, split = d["x"], d["y"], d["split"]
    normal = np.isin(y, NORMAL)
    tr = normal & (split == 0)
    va_norm = normal & (split == 1)
    va_abn = np.isin(y, ABNORMAL) & (split == 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] train normal={tr.sum()} val normal={va_norm.sum()} val abnormal={va_abn.sum()}")

    kw = {"num_workers": args.workers, "persistent_workers": args.workers > 0}
    train_loader = DataLoader(WindowDataset(x[tr], y[tr], True), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kw)
    norm_loader = DataLoader(WindowDataset(x[va_norm], y[va_norm], False), batch_size=256, **kw)
    abn_loader = DataLoader(WindowDataset(x[va_abn], y[va_abn], False), batch_size=256, **kw)

    model = PoseSequenceAutoencoder(input_dim=51, latent_dim=64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, _ in train_loader:
            xb = xb.to(dev)
            opt.zero_grad()
            loss = torch.mean(torch.abs(xb - model(xb)))
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        e_norm = errors(model, norm_loader, dev)
        e_abn = errors(model, abn_loader, dev)
        val = float(e_norm.mean())
        print(f"Epoch {epoch:02d} train_l1={tot / n:.4f} val_normal_l1={val:.4f} "
              f"val_abnormal_l1={e_abn.mean():.4f} AUROC={auroc(e_norm, e_abn):.3f}", flush=True)
        if val < best:
            best = val
            torch.save({"model_state_dict": model.state_dict()}, out)

    model.load_state_dict(torch.load(out, map_location=dev, weights_only=False)["model_state_dict"])
    e_norm = errors(model, norm_loader, dev)
    e_abn = errors(model, abn_loader, dev)
    p95, p995 = np.percentile(e_norm, [95, 99.5])
    calib = {
        "threshold": round(float(p95), 4),                                  # score 0.5 at p95 of normal
        "steepness": round(float(np.log(9.0) / max(1e-4, p995 - p95)), 2),  # score 0.9 at p99.5 of normal
        "auroc_normal_vs_abnormal": round(auroc(e_norm, e_abn), 3),
        "val_normal_error_p50_p95_p995": [round(float(v), 4) for v in np.percentile(e_norm, [50, 95, 99.5])],
        "val_abnormal_error_p50": round(float(np.median(e_abn)), 4),
    }
    out.with_name("pose_autoencoder_calibration.json").write_text(json.dumps(calib, indent=2))

    export_onnx(model, x.shape[1], out.with_suffix(".onnx"))
    print(f"[Done] {calib}")


if __name__ == "__main__":
    main()
