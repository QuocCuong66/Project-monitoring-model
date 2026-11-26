"""Evaluate labeled skeleton windows; report validation separately from test results."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal.temporal_model import SpatialTemporalTransformer
from src.action.action_classifier import ACTION_CLASSES
from training.train_action import WindowDataset, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Action Model")
    parser.add_argument("--weights", type=str, default="models/checkpoints/best_st_transformer.pt")
    parser.add_argument("--output-dir", type=str, default="results/evaluation_reports")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", default="data/processed/ntu_windows.npz")
    parser.add_argument("--split", type=int, choices=[1, 2], default=1,
                        help="1 = validation; 2 = independent test, if available")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = SpatialTemporalTransformer(input_dim=51, num_classes=len(ACTION_CLASSES))

    if Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=dev, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            if ckpt.get("classes") != ACTION_CLASSES:
                raise ValueError("Checkpoint class order does not match ACTION_CLASSES")
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[Info] Loaded weights from {args.weights}")
    else:
        raise FileNotFoundError(args.weights)

    model.to(dev)

    torch.set_num_threads(4)
    x, y, source = load_split(Path(args.data), args.split)
    def score(mask: np.ndarray) -> Dict[str, Any]:
        loader = DataLoader(WindowDataset(x[mask], y[mask], False), batch_size=128)
        return evaluate(model, loader, dev)
    summary = score(np.ones(len(y), dtype=bool))
    summary.update({"split": "validation" if args.split == 1 else "test",
                    "weights": args.weights, "data": args.data, "samples": len(y),
                    "note": "Window classification, not end-to-end alert accuracy."})
    summary["source_metrics"] = {str(int(s)): score(source == s) for s in np.unique(source)}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("source_metrics", "confusion", "per_class")}, indent=2))


def load_split(path: Path, split_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real raw windows; refuse missing splits, malformed data and invalid labels."""
    with np.load(path, allow_pickle=False) as data:
        x, y, split = data["x"], data["y"], data["split"]
        source = data["src"] if "src" in data else np.full(len(y), -1)
    if x.ndim != 4 or x.shape[2:] != (17, 3) or x.shape[1] < 1:
        raise ValueError("Expected (N, T, 17, 3) raw skeleton windows")
    if y.shape != (len(x),) or split.shape != y.shape or source.shape != y.shape:
        raise ValueError("Labels, split and source must align with windows")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= len(ACTION_CLASSES))):
        raise ValueError("Invalid action labels")
    mask = split == split_id
    if not mask.any():
        raise ValueError(f"No real samples for split {split_id}")
    if not np.isfinite(x[mask]).all():
        raise ValueError("Non-finite skeleton coordinates")
    return x[mask], y[mask], source[mask]


if __name__ == "__main__":
    main()
