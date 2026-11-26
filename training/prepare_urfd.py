"""Extract COCO-17 poses from official URFD cam0 RGB ZIPs without unpacking PNGs.

Official posture labels are coarse. Allowed-label masks avoid fabricating exact
walking/sitting/standing labels. ADL transition frames are never labeled falling.
Splits are recording-disjoint, NOT subject-disjoint (subject IDs unavailable here).
Camera 1 is archived but excluded: official posture labels refer to camera 0.
"""

import argparse
import csv
import json
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from src.pose.rtmo_estimator import RTMOPoseEstimator  # noqa: E402


def allowed_classes(sequence: str, posture: int) -> list[int]:
    """Use source posture labels without inventing unavailable fine-grained labels."""
    if posture not in (-1, 0, 1):
        raise ValueError(f"Invalid URFD posture {posture}")
    if sequence.startswith("fall-") and posture == 0:
        names = ["falling"]
    elif posture == 1:
        names = ["lying"]  # lying on the ground does not establish immobility
    else:
        names = ["walking", "standing", "sitting", "bending", "getting_up"]
        if posture == 0:  # deliberate ADL transition may already be lying
            names.append("lying")
    return [ACTION_CLASSES.index(name) for name in names]


def recording_splits(names: list[str], seed: int, train: float, val: float) -> dict[str, int]:
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("Invalid train/validation fractions")
    rng = np.random.default_rng(seed)
    result: dict[str, int] = {}
    for prefix in ("adl-", "fall-"):
        group = sorted(n for n in names if n.startswith(prefix))
        rng.shuffle(group)
        a, b = int(len(group) * train), int(len(group) * (train + val))
        if not 0 < a < b < len(group):
            raise ValueError("Need recordings in every split")
        result.update({name: 0 if i < a else 1 if i < b else 2 for i, name in enumerate(group)})
    return result


def read_labels(raw: Path) -> dict[str, dict[int, int]]:
    labels: dict[str, dict[int, int]] = {}
    for path in sorted(raw.glob("urfall-cam0-*.csv")):
        for row in csv.reader(path.read_text().splitlines()):
            if not row:
                continue
            name, frame, label = row[0], int(row[1]), int(row[2])
            allowed_classes(name, label)
            labels.setdefault(name, {})[frame] = label
    if len(labels) != 70:
        raise ValueError(f"Expected labels for 70 recordings, found {len(labels)}")
    return labels


def extract(path: Path, out: Path, estimator: RTMOPoseEstimator, cfg: dict) -> None:
    poses, frames = [], []
    with zipfile.ZipFile(path) as archive:
        names = sorted((n for n in archive.namelist() if n.lower().endswith(".png")),
                       key=lambda n: int(re.search(r"-(\d+)\.png$", n)[1]))
        for name in names[::cfg["sample_stride"]]:
            frame_id = int(re.search(r"-(\d+)\.png$", name)[1])
            image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {name}")
            _, items = estimator.detect_and_estimate_single_pass(image)
            kp = max(items, key=lambda item: item.confidence).keypoints if items else np.zeros((17, 3))
            poses.append(np.asarray(kp, dtype=np.float32))
            frames.append(frame_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.stack(poses), frame=np.asarray(frames, dtype=np.int32))


def make_windows(poses: Path, labels: dict[str, dict[int, int]], cfg: dict,
                 splits: dict[str, int], output: Path) -> dict:
    xs, masks, targets, groups, split_values, frame_ends = [], [], [], [], [], []
    skipped = Counter()
    for sequence in sorted(labels):
        with np.load(poses / f"{sequence}.npz", allow_pickle=False) as data:
            x, frames = data["x"], data["frame"]
        for end in range(cfg["window_size"], len(x) + 1, cfg["window_stride"]):
            frame = int(frames[end - 1])
            posture = labels[sequence].get(frame)
            if posture is None:
                skipped["missing_official_label"] += 1
                continue
            window = x[end - cfg["window_size"]:end]
            valid = (window[..., 2] >= cfg["keypoint_confidence"]).sum(1) >= cfg["min_visible_joints"]
            if valid.mean() < cfg["min_valid_fraction"]:
                skipped["insufficient_pose"] += 1
                continue
            observed = [labels[sequence].get(int(f)) for f in frames[end - cfg["window_size"]:end]]
            transition = sequence.startswith("fall-") and 0 in observed
            mask = np.zeros(len(ACTION_CLASSES), dtype=bool)
            mask[allowed_classes(sequence, 0 if transition else posture)] = True
            xs.append(window.astype(np.float16))
            masks.append(mask)
            targets.append(int(transition))
            groups.append(sequence)
            split_values.append(splits[sequence])
            frame_ends.append(frame)
    if not xs:
        raise ValueError("No labeled pose windows")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.stack(xs), allowed=np.stack(masks),
                        event=np.array(targets, dtype=np.int64), group=np.array(groups),
                        split=np.array(split_values, dtype=np.int8), frame=np.array(frame_ends))
    report = {"windows": len(xs), "split_counts": dict(Counter(split_values)), "skipped": dict(skipped),
              "windows_per_recording": dict(Counter(groups)),
              "recordings_without_valid_windows": sorted(set(labels) - set(groups)),
              "recording_splits": splits, "protocol": "recording-disjoint, not subject-disjoint",
              "cam1": "Downloaded but excluded: no camera-1 posture labels",
              "labels": "Official coarse posture labels; marginal allowed-class supervision",
              "positive": "fall recording windows containing an official transition (0) frame; settled lying is not a fall onset"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/urfd.yaml"))
    parser.add_argument("--wait-download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = config["preparation"]
    raw, poses = ROOT / config["dataset"]["raw_dir"], ROOT / config["dataset"]["pose_dir"]
    model = yaml.safe_load((ROOT / cfg["pose_config"]).read_text(encoding="utf-8"))["pose"]
    estimator = RTMOPoseEstimator(model_path=ROOT / model["weights_path"],
                                 conf_threshold=model["conf_threshold"], device=cfg["pose_device"])
    if estimator.session is None:
        raise RuntimeError("Pose model unavailable")
    names = [f"{kind}-{i:02d}" for kind, count in (("adl", 40), ("fall", 30)) for i in range(1, count + 1)]
    pending = set(names)
    last_status = 0.0
    while pending:
        progressed = False
        for name in sorted(pending):
            out = poses / f"{name}.npz"
            path = raw / f"{name}-cam0-rgb.zip"
            if out.exists():
                pending.remove(name)
                continue
            if not path.exists():
                continue
            extract(path, out, estimator, cfg)
            pending.remove(name)
            progressed = True
            print(f"[{70-len(pending)}/70] poses {name}", flush=True)
        if pending:
            if not args.wait_download:
                raise FileNotFoundError(f"Missing RGB archives: {sorted(pending)}")
            if time.monotonic() - last_status > 30:
                print(f"Waiting for {len(pending)} RGB archives", flush=True)
                last_status = time.monotonic()
            if not progressed:
                time.sleep(5)
    labels = read_labels(raw)
    splits = recording_splits(names, cfg["seed"], cfg["train_fraction"], cfg["val_fraction"])
    print(json.dumps(make_windows(poses, labels, cfg, splits, ROOT / "data/processed/urfd_windows.npz"), indent=2), flush=True)


if __name__ == "__main__":
    main()
