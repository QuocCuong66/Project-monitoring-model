"""Build COCO-17 mixed-source training data, preserving label uncertainty and split groups."""

import hashlib
import io
import json
import pickle
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.action.action_classifier import ACTION_CLASSES  # noqa: E402
from src.temporal.sequence_features import build_model_sequence  # noqa: E402


def normal_mask(classes: list[str]) -> np.ndarray:
    mask = np.zeros(len(ACTION_CLASSES), dtype=bool)
    mask[[ACTION_CLASSES.index(c) for c in classes]] = True
    return mask


def ucf_identity(name: str) -> tuple[str, str]:
    match = re.search(r"v_([A-Za-z]+)_g(\d+)_c\d+", name)
    if not match:
        raise ValueError(f"Unrecognized UCF group: {name}")
    return match[1], f"{match[1]}_g{match[2]}"


def clip_windows(track: np.ndarray, cfg: dict, source_fps: float) -> list[np.ndarray]:
    """Resample the physical timeline to live 15 Hz before taking windows."""
    count = max(1, int((len(track) - 1) * cfg["target_fps"] / source_fps) + 1)
    index = np.rint(np.arange(count) * source_fps / cfg["target_fps"]).astype(int)
    track = track[np.clip(index, 0, len(track) - 1)]
    length = cfg["window_size"]
    ends = sorted(set(np.rint(np.linspace(min(length, len(track)), len(track), cfg["windows_per_clip"])).astype(int)))
    return [track[max(0, end - length):end] for end in ends]


def annotation_track(annotation: dict) -> np.ndarray:
    kp, score = annotation["keypoint"], annotation["keypoint_score"]
    if kp.ndim != 4 or kp.shape[2:] != (17, 2) or score.shape != kp.shape[:-1]:
        raise ValueError("Expected MMAction2 COCO-17 image coordinates + measured confidence")
    body = int((score > 0.3).sum((1, 2)).argmax())
    return np.concatenate([kp[body], score[body, ..., None]], axis=-1).astype(np.float32)


class Piece:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.x: list[np.ndarray] = []
        self.allowed: list[np.ndarray] = []
        self.split: list[int] = []
        self.group: list[str] = []
        self.y: list[int] = []
        self.skipped: Counter = Counter()

    def add(self, window: np.ndarray, allowed: np.ndarray, split: int, group: str, y: int = -1,
            validate_quality: bool = True) -> None:
        valid = (window[..., 2] >= self.cfg["keypoint_confidence"]).sum(1) >= self.cfg["min_visible_joints"]
        if not np.isfinite(window).all() or (validate_quality and valid.mean() < self.cfg["min_valid_fraction"]):
            self.skipped["pose_quality"] += 1
            return
        self.x.append(build_model_sequence(window, self.cfg["window_size"]))
        self.allowed.append(allowed)
        self.split.append(split)
        self.group.append(group)
        self.y.append(y)

    def save(self, path: Path, extra: dict) -> None:
        if not self.x:
            raise ValueError(f"No usable windows for {path.name}")
        for a in (0, 1, 2):
            for b in range(a + 1, 3):
                ga = {g for g, s in zip(self.group, self.split) if s == a}
                gb = {g for g, s in zip(self.group, self.split) if s == b}
                if ga & gb:
                    raise ValueError("Group leakage")
        np.savez_compressed(path, x=np.stack(self.x), allowed=np.stack(self.allowed),
                            split=np.asarray(self.split, np.int8), group=np.asarray(self.group), y=np.asarray(self.y, np.int64))
        report = {"windows": len(self.x), "unique_groups": len(set(self.group)),
                  "split_counts": {str(s): self.split.count(s) for s in (0, 1, 2)},
                  "skipped": dict(self.skipped), **extra}
        path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(path.stem, json.dumps(report), flush=True)


def prepare_old(folder: Path, cfg: dict) -> None:
    out = folder / "ntu60.npz"
    if out.exists():
        return
    piece = Piece(cfg)
    rng = np.random.default_rng(cfg["seed"])
    with np.load(ROOT / "data/processed/ntu_hard_negatives.npz", allow_pickle=False) as data:
        x, y, split = data["x"], data["y"], data["split"]
        keep = list(np.flatnonzero(split == 1))  # retain the full, unchanged development benchmark
        for cls in range(len(ACTION_CLASSES)):
            candidates = np.flatnonzero((split == 0) & (y == cls))
            keep.extend(rng.choice(candidates, min(len(candidates), cfg["max_old_train_per_class"]), replace=False))
        eye = np.eye(len(ACTION_CLASSES), dtype=bool)
        for i in keep:
            # The original archive has subject-disjoint partitions but no per-window clip IDs.
            piece.add(x[i].astype(np.float32), eye[y[i]], int(split[i]), f"legacy_{split[i]}_{i}", int(y[i]), validate_quality=False)
    piece.save(out, {"origin": "NTU60 legacy labeled archive", "protocol": "Legacy cross-subject development split; previously used for selection"})


def prepare_mmaction(name: str, source: dict, folder: Path, cfg: dict) -> None:
    out = folder / f"{name}.npz"
    if out.exists():
        return
    with (ROOT / source["path"]).open("rb") as stream:
        data = pickle.load(stream)  # fixed official MMAction2 download only
    annotations, splits = data["annotations"], data["split"]
    label_names = (ROOT / source["label_map_path"]).read_text(encoding="utf-8").splitlines() if name == "hmdb51" else []
    print(name, "annotations", len(annotations), "split keys", list(splits), flush=True)
    piece = Piece(cfg)
    safe = normal_mask(cfg["exercise_classes"])
    rng = np.random.default_rng(cfg["seed"])
    if name == "ntu120":
        train_ids = set(splits["xsub_train"])
        test_ids = set(splits["xsub_val"])
        validation_subjects = set(cfg["new_validation_subjects"])
    else:
        train_key = next((key for key in ("split1_train", "train1", "train") if key in splits), None)
        test_key = next((key for key in ("split1_test", "test1", "test") if key in splits), None)
        if train_key is None or test_key is None:
            raise ValueError(f"Unknown split schema: {list(splits)}")
        train_ids, test_ids = set(splits[train_key]), set(splits[test_key])
        candidates: dict[str, list[str]] = {}
        for ann in annotations:
            clip = ann["frame_dir"]
            action, group = ucf_identity(clip) if name == "ucf101" else (str(ann["label"]), clip)
            if clip in train_ids:
                candidates.setdefault(action, []).append(group)
        val_groups: set[str] = set()
        for groups in candidates.values():
            unique = sorted(set(groups))
            rng.shuffle(unique)
            val_groups.update(unique[:max(1, int(len(unique) * 0.2))])
    selected = 0
    for ann in annotations:
        clip, label = ann["frame_dir"], int(ann["label"])
        if name == "ntu120":
            # Only new single-person, intentional actions A061..A105. Never duplicate NTU60.
            if not 60 <= label <= 104:
                continue
            actor = int(re.search(r"P(\d{3})", clip)[1])
            if clip in test_ids:
                if actor <= 40:
                    continue  # old NTU60 development actors are not a fresh test
                split = 2
            elif clip in train_ids:
                split = 1 if actor in validation_subjects else 0
            else:
                continue
            group = f"ntu120_P{actor:03d}"
        else:
            if name == "ucf101":
                action, group = ucf_identity(clip)
                if action not in cfg["ucf_classes"]:
                    continue
            else:
                action = label_names[label]
                if action not in cfg["hmdb_classes"]:
                    continue
                group = clip
            if clip in test_ids:
                split = 2
            elif clip in train_ids:
                split = 1 if group in val_groups else 0
            else:
                continue
        for window in clip_windows(annotation_track(ann), cfg, cfg["source_fps"][name]):
            piece.add(window, safe, split, group)
        selected += 1
        if selected % 1000 == 0:
            print(name, "selected clips", selected, "windows", len(piece.x), flush=True)
    piece.save(out, {"raw_annotations": len(annotations), "selected_clips": selected,
                     "labels": "Known intentional activity; marginal supervision over safe action classes",
                     "protocol": "Actor-disjoint NTU additions; official video/group splits for UCF/HMDB",
                     "source_fps": cfg["source_fps"][name]})


def prepare_urfd(folder: Path, cfg: dict) -> None:
    out = folder / "urfd.npz"
    if out.exists():
        return
    piece = Piece(cfg)
    with np.load(ROOT / "data/processed/urfd_windows.npz", allow_pickle=False) as data:
        for x, allowed, split, group in zip(data["x"], data["allowed"], data["split"], data["group"]):
            piece.add(x.astype(np.float32), allowed, int(split), str(group))
    piece.save(out, {"protocol": "Recording-disjoint; test used in earlier experiments, not untouched"})


def prepare_upfall(folder: Path, cfg: dict) -> None:
    out = folder / "upfall.npz"
    if out.exists():
        return
    piece = Piece(cfg)
    mask = normal_mask(["falling"])
    records = []
    for path in sorted((ROOT / cfg["upfall_dir"]).glob("SUBJECT*.zip")):
        actor = int(re.search(r"SUBJECT(\d+)", path.name)[1])
        split = next(s for s, key in enumerate(("upfall_train_subjects", "upfall_val_subjects", "upfall_test_subjects")) if actor in cfg[key])
        records.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                data = np.genfromtxt(io.StringIO(archive.read(name).decode("utf-8-sig")), delimiter=",", skip_header=1)
                if data.shape[1] != 100:
                    raise ValueError("Expected 33 xyz joints and an impact label")
                impacts = np.flatnonzero(data[:, -1] == 1)
                if len(impacts) == 0 or impacts[0] < 10:
                    continue
                end = int(impacts[0]) + 1  # only first impact; never mark all post-fall lying as falling
                joints = data[max(0, end - cfg["window_size"]):end, :99].reshape(-1, 33, 3)
                xy = joints[:, cfg["upfall_coco_indices"], :2] * np.asarray(cfg["upfall_crop_size"])
                presence = np.isfinite(xy).all(-1)[..., None].astype(np.float32)
                window = np.concatenate([np.nan_to_num(xy), presence], axis=-1).astype(np.float32)
                piece.add(window, mask, split, f"upfall_P{actor}", ACTION_CLASSES.index("falling"))
    piece.save(out, {"protocol": "Subjects 1/2/3 train, 4 val, 5 test; both cameras grouped by subject",
                     "source": "https://zenodo.org/records/12773013", "downloads": records,
                     "conversion": "MediaPipe normalized image xy -> cropped 500x700 COCO-17; z discarded; binary presence, not measured confidence",
                     "limitations": "Only first-impact positive windows; no ADL, no measured visibility, and sampling timestamps unavailable. No false-alarm metric from this source."})


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/rgb_expansion.yaml").read_text(encoding="utf-8"))
    cfg = config["preparation"]
    folder = ROOT / "data/processed/rgb_expansion"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest = folder / "preparation.json"
    if manifest.exists() and json.loads(manifest.read_text())["config_sha256"] != fingerprint:
        raise ValueError("Preparation config changed; choose a new output cache before reusing data")
    prepare_old(folder, cfg)
    prepare_urfd(folder, cfg)
    prepare_upfall(folder, cfg)
    for name, source in config["sources"].items():
        prepare_mmaction(name, source, folder, cfg)
    manifest.write_text(json.dumps({"config_sha256": fingerprint, "config": cfg,
                                    "pieces": [p.name for p in sorted(folder.glob("*.npz"))]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
