"""Prepare subject-disjoint phase-labeled videos using the deployed COCO-17 pose model.

Only files explicitly referenced by the curator's mmaction annotation list are used.
Unreferenced alternate exports and audit-only archives are excluded from fitting.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pose.rtmo_estimator import RTMOPoseEstimator  # noqa: E402
from src.pose.pose_estimator import SinglePassDetectionItem  # noqa: E402
from src.utils.ort_providers import dml_provider  # noqa: E402
from training.prepare_rgb_expansion import Piece, normal_mask  # noqa: E402


def subject_split(subject: int, config: dict) -> int:
    if config.get("test_only"):
        return 2
    memberships = [i for i, name in enumerate(("train", "val", "test"))
                   if subject in config[f"{name}_subjects"]]
    if len(memberships) != 1:
        raise ValueError(f"Unknown/overlapping subject {subject}")
    return memberships[0]


def parse_annotation(line: str, config: dict) -> dict:
    relative, target = line.strip().rsplit(maxsplit=1)
    relative = relative.replace("\\", "/")
    if relative.startswith("/") or ":" in relative or ".." in Path(relative).parts:
        raise ValueError("Unsafe annotated path")
    # The rightmost six fields are independent of underscores in recording names.
    base, label, start, end, subject, camera, dataset = Path(relative).stem.rsplit("_", 6)
    if int(label) != int(target) or float(end) <= float(start):
        raise ValueError("Inconsistent label or time interval")
    return {"relative": relative, "label": int(label), "start": float(start),
            "end": float(end), "subject": int(subject), "camera": camera,
            "recording": base, "group": f"{dataset}:{subject}:{base}",
            "overlong_tail_export": bool(config.get("overlong_tail_export", False)),
            "split": subject_split(int(subject), config)}


def canonical_rows(archive: zipfile.ZipFile, config: dict) -> tuple[list[dict], dict]:
    names = archive.namelist()
    annotation = [n for n in names if Path(n).name.startswith("mmaction2_annotations") and n.endswith(".txt")]
    if len(annotation) != 1:
        raise ValueError("Expected one canonical annotation list")
    by_tail: dict[str, list[str]] = {}
    for name in names:
        if name.lower().endswith((".avi", ".mp4")):
            by_tail.setdefault(Path(name).name, []).append(name)
    rows, seen, skipped, conflicted = [], {}, Counter(), set()
    for line in archive.read(annotation[0]).decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = parse_annotation(line, config)
        matches = [n for n in by_tail.get(Path(row["relative"]).name, [])
                   if n.endswith("/" + row["relative"]) or "/" not in row["relative"]]
        if not matches:
            skipped["missing_annotated_file"] += 1
            continue
        if len(matches) != 1:
            raise ValueError(f"Annotated file ambiguous: {row['relative']}")
        row["member"] = matches[0]
        info = archive.getinfo(matches[0])
        # CRC+length is a fast candidate key only; confirm equality by SHA256.
        key = info.CRC, info.file_size
        if key in seen:
            digest = hashlib.sha256(archive.read(row["member"])).hexdigest()
            previous = seen[key]
            old_digest = hashlib.sha256(archive.read(previous["member"])).hexdigest()
            if digest == old_digest:
                if row["split"] != previous["split"]:
                    raise ValueError("Duplicate content has conflicting split")
                if row["label"] != previous["label"] or previous["member"] in conflicted:
                    conflicted.update((previous["member"], row["member"]))
                    skipped["conflicting_label_duplicate"] += 1
                    continue
                skipped["duplicate_content"] += 1
                continue
        seen[key] = row
        rows.append(row)
    rows = [row for row in rows if row["member"] not in conflicted]
    falling_groups = {r["group"] for r in rows if r["label"] == 1}
    for row in rows:
        # Never count lying after a fall as a benign negative in FPR metrics.
        row["normal_recording"] = row["group"] not in falling_groups
    return rows, {"annotation": annotation[0], "canonical_clips": len(rows),
                  "unreferenced_exports": sum(len(v) for v in by_tail.values()) - len(rows),
                  "duplicates": dict(skipped), "quarantined_conflicting_members": sorted(conflicted)}


def phase_interval(row: dict, duration: float, tolerance: float) -> tuple[float, float]:
    """Trim verified EDF tail exports; reject unexplained duration mismatches.

    Frame-overlap audit establishes that affected EDF files begin at annotated
    start but extend beyond annotated end (duration=end, or EOF truncation).
    Duration alone cannot establish alignment for a new, unaudited source.
    """
    expected = row["end"]-row["start"]
    if abs(duration-expected) <= tolerance:
        return 0., expected
    if row.get("overlong_tail_export") and duration >= expected-tolerance:
        return 0., expected
    raise ValueError(f"Media duration {duration:.3f} disagrees with label {row['start']}..{row['end']}")


def sample_pose(video: Path, model: RTMOPoseEstimator, cfg: dict, row: dict) -> dict[str, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Invalid video or FPS: {video}")
    source_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        start, stop = phase_interval(row, source_frames/fps, cfg["media_duration_tolerance_seconds"])
    except ValueError:
        capture.release()
        raise
    start_frame = int(np.ceil(start*fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-start_frame) > 1:
        capture.release()
        raise ValueError("Video seek failed")
    poses, times, boxes, ambiguous = [], [], [], []
    frame_index, next_time = start_frame, 0.0
    previous = None
    try:
        while True:
            if frame_index/fps >= stop:
                break
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps-start
            frame_index += 1
            if timestamp + .5/fps < next_time:
                continue
            next_time += 1/cfg["target_fps"]
            _, items = model.detect_and_estimate_single_pass(frame)
            usable = [p for p in items if p.confidence >= cfg["person_confidence"]]
            ambiguous.append(len(usable) > 1)
            if usable:
                def priority(item: SinglePassDetectionItem) -> float:
                    b = np.asarray(item.bbox)
                    center = (b[:2] + b[2:]) / 2
                    distance = 0 if previous is None else np.linalg.norm(center-previous) / max(frame.shape[:2])
                    return float(item.confidence-distance)
                person = max(usable, key=priority)
                box = np.asarray(person.bbox, np.float32)
                previous = (box[:2]+box[2:])/2
                poses.append(np.asarray(person.keypoints, np.float32))
                boxes.append(box)
            else:
                poses.append(np.zeros((17, 3), np.float32))
                boxes.append(np.zeros(4, np.float32))
            times.append(timestamp)
    finally:
        capture.release()
    index = np.arange(len(times))
    timeline = np.asarray(times)
    if len(times) and fps < cfg["target_fps"]:
        timeline, index = upsample_timeline(np.asarray(times), cfg["target_fps"])
    return {"pose": np.asarray(poses, np.float32)[index], "timestamps": timeline,
            "bbox": np.asarray(boxes, np.float32)[index], "multiple_people": np.asarray(ambiguous)[index],
            "measured_timestamps": np.asarray(times)[index],
            "source_fps": np.asarray(fps), "decoded_frames": np.asarray(frame_index),
            "source_frames": np.asarray(source_frames), "phase_cropped": np.asarray(True),
            "phase_schema": np.asarray(3), "media_duration": np.asarray(source_frames/fps),
            "media_offset": np.asarray(start)}


def upsample_timeline(times: np.ndarray, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest measured-frame resampling preserves duration for low-FPS video."""
    grid = np.arange(int(np.floor(times[-1]*target_fps))+1)/target_fps
    right = np.clip(np.searchsorted(times, grid), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    index = np.where(np.abs(times[left]-grid) <= np.abs(times[right]-grid), left, right)
    return grid, index


def prepare(name: str, config: dict, model: RTMOPoseEstimator) -> None:
    cfg, source = config["preparation"], config["sources"][name]
    raw = ROOT/config["dataset"]["raw_dir"]/source["filename"]
    receipt = json.loads(raw.with_suffix(".download.json").read_text(encoding="utf-8"))
    if receipt["publisher_md5"] != source["md5"] or raw.stat().st_size != receipt["bytes"]:
        raise ValueError("Verified download receipt missing or inconsistent")
    output = ROOT/config["dataset"]["processed_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (all((output/f"{name}{suffix}").exists() for suffix in (".npz", ".json", ".windows.json"))
            and json.loads((output/f"{name}.json").read_text()).get("preparation_schema") == 3):
        print(name, "already prepared", flush=True)
        return
    cache = ROOT/config["dataset"]["pose_dir"]/name
    videos = ROOT/config["dataset"]["extracted_dir"]/name
    cache.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    piece, metadata, counts = Piece(cfg), [], Counter()
    with zipfile.ZipFile(raw) as archive:
        rows, provenance = canonical_rows(archive, source)
        for index, row in enumerate(rows):
            if row["label"] == cfg["skip_other_label"]:
                counts["unknown_other_label"] += 1
                continue
            key = hashlib.sha256(row["member"].encode()).hexdigest()[:24]
            local, cached = videos/(key+Path(row["member"]).suffix), cache/(key+".npz")
            if cached.exists():
                with np.load(cached, allow_pickle=False) as old:
                    obsolete_crop = "phase_cropped" in old and ("phase_schema" not in old or int(old["phase_schema"]) != 3)
                if obsolete_crop:
                    cached.unlink()  # Only this generated, dataset-owned cache file.
            if not cached.exists():
                if not local.exists():
                    local.write_bytes(archive.read(row["member"]))
                try:
                    data = sample_pose(local, model, cfg, row)
                except ValueError as error:
                    counts["media_duration_or_seek_rejected"] += 1
                    print(name, row["relative"], str(error), flush=True)
                    continue
                np.savez_compressed(cached, **data)
            with np.load(cached, allow_pickle=False) as pack:
                track, times, multiple = pack["pose"], pack["timestamps"], pack["multiple_people"]
                if "phase_cropped" in pack:
                    offset = float(pack["media_offset"])
                    duration = float(pack["media_duration"])
                else:
                    # Reuse raw measured poses from the initial media audit.
                    duration = float(pack["decoded_frames"])/float(pack["source_fps"])
                    try:
                        offset, stop = phase_interval(row, duration,
                                                      cfg["media_duration_tolerance_seconds"])
                    except ValueError:
                        counts["media_duration_or_seek_rejected"] += 1
                        continue
                    selected = (times >= offset) & (times < stop)
                    track, times, multiple = track[selected], times[selected]-offset, multiple[selected]
                if duration > row["end"]-row["start"]+cfg["media_duration_tolerance_seconds"]:
                    counts["tail_export_trimmed"] += 1
                if not len(track):
                    counts["empty_video"] += 1
                    continue
                if multiple.mean() > cfg["max_multiple_person_fraction"]:
                    counts["ambiguous_target"] += 1
                    continue
            allowed = normal_mask(cfg["source_class_map"][row["label"]])
            exact = int(np.flatnonzero(allowed)[0]) if allowed.sum() == 1 else -1
            ends = sorted(set(np.rint(np.linspace(min(len(track), cfg["window_size"]),
                                                 len(track), cfg["windows_per_clip"])).astype(int)))
            for end in ends:
                before = len(piece.x)
                piece.add(track[max(0, end-cfg["window_size"]):end], allowed,
                          row["split"], row["group"], exact)
                if len(piece.x) > before:
                    metadata.append({**row, "pose_file": str(cached.relative_to(ROOT)),
                                     "media_offset_seconds": offset,
                                     "window_end": int(end), "observed_seconds": float(times[end-1]-times[0]),
                                     "short_history": bool(end < cfg["window_size"])})
            counts["decoded_clips"] += 1
            if index % 20 == 0:
                print(name, index+1, "/", len(rows), "windows", len(piece.x), flush=True)
    piece.save(output/f"{name}.npz", {**provenance, "preparation_schema": 3, "clip_counts": dict(counts),
                                     "source": source, "protocol": "subject-disjoint; audit-only sources excluded"})
    (output/f"{name}.windows.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--wait-for-download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/"configs/diverse_v2.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT/config["preparation"]["pose_model_config"]).read_text(encoding="utf-8"))["pose"]
    model = RTMOPoseEstimator(model_path=ROOT/model_cfg["weights_path"],
                              conf_threshold=model_cfg["conf_threshold"], device=config["preparation"]["pose_device"])
    if model.session is None:
        raise RuntimeError("Pose model unavailable")
    # Offline extraction avoids default ORT/OpenCV thread oversubscription.
    # The weights and preprocessing/postprocessing stay identical to deployment.
    cv2.setNumThreads(config["preparation"]["pose_cpu_threads"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["preparation"]["pose_cpu_threads"]
    options.enable_mem_pattern = False
    options.add_free_dimension_override_by_name("batch", 1)
    providers = [dml_provider(), "CPUExecutionProvider"] if model.active_provider == "DmlExecutionProvider" else model.session.get_providers()
    model.session = ort.InferenceSession(str(model.model_path), sess_options=options, providers=providers)
    for name in args.sources or config["sources"]:
        if config["sources"][name].get("audit_only"):
            print("Excluded after raw-data audit:", name, flush=True)
            continue
        receipt = ROOT/config["dataset"]["raw_dir"]/Path(config["sources"][name]["filename"]).with_suffix(".download.json")
        if args.wait_for_download and not receipt.exists():
            print("Waiting for verified download:", name, flush=True)
            while not receipt.exists():
                time.sleep(5)
        prepare(name, config, model)


if __name__ == "__main__":
    main()
