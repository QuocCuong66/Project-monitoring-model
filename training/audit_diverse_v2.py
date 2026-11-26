"""Audit canonical labels, duplicates and media durations without invoking a model.

This stage may inspect held-out media quality, never model performance.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.prepare_diverse_v2 import canonical_rows, phase_interval  # noqa: E402


def local_video(archive: zipfile.ZipFile, row: dict, folder: Path) -> Path:
    path = folder/(hashlib.sha256(row["member"].encode()).hexdigest()[:24]+Path(row["member"]).suffix)
    if not path.exists():
        path.write_bytes(archive.read(row["member"]))
    return path


def frame_at(path: Path, timestamp: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
        success, image = capture.read()
        return cv2.resize(image, (160, 120)).astype(np.float32) if success else None
    finally:
        capture.release()


def audit(name: str, config: dict) -> dict:
    source = config["sources"][name]
    raw = ROOT/config["dataset"]["raw_dir"]
    folder = ROOT/config["dataset"]["extracted_dir"]/name
    folder.mkdir(parents=True, exist_ok=True)
    counts, rejected, overlaps = Counter(), [], []
    with zipfile.ZipFile(raw/source["filename"]) as archive:
        rows, provenance = canonical_rows(archive, source)
        for row in rows:
            if row["label"] == config["preparation"]["skip_other_label"]:
                continue
            video = local_video(archive, row, folder)
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps else -1
            capture.release()
            try:
                phase_interval(row, duration, config["preparation"]["media_duration_tolerance_seconds"])
                counts["consistent"] += 1
            except ValueError:
                counts["mismatch"] += 1
                rejected.append({"relative": row["relative"], "start": row["start"], "end": row["end"], "duration": duration})
        if source.get("overlong_tail_export"):
            for subject, camera in sorted({(r["subject"], r["camera"]) for r in rows}):
                sequence = sorted([r for r in rows if r["subject"] == subject and r["camera"] == camera], key=lambda r: r["start"])
                a, b = sequence[1:3]
                va, vb = local_video(archive, a, folder), local_video(archive, b, folder)
                before, after = frame_at(va, 0), frame_at(vb, 0)
                overlap = frame_at(va, b["start"]-a["start"])
                if overlap is not None and before is not None and after is not None:
                    overlaps.append({"subject": subject, "camera": camera,
                                     "prefix_mse": float(np.mean((before-after)**2)),
                                     "tail_overlap_mse": float(np.mean((overlap-after)**2))})
    result = {"counts": dict(counts), "provenance": provenance, "rejected": rejected, "overlap_checks": overlaps}
    (raw/f"{name}_media_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(name, dict(counts), "overlap_checks", len(overlaps), flush=True)
    return result


def main() -> None:
    config = yaml.safe_load((ROOT/"configs/diverse_v2.yaml").read_text(encoding="utf-8"))
    for name in config["sources"]:
        audit(name, config)


if __name__ == "__main__":
    main()
