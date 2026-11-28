"""Dataset fetcher for the benchmark suite.

Usage:
    python benchmark/scripts/download_datasets.py --dataset coco2017_val
    python benchmark/scripts/download_datasets.py --dataset coco2017_val --verify-only

Behavior (per DATASET.md / benchmark policy):
- Checks whether the dataset already exists before downloading anything.
- Attempts a direct download from the official URL.
- If the download fails (network block, timeout, ...), prints the official
  URL and manual download/placement instructions instead of guessing or
  silently skipping.
- Never auto-downloads datasets that require an interactive license
  click-through (e.g. sites requiring signed EULAs). COCO 2017 val +
  keypoints annotations have no such gate, so a direct HTTP(S) download is
  attempted, but is expected to fail on networks that block
  images.cocodataset.org.
- Computes and stores a sha256 of whatever ends up on disk in metadata.json.
  This is NOT validated against an official checksum (COCO does not publish
  one for these archives) -- it exists purely so future runs can detect that
  the local file changed/corrupted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

BENCHMARK_DIR = Path(__file__).resolve().parents[1]


@dataclass
class DatasetFile:
    url: str
    dest_relative: str  # path relative to the dataset's root dir
    extract: bool = False  # if True, extract this zip directly into the dataset root
    check_dir: str = ""  # dir (relative to root) expected after extraction, used to skip re-extracting
    keep_members: tuple = ()  # if non-empty, only keep these members after extracting (rest deleted)


@dataclass
class DatasetSpec:
    name: str
    root: Path
    license_note: str
    files: List[DatasetFile]


def coco2017_val_spec() -> DatasetSpec:
    root = BENCHMARK_DIR / "datasets" / "coco"
    return DatasetSpec(
        name="coco2017_val",
        root=root,
        license_note=(
            "COCO images: mixed Flickr licenses (see https://cocodataset.org/#termsofuse). "
            "Annotations: CC BY 4.0. No login/EULA click-through required."
        ),
        files=[
            DatasetFile(
                url="http://images.cocodataset.org/zips/val2017.zip",
                dest_relative="val2017.zip",
                extract=True,
                # The zip's own top-level entry is "val2017/*.jpg", so extracting
                # into the dataset root (not a "val2017" subdir) is what produces
                # <root>/val2017/*.jpg instead of a doubly-nested val2017/val2017/.
                check_dir="val2017",
            ),
            DatasetFile(
                url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                dest_relative="annotations_trainval2017.zip",
                extract=True,
                check_dir="annotations",
                # The zip contains train+val for instances/captions/person_keypoints;
                # we only need the val person_keypoints file for pose benchmarking.
                keep_members=("annotations/person_keypoints_val2017.json",),
            ),
        ],
    )


DATASET_REGISTRY = {
    "coco2017_val": coco2017_val_spec,
}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, dest: Path, timeout: int = 15) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "benchmark-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        with open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {dest.name}: {pct:5.1f}% ({downloaded / 1e6:.1f}/{total / 1e6:.1f} MB)", end="")
        print()
        tmp_path.rename(dest)


def extract_zip(zip_path: Path, extract_dir: Path, keep_members: tuple = ()) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if keep_members:
            members = [m for m in members if m in keep_members]
        for member in members:
            zf.extract(member, extract_dir)


def print_manual_instructions(spec: DatasetSpec, missing: List[DatasetFile]) -> None:
    print("\n" + "=" * 78)
    print(f"AUTOMATIC DOWNLOAD FAILED for dataset '{spec.name}'.")
    print("This is expected if images.cocodataset.org is blocked on your network.")
    print(f"License/terms: {spec.license_note}")
    print("\nPlease download the following file(s) manually and place them at the")
    print("exact paths shown, then re-run this script with --verify-only:\n")
    for f in missing:
        dest_path = spec.root / f.dest_relative
        print(f"  URL:  {f.url}")
        print(f"  Save to: {dest_path}")
        print()
    print("Tips if the official host is blocked: try a different network/VPN, or")
    print("use the community mirror at https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset")
    print("(verify it matches the official file sizes before trusting it as ground truth).")
    print("=" * 78)


def write_metadata(spec: DatasetSpec) -> None:
    meta = {
        "dataset": spec.name,
        "license_note": spec.license_note,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    for f in spec.files:
        p = spec.root / f.dest_relative
        if p.exists():
            meta["files"].append({
                "path": f.dest_relative,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            })
    meta_path = spec.root / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


def run(dataset_id: str, verify_only: bool) -> int:
    if dataset_id not in DATASET_REGISTRY:
        print(f"Unknown dataset '{dataset_id}'. Available: {list(DATASET_REGISTRY)}")
        return 2

    spec = DATASET_REGISTRY[dataset_id]()
    spec.root.mkdir(parents=True, exist_ok=True)
    missing_after_attempt: List[DatasetFile] = []

    for f in spec.files:
        dest = spec.root / f.dest_relative
        check_dir = spec.root / f.check_dir if f.check_dir else None
        already_extracted = check_dir is not None and check_dir.exists() and any(check_dir.iterdir())

        if dest.exists() or already_extracted:
            print(f"[OK] {f.dest_relative} already present, skipping download.")
        elif verify_only:
            print(f"[MISSING] {f.dest_relative} not found (run without --verify-only to attempt download,")
            print(f"          or place it manually at {dest}, or extract directly into {check_dir}).")
            missing_after_attempt.append(f)
            continue
        else:
            print(f"Attempting download: {f.url}")
            try:
                download_with_progress(f.url, dest)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"  FAILED: {e}")
                missing_after_attempt.append(f)
                continue

        if f.extract and dest.exists() and not already_extracted:
            print(f"Extracting {dest.name} -> {spec.root}")
            extract_zip(dest, spec.root, keep_members=f.keep_members)

    if missing_after_attempt:
        print_manual_instructions(spec, missing_after_attempt)

    write_metadata(spec)
    return 1 if missing_after_attempt else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    parser.add_argument("--verify-only", action="store_true",
                         help="Do not attempt download; only check/checksum what's already on disk.")
    args = parser.parse_args()
    sys.exit(run(args.dataset, args.verify_only))


if __name__ == "__main__":
    main()
