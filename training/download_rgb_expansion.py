"""Download official MMAction2 COCO-17 skeleton data compatible with the live model."""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.download_urfd import transfer  # noqa: E402


def download(source: dict[str, str]) -> None:
    path = ROOT / source["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.head(source["url"], timeout=60)
    response.raise_for_status()
    size = int(response.headers["Content-Length"])
    if not path.exists() or path.stat().st_size != size:
        partial = path.with_suffix(".pkl.part")
        for attempt in range(4):
            try:
                transfer(source["url"], partial, 90, chunk_bytes=64 * 1024 * 1024)
                if partial.stat().st_size != size:
                    raise ValueError("Download length mismatch")
                with partial.open("rb") as stream:
                    if stream.read(1) != b"\x80":
                        raise ValueError("Expected binary pickle")
                partial.replace(path)
                break
            except (requests.RequestException, ValueError, OSError):
                if attempt == 3:
                    raise
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {"url": source["url"], "bytes": size, "sha256": digest.hexdigest()}
    path.with_suffix(".download.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(path.name, json.dumps(record), flush=True)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/rgb_expansion.yaml").read_text(encoding="utf-8"))
    for source in cfg["sources"].values():
        if "label_map_url" in source:
            response = requests.get(source["label_map_url"], timeout=60)
            response.raise_for_status()
            label_path = ROOT / source["label_map_path"]
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(response.text, encoding="utf-8")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, cfg["sources"].values()))


if __name__ == "__main__":
    main()
