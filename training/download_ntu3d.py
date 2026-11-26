"""Download the publicly published MMAction2 NTU60/120 3D skeleton releases."""

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


def download(name: str, source: dict) -> dict:
    path = ROOT / source["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = requests.head(source["url"], timeout=60)
    header.raise_for_status()
    expected = int(header.headers["Content-Length"])
    if not path.exists() or path.stat().st_size != expected:
        partial = path.with_suffix(".pkl.part")
        for attempt in range(4):
            try:
                transfer(source["url"], partial, 90, chunk_bytes=64 * 1024 * 1024)
                if partial.stat().st_size != expected:
                    raise ValueError("File length differs from source")
                with partial.open("rb") as stream:
                    if stream.read(1) != b"\x80":
                        raise ValueError("Expected a binary pickle, not an HTML error page")
                partial.replace(path)
                break
            except (requests.RequestException, OSError, ValueError):
                if attempt == 3:
                    raise
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {"name": name, "url": source["url"], "bytes": expected, "sha256": digest.hexdigest(),
              "format": "MMAction2 25-joint 3D pickle, not raw .skeleton text", "etag": header.headers.get("ETag")}
    path.with_suffix(".download.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/ntu3d.yaml").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: download(*item), cfg["sources"].items()))


if __name__ == "__main__":
    main()
