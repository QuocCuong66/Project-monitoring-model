"""Download newly labeled staged/OOD video datasets; verify publisher MD5 + ZIP CRC."""

import hashlib
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.download_urfd import transfer  # noqa: E402


def segmented_transfer(url: str, partial: Path, timeout: int) -> None:
    """Resume the contiguous prefix, then fetch independent bounded ranges."""
    response = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout)
    response.raise_for_status()
    if response.status_code != 206:
        transfer(url, partial, timeout)
        return
    total = int(response.headers["Content-Range"].split("/")[-1])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError("Partial file exceeds source")
    chunks = [(start, min(start+8*1024*1024-1, total-1)) for start in range(offset, total, 8*1024*1024)]
    def block(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        destination = partial.with_name(partial.name+f".{start}.range")
        if destination.exists() and destination.stat().st_size == end-start+1:
            return destination
        for attempt in range(4):
            try:
                r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout)
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{total}" or len(r.content) != end-start+1:
                    raise ValueError("Incorrect ranged response")
                destination.write_bytes(r.content)
                return destination
            except Exception:
                if attempt == 3:
                    raise
        raise RuntimeError("Unreachable")
    # map yields in source order; only append verified contiguous blocks.
    with ThreadPoolExecutor(max_workers=4) as pool, partial.open("ab") as stream:
        for path in pool.map(block, chunks):
            stream.write(path.read_bytes())
            stream.flush()
            path.unlink()


def fetch(source: dict, folder: Path, timeout: int) -> dict:
    path = folder / source["filename"]
    if not path.exists():
        partial = path.with_suffix(".zip.part")
        for attempt in range(4):
            try:
                segmented_transfer(source["url"], partial, timeout)
                partial.replace(path)
                break
            except Exception:
                if attempt == 3:
                    raise
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            md5.update(block); sha.update(block)
    if md5.hexdigest() != source["md5"]:
        raise ValueError(f"Publisher checksum mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC failure: {path}")
    record = {"url": source["url"], "bytes": path.stat().st_size,
              "publisher_md5": md5.hexdigest(), "sha256": sha.hexdigest()}
    path.with_suffix(".download.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(path.name, json.dumps(record),flush=True)
    return record


def main() -> None:
    config = yaml.safe_load((ROOT/"configs/diverse_v2.yaml").read_text(encoding="utf-8"))
    folder = ROOT/config["dataset"]["raw_dir"]
    folder.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=config["dataset"]["download_workers"]) as pool:
        list(pool.map(lambda source:fetch(source,folder,config["dataset"]["timeout_seconds"]),config["sources"].values()))


if __name__ == "__main__":
    main()
