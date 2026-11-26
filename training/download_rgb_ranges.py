"""Finish a slow public MMAction download with independently resumable HTTP ranges."""

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.download_rgb_expansion import download  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/rgb_expansion.yaml").read_text(encoding="utf-8"))
    source = config["sources"]["ntu120"]
    path = ROOT / source["path"]
    if path.exists():
        download(source)
        return
    response = requests.head(source["url"], timeout=60)
    response.raise_for_status()
    total = int(response.headers["Content-Length"])
    partial = path.with_suffix(".pkl.part")
    offset = partial.stat().st_size if partial.exists() else 0
    chunks = path.parent / (path.stem + ".ranges")
    chunks.mkdir(parents=True, exist_ok=True)
    size = 32 * 1024 * 1024
    ranges = [(start, min(start + size, total) - 1) for start in range(offset, total, size)]

    def fetch(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        out = chunks / f"{start:012d}-{end:012d}.part"
        for attempt in range(5):
            have = out.stat().st_size if out.exists() else 0
            if have == end - start + 1:
                return out
            try:
                with requests.get(source["url"], headers={"Range": f"bytes={start+have}-{end}"},
                                  stream=True, timeout=90) as result:
                    result.raise_for_status()
                    expected = f"bytes {start+have}-{end}/{total}"
                    if result.status_code != 206 or result.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor requested byte range")
                    with out.open("ab") as stream:
                        for block in result.iter_content(1024 * 1024):
                            stream.write(block)
                if out.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                print(f"range {start}-{end} complete", flush=True)
                return out
            except (requests.RequestException, ValueError, OSError):
                if attempt == 4:
                    raise
        raise RuntimeError("Range failed")

    print(f"Preserving {offset/1e6:.1f} MB; fetching {len(ranges)} remaining ranges", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(fetch, ranges))
    with partial.open("ab") as stream:
        for item in downloaded:
            with item.open("rb") as incoming:
                shutil.copyfileobj(incoming, stream)
    if partial.stat().st_size != total:
        raise ValueError("Assembled file length mismatch")
    partial.replace(path)
    download(source)  # records the complete file SHA-256 in the standard manifest
    for item in downloaded:
        item.unlink()  # only temporary chunk files created by this invocation


if __name__ == "__main__":
    main()
