"""Download official URFD RGB/depth archives and CSVs, with resume and integrity records."""

import argparse
import hashlib
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.urls.extend(v.strip() for k, v in attrs if k == "href" and v)


def discover(html: str, page: str, rgb: bool = True, depth: bool = True) -> list[str]:
    parser = Links()
    parser.feed(html)
    found: set[str] = set()
    for href in parser.urls:
        url = urljoin(page, href)
        name = Path(urlparse(url).path).name
        if urlparse(url).hostname != urlparse(page).hostname:
            continue
        archive = re.fullmatch(r"(?:fall|adl)-\d{2}-cam[01]-(rgb|d)\.zip", name)
        csv = re.fullmatch(r"(?:(?:fall|adl)-\d{2}-(?:data|acc)|urfall-cam0-(?:falls|adls))\.csv", name)
        if csv or (archive and ((archive[1] == "rgb" and rgb) or (archive[1] == "d" and depth))):
            found.add(url)
    return sorted(found)


def validate(path: Path) -> str:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise ValueError(f"Corrupt or empty archive: {path}")
    elif not path.read_bytes().strip() or b"<html" in path.read_bytes()[:200].lower():
        raise ValueError(f"Invalid CSV: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transfer(url: str, partial: Path, timeout: int, chunk_bytes: int = 8 * 1024 * 1024) -> None:
    """Bound HTTP range responses so slow transfers can resume at any saved byte."""
    with requests.Session() as session:
        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-{offset + chunk_bytes - 1}"}
            with session.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code == 416:
                    total = response.headers.get("Content-Range", "").split("/")[-1]
                    if total.isdigit() and offset == int(total):
                        return
                    raise ValueError("Local partial exceeds remote length")
                response.raise_for_status()
                append = response.status_code == 206
                total = None
                if append:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise ValueError("Server returned an inconsistent Content-Range")
                    total = int(match[3])
                received = 0
                with partial.open("ab" if append else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                expected = response.headers.get("Content-Length")
                if not received or (expected and received != int(expected)):
                    raise ValueError("Incomplete response")
                if total is None or partial.stat().st_size == total:
                    return
                if partial.stat().st_size > total:
                    raise ValueError("Response exceeded remote length")


def fetch(url: str, folder: Path, timeout: int, retries: int, previous: dict) -> dict:
    path = folder / Path(urlparse(url).path).name
    if path.exists() and previous.get("bytes") == path.stat().st_size:
        digest = validate(path)
        if digest == previous.get("sha256"):
            return previous
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries):
        try:
            transfer(url, partial, timeout)
            # Validate the .part using the final extension, without extracting files.
            if path.suffix == ".zip":
                with zipfile.ZipFile(partial) as archive:
                    if not archive.namelist() or archive.testzip() is not None:
                        partial.unlink(missing_ok=True)
                        raise ValueError("Archive CRC failed")
            partial.replace(path)
            return {"url": url, "file": path.name, "bytes": path.stat().st_size, "sha256": validate(path)}
        except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to download {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/urfd.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["dataset"]
    folder = ROOT / cfg["raw_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(cfg["page_url"], timeout=cfg["timeout_seconds"])
    response.raise_for_status()
    (folder / "source.html").write_text(response.text, encoding="utf-8")
    urls = discover(response.text, cfg["page_url"], cfg["download_rgb"], cfg["download_depth"])
    # Get annotations and RGB training inputs first; depth and camera 1 remain
    # queued for the complete archive while pose preparation can already run.
    urls.sort(key=lambda u: (0 if u.endswith(".csv") else 1 if "cam0-rgb.zip" in u else 2, u))
    if not urls:
        raise ValueError("No dataset links found")
    manifest_path = folder / "download_manifest.json"
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    print(f"Downloading {len(urls)} official files to {folder}", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(fetch, url, folder, cfg["timeout_seconds"], cfg["retries"],
                               records.get(Path(urlparse(url).path).name, {})): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
                records[record["file"]] = record
                tmp = manifest_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
                tmp.replace(manifest_path)
                print(f"[{i}/{len(urls)}] {record['file']} {record['bytes']/1e6:.1f} MB; total={sum(r['bytes'] for r in records.values())/1e9:.3f} GB", flush=True)
            except Exception as error:
                failures.append(futures[future])
                print(f"FAILED {futures[future]}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to resume")
    print("All downloads complete and validated.", flush=True)


if __name__ == "__main__":
    main()
