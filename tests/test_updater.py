"""In-app update: version compare, GitHub release parsing, safe download, install guard."""

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Optional

import pytest

import src.update.updater as upd
from src.update.updater import STATE_ERROR, STATE_READY, UpdateConfig, Updater, is_newer
from src.version import __version__

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = b"MZ fake installer " * 1000
ASSET_URL = "https://github.com/QuocCuong66/Project-monitoring-model/releases/download/v9.9.9/Fall_and_Stroke_Warning_System_Setup.exe"


class FakeResponse:
    def __init__(self, status: int = 200, json_data: Any = None, body: bytes = b"", url: str = "") -> None:
        self.status_code = status
        self._json = json_data
        self._body = body
        self.url = url
        self.headers = {"Content-Length": str(len(body))} if body else {}

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int = 1) -> Any:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class FakeGitHub:
    """Stands in for requests.Session: serves one release and its installer."""

    def __init__(self, tag: str = "v9.9.9", digest: Optional[str] = None, final_url: str = ASSET_URL,
                 asset_url: str = ASSET_URL, size: Optional[int] = None) -> None:
        self.release = {
            "tag_name": tag, "name": f"Release {tag}", "body": "- Two alert streams\n- Webcam reconnect",
            "published_at": "2026-10-04T10:00:00Z", "html_url": "https://github.com/x/y/releases/tag/" + tag,
            "assets": [
                {"name": "source.zip", "browser_download_url": "https://github.com/x.zip", "size": 1},
                {"name": "Fall_and_Stroke_Warning_System_Setup_v9.9.9.exe", "browser_download_url": asset_url + "?v",
                 "size": len(PAYLOAD)},
                {"name": "Fall_and_Stroke_Warning_System_Setup.exe", "browser_download_url": asset_url,
                 "size": size if size is not None else len(PAYLOAD),
                 "digest": digest or "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()},
            ],
        }
        self.final_url = final_url

    def get(self, url: str, **kw: Any) -> FakeResponse:
        if url.endswith("/releases/latest"):
            return FakeResponse(json_data=self.release)
        return FakeResponse(body=PAYLOAD, url=self.final_url)


def _updater(tmp_path: Path, gh: FakeGitHub, current: str = "0.3.0") -> Updater:
    cfg = UpdateConfig(preferred_asset_name="Fall_and_Stroke_Warning_System_Setup.exe")
    return Updater(config=cfg, current_version=current, session=gh, download_dir=tmp_path,  # type: ignore[arg-type]
                   exit_process=lambda: None)


def _wait_done(u: Updater) -> dict[str, Any]:
    for _ in range(200):
        st = u.status()
        if st["state"] != "downloading":
            return st
        time.sleep(0.02)
    return u.status()


def test_version_comparison() -> None:
    assert is_newer("v0.4.0", "0.3.0")
    assert is_newer("0.3.1", "0.3")
    assert not is_newer("0.3.0", "0.3.0")
    assert not is_newer("0.2.9", "0.3.0")
    assert not is_newer("0.4.0-beta.1", "0.4.0")
    assert is_newer("0.4.0", "0.4.0-beta.1")


def test_version_is_the_same_everywhere() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup = (ROOT / "installer" / "setup.iss").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__  # type: ignore[union-attr]
    assert re.search(r'#define MyAppVersion "([^"]+)"', setup).group(1) == __version__  # type: ignore[union-attr]


def test_check_finds_newer_release_and_prefers_fixed_installer_name(tmp_path: Path) -> None:
    st = _updater(tmp_path, FakeGitHub()).check(force=True)
    assert st["update_available"] is True
    assert st["latest"]["version"] == "9.9.9"
    assert st["latest"]["asset_name"] == "Fall_and_Stroke_Warning_System_Setup.exe"
    assert st["latest"]["asset_sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


def test_same_version_is_up_to_date(tmp_path: Path) -> None:
    st = _updater(tmp_path, FakeGitHub(tag="v0.3.0")).check(force=True)
    assert st["update_available"] is False
    with pytest.raises(ValueError):
        _updater(tmp_path, FakeGitHub(tag="v0.3.0")).download()


def test_download_verifies_and_keeps_the_installer(tmp_path: Path) -> None:
    u = _updater(tmp_path, FakeGitHub())
    u.check(force=True)
    u.download()
    st = _wait_done(u)
    assert st["state"] == STATE_READY
    files = list(tmp_path.glob("*.exe"))
    assert len(files) == 1 and files[0].read_bytes() == PAYLOAD


def test_checksum_mismatch_is_rejected_and_deleted(tmp_path: Path) -> None:
    u = _updater(tmp_path, FakeGitHub(digest="sha256:" + "0" * 64))
    u.check(force=True)
    u.download()
    st = _wait_done(u)
    assert st["state"] == STATE_ERROR and "SHA-256" in st["error"]
    assert not list(tmp_path.glob("*.exe")) and not list(tmp_path.glob("*.part"))


def test_redirect_to_a_foreign_host_is_rejected(tmp_path: Path) -> None:
    u = _updater(tmp_path, FakeGitHub(final_url="https://evil.example.com/setup.exe"))
    u.check(force=True)
    u.download()
    st = _wait_done(u)
    assert st["state"] == STATE_ERROR
    assert not list(tmp_path.glob("*.exe"))


def test_asset_outside_github_is_never_downloaded(tmp_path: Path) -> None:
    u = _updater(tmp_path, FakeGitHub(asset_url="http://github.com/setup.exe"))   # not https
    u.check(force=True)
    with pytest.raises(ValueError):
        u.download()


def test_install_refused_when_running_from_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    u = _updater(tmp_path, FakeGitHub())
    u.check(force=True)
    u.download()
    assert _wait_done(u)["state"] == STATE_READY
    monkeypatch.setattr(upd, "is_installed_app", lambda: False)
    with pytest.raises(ValueError, match="mã nguồn"):
        u.install()


def test_install_runs_installer_silently_then_restarts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[list[str]] = []
    exited: list[bool] = []
    monkeypatch.setattr(upd, "is_installed_app", lambda: True)
    monkeypatch.setattr(upd.subprocess, "Popen", lambda args, **kw: launched.append(args))
    u = _updater(tmp_path, FakeGitHub())
    u._exit_process = lambda: exited.append(True)
    u.check(force=True)
    u.download()
    assert _wait_done(u)["state"] == STATE_READY
    u.install()
    script = (tmp_path / "run_update.cmd").read_text(encoding="utf-8")
    assert "/VERYSILENT" in script and "start \"\"" in script
    assert launched and launched[0][0] == "cmd.exe"
    time.sleep(1.8)
    assert exited == [True]


def test_github_unreachable_is_reported_not_raised(tmp_path: Path) -> None:
    import requests

    class Offline:
        def get(self, *a: Any, **k: Any) -> Any:
            raise requests.ConnectionError("offline")
    st = _updater(tmp_path, Offline()).check(force=True)  # type: ignore[arg-type]
    assert st["update_available"] is False and "GitHub" in st["check_error"]
