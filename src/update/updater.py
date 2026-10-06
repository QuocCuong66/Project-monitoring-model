"""Check GitHub Releases for a newer version, download the installer and install it.

Flow (Settings -> "Cập nhật phiên bản"):
1. check()    - ask the GitHub Releases API for the latest release; compare with src.version
2. download() - fetch the installer in the background (progress reported by status()); only from the
                configured GitHub hosts, size checked, SHA-256 checked when GitHub publishes a digest
3. install()  - installed app only: run the Inno Setup installer silently from a small helper script,
                quit this process so its files can be replaced, then the helper starts the new version.
                The installer is per-user (PrivilegesRequired=lowest), so no administrator prompt.

When running from source (python main.py) step 3 is not available: the dashboard tells the user to
update with git or to download the installer.
"""

import hashlib
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import requests

from src.utils.config import load_config
from src.version import __version__

STATE_IDLE = "idle"
STATE_DOWNLOADING = "downloading"
STATE_READY = "ready"
STATE_INSTALLING = "installing"
STATE_ERROR = "error"


def parse_version(text: str) -> tuple[tuple[int, ...], bool]:
    """'v1.2.3' -> ((1, 2, 3), False); '1.3.0-beta.1' -> ((1, 3, 0), True). Pre-releases sort lower."""
    text = (text or "").strip().lstrip("vV")
    core, _, pre = text.partition("-")
    nums = tuple(int(n) for n in re.findall(r"\d+", core)[:4])
    nums = nums + (0,) * (3 - len(nums)) if len(nums) < 3 else nums
    return nums, bool(pre)


def is_newer(candidate: str, current: str) -> bool:
    (c_nums, c_pre), (cur_nums, cur_pre) = parse_version(candidate), parse_version(current)
    if c_nums != cur_nums:
        return c_nums > cur_nums
    return cur_pre and not c_pre        # 1.0.0 is newer than 1.0.0-beta


@dataclass
class UpdateConfig:
    enabled: bool = True
    github_repo: str = "QuocCuong66/elderly-ai-installer"   # public installer repo (source repo is private)
    api_base: str = "https://api.github.com"
    installer_asset_keyword: str = "_Setup"
    preferred_asset_name: str = ""
    allowed_download_hosts: list[str] = field(default_factory=lambda: [
        "github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com",
    ])
    check_on_startup: bool = True
    cache_minutes: float = 60.0
    request_timeout_seconds: float = 15.0
    max_installer_mb: float = 2048.0
    installer_args: list[str] = field(default_factory=lambda: [
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS",
    ])
    user_settings_dir: str = "FallStrokeWarning"

    @classmethod
    def load(cls) -> "UpdateConfig":
        try:
            raw = (load_config("update.yaml") or {}).get("update", {}) or {}
        except Exception:
            raw = {}
        try:
            folder = (load_config("notifications.yaml") or {}).get("user_settings_dir", "FallStrokeWarning")
        except Exception:
            folder = "FallStrokeWarning"
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known, user_settings_dir=folder)


@dataclass
class ReleaseInfo:
    version: str
    tag: str
    name: str
    notes: str
    published_at: str
    html_url: str
    asset_name: Optional[str] = None
    asset_url: Optional[str] = None
    asset_size: int = 0
    asset_sha256: Optional[str] = None


def is_installed_app() -> bool:
    """Running from the installed (PyInstaller) build on Windows, so the installer can replace it."""
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


class Updater:
    """Thread-safe update state shared by the API endpoints."""

    def __init__(
        self,
        config: Optional[UpdateConfig] = None,
        current_version: str = __version__,
        session: Optional[requests.Session] = None,
        download_dir: Optional[Path] = None,
        exit_process: Callable[[], None] = lambda: os._exit(0),
    ) -> None:
        self.cfg = config or UpdateConfig.load()
        self.current_version = current_version
        self.session = session or requests.Session()
        base = Path(os.getenv("APPDATA") or Path.home())
        self.download_dir = download_dir or base / self.cfg.user_settings_dir / "updates"
        self._exit_process = exit_process
        self._lock = threading.RLock()
        self._latest: Optional[ReleaseInfo] = None
        self._checked_at = 0.0
        self._check_error: Optional[str] = None
        self._state = STATE_IDLE
        self._downloaded = 0
        self._total = 0
        self._error: Optional[str] = None
        self._installer_path: Optional[Path] = None

    # ------------------------------------------------------------------ check
    def _pick_asset(self, assets: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        exes = [a for a in assets if str(a.get("name", "")).lower().endswith(".exe")
                and self.cfg.installer_asset_keyword.lower() in str(a.get("name", "")).lower()]
        preferred = [a for a in exes if a.get("name") == self.cfg.preferred_asset_name]
        return (preferred or exes or [None])[0]

    def check(self, force: bool = False) -> dict[str, Any]:
        """Ask GitHub for the latest release (cached for cache_minutes unless force)."""
        with self._lock:
            fresh = time.time() - self._checked_at < self.cfg.cache_minutes * 60
            if not self.cfg.enabled or (fresh and not force and (self._latest or self._check_error)):
                return self.status()
        url = f"{self.cfg.api_base.rstrip('/')}/repos/{self.cfg.github_repo}/releases/latest"
        latest: Optional[ReleaseInfo] = None
        error: Optional[str] = None
        try:
            resp = self.session.get(url, timeout=self.cfg.request_timeout_seconds,
                                    headers={"Accept": "application/vnd.github+json"})
            if resp.status_code == 404:
                error = "Chưa có bản phát hành nào trên GitHub."
            elif resp.status_code == 403:
                error = "GitHub tạm giới hạn số lần kiểm tra. Vui lòng thử lại sau ít phút."
            else:
                resp.raise_for_status()
                data = resp.json()
                asset = self._pick_asset(data.get("assets") or [])
                digest = str((asset or {}).get("digest") or "")
                latest = ReleaseInfo(
                    version=parse_version_text(data.get("tag_name", "")),
                    tag=str(data.get("tag_name", "")),
                    name=str(data.get("name") or data.get("tag_name") or ""),
                    notes=str(data.get("body") or "")[:4000],
                    published_at=str(data.get("published_at") or ""),
                    html_url=str(data.get("html_url") or ""),
                    asset_name=(asset or {}).get("name"),
                    asset_url=(asset or {}).get("browser_download_url"),
                    asset_size=int((asset or {}).get("size") or 0),
                    asset_sha256=digest.split(":", 1)[1].lower() if digest.lower().startswith("sha256:") else None,
                )
        except requests.RequestException:
            error = "Không kết nối được tới GitHub để kiểm tra bản mới (kiểm tra mạng Internet)."
        except (ValueError, KeyError):
            error = "Phản hồi từ GitHub không hợp lệ."
        with self._lock:
            self._checked_at = time.time()
            self._check_error = error
            if latest is not None:
                self._latest = latest
            return self.status()

    # ------------------------------------------------------------------ status
    def update_available(self) -> bool:
        with self._lock:
            return self._latest is not None and is_newer(self._latest.version, self.current_version)

    def status(self) -> dict[str, Any]:
        with self._lock:
            latest = asdict(self._latest) if self._latest else None
            return {
                "current_version": self.current_version,
                "latest": latest,
                "update_available": self.update_available(),
                "checked_at": self._checked_at or None,
                "check_error": self._check_error,
                "can_install": is_installed_app(),
                "state": self._state,
                "downloaded_bytes": self._downloaded,
                "total_bytes": self._total,
                "error": self._error,
            }

    # ------------------------------------------------------------------ download
    def _host_allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        return parts.scheme == "https" and (parts.hostname or "").lower() in {
            h.lower() for h in self.cfg.allowed_download_hosts
        }

    def download(self) -> dict[str, Any]:
        """Start downloading the latest installer in the background."""
        with self._lock:
            if self._state in (STATE_DOWNLOADING, STATE_INSTALLING):
                return self.status()
            release = self._latest
            if release is None or not self.update_available():
                raise ValueError("Không có bản mới để tải. Hãy bấm Kiểm tra trước.")
            if not release.asset_url or not release.asset_name:
                raise ValueError("Bản phát hành mới không có file bộ cài (.exe).")
            if not self._host_allowed(release.asset_url):
                raise ValueError("Đường dẫn tải bộ cài không thuộc máy chủ GitHub được phép.")
            if release.asset_size > self.cfg.max_installer_mb * 1024 * 1024:
                raise ValueError("File bộ cài lớn bất thường, đã từ chối tải.")
            self._state, self._error = STATE_DOWNLOADING, None
            self._downloaded, self._total = 0, release.asset_size
            self._installer_path = None
        threading.Thread(target=self._download_worker, args=(release,), daemon=True, name="UpdateDownload").start()
        return self.status()

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state, self._error = STATE_ERROR, message

    def _download_worker(self, release: ReleaseInfo) -> None:
        assert release.asset_url and release.asset_name
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", release.asset_name)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        target = self.download_dir / f"v{release.version}_{safe_name}"
        part = target.with_suffix(target.suffix + ".part")
        sha = hashlib.sha256()
        try:
            with self.session.get(release.asset_url, stream=True, timeout=self.cfg.request_timeout_seconds) as resp:
                # Redirects (GitHub -> release asset storage) must stay on GitHub hosts too
                if not self._host_allowed(resp.url):
                    self._fail("Bộ cài bị chuyển hướng tới máy chủ không được phép, đã huỷ tải.")
                    return
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or release.asset_size or 0)
                with self._lock:
                    self._total = total
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        sha.update(chunk)
                        with self._lock:
                            self._downloaded += len(chunk)
        except (requests.RequestException, OSError):
            part.unlink(missing_ok=True)
            self._fail("Tải bộ cài thất bại (mất mạng hoặc hết dung lượng ổ đĩa). Vui lòng thử lại.")
            return

        size = part.stat().st_size
        if release.asset_size and size != release.asset_size:
            part.unlink(missing_ok=True)
            self._fail("File tải về không đủ dung lượng (bị lỗi khi tải). Vui lòng thử lại.")
            return
        if release.asset_sha256 and sha.hexdigest() != release.asset_sha256:
            part.unlink(missing_ok=True)
            self._fail("Mã kiểm tra (SHA-256) của bộ cài không khớp với GitHub, đã xoá file để an toàn.")
            return
        part.replace(target)
        with self._lock:
            self._installer_path = target
            self._state = STATE_READY

    # ------------------------------------------------------------------ install
    def install(self) -> dict[str, Any]:
        """Run the downloaded installer silently, quit, and let the helper restart the new version."""
        with self._lock:
            if self._state != STATE_READY or self._installer_path is None or not self._installer_path.exists():
                raise ValueError("Chưa tải xong bộ cài.")
            if not is_installed_app():
                raise ValueError("Đang chạy từ mã nguồn: không thể tự cài. Hãy cập nhật bằng git hoặc dùng bộ cài.")
            installer = self._installer_path
            self._state = STATE_INSTALLING
        app_exe = Path(sys.executable).resolve()
        script = self.download_dir / "run_update.cmd"
        args = " ".join(self.cfg.installer_args)
        script.write_text(
            "@echo off\r\n"
            "rem Fall & Stroke Warning System: install the downloaded update, then start the new version\r\n"
            "timeout /t 3 /nobreak >nul\r\n"
            f"\"{installer}\" {args}\r\n"
            f"start \"\" \"{app_exe}\"\r\n",
            encoding="utf-8",
        )
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(["cmd.exe", "/c", str(script)], creationflags=flags, close_fds=True,
                         cwd=str(self.download_dir))

        def _quit() -> None:
            time.sleep(1.5)            # let the HTTP response reach the dashboard first
            self._exit_process()
        threading.Thread(target=_quit, daemon=True, name="UpdateExit").start()
        return self.status()


def parse_version_text(tag: str) -> str:
    """'v0.4.0' -> '0.4.0'."""
    return (tag or "").strip().lstrip("vV")


_updater: Optional[Updater] = None
_updater_lock = threading.Lock()


def get_updater() -> Updater:
    """Process-wide updater used by the API."""
    global _updater
    with _updater_lock:
        if _updater is None:
            _updater = Updater()
        return _updater
