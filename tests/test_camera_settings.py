"""The default camera (may contain an RTSP password) lives in %APPDATA%, never in configs/."""

from pathlib import Path

import pytest

from src.camera import camera_settings
from src.utils.config import get_project_root

RTSP = "rtsp://admin:S3cret@192.168.1.50:554/Streaming/Channels/101"


def test_save_and_reload_rtsp(tmp_path: Path) -> None:
    path = tmp_path / "camera.yaml"
    camera_settings.save_user_camera_settings({"source_type": "rtsp", "device_index": 0, "rtsp_url": f"  {RTSP} "}, path)
    data = camera_settings.load_user_camera_settings(path)
    assert data["source_type"] == "rtsp"
    assert data["rtsp"]["url"] == RTSP
    assert data["webcam"]["device_index"] == 0


def test_settings_path_is_under_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = camera_settings.get_camera_settings_path()
    assert path.is_relative_to(tmp_path)
    assert not path.is_relative_to(get_project_root())


def test_user_settings_override_shipped_defaults() -> None:
    base = {"source_type": "webcam", "webcam": {"device_index": 0, "width": 1280}, "rtsp": {"url": "rtsp://placeholder"}}
    user = {"source_type": "rtsp", "rtsp": {"url": RTSP}, "webcam": {"device_index": 1}}
    merged = camera_settings.merge_camera_config(base, user)
    assert merged["source_type"] == "rtsp"
    assert merged["rtsp"]["url"] == RTSP
    assert merged["webcam"] == {"device_index": 1, "width": 1280}


def test_missing_or_corrupt_file_returns_empty(tmp_path: Path) -> None:
    assert camera_settings.load_user_camera_settings(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("source_type: [unclosed", encoding="utf-8")
    assert camera_settings.load_user_camera_settings(bad) == {}
