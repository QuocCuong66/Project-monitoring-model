"""Per-user default camera, stored outside the install / source directory.

The default camera may include an RTSP URL with the camera password, so it is kept in
%APPDATA%\\<user_settings_dir>\\camera.yaml (same folder as the Telegram settings):
never written into configs/camera.yaml, never committed to git, never packed into the
installer, and kept across app updates. configs/camera.yaml only holds shipped defaults.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.config import load_config

SETTINGS_FILE = "camera.yaml"


def get_camera_settings_path() -> Path:
    try:
        folder = (load_config("notifications.yaml") or {}).get("user_settings_dir", "FallStrokeWarning")
    except Exception:
        folder = "FallStrokeWarning"
    base = Path(os.getenv("APPDATA") or Path.home())
    return base / folder / SETTINGS_FILE


def load_user_camera_settings(path: Optional[Path] = None) -> dict[str, Any]:
    path = path or get_camera_settings_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def save_user_camera_settings(selection: dict[str, Any], path: Optional[Path] = None) -> Path:
    """Persist the camera chosen in the UI (source_type, device_index, rtsp_url, video_path)."""
    path = path or get_camera_settings_path()
    data = load_user_camera_settings(path)
    data["source_type"] = selection.get("source_type", data.get("source_type", "webcam"))
    if selection.get("device_index") is not None:
        data.setdefault("webcam", {})["device_index"] = int(selection["device_index"])
    if selection.get("rtsp_url"):
        data.setdefault("rtsp", {})["url"] = str(selection["rtsp_url"]).strip()
    if selection.get("video_path"):
        data.setdefault("video", {})["path"] = str(selection["video_path"]).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def merge_camera_config(base: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Overlay the user's saved camera onto the shipped configs/camera.yaml defaults."""
    merged = dict(base or {})
    for key, value in (user or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_camera_config() -> dict[str, Any]:
    return merge_camera_config(load_config("camera.yaml") or {}, load_user_camera_settings())
