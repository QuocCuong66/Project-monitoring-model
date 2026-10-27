"""Configuration loading utilities."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.
    
    Supports normal development as well as PyInstaller frozen executables.
    FSW_PROJECT_ROOT overrides it (macOS .app: the bundle is read-only, app_runner points this
    at a writable folder in Application Support).
    """
    override = os.getenv("FSW_PROJECT_ROOT")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / "configs").exists() or (exe_dir / "frontend").exists():
            return exe_dir
        if hasattr(sys, "_MEIPASS"):
            meipass = Path(sys._MEIPASS)
            if (meipass / "configs").exists() or (meipass / "frontend").exists():
                return meipass
        internal_dir = exe_dir / "_internal"
        if (internal_dir / "configs").exists() or (internal_dir / "frontend").exists():
            return internal_dir
        return exe_dir
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Safely load a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration values.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def load_config(config_name: str = "model.yaml") -> Dict[str, Any]:
    """Load configuration from the configs/ directory.

    Args:
        config_name: Name of YAML file in configs/ (e.g. 'model.yaml').

    Returns:
        Dictionary of settings.
    """
    config_path = get_project_root() / "configs" / config_name
    return load_yaml(config_path)


def get_env(key: str, default: Optional[Any] = None) -> Any:
    """Get environment variable with fallback default."""
    return os.getenv(key, default)


def resolve_model_path(path: Optional[str | Path]) -> Optional[Path]:
    """Resolve model weight paths across dev environments and PyInstaller bundles.
    
    Checks current working directory, project root, _internal/ directory, and PyInstaller temporary folders.
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    root = get_project_root()
    candidates = [
        root / p,
        root / "_internal" / p,
    ]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([
            exe_dir / p,
            exe_dir / "_internal" / p,
        ])
        if hasattr(sys, "_MEIPASS"):
            meipass = Path(sys._MEIPASS)
            candidates.extend([
                meipass / p,
                meipass / "_internal" / p,
            ])
    for c in candidates:
        if c.exists():
            return c.resolve()
    return p

