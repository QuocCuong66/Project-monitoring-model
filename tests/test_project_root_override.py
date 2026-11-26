"""FSW_PROJECT_ROOT lets the macOS .app point the project root at a writable folder."""

from pathlib import Path

from src.utils.config import get_project_root


def test_env_override_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FSW_PROJECT_ROOT", str(tmp_path))
    assert get_project_root() == tmp_path


def test_default_without_override(monkeypatch) -> None:
    monkeypatch.delenv("FSW_PROJECT_ROOT", raising=False)
    assert (get_project_root() / "configs").is_dir()
