"""Unit tests for the shared ONNX Runtime provider helpers."""

from typing import Any

import pytest

from src.utils import ort_providers


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: dict[str, Any]) -> None:
    monkeypatch.setattr(ort_providers, "load_config", lambda name: {"runtime": runtime})


def test_defaults_to_discrete_gpu_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, {})
    assert ort_providers.dml_provider_options() == {
        "performance_preference": "high_performance",
        "device_filter": "gpu",
    }


def test_reads_preference_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, {"dml_performance_preference": "Minimum_Power"})
    assert ort_providers.dml_provider_options()["performance_preference"] == "minimum_power"


def test_invalid_preference_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, {"dml_performance_preference": "turbo"})
    assert ort_providers.dml_provider_options()["performance_preference"] == "high_performance"


def test_dml_provider_entry_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, {})
    name, options = ort_providers.dml_provider()
    assert name == "DmlExecutionProvider"
    assert options["device_filter"] == "gpu"
