"""ONNX Runtime execution-provider helpers shared by every inference session."""

from typing import Any

from src.utils.config import load_config

# Values accepted by onnxruntime-directml's "performance_preference" option.
_DML_PREFERENCES = ("default", "high_performance", "minimum_power")


def dml_provider_options() -> dict[str, str]:
    """DirectML options selecting which GPU adapter runs inference.

    DirectML picks adapter 0 by default, which on hybrid-graphics laptops is the
    integrated GPU (e.g. Intel UHD) rather than the discrete NVIDIA/AMD card.
    "high_performance" asks DXGI for the fastest adapter instead.
    """
    runtime_cfg = load_config("inference.yaml").get("runtime", {})
    pref = str(runtime_cfg.get("dml_performance_preference", "high_performance")).lower()
    if pref not in _DML_PREFERENCES:
        pref = "high_performance"
    return {"performance_preference": pref, "device_filter": "gpu"}


def dml_provider() -> tuple[str, dict[str, Any]]:
    """DirectML provider entry ready to pass to ``ort.InferenceSession(providers=...)``."""
    return ("DmlExecutionProvider", dml_provider_options())
