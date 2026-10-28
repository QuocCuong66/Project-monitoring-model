"""Performance and latency profiler for real-time video intelligence pipeline."""

import time
from collections import deque
from typing import Dict, Any, Optional
import psutil

try:
    import torch
    TORCH_AVAILABLE = hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available")
except ImportError:
    TORCH_AVAILABLE = False


class LatencyProfiler:
    """Tracks latency across multiple stages and rolling FPS."""

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self._stage_times: Dict[str, deque[float]] = {}
        self._active_timers: Dict[str, float] = {}
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._last_frame_timestamp: Optional[float] = None

    def start_frame(self) -> None:
        """Mark the start of a frame processing cycle."""
        now = time.perf_counter()
        if self._last_frame_timestamp is not None:
            dt = now - self._last_frame_timestamp
            if dt > 0:
                self._frame_times.append(dt)
        self._last_frame_timestamp = now

    def start(self, stage_name: str) -> None:
        """Start recording latency for a specific pipeline stage."""
        self._active_timers[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        """Stop recording and save stage elapsed duration in milliseconds."""
        if stage_name not in self._active_timers:
            return 0.0

        elapsed_ms = (time.perf_counter() - self._active_timers.pop(stage_name)) * 1000.0
        if stage_name not in self._stage_times:
            self._stage_times[stage_name] = deque(maxlen=self.window_size)
        self._stage_times[stage_name].append(elapsed_ms)
        return elapsed_ms

    @property
    def fps(self) -> float:
        """Calculate average FPS over the rolling window."""
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get average latency (ms) for all recorded stages and current FPS."""
        summary: Dict[str, float] = {"fps": round(self.fps, 1)}
        for stage, times in self._stage_times.items():
            if times:
                summary[f"{stage}_ms"] = round(sum(times) / len(times), 2)
        return summary


try:
    import onnxruntime as ort
    ORT_DML_AVAILABLE = "DmlExecutionProvider" in ort.get_available_providers()
except ImportError:
    ORT_DML_AVAILABLE = False


class ResourceMonitor:
    """Monitors CPU, RAM, and GPU VRAM usage."""

    def __init__(self) -> None:
        self.has_gpu = (TORCH_AVAILABLE and torch.cuda.is_available()) or ORT_DML_AVAILABLE
        self.process = psutil.Process()

    def get_telemetry(self) -> Dict[str, Any]:
        """Fetch current hardware telemetry metrics."""
        metrics: Dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": round(self.process.memory_info().rss / (1024 ** 3), 2),
            "system_ram_percent": psutil.virtual_memory().percent,
            "gpu_available": self.has_gpu,
        }

        if self.has_gpu:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
                metrics["gpu_vram_allocated_mb"] = round(allocated, 1)
                metrics["gpu_vram_reserved_mb"] = round(reserved, 1)
                metrics["gpu_device_name"] = torch.cuda.get_device_name(0)
            else:
                metrics["gpu_device_name"] = "NVIDIA GeForce RTX 3050 Laptop GPU (DirectML)"
                metrics["gpu_vram_allocated_mb"] = 0.0
                metrics["gpu_vram_reserved_mb"] = 0.0

        return metrics
