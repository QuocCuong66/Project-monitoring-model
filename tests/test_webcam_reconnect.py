"""A webcam taken over by another app must be reopened, not read forever (frozen picture)."""

import time
from typing import Any, Optional

import numpy as np
import pytest

import src.camera.webcam as webcam_mod
from src.camera.webcam import WebcamStream

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeCamera:
    """cv2.VideoCapture stand-in: `script` decides, per open, how many frames arrive before failing."""

    opens: list[int] = []
    script: list[Optional[int]] = []      # frames delivered per open (None = endless)

    def __init__(self, index: int, backend: int) -> None:
        FakeCamera.opens.append(backend)
        self.left = FakeCamera.script.pop(0) if FakeCamera.script else None
        self.open = True

    def isOpened(self) -> bool:
        return self.open

    def set(self, *_: Any) -> bool:
        return True

    def get(self, *_: Any) -> float:
        return 30.0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.left is None:
            return True, FRAME
        if self.left > 0:
            self.left -= 1
            return True, FRAME
        return False, None                  # preempted: "can't grab frame"

    def release(self) -> None:
        self.open = False


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch) -> type[FakeCamera]:
    FakeCamera.opens = []
    FakeCamera.script = []
    monkeypatch.setattr(webcam_mod.cv2, "VideoCapture", FakeCamera)
    return FakeCamera


def _stream(**kw: Any) -> WebcamStream:
    return WebcamStream(device_index=0, warmup_seconds=0.2, reopen_after_seconds=0.3,
                        max_reconnect_delay_seconds=0.2, **kw)


def _pump(s: WebcamStream, seconds: float) -> int:
    got, end = 0, time.time() + seconds
    while time.time() < end:
        ok, _ = s._read_raw_frame()
        got += int(ok)
    return got


def test_camera_taken_over_mid_stream_is_reopened(fake_cv2: type[FakeCamera]) -> None:
    fake_cv2.script = [5, None]             # first handle dies after 5 frames, the reopened one works
    s = _stream()
    s._backends = lambda: [webcam_mod.cv2.CAP_DSHOW]  # type: ignore[method-assign]
    assert s._open_capture()
    s._running = True
    frames = _pump(s, 2.0)
    assert len(fake_cv2.opens) == 2         # reopened once
    assert frames > 20                      # and frames flow again
    assert s.connection_status()["state"] == "connected"


def test_reports_reconnecting_while_the_camera_stays_busy(fake_cv2: type[FakeCamera]) -> None:
    fake_cv2.script = [3] + [0] * 50        # every reopen opens but delivers nothing (still taken)
    s = _stream()
    s._backends = lambda: [webcam_mod.cv2.CAP_DSHOW]  # type: ignore[method-assign]
    assert s._open_capture()
    s._running = True
    _pump(s, 1.5)
    status = s.connection_status()
    assert status["state"] == "reconnecting"
    assert status["reconnect_attempts"] >= 1
    assert "chiếm" in (status["last_error"] or "")


def test_handle_that_opens_without_frames_is_not_accepted(fake_cv2: type[FakeCamera]) -> None:
    fake_cv2.script = [0]
    s = _stream()
    s._backends = lambda: [webcam_mod.cv2.CAP_DSHOW]  # type: ignore[method-assign]
    assert not s._open_capture()


def test_msmf_not_tried_when_directshow_lists_the_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.camera.devices as devices
    monkeypatch.setattr(webcam_mod.sys, "platform", "win32")
    monkeypatch.setattr(devices, "list_webcam_devices", lambda: [{"device_index": 0, "name": "Integrated Camera"}])
    assert WebcamStream(device_index=0, api_preference="dshow")._backends() == [webcam_mod.cv2.CAP_DSHOW]
    monkeypatch.setattr(devices, "list_webcam_devices", lambda: [])
    assert webcam_mod.cv2.CAP_MSMF in WebcamStream(device_index=0, api_preference="dshow")._backends()
