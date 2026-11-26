"""Webcam listing must never open devices other than the one the user selected."""

import sys

import pytest

from src.camera import camera_permission
from src.camera.devices import build_webcam_choices, is_virtual_camera, list_webcam_devices


class _FakeCapture:
    opened: list[int] = []

    def __init__(self, index: int, *_args) -> None:
        _FakeCapture.opened.append(index)
        self.index = index

    def isOpened(self) -> bool:
        return self.index == 0

    def read(self):
        import numpy as np
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeCapture.opened = []


def test_phone_and_software_cameras_are_flagged_virtual() -> None:
    assert is_virtual_camera("REDMI Note 15 Pro (Windows Virtual Camera)")
    assert is_virtual_camera("OBS Virtual Camera")
    assert is_virtual_camera("Some Webcam", device_path="")
    assert not is_virtual_camera("Integrated Camera", device_path=r"\\?\usb#vid_13d3&pid_54b6")


def test_choices_mark_primary_and_extra_cameras() -> None:
    listed = [
        {"device_index": 0, "name": "Integrated Camera", "is_virtual": False},
        {"device_index": 1, "name": "REDMI (Windows Virtual Camera)", "is_virtual": True},
        {"device_index": 2, "name": "USB Cam", "is_virtual": False},
    ]
    choices = build_webcam_choices(listed, active_index=0, extra_indices={2})
    assert [c["is_active"] for c in choices] == [True, False, False]
    assert [c["in_use_by_extra"] for c in choices] == [False, False, True]
    assert choices[1]["is_virtual"] is True


def test_choices_without_names_only_offer_the_selected_camera() -> None:
    choices = build_webcam_choices(None, active_index=3)
    assert [c["device_index"] for c in choices] == [3]
    assert choices[0]["is_active"] is True


def test_permission_check_opens_only_the_selected_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_permission.cv2, "VideoCapture", _FakeCapture)
    ok, _msg, working = camera_permission.test_camera_access(0)
    assert ok and working == [0]
    assert set(_FakeCapture.opened) == {0}


def test_failed_permission_check_does_not_fall_back_to_other_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_permission.cv2, "VideoCapture", _FakeCapture)
    ok, _msg, working = camera_permission.test_camera_access(1)
    assert not ok and working == []
    assert set(_FakeCapture.opened) == {1}


@pytest.mark.skipif(sys.platform != "win32", reason="DirectShow enumeration is Windows-only")
def test_listing_does_not_open_any_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    devices = list_webcam_devices()
    assert devices is not None
    assert _FakeCapture.opened == []
    assert [d["device_index"] for d in devices] == list(range(len(devices)))
