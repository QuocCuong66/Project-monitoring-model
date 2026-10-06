"""List webcams by name WITHOUT opening them.

Opening a capture device just to see whether it exists has side effects: a phone linked to
Windows as a "connected camera" (Phone Link / Mobile devices) wakes up and starts streaming
every time its index is opened. So the device list comes from the DirectShow enumerator
(the same order OpenCV's CAP_DSHOW uses for device indices), which only reads names, and the
only camera ever opened is the one the user selected.
"""

import ctypes
import re
import sys
from ctypes import POINTER, byref, c_long, c_uint32, c_void_p, c_wchar_p
from typing import Any, Optional

# Phone-as-webcam / software cameras: listed, but labelled so they are never picked by accident
_VIRTUAL_CAMERA_PATTERN = re.compile(
    r"virtual|phone|mobile|droidcam|iriun|camo|epoccam|obs|snap camera|xsplit|manycam|"
    r"galaxy|iphone|pixel|xiaomi|redmi|oppo|vivo|realme|huawei|honor|nokia|motorola",
    re.IGNORECASE,
)


def is_virtual_camera(name: str, device_path: Optional[str] = None) -> bool:
    """Phone / software camera (no hardware device path, or a known virtual-camera name)."""
    if _VIRTUAL_CAMERA_PATTERN.search(name or ""):
        return True
    return device_path is not None and not device_path.strip()


class _GUID(ctypes.Structure):
    _fields_ = [("d1", c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    def __init__(self, text: str) -> None:
        super().__init__()
        ctypes.oledll.ole32.CLSIDFromString(c_wchar_p(text), byref(self))


class _VARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("ptr", c_void_p), ("pad", c_void_p)]


_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"
_IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
_IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
_VT_BSTR = 8


def _method(obj: c_void_p, index: int, *argtypes: Any) -> Any:
    """Bind COM vtable slot `index` of `obj` (returns an HRESULT as a plain int)."""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    proto = ctypes.WINFUNCTYPE(c_long, c_void_p, *argtypes)
    fn = proto(vtbl[index])
    return lambda *args: fn(obj, *args)


def _release(obj: c_void_p) -> None:
    if obj:
        _method(obj, 2)()


def _read_property(bag: c_void_p, name: str) -> Optional[str]:
    var = _VARIANT()
    oleaut32 = ctypes.windll.oleaut32
    oleaut32.VariantInit(byref(var))
    try:
        hr = _method(bag, 3, c_wchar_p, POINTER(_VARIANT), c_void_p)(name, byref(var), None)
        if hr != 0 or var.vt != _VT_BSTR or not var.ptr:
            return None
        return ctypes.wstring_at(var.ptr)
    finally:
        oleaut32.VariantClear(byref(var))


def _enumerate_dshow_video_devices() -> list[dict[str, Any]]:
    ole32 = ctypes.windll.ole32
    hr_init = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED; S_FALSE if already initialised
    devices: list[dict[str, Any]] = []
    dev_enum = c_void_p()
    enum_moniker = c_void_p()
    try:
        hr = ole32.CoCreateInstance(
            byref(_GUID(_CLSID_SYSTEM_DEVICE_ENUM)), None, 1,  # CLSCTX_INPROC_SERVER
            byref(_GUID(_IID_ICREATE_DEV_ENUM)), byref(dev_enum),
        )
        if hr != 0:
            return devices
        hr = _method(dev_enum, 3, POINTER(_GUID), POINTER(c_void_p), c_uint32)(
            byref(_GUID(_CLSID_VIDEO_INPUT_CATEGORY)), byref(enum_moniker), 0)
        if hr != 0 or not enum_moniker:  # S_FALSE (1): no video devices at all
            return devices

        next_moniker = _method(enum_moniker, 3, c_uint32, POINTER(c_void_p), POINTER(c_uint32))
        iid_bag = _GUID(_IID_IPROPERTY_BAG)
        while True:
            moniker = c_void_p()
            fetched = c_uint32(0)
            if next_moniker(1, byref(moniker), byref(fetched)) != 0 or fetched.value == 0:
                break
            bag = c_void_p()
            try:
                # IMoniker::BindToStorage(pbc, pmkToLeft, riid, ppvObj)
                hr = _method(moniker, 9, c_void_p, c_void_p, POINTER(_GUID), POINTER(c_void_p))(
                    None, None, byref(iid_bag), byref(bag))
                name = _read_property(bag, "FriendlyName") if hr == 0 else None
                path = _read_property(bag, "DevicePath") if hr == 0 else None
            finally:
                _release(bag)
                _release(moniker)
            index = len(devices)
            name = name or f"Webcam {index}"
            devices.append({
                "device_index": index,
                "name": name,
                "is_virtual": is_virtual_camera(name, path if path is not None else ""),
            })
    finally:
        _release(enum_moniker)
        _release(dev_enum)
        if hr_init in (0, 1):
            ole32.CoUninitialize()
    return devices


def build_webcam_choices(
    listed: Optional[list[dict[str, Any]]],
    active_index: Optional[int],
    extra_indices: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Webcam choices for the UI: which one the primary camera uses, which are extra cameras.

    `listed` is list_webcam_devices(); when it is None (names unavailable) only the active /
    default camera is offered, since finding the others would mean opening them.
    """
    extra_indices = extra_indices or set()
    if listed is None:
        idx = active_index if active_index is not None else 0
        listed = [{"device_index": idx, "name": f"Webcam {idx}", "is_virtual": False}]
    choices = []
    for dev in listed:
        idx = int(dev["device_index"])
        choices.append({
            "device_index": idx,
            "name": dev["name"],
            "is_virtual": bool(dev.get("is_virtual")),
            "is_active": idx == active_index,
            "in_use_by_extra": idx in extra_indices,
        })
    return choices


def list_webcam_devices() -> Optional[list[dict[str, Any]]]:
    """Webcams in OpenCV CAP_DSHOW index order, without opening any of them.

    Returns None when the names cannot be listed (non-Windows / COM failure); callers then
    show only the configured camera rather than probing indices.
    """
    if sys.platform != "win32":
        return None
    try:
        return _enumerate_dshow_video_devices()
    except Exception:
        return None
