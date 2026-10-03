"""Multi-camera mode: per-camera AI state, shared models, camera identity on alerts, camera list API."""

import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from fastapi import HTTPException

from src.action.action_classifier import ActionPrediction
from src.alerts.alert_manager import AlertManager
from src.alerts.event_logger import EventLogger
from src.anomaly.anomaly_detector import AnomalyResult
from src.camera.camera_settings import load_extra_cameras, save_extra_cameras
from src.camera.stream import FramePacket
from src.detection.detector import Detection
from src.pipeline.decoupled_pipeline import DecoupledPipeline
from src.pipeline.multi_camera import (
    CameraConfigError,
    CameraSession,
    MultiCameraMonitor,
    normalize_camera_entry,
)
from src.pose.pose_estimator import SinglePassDetectionItem
from src.risk.risk_engine import RiskAssessment

PERSON_BOX = (200.0, 100.0, 300.0, 400.0)


class _SilentDispatcher:
    """Never reaches the user's real Telegram / webhook settings."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def dispatch(self, event: Any) -> None:
        self.events.append(event)


class FakePose:
    """Sees one standing person when the frame is marked (pixel [0,0] = 255), nobody otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    def detect_and_estimate_single_pass(self, frame: np.ndarray):
        self.calls += 1
        if frame[0, 0, 0] != 255:
            return [], []
        x1, y1, x2, y2 = PERSON_BOX
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 0] = np.linspace(x1 + 10, x2 - 10, 17)
        kp[:, 1] = np.linspace(y1 + 10, y2 - 10, 17)
        kp[:, 2] = 0.9
        return [Detection(bbox=PERSON_BOX, confidence=0.95)], [SinglePassDetectionItem(PERSON_BOX, 0.95, kp)]

    def match_tracks_to_keypoints(self, tracks, items, iou_threshold: float = 0.45):
        return {t.track_id: items[0].keypoints for t in tracks} if items else {}


class FakeAction:
    def predict(self, seq, kinematics=None) -> ActionPrediction:
        return ActionPrediction("standing", 0.9, {"standing": 0.9}, 0.0, 0.0)


class FakeAnomaly:
    def score(self, seq) -> AnomalyResult:
        return AnomalyResult(0.05, False, 0.05, 0.35)


def _base_pipeline(tmp_path: Path) -> DecoupledPipeline:
    return DecoupledPipeline(
        pose_estimator=FakePose(),
        action_classifier=FakeAction(),
        anomaly_detector=FakeAnomaly(),
        alert_manager=AlertManager(logger=EventLogger(log_dir=tmp_path / "events"), dispatcher=_SilentDispatcher()),
    )


def _frame(with_person: bool) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    if with_person:
        frame[0, 0, 0] = 255
    return frame


def _assessment(track_id: int = 1) -> RiskAssessment:
    return RiskAssessment(track_id=track_id, risk_score=0.9, state="ALERT_SENT", state_changed=True,
                          should_alert=True, evidence={"primary_action": "standing"})


# --- Per-camera state, shared models ----------------------------------------

def test_camera_sessions_share_models_but_not_state(tmp_path: Path) -> None:
    base = _base_pipeline(tmp_path)
    a = base.create_camera_session("cam_a", "Phòng A")
    b = base.create_camera_session("cam_b", "Phòng B")

    for session in (a, b):
        # Heavy models are loaded once and called under one lock (no extra VRAM per camera)
        assert session.pose_estimator is base.pose_estimator
        assert session.action_classifier is base.action_classifier
        assert session.anomaly_detector is base.anomaly_detector
        assert session.inference_lock is base.inference_lock
        # One event log and one dispatcher (Telegram / WebSocket) for every camera
        assert session.alert_manager.logger is base.alert_manager.logger
        assert session.alert_manager.dispatcher is base.alert_manager.dispatcher

    # Temporal reasoning state is strictly per camera
    for attr in ("tracker", "sequence_buffer", "risk_engine", "gated_trigger", "motion_gater",
                 "behavior_analyzer", "alert_manager", "shared_state"):
        assert getattr(a, attr) is not getattr(b, attr), attr
    assert a.alert_manager.clip_recorder is not b.alert_manager.clip_recorder
    assert (a.alert_manager.camera_id, a.alert_manager.camera_name) == ("cam_a", "Phòng A")


def test_person_in_one_room_never_appears_in_another(tmp_path: Path) -> None:
    base = _base_pipeline(tmp_path)
    a = base.create_camera_session("cam_a", "Phòng A")
    b = base.create_camera_session("cam_b", "Phòng B")
    a.motion_gater.enabled = b.motion_gater.enabled = False

    t = 1000.0
    for i in range(12):
        a.submit_frame(FramePacket(_frame(True), i, t, "a"))
        a._run_ai_step()
        b.submit_frame(FramePacket(_frame(False), i, t, "b"))
        b._run_ai_step()
        t += 1.0 / 15.0

    assert len(a.shared_state.tracks) == 1
    assert b.shared_state.tracks == {}
    tid = next(iter(a.shared_state.tracks))
    assert len(a.sequence_buffer.get_snapshots(tid)) >= 3
    assert len(b.sequence_buffer.get_snapshots(tid)) == 0


# --- Camera identity on alerts ------------------------------------------------

def test_alerts_carry_camera_and_do_not_collide(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path)
    dispatcher = _SilentDispatcher()
    cam_a = AlertManager(logger=logger, dispatcher=dispatcher, camera_id="cam_a", camera_name="Phòng A")
    cam_b = AlertManager(logger=logger, dispatcher=dispatcher, camera_id="cam_b", camera_name="Phòng B")

    # Same track id, same second, two cameras: two distinct events (cooldowns are per camera)
    evt_a = cam_a.process_assessment(_assessment(1), timestamp=100.0)
    evt_b = cam_b.process_assessment(_assessment(1), timestamp=100.0)
    assert cam_a.process_assessment(_assessment(1), timestamp=101.0) is None  # same camera: still cooling down

    assert evt_a is not None and evt_b is not None
    assert evt_a.event_id != evt_b.event_id
    assert "cam_a" in evt_a.event_id and "cam_b" in evt_b.event_id
    stored = {e["event_id"]: e for e in logger.get_recent_events()}
    assert stored[evt_a.event_id]["camera_name"] == "Phòng A"
    assert stored[evt_b.event_id]["camera_id"] == "cam_b"
    assert evt_a.message == "Possible medical emergency / abnormal behavior detected. Please check the person."


def test_concurrent_cameras_never_lose_events(tmp_path: Path) -> None:
    logger = EventLogger(log_dir=tmp_path)

    def log_many(cam: str) -> None:
        for i in range(25):
            logger.log_event(person_id=i, risk_score=0.9, evidence={}, camera_id=cam)

    threads = [threading.Thread(target=log_many, args=(f"cam_{n}",)) for n in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(logger.get_recent_events(limit=1000)) == 100


def test_fall_clip_name_includes_camera(tmp_path: Path) -> None:
    from src.alerts.clip_recorder import EventClipRecorder

    rec = EventClipRecorder(clips_dir=tmp_path, cfg={"enabled": True, "pre_seconds": 1.0, "post_seconds": 1.0,
                                                     "fps": 10.0, "fall_actions": ["falling"]})
    path = rec.start_clip(3, 10.0, camera_id="cam_bedroom")
    assert path is not None and "_cam_bedroom_trk3" in path


# --- Camera list management ----------------------------------------------------

class FakeStream:
    def __init__(self, cam: dict[str, Any]) -> None:
        self.cam = cam
        self.started = False
        self.released = False

    def start(self) -> bool:
        self.started = True
        return True

    def is_opened(self) -> bool:
        return self.started and not self.released

    def read(self, timeout: float = 0.5) -> Optional[FramePacket]:
        time.sleep(0.01)
        return None

    def release(self) -> None:
        self.released = True


class FakeSession:
    def __init__(self, cam_id: str, name: str) -> None:
        self.cam_id = cam_id
        self.name = name
        self.blur_faces = False
        self.running = False
        self.submitted: list[FramePacket] = []
        self.rendered = 0

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def submit_frame(self, packet: FramePacket) -> None:
        self.submitted.append(packet)

    def render_frame(self, frame: np.ndarray, timestamp: float):
        self.rendered += 1
        return frame, []


def _monitor(saved: Optional[list] = None, max_cameras: int = 2) -> tuple[MultiCameraMonitor, list]:
    persisted: list = []
    monitor = MultiCameraMonitor(
        settings={"max_cameras": max_cameras},
        stream_factory=FakeStream,
        session_factory=FakeSession,
        persist=lambda cams: persisted.append(cams),
    )
    if saved:
        monitor.load_saved(saved)
    return monitor, persisted


def test_add_list_remove_camera() -> None:
    monitor, persisted = _monitor()
    status = monitor.add_camera({"name": "Phòng ngủ", "source_type": "rtsp",
                                 "rtsp_url": "rtsp://admin:secret@10.0.0.5:554/stream1"})
    try:
        cam_id = status["id"]
        assert status["name"] == "Phòng ngủ"
        assert "secret" not in status["rtsp_url"]  # never sent back to the browser
        assert persisted[-1][0]["rtsp_url"] == "rtsp://admin:secret@10.0.0.5:554/stream1"  # saved in full
        session = monitor.get_session(cam_id)
        assert session is not None and session.stream.started and session.pipeline.running

        assert [c["id"] for c in monitor.list_status()] == [cam_id]
        assert monitor.remove_camera(cam_id) is True
        assert session.stream.released and not session.pipeline.running
        assert monitor.list_status() == [] and persisted[-1] == []
        assert monitor.remove_camera(cam_id) is False
    finally:
        monitor.stop_all()


def test_camera_limit_and_webcam_conflicts() -> None:
    monitor, _ = _monitor(max_cameras=2)
    try:
        monitor.add_camera({"source_type": "webcam", "device_index": 1})
        with pytest.raises(CameraConfigError):
            monitor.add_camera({"source_type": "webcam", "device_index": 1})  # already an extra camera
        with pytest.raises(CameraConfigError):
            monitor.add_camera({"source_type": "webcam", "device_index": 0}, primary_webcam_index=0)
        monitor.add_camera({"source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.6/s"})
        with pytest.raises(CameraConfigError):
            monitor.add_camera({"source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.7/s"})  # over max_cameras
        assert monitor.webcam_indices_in_use() == {1}
    finally:
        monitor.stop_all()


def test_invalid_camera_definitions_are_rejected() -> None:
    with pytest.raises(CameraConfigError):
        normalize_camera_entry({"source_type": "rtsp", "rtsp_url": "ftp://x"}, set())
    with pytest.raises(CameraConfigError):
        normalize_camera_entry({"source_type": "thermal"}, set())
    with pytest.raises(CameraConfigError):
        normalize_camera_entry({"id": "main", "source_type": "webcam"}, set())
    with pytest.raises(CameraConfigError):
        normalize_camera_entry({"id": "a b", "source_type": "webcam"}, set())


def test_saved_cameras_restart_and_disabled_ones_stay_off() -> None:
    saved = [
        {"id": "cam_bed", "name": "Phòng ngủ", "source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.5/s"},
        {"id": "cam_kit", "name": "Bếp", "source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.6/s", "enabled": False},
        {"id": "cam_bad", "source_type": "rtsp", "rtsp_url": "not-a-url"},
    ]
    monitor, _ = _monitor(saved=saved, max_cameras=3)
    try:
        assert monitor.get_session("cam_bed") is not None
        assert monitor.get_session("cam_kit") is None
        assert [c["id"] for c in monitor.list_status()] == ["cam_bed", "cam_kit"]
    finally:
        monitor.stop_all()


def test_blur_toggle_reaches_every_camera() -> None:
    monitor, _ = _monitor()
    try:
        status = monitor.add_camera({"source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.5/s"})
        monitor.set_blur(True)
        assert monitor.get_session(status["id"]).pipeline.blur_faces is True
        later = monitor.add_camera({"source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.6/s"})
        assert monitor.get_session(later["id"]).pipeline.blur_faces is True
    finally:
        monitor.stop_all()


def test_session_feeds_ai_every_frame_but_encodes_tiles_at_tile_fps() -> None:
    cam = {"id": "cam_a", "name": "A", "source_type": "rtsp", "rtsp_url": "rtsp://10.0.0.5/s"}
    pipeline = FakeSession("cam_a", "A")
    session = CameraSession(cam, FakeStream(cam), pipeline, tile_fps=2.0, tile_max_width=320)
    for i in range(5):
        session.process_packet(FramePacket(_frame(False), i, float(i), "a"))

    assert len(pipeline.submitted) == 5
    assert pipeline.rendered == 1  # the other frames arrived within the 0.5 s tile period
    jpeg, seq = session.latest_jpeg()
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8" and seq == 1
    status = session.status()
    assert status["online"] is True and status["id"] == "cam_a"


def test_extra_cameras_persist_in_user_settings(tmp_path: Path) -> None:
    path = tmp_path / "camera.yaml"
    cams = [{"id": "cam_bed", "name": "Phòng ngủ", "source_type": "rtsp", "rtsp_url": "rtsp://u:p@h/s"}]
    save_extra_cameras(cams, path)
    assert load_extra_cameras(path) == cams
    save_extra_cameras([], path)
    assert load_extra_cameras(path) == []


# --- API -----------------------------------------------------------------------

def test_camera_api_lists_adds_and_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api.server as srv

    monitor, _ = _monitor()
    monkeypatch.setattr(srv, "multi_camera_monitor", monitor)
    try:
        res = srv.add_extra_camera(srv.ExtraCameraRequest(
            name="Phòng ngủ", source_type="rtsp", rtsp_url="rtsp://10.0.0.5:554/s",
            rtsp_username="admin", rtsp_password="p@ss:word",
        ))
        cam = res["camera"]
        assert "p@ss" not in cam["rtsp_url"] and "****" in cam["rtsp_url"]

        listed = srv.list_cameras()
        assert listed["max_cameras"] == 2 and [c["id"] for c in listed["cameras"]] == [cam["id"]]

        with pytest.raises(HTTPException) as bad:
            srv.add_extra_camera(srv.ExtraCameraRequest(source_type="rtsp", rtsp_url="ftp://nope"))
        assert bad.value.status_code == 400

        assert srv.remove_extra_camera(cam["id"])["status"] == "success"
        with pytest.raises(HTTPException) as missing:
            srv.remove_extra_camera(cam["id"])
        assert missing.value.status_code == 404
    finally:
        monitor.stop_all()
